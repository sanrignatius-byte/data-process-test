#!/usr/bin/env python3
"""
大规模多模态论文采集工具 v2 (PDF + LaTeX 严格成对下载)
基于引文网络的雪球采样 (Snowball Sampling via BFS)

改进要点（相对原版）：
  1. collections.deque 替代 list 作 queue，pop 从 O(N) 降到 O(1)
  2. pending_set 双重去重：避免同一 ID 被多篇引文重复加入 queue
  3. 断点续跑：启动时自动加载 seen_arxiv_ids / successful_pairs /
     pending_queue，crash 后直接重启即可
  4. Content-Type 检测替代魔法字节：更可靠地判断 LaTeX 源是否存在
  5. 指数退避重试：3 次机会，避免临时网络抖动丢掉整篇论文
  6. 仅对下载成功的论文展开引文（节省 S2 API 配额）
  7. 状态每 20 篇保存一次（含 pending queue 序列化）
  8. ETA 估算 + 滚动速率显示
"""

import argparse
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------

S2_GRAPH_API = "https://api.semanticscholar.org/graph/v1"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"

USER_AGENT = (
    "Mozilla/5.0 (compatible; LargeDatasetCollector/2.0; "
    "+mailto:your@email.com)"  # arXiv 偏好带 contact 的 UA
)

# arXiv ToS：批量任务至少 3 秒间隔，建议 4 秒
ARXIV_DELAY = 4.0
# S2 API 无 key 时限速更严
S2_DELAY_WITH_KEY = 1.0
S2_DELAY_NO_KEY = 3.5

# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class PaperCollector:
    def __init__(
        self,
        output_dir: Path,
        target_count: int,
        api_key: Optional[str] = None,
        save_every: int = 20,
    ):
        self.output_dir = output_dir
        self.target_count = target_count
        self.api_key = api_key
        self.save_every = save_every

        self.pdf_dir = output_dir / "pdfs"
        self.src_dir = output_dir / "latex_sources"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.src_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})

        # ── 持久化状态 ───────────────────────────────────────────────────────
        self.seen_arxiv_ids: Set[str] = set()   # 已处理（不一定成功）
        self.pending_set: Set[str] = set()       # 当前在 queue 中的 ID（去重用）
        self.successful_pairs: List[str] = []    # 成功配对的 arXiv ID 列表

        # ── 速率限制状态 ─────────────────────────────────────────────────────
        self._last_s2_call: float = 0.0
        self._last_arxiv_call: float = 0.0

        # ── 速率统计（用于 ETA）──────────────────────────────────────────────
        self._start_time: float = 0.0
        self._initial_success_count: int = 0

    # -----------------------------------------------------------------------
    # 状态持久化
    # -----------------------------------------------------------------------

    @property
    def _state_dir(self) -> Path:
        d = self.output_dir / ".state"
        d.mkdir(exist_ok=True)
        return d

    def save_state(self, queue: deque) -> None:
        """将完整采集状态序列化到磁盘（断点续跑用）。"""
        state = {
            "seen_arxiv_ids": sorted(self.seen_arxiv_ids),
            "successful_pairs": self.successful_pairs,
            # 保存队列（顺序有意义，保留 BFS 层级）
            "pending_queue": list(queue),
        }
        tmp = self._state_dir / "state.json.tmp"
        final = self._state_dir / "state.json"
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(final)  # 原子写（避免 crash 时写坏状态文件）

        # 同时输出易读的成功 ID 列表
        (self.output_dir / "collected_pairs.txt").write_text(
            "\n".join(self.successful_pairs), encoding="utf-8"
        )

    def load_state(self) -> deque:
        """从磁盘加载上次的状态，返回已恢复的队列。"""
        state_file = self._state_dir / "state.json"
        if not state_file.exists():
            return deque()

        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            self.seen_arxiv_ids = set(state.get("seen_arxiv_ids", []))
            self.successful_pairs = state.get("successful_pairs", [])
            pending = list(state.get("pending_queue", []))
            self.pending_set = set(pending)
            print(
                f"[Resume] Loaded state: {len(self.seen_arxiv_ids)} seen, "
                f"{len(self.successful_pairs)} succeeded, "
                f"{len(pending)} pending in queue."
            )
            return deque(pending)
        except Exception as e:
            print(f"[Warn] Failed to load state: {e}. Starting fresh.")
            return deque()

    # -----------------------------------------------------------------------
    # 速率控制
    # -----------------------------------------------------------------------

    def _rate_limit(self, domain: str) -> None:
        now = time.time()
        if domain == "s2":
            delay = S2_DELAY_WITH_KEY if self.api_key else S2_DELAY_NO_KEY
            elapsed = now - self._last_s2_call
            if elapsed < delay:
                time.sleep(delay - elapsed)
            self._last_s2_call = time.time()
        elif domain == "arxiv":
            elapsed = now - self._last_arxiv_call
            if elapsed < ARXIV_DELAY:
                time.sleep(ARXIV_DELAY - elapsed)
            self._last_arxiv_call = time.time()

    # -----------------------------------------------------------------------
    # 工具方法
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_arxiv_id(raw_id: str) -> Optional[str]:
        if not raw_id:
            return None
        cleaned = str(raw_id).replace("arXiv:", "").replace("arxiv:", "").strip()
        cleaned = re.sub(r"v\d+$", "", cleaned)  # 去版本号，确保去重
        # 基本格式校验（YYMM.NNNNN 或旧格式 cs/0601001）
        if re.match(r"^\d{4}\.\d{4,5}$", cleaned) or re.match(r"^[a-z\-]+/\d+$", cleaned):
            return cleaned
        return None

    def _get_eta(self) -> str:
        """计算预计剩余时间。"""
        elapsed = time.time() - self._start_time
        done_this_run = len(self.successful_pairs) - self._initial_success_count
        if done_this_run <= 0 or elapsed <= 0:
            return "calculating..."
        rate = done_this_run / elapsed  # papers/sec
        remaining = self.target_count - len(self.successful_pairs)
        if rate <= 0:
            return "unknown"
        eta_sec = remaining / rate
        h, rem = divmod(int(eta_sec), 3600)
        m = rem // 60
        return f"{h}h{m:02d}m"

    # -----------------------------------------------------------------------
    # Semantic Scholar API
    # -----------------------------------------------------------------------

    def fetch_references(self, arxiv_id: str) -> List[str]:
        """通过 Semantic Scholar 获取引文列表（返回清洗后的 arXiv IDs）。"""
        # Step 1: Resolve to S2 paper ID
        self._rate_limit("s2")
        try:
            res = self.session.get(
                f"{S2_GRAPH_API}/paper/arXiv:{arxiv_id}",
                params={"fields": "paperId"},
                timeout=20,
            )
            if res.status_code != 200:
                return []
            s2_id = res.json().get("paperId")
        except Exception as e:
            print(f"  [S2 Error] Resolve {arxiv_id}: {e}")
            return []

        if not s2_id:
            return []

        # Step 2: Paginate through references
        ref_url = f"{S2_GRAPH_API}/paper/{s2_id}/references"
        params: Dict = {"fields": "externalIds", "limit": 500, "offset": 0}
        found: List[str] = []

        while True:
            self._rate_limit("s2")
            try:
                res = self.session.get(ref_url, params=params, timeout=30)
                if res.status_code == 429:
                    wait = int(res.headers.get("Retry-After", 30))
                    print(f"  [S2] Rate limited. Waiting {wait}s…")
                    time.sleep(wait)
                    continue
                if res.status_code != 200:
                    break
                data = res.json().get("data", [])
                if not data:
                    break
                for item in data:
                    ext = (item.get("citedPaper") or {}).get("externalIds") or {}
                    aid = self._normalize_arxiv_id(ext.get("ArXiv", ""))
                    if aid:
                        found.append(aid)
                if len(data) < 500:
                    break
                params["offset"] += 500
            except Exception as e:
                print(f"  [S2 Error] References {arxiv_id}: {e}")
                break

        return found

    # -----------------------------------------------------------------------
    # 下载（原子化 + 重试）
    # -----------------------------------------------------------------------

    def _download_with_retry(
        self,
        url: str,
        dest: Path,
        domain: str,
        max_retries: int = 3,
        validate_fn=None,
    ) -> bool:
        """带指数退避重试的流式下载。

        Args:
            validate_fn: 可选函数，接收 (response, first_chunk) 返回 bool。
                         返回 False 则视为下载无效（不重试）。
        """
        for attempt in range(max_retries):
            self._rate_limit(domain)
            try:
                r = self.session.get(url, stream=True, timeout=60)
                if r.status_code == 404:
                    return False  # 不重试
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 30))
                    time.sleep(wait)
                    continue
                if r.status_code != 200:
                    time.sleep(2 ** attempt)
                    continue

                first_chunk = next(r.iter_content(chunk_size=2048), b"")
                if validate_fn and not validate_fn(r, first_chunk):
                    return False  # 内容无效，不重试

                with open(dest, "wb") as f:
                    f.write(first_chunk)
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                if dest.stat().st_size > 0:
                    return True

                dest.unlink(missing_ok=True)
                time.sleep(2 ** attempt)

            except requests.exceptions.RequestException as e:
                print(f"  [Net Error] {url}: {e} (attempt {attempt + 1}/{max_retries})")
                if dest.exists():
                    dest.unlink(missing_ok=True)
                time.sleep(2 ** attempt * 2)

        return False

    def download_pair(self, arxiv_id: str) -> bool:
        """原子化下载：先获取 LaTeX 源，验证通过后再下载 PDF。"""
        pdf_out = self.pdf_dir / f"{arxiv_id}.pdf"
        # arXiv e-print 实际 Content-Type 决定真实格式，先用 .bin 存
        src_out = self.src_dir / f"{arxiv_id}.tar.gz"

        if pdf_out.exists() and pdf_out.stat().st_size > 0 \
                and src_out.exists() and src_out.stat().st_size > 0:
            return True  # 已下载，跳过

        # ── Step 1: 下载 LaTeX 源文件 ───────────────────────────────────────
        src_url = ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id)
        src_tmp = self.src_dir / f"{arxiv_id}.tmp"

        def validate_latex_source(resp, first_chunk: bytes) -> bool:
            """Content-Type 优先；魔法字节作补充。"""
            ct = resp.headers.get("Content-Type", "").lower()
            # PDF-only 提交或 HTML 错误页
            if "pdf" in ct or "html" in ct:
                return False
            # 有些旧论文 Content-Type 是 application/octet-stream，
            # 这时用魔法字节兜底
            if "pdf" not in ct:
                if first_chunk.startswith(b"%PDF-"):
                    return False  # 降级为 PDF 提交
                if b"<!DOCTYPE" in first_chunk[:200] or b"<html" in first_chunk[:200]:
                    return False  # HTML 错误页
            return True

        ok = self._download_with_retry(
            src_url, src_tmp, "arxiv", validate_fn=validate_latex_source
        )
        if not ok:
            src_tmp.unlink(missing_ok=True)
            return False

        # ── Step 2: 下载 PDF ─────────────────────────────────────────────────
        pdf_url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
        ok = self._download_with_retry(pdf_url, pdf_out, "arxiv")
        if not ok:
            pdf_out.unlink(missing_ok=True)
            src_tmp.unlink(missing_ok=True)
            return False

        # ── 原子提交：两者均成功 ─────────────────────────────────────────────
        src_tmp.replace(src_out)
        return True

    # -----------------------------------------------------------------------
    # 主循环
    # -----------------------------------------------------------------------

    def run_snowball_sampling(self, seed_arxiv_ids: List[str]) -> None:
        """广度优先引文展开 + 配对下载。支持断点续跑。"""
        queue = self.load_state()
        self._start_time = time.time()
        self._initial_success_count = len(self.successful_pairs)

        # 注入种子（已加载的 queue 中可能已含部分种子，去重）
        for raw in seed_arxiv_ids:
            aid = self._normalize_arxiv_id(raw)
            if aid and aid not in self.seen_arxiv_ids and aid not in self.pending_set:
                queue.append(aid)
                self.pending_set.add(aid)

        print(f"\n{'='*64}")
        print(f"Snowball Collection  target={self.target_count}  seeds={len(seed_arxiv_ids)}")
        print(f"S2 API key: {'yes' if self.api_key else 'no (slower)'}")
        print(f"Already collected:  {len(self.successful_pairs)}")
        print(f"Queue depth:        {len(queue)}")
        print(f"{'='*64}\n")

        while queue and len(self.successful_pairs) < self.target_count:
            current_id = queue.popleft()
            self.pending_set.discard(current_id)

            if current_id in self.seen_arxiv_ids:
                continue
            self.seen_arxiv_ids.add(current_id)

            n_done = len(self.successful_pairs)
            eta = self._get_eta()
            print(
                f"[{n_done}/{self.target_count}  ETA:{eta}  Q:{len(queue)}] "
                f"→ {current_id}"
            )

            # ── 下载配对 ──────────────────────────────────────────────────────
            if self.download_pair(current_id):
                self.successful_pairs.append(current_id)
                print(f"  ✓ PDF + LaTeX acquired")

                # 仅对成功下载的论文展开引文（节省 S2 配额）
                refs = self.fetch_references(current_id)
                new_count = 0
                for ref in refs:
                    if ref not in self.seen_arxiv_ids and ref not in self.pending_set:
                        queue.append(ref)
                        self.pending_set.add(ref)
                        new_count += 1
                if new_count:
                    print(f"  → Queued {new_count} new refs (total Q={len(queue)})")
            else:
                print(f"  ✗ Skipped: no LaTeX source or download failed")

            # ── 定期保存 ──────────────────────────────────────────────────────
            if len(self.successful_pairs) % self.save_every == 0:
                self.save_state(queue)

        # 最终保存
        self.save_state(queue)

        elapsed = time.time() - self._start_time
        h, rem = divmod(int(elapsed), 3600)
        m = rem // 60
        added = len(self.successful_pairs) - self._initial_success_count

        print(f"\n{'='*64}")
        print(f"Done.  Collected {len(self.successful_pairs)} pairs total "
              f"(+{added} this run)  in {h}h{m:02d}m")
        if not queue and len(self.successful_pairs) < self.target_count:
            print(
                f"[Warn] Queue exhausted at {len(self.successful_pairs)} pairs. "
                f"Add more seed papers to expand the citation network."
            )
        print(f"Output: {self.output_dir}")
        print(f"{'='*64}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snowball sampling downloader for PDF+LaTeX pairs (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 从 5 篇 fairness/ML 综述展开，收集 5000 对
  python scripts/download_large_dataset.py \\
      --seeds 1908.09635 2010.04053 2110.00168 \\
      --count 5000 --output data/large_dataset --api-key YOUR_KEY

  # 断点续跑（直接重启，状态自动从 .state/state.json 恢复）
  python scripts/download_large_dataset.py \\
      --seeds 1908.09635 --count 5000 --output data/large_dataset
        """,
    )
    parser.add_argument(
        "--seeds", nargs="+", required=True,
        help="初始 Survey arXiv ID 列表（如 1908.09635 2401.12345）"
    )
    parser.add_argument(
        "--count", type=int, default=5000,
        help="目标配对数量（默认 5000）"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/large_multimodal_dataset"),
        help="输出目录"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Semantic Scholar API Key（强烈推荐，速率上限 10x）"
    )
    parser.add_argument(
        "--save-every", type=int, default=20,
        help="每 N 篇保存一次状态（默认 20）"
    )
    args = parser.parse_args()

    collector = PaperCollector(
        output_dir=args.output,
        target_count=args.count,
        api_key=args.api_key,
        save_every=args.save_every,
    )
    collector.run_snowball_sampling(args.seeds)


if __name__ == "__main__":
    main()
