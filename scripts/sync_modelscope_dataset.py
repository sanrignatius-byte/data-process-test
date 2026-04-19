#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests
from modelscope.hub.api import HubApi


REPO_ID = "IgnatiusMao/M4query_test"
ROOT = Path("/projects/myyyx1/data-process-test")
PACK_DIR = ROOT / "data/03_queries/M4query_v1"
ZIP_PATH = ROOT / "data/03_queries/M4query_delivery_v1.zip"
STATE_PATH = ROOT / "logs/modelscope_sync_state.json"


@dataclass
class Group:
    index: int
    docs: List[str]
    file_count: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_api(timeout: int, max_retries: int) -> HubApi:
    api = HubApi()
    original_create_commit = api.create_commit

    def patched_create_commit(*args, **kwargs):
        kwargs.setdefault("timeout", timeout)
        kwargs.setdefault("max_retries", max_retries)
        return original_create_commit(*args, **kwargs)

    api.create_commit = patched_create_commit
    return api


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "created_at": utc_now(),
        "completed_groups": [],
        "manual_uploads": [],
        "last_error": None,
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def all_local_files(base: Path) -> List[str]:
    return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())


def remote_files(api: HubApi) -> List[str]:
    files: List[str] = []
    page = 1
    while True:
        items = api.get_dataset_files(
            repo_id=REPO_ID,
            recursive=True,
            page_number=page,
            page_size=100,
        )
        if not items:
            break
        files.extend(i["Path"] for i in items if i.get("Type") != "tree")
        if len(items) < 100:
            break
        page += 1
    return sorted(files)


def print_remote_summary(api: HubApi, label: str) -> None:
    local = set(all_local_files(PACK_DIR))
    remote = set(remote_files(api))
    missing = sorted(local - remote)
    extra = sorted(remote - local)
    print(f"[{label}] remote_total={len(remote)}")
    print(f"[{label}] missing={len(missing)}")
    print(f"[{label}] extra={len(extra)}")
    print(
        f"[{label}] missing_by_root="
        f"{dict(Counter((p.split('/', 1)[0] if '/' in p else p) for p in missing))}"
    )
    print(
        f"[{label}] extra_by_root="
        f"{dict(Counter((p.split('/', 1)[0] if '/' in p else p) for p in extra))}"
    )
    print(f"[{label}] root={sorted(p for p in remote if '/' not in p)}")


def compute_groups(doc_root: Path, threshold: int) -> List[Group]:
    groups: List[Group] = []
    current_docs: List[str] = []
    current_total = 0

    docs = sorted([p for p in doc_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    for doc in docs:
        file_count = sum(1 for p in doc.rglob("*") if p.is_file())
        if current_docs and current_total + file_count > threshold:
            groups.append(Group(index=len(groups) + 1, docs=current_docs, file_count=current_total))
            current_docs = []
            current_total = 0
        current_docs.append(doc.name)
        current_total += file_count

    if current_docs:
        groups.append(Group(index=len(groups) + 1, docs=current_docs, file_count=current_total))
    return groups


def locate_file_by_oid(doc_root: Path, docs: List[str], oid: str, size: int) -> Path | None:
    candidates: List[Path] = []
    for doc in docs:
        for path in (doc_root / doc).rglob("*"):
            if path.is_file() and path.stat().st_size == size:
                candidates.append(path)

    for path in candidates:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == oid:
            return path
    return None


def manual_upload(api: HubApi, path: Path) -> None:
    rel = path.relative_to(PACK_DIR / "documents").as_posix()
    print(f"[manual] upload {rel}")
    info = api.upload_file(
        repo_id=REPO_ID,
        repo_type="dataset",
        path_or_fileobj=str(path),
        path_in_repo=f"documents/{rel}",
        commit_message=f"Manual sync for {rel}",
        disable_tqdm=True,
    )
    print(f"[manual] commit={info.commit_url}")


def sync_group(api: HubApi, doc_root: Path, group: Group, state: dict) -> None:
    patterns = [f"{doc}/**" for doc in group.docs]
    print(f"[group {group.index}] docs={group.docs}")
    print(f"[group {group.index}] file_count={group.file_count}")

    attempts = 0
    while True:
        try:
            info = api.upload_folder(
                repo_id=REPO_ID,
                repo_type="dataset",
                folder_path=str(doc_root),
                path_in_repo="documents",
                allow_patterns=patterns,
                commit_message=f"Sync ModelScope document group {group.index}",
                max_workers=8,
            )
            print(f"[group {group.index}] done commit={info.commit_url}")
            state["completed_groups"].append(group.index)
            state["last_error"] = None
            save_state(STATE_PATH, state)
            return
        except requests.exceptions.HTTPError as exc:
            attempts += 1
            message = str(exc)
            print(f"[group {group.index}] http_error attempt={attempts}: {message}")
            match = re.search(r'"oid": "([0-9a-f]{64})", "size": (\d+)', message)
            if not match or attempts > 3:
                state["last_error"] = {
                    "group": group.index,
                    "time": utc_now(),
                    "message": message,
                }
                save_state(STATE_PATH, state)
                raise
            oid = match.group(1)
            size = int(match.group(2))
            target = locate_file_by_oid(doc_root, group.docs, oid, size)
            if target is None:
                state["last_error"] = {
                    "group": group.index,
                    "time": utc_now(),
                    "message": f"Could not map oid={oid} size={size}",
                }
                save_state(STATE_PATH, state)
                raise RuntimeError(f"Could not map oid={oid} size={size} in group {group.index}")
            manual_upload(api, target)
            rel = target.relative_to(ROOT).as_posix()
            if rel not in state["manual_uploads"]:
                state["manual_uploads"].append(rel)
                save_state(STATE_PATH, state)


def upload_zip(api: HubApi) -> None:
    print("[zip] upload start")
    info = api.upload_file(
        repo_id=REPO_ID,
        repo_type="dataset",
        path_or_fileobj=str(ZIP_PATH),
        path_in_repo=ZIP_PATH.name,
        commit_message="Upload rebuilt M4query_delivery_v1.zip",
        disable_tqdm=True,
    )
    print(f"[zip] commit={info.commit_url}")


def try_delete_api_logs(api: HubApi) -> None:
    print("[cleanup] trying to delete api_logs.zip")
    result = api.delete_files(
        repo_id=REPO_ID,
        repo_type="dataset",
        delete_patterns=["api_logs.zip"],
    )
    print(f"[cleanup] result={result}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=450)
    parser.add_argument("--commit-timeout", type=int, default=900)
    parser.add_argument("--commit-retries", type=int, default=5)
    parser.add_argument("--skip-zip", action="store_true")
    parser.add_argument("--skip-delete-api-logs", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("ModelScope Dataset Sync")
    print(f"start={utc_now()}")
    print(f"repo={REPO_ID}")
    print(f"pack_dir={PACK_DIR}")
    print(f"zip={ZIP_PATH}")
    print("=" * 80)

    api = build_api(timeout=args.commit_timeout, max_retries=args.commit_retries)
    state = load_state(STATE_PATH)
    save_state(STATE_PATH, state)

    print_remote_summary(api, "before")

    doc_root = PACK_DIR / "documents"
    groups = compute_groups(doc_root, args.threshold)
    print(f"[plan] threshold={args.threshold} groups={len(groups)}")
    for group in groups:
        print(f"[plan] group {group.index}: file_count={group.file_count} docs={group.docs}")

    completed = set(state["completed_groups"])
    for group in groups:
        if group.index in completed:
            print(f"[skip] group {group.index} already completed")
            continue
        sync_group(api, doc_root, group, state)

    if not args.skip_zip:
        upload_zip(api)

    if not args.skip_delete_api_logs:
        try_delete_api_logs(api)

    print_remote_summary(api, "after")
    print(f"done={utc_now()}")


if __name__ == "__main__":
    main()
