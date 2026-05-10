# B4 / C5 API 状态确认

**Date**: 2026-05-10
**Verdict**: 🔴 **BLOCKED** — Company API endpoint 当前不可用

---

## 检测结果

| 测试 | 结果 |
|---|---|
| `COMPANY_API_URL` env | ✅ 已配置（`.env` 内） |
| `COMPANY_API_KEY` env | ✅ 已配置 |
| `POST /v1/chat/completions` ping | ❌ **HTTP 401** |
| 错误消息 | `{"error":{"code":"","message":"无效的令牌","type":"new_api_error"}}` |
| Last successful call | `api_logs/calls/gpt-5.4/2026-04/2026-04-21.jsonl` 最后一条 timestamp 2026-04-21T10:35:41 |
| Days since alive | **19 天** (4/21 → 5/10) |

---

## 受影响 todo

| # | TODO | 估计阻塞天数 | 备注 |
|---|---|---:|---|
| **B4** | 全 27209 elements LLM enrich | 19+ | 当前 10988/27209 = 40.4%，余 16221 elements 无法继续 |
| **C5** | 多粒度 enrich (DocResearcher) | 19+ | 与 [claim:C8](../research-wiki/claims/C8_modora_visual_enrichment_net_negative.md) 矛盾，即便 API 活也建议先做 smoke50 再决策 |
| Method C 长链 enrich | — | 已废 | 5/3 已停 |

---

## 不阻塞的项

| 项 | 备注 |
|---|---|
| F-formula (math-aware encoder) | 不需 API，用本地 embedding 模型 |
| smoke50 重跑（B1 Phase 2 后） | 不需 API |
| C2 chunk dilution claim | 不需 API |
| 任何 graph rerank ablation | 不需 API |

---

## 处置建议

1. **联系 cluster / mentor 请求新 API token**：当前 `COMPANY_API_KEY` 19 天前失效，可能是周期性轮换或账号问题
2. **B4 不强推**：[claim:C8](../research-wiki/claims/C8_modora_visual_enrichment_net_negative.md) 已证 visual-style enrich 对 text-style retrieval 净负，即便 API 恢复，B4 大概率不会改变 0.6913 ceiling
3. **C5 等 B1 Phase 2 + smoke50 重跑结果**：如果 formula R@10 突破 0.56，C5 多粒度 enrich 可能成为新方向（不靠 visual 描述，靠 cross-doc summary）；如果不突破，C5 优先级与 C8 一致继续低
4. **替代方案**（仅当紧急时）：`.env` 中 `OPENAI_API_KEY` 配置存在；若 user 同意切到 OpenAI 直连，需要：
   - 改 `src/llm/local_api_logger.py` 的 base_url
   - 重新核算 `data_process_test` budget（OpenAI 直连成本 ~10x company API）
   - **不建议**——除非 mentor 明确同意

---

## Wiki 状态更新

`research-wiki/index.md` Track C 1040-doc 现状表 LLM enrich 行从 "⚠️ 部分 40.4%" → 加注 "API 19 天 dead，blocked since 4/21"。

mentor todo list 草稿（[MENTOR_TODO_DRAFT_20260510.md](MENTOR_TODO_DRAFT_20260510.md)）B4 已标记为 🔴 BLOCKED。

---

## 验证脚本

```bash
python3 -c "
import json, urllib.request
url, key = '', ''
for line in open('.env'):
    if line.startswith('COMPANY_API_URL='): url = line.split('=',1)[1].strip()
    elif line.startswith('COMPANY_API_KEY='): key = line.split('=',1)[1].strip()
body = json.dumps({'model':'gpt-5.4','messages':[{'role':'user','content':'ping'}],'max_tokens':5}).encode()
req = urllib.request.Request(url, data=body, headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'})
try:
    r = urllib.request.urlopen(req, timeout=15); print('OK')
except Exception as e:
    print('FAIL', e)
"
```

下次 user 想验证 API 是否恢复时直接跑这个。
