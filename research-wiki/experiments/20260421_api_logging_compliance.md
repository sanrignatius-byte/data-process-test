# exp:20260421_api_logging_compliance

**Date**: 2026-04-21  
**Status**: ACTIVE RULE  
**Scope**: all LLM / company-proxy calls in `data-process-test`

---

## 铁律

1. **所有调用 LLM / 公司代理 API 的脚本，必须走 `local_api_logger`。**
2. **API 调用日志必须输出到 `/projects/myyyx1/data-process-test/api_logs`。**
3. `src/utils/token_logger.py` 这类项目内 token 审计只能作为补充，不可替代 `api_logs`。
4. 任何未进入 `api_logs` 的公司代理调用，都视为不合规，需要单独审计。

---

## 标准调用路径

```text
script
  -> src.api.call_llm(...)
  -> local_api_logger.wrap_requests_call(...)
  -> /projects/myyyx1/data-process-test/api_logs
```

对 `company` provider，规范路径应通过 [src/api/__init__.py](/projects/_hdd/myyyx1/data-process-test/src/api/__init__.py) 中的 `call_llm(..., provider="company")` 落到 `local_api_logger`。

---

## 当前要求

- 新增脚本如果需要调用公司代理，必须复用 `src.api.call_llm(..., provider="company")` 或等价的 `local_api_logger` 包装层。
- 实验记录和复盘要单独检查该 job 是否遵守 `local_api_logger -> api_logs`。
- 给领导看的汇报文档不写这类过程约束；只在 wiki 和内部技术记录中维护。

---

## 审计备注

- 2026-04-21 开始把这条规则显式写入 wiki。
- 同日开始追查最近一段时间是否存在绕开 `local_api_logger -> api_logs` 的实际运行入口。
