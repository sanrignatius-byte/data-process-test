# 第一次调试指南（First Run Debug Guide）

> 日期：2026-03-09
> 目标：在当前环境中跑通 enrichment + query 生成管道，排除阻塞项

---

## 零、当前环境诊断

```
Python:        3.11.14 ✅
anthropic SDK: ❌ 未安装
openai SDK:    ❌ 未安装
requests:      2.32.5 ✅
.env:          ❌ 不存在
local_api_logger/: ❌ 不存在
mineru_output/:    ❌ 无图片文件（图片在集群上）
```

### 核心数据文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `data/multimodal_elements.json` | ✅ 4.7M | 1316 元素 |
| `data/latex_reference_graph.json` | ✅ 6.7M | 73 篇引用 DAG |
| `data/latex_hub_multihop_candidates.json` | ✅ 497K | 500 候选 |
| `data/hub_candidates_enriched.json` | ✅ 718K | 206 对（41.2% 映射率） |
| `data/multimodal_elements_enriched.json` | ❌ 未生成 | 需先跑 Step 0 |

---

## 一、安装依赖

```bash
# 选择一种 provider 安装对应 SDK
pip install anthropic          # 方案 A：直连 Anthropic
pip install openai             # 方案 B：OpenAI 或兼容代理
# 方案 C：公司 API 只需 requests（已有）+ local_api_logger 模块
```

---

## 二、配置 API Key

```bash
# 方案 A：Anthropic 直连
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
export $(grep -v '^#' .env | xargs)

# 方案 B：OpenAI
echo 'OPENAI_API_KEY=sk-...' > .env
export $(grep -v '^#' .env | xargs)

# 方案 C：公司 API
echo 'COMPANY_API_KEY=sk-...' > .env
echo 'COMPANY_API_URL=https://yunwu.ai/v1/chat/completions' >> .env
export $(grep -v '^#' .env | xargs)
# 并把 local_api_logger/ 目录放到项目根目录
```

---

## 三、分步调试（从最小到全量）

### Step 0: 验证 API 连通性

```bash
# Anthropic
python -c "
import anthropic
c = anthropic.Anthropic()
r = c.messages.create(model='claude-sonnet-4-5-20250929', max_tokens=10,
    messages=[{'role':'user','content':'Say OK'}])
print(r.content[0].text, '| tokens:', r.usage.input_tokens, r.usage.output_tokens)
"

# OpenAI
python -c "
from openai import OpenAI
c = OpenAI()
r = c.chat.completions.create(model='gpt-4o', max_tokens=10,
    messages=[{'role':'user','content':'Say OK'}])
print(r.choices[0].message.content)
"

# 公司 API
python main.py
```

### Step 1: Dry-run 验证 prompt 构建（不花钱）

```bash
# enrich_elements_modora — 看 prompt 是否正确
python scripts/enrich_elements_modora.py \
    --input data/multimodal_elements.json \
    --output /dev/null \
    --dry-run \
    --limit 3

# generate_multihop_l1_queries — 看候选对 + prompt
python scripts/generate_multihop_l1_queries.py \
    --candidates data/hub_candidates_enriched.json \
    --output /dev/null \
    --dry-run \
    --limit 3 \
    --no-images
```

**检查点**：
- [ ] prompt 内容合理（caption/context 非空）
- [ ] 无 KeyError / TypeError 崩溃
- [ ] element_a / element_b 类型正确

### Step 2: 小批量真实调用（2-5 条，验证端到端）

```bash
# 先跑 element enrichment（--no-images 因为本地没图片）
python scripts/enrich_elements_modora.py \
    --input data/multimodal_elements.json \
    --output data/_test_enriched_5.json \
    --provider anthropic \
    --model claude-sonnet-4-5-20250929 \
    --no-images \
    --limit 5 \
    --delay 0.5

# 检查输出
python -c "
import json
with open('data/_test_enriched_5.json') as f:
    d = json.load(f)
enriched = 0
for doc in d['documents'].values():
    for el in doc.get('elements', {}).values():
        if 'enriched_title' in el:
            enriched += 1
            print(f\"  {el['element_id']}: {el['enriched_title']}\")
print(f'Total enriched: {enriched}')
"
```

**检查点**：
- [ ] `log_run` 输出行出现（`[token_log] Recorded → $X.XXXX ...`）
- [ ] `enriched_title` 字段非空
- [ ] `enriched_content` 是 2-4 句结构化描述
- [ ] `enriched_metadata` 包含 `keywords` 数组
- [ ] `logs/token_usage.db` 已创建

### Step 3: 小批量 query 生成（5 条）

```bash
python scripts/generate_multihop_l1_queries.py \
    --candidates data/hub_candidates_enriched.json \
    --output data/_test_queries_5.jsonl \
    --provider anthropic \
    --model claude-sonnet-4-5-20250929 \
    --no-images \
    --limit 5 \
    --delay 0.5

# 检查 QC 结果
python -c "
import json
with open('data/_test_queries_5.jsonl') as f:
    lines = [json.loads(l) for l in f]
for r in lines:
    status = 'PASS' if r.get('qc_pass') else 'FAIL'
    fails = r.get('qc_failures', [])
    print(f\"  {r['pair_id']} [{r['pair_type']}] {status} {fails}\")
print(f'Total: {len(lines)}, Pass: {sum(1 for r in lines if r.get(\"qc_pass\"))}')
"
```

**检查点**：
- [ ] JSON 解析成功（无 PARSE FAIL）
- [ ] 至少有 query + answer + evidence_spans 字段
- [ ] QC pass 率 > 30%（小样本波动大但不应为 0）
- [ ] token log 已记录

### Step 4: 查看 token 审计

```bash
python src/utils/token_logger.py --all
```

---

## 四、已知问题与解决方案

### 问题 1：图片路径全部解析失败
**现象**：`image_path` 指向集群路径 `/projects/_hdd/myyyx1/...`，本地不存在
**影响**：figure 类元素无法发送图片给 API，退化为纯文本 enrichment
**解决**：
- 短期：使用 `--no-images` 跳过图片（text-only enrichment 也有价值）
- 长期：把 `data/mineru_output/` 从集群 rsync 到本地

### 问题 2：`ModuleNotFoundError: No module named 'anthropic'`
**解决**：`pip install anthropic`

### 问题 3：`ModuleNotFoundError: No module named 'local_api_logger'`
**解决**：把 `local_api_logger/` 目录放到项目根目录。仅 `--provider company` 需要。

### 问题 4：映射率只有 41.2%（500→206）
**原因**：Phase 1/2 标签匹配率 49.8%，很多 LaTeX label 无法映射到 MinerU element
**不是 bug**：这是上游数据质量限制（MinerU 编号与 LaTeX 编号 offset 不一致）
**本轮修复后**：Phase 3 顺序匹配更准确，预期映射率会小幅提升

### 问题 5：`429 insufficient_quota` / `RateLimitError`
**解决**：加大 `--delay`（0.5→1.0），或换有额度的 key

---

## 五、全量运行命令（调试通过后执行）

```bash
# === MoDora Pipeline 全量 ===

# Step 0: Element enrichment（约 1316 元素，预计 $2-4）
python scripts/enrich_elements_modora.py \
    --input data/multimodal_elements.json \
    --output data/multimodal_elements_enriched.json \
    --provider anthropic \
    --model claude-sonnet-4-5-20250929 \
    --no-images \
    --delay 0.3

# Step 1: Hub enrichment（传入 enriched elements）
python scripts/enrich_hub_candidates.py \
    --hub-candidates data/latex_hub_multihop_candidates.json \
    --elements data/multimodal_elements.json \
    --latex-graph data/latex_reference_graph.json \
    --enriched-elements data/multimodal_elements_enriched.json \
    --output data/hub_candidates_enriched_v2.json

# Step 2: Query generation
python scripts/generate_multihop_l1_queries.py \
    --candidates data/hub_candidates_enriched_v2.json \
    --output data/l1_dual_evidence_queries_hub_enriched_v1.jsonl \
    --pass-only \
    --provider anthropic \
    --model claude-sonnet-4-5-20250929 \
    --no-images \
    --delay 0.3

# Step 3: 查看审计报告
python src/utils/token_logger.py --all
```

---

## 六、调试决策树

```
开始
 │
 ├─ pip install 成功？
 │   ├─ 否 → 检查 Python 版本和 pip 权限
 │   └─ 是 ↓
 │
 ├─ API 连通性测试通过？
 │   ├─ 否 → 检查 key / 网络 / quota
 │   └─ 是 ↓
 │
 ├─ dry-run 无报错？
 │   ├─ 否 → 检查数据文件格式（可能上游脚本版本不匹配）
 │   └─ 是 ↓
 │
 ├─ 5 条小批量输出正常？
 │   ├─ 否 → 检查 JSON parse / QC 逻辑
 │   └─ 是 ↓
 │
 ├─ token_usage.db 有记录？
 │   ├─ 否 → 检查 log_run() 调用是否被跳过（dry-run 模式会跳过）
 │   └─ 是 ↓
 │
 └─ 可以全量运行 ✓
```
