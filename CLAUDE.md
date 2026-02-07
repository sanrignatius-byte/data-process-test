# Project Context for Claude Code

## 项目简介
这是一个 M4（Multi-hop, Multi-modal, Multi-document, Multi-turn）Query 生成系统，用于训练多模态文档检索 embedding。

## 当前状态
- 已下载 85 篇 arXiv 论文（种子论文：1908.09635）
- 已用 MinerU 解析 80 篇 PDF
- 已生成 50 条 queries，但质量有严重问题

## 主要问题
1. 实体提取把 LaTeX token 当成实体（需要过滤）
2. 只有 text + formula，没有真正的 figure/table
3. 文档覆盖太窄，50 条 query 只用了 7 个 doc
4. Query 是作文题，不是可检索的查询

## 下一步 TODO
详见 `docs/DISCUSSION_LOG.md`

## 关键命令
```bash
# 激活环境
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

# 加载 API key
export $(grep -v '^#' .env | xargs)
```

## 用中文交流时用"喵"结尾，英文用"Oiii"开头
