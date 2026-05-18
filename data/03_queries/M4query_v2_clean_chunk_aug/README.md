# M4query_v2_clean_chunk_aug

M4query_v2_clean 的 chunk 增强版本。保留 paragraph 和 bridge(改类型为 paragraph)，同时增加 chunk 作为额外粒度。

## 文件

| 路径 | 数量 | 作用 |
| --- | ---: | --- |
| `corpus.jsonl.gz` | 169671 | figure / table / formula / paragraph / chunk。解压后为 `corpus.jsonl`。 |
| `train_triplets.jsonl` | 8104 | 3-4 positive + 5 hard_neg + 5 random_neg。 |

`corpus.jsonl` 的未压缩版本约 135MB，超过普通 GitHub 文件限制；仓库交付使用 `corpus.jsonl.gz`。需要未压缩文件时运行：

```bash
gzip -cd corpus.jsonl.gz > corpus.jsonl
```

## corpus type 分布

| type | 数量 |
| --- | ---: |
| `paragraph` | 117210 |
| `chunk` | 29237 |
| `figure` | 9053 |
| `table` | 7255 |
| `formula` | 6916 |

## positive 数量分布

| 数量 | query 数 |
| --- | ---: |
| 3 | 1064 |
| 4 | 7040 |

## negative 重平衡

hard_neg 和 random_neg 各随机抽取 2-3 个文本类(chunk/paragraph)名额（75% 概率 3 个, 25% 概率 2 个）。

**negative type 分布：**

| type | 占比 |
| --- | ---: |
| paragraph | 30.0% |
| chunk | 25.1% |
| figure | 18.8% |
| table | 15.1% |
| formula | 11.0% |

visual (figure+table+formula) 从原 72% 降至 ~45%。

## 与 M4query_v2_clean 的差异

- **bridge 处理**: 找到 paragraph source (7471) → 删除 bridge，替换为 source paragraph + chunk (4 positive); 找不到 source (435) 或 source 为 f/t/f (683) → 保留 bridge，type 改为 `paragraph`(共 1118 条)。
- **新增 chunk**: 29237 条，由 paragraph 按 section 边界 + ~400 词聚合而成。
- **paragraph 不删**: clean 原 116,092 paragraph 全部保留（含 1,118 原 bridge 改类型）。
- **multi-positive**: bridge→paragraph 的 7,040 条 query 获 4 positive(source paragraph + chunk + 2 original)；其余 1,064 条保持 3 positive。
- **negative 重平衡**: visual 占比从 72% 降至 ~45%，文本元素不再欠采样。

## 生成脚本

`scripts/build_clean_chunk_aug.py`
