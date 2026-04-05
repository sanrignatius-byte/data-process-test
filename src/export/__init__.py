"""训练数据导出层 —— 从 query 到模型可训练数据的最后一公里。

核心类：DatasetBuilder（总指挥）和 DatasetConfig（配置）。
"""

from src.export.dataset_builder import DatasetBuilder, DatasetConfig

__all__ = ["DatasetBuilder", "DatasetConfig"]
