"""可插拔负样本采样 —— 给对比学习挑干扰项。

三种策略：heuristic（规则/随机）、graph_aware（图感知，目前 stub）。
通过 build_sampler() 工厂函数按 config 实例化。
"""

from src.sampling.negative_sampler import (
    GraphAwareNegativeSampler,
    HeuristicNegativeSampler,
    NegativeSampler,
    build_sampler,
)

__all__ = [
    "NegativeSampler",
    "HeuristicNegativeSampler",
    "GraphAwareNegativeSampler",
    "build_sampler",
]
