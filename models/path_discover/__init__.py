"""
Path Discovery Module - A* Based Path Generator

这个模块实现了基于A*搜索的智能路径生成器，包含以下核心组件：

1. AStarPathGenerator: 主要的路径生成器
   - 使用A*搜索算法作为骨架
   - 四个可学习的评分函数：s_rel, s_tran, s_loc, s_tri
   - 支持多hop路径生成
   - 预留GAN-RL训练接口

2. 四个评分函数模型：
   - RelationPredictorNetwork (s_rel): 条件策略网络，根据问题和路径上下文预测最佳关系
   - TransitionModel (s_tran): 一阶马尔可夫模型，确保路径的局部连贯性
   - LocalSemanticModel (s_loc): 语义相似度匹配模型，保持路径与问题的语义对齐
   - TripletConfidenceModel (s_tri): 数据质量评估模型，评估三元组的可信度

核心算法：
- f(n) = g(n) + h(n) 的A*搜索
- edge_score = w1*s_rel + w2*s_tran + w3*s_loc + w4*s_tri
- 优先队列驱动的高效搜索
- 可配置的beam search宽度控制

支持的功能：
- 多hop路径发现（1-hop到任意hop）
- 随机化探索（训练时使用）
- 批量路径生成
- 与现有PathRanker判别器的无缝对接

预留的训练接口：
- 策略梯度损失计算
- GAN-RL对抗训练循环
- 判别器反馈集成
"""

from .astar_path_generator import (
    AStarPathGenerator,
    RelationPredictorNetwork, 
    TransitionModel,
    LocalSemanticModel,
    TripletConfidenceModel,
    SearchNode
)
from .differentiable_path_generator import DifferentiablePathGenerator
from .gan_rl_trainer import GANRLTrainer

__all__ = [
    'AStarPathGenerator',
    'DifferentiablePathGenerator',
    'RelationPredictorNetwork',
    'TransitionModel', 
    'LocalSemanticModel',
    'TripletConfidenceModel',
    'SearchNode',
    'GANRLTrainer'
]

__version__ = '1.0.0'
__author__ = 'Claude Code Assistant'