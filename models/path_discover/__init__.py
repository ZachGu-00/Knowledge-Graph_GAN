"""
Path Discovery Module - Beam Search Based Path Generator

这个模块实现了基于Beam Search的智能路径生成器，包含以下核心组件：

1. BeamSearchPathGenerator: 主要的路径生成器
   - 使用Beam Search搜索算法作为骨架
   - 可学习的p_rel_network边缘评分网络
   - 支持多hop路径生成
   - 支持GAN-RL对抗训练

2. 核心特性：
   - 可微分路径生成，支持REINFORCE训练
   - 持续查询探索机制
   - 温度控制的随机探索
   - 与PathRanker判别器的对抗训练

核心算法：
- Beam Search驱动的路径搜索
- p_rel_network学习边缘重要性
- REINFORCE策略梯度优化
- 温度参数控制探索vs利用

支持的功能：
- 多hop路径发现（1-hop到任意hop）
- 随机化探索（训练时使用）
- 可微分路径生成
- 持续探索直到找到正确答案
- 与PathRanker判别器的对抗学习

训练接口：
- REINFORCE策略梯度损失
- GAN-RL对抗训练循环
- 判别器反馈集成
- 智能奖励塑造
"""

from .beam_search_generator import BeamSearchPathGenerator
from .gan_rl_trainer_fixed import GANRLTrainerFixed

__all__ = [
    'BeamSearchPathGenerator',
    'GANRLTrainerFixed'
]

__version__ = '2.0.0'
__author__ = 'Claude Code Assistant'