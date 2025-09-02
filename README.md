# KG-GAN: Adversarial Learning for Knowledge Graph Path Reasoning

A GAN-based adversarial learning system for multi-hop reasoning on knowledge graphs.

## Overview

This project implements an adversarial learning framework that combines:
- **Generator (Discoverer)**: Generates reasoning paths in knowledge graphs using A* search + differentiable path generation
- **Discriminator (Ranker)**: Evaluates path quality using SBERT + neural ranking
- **Adversarial Training**: GAN + Reinforcement Learning with ground truth correction mechanism

## Key Features

- **Ground Truth Correction**: Prevents generator-discriminator collusion
- **Intelligent Reward Shaping**: Dynamic weighting based on GT-Discriminator agreement
- **Persistent Query Exploration**: Continues attempts until correct path found
- **Best Model Auto-saving**: Saves models based on comprehensive metrics

## Quick Start

```bash
# Install dependencies
pip install torch transformers sentence-transformers networkx numpy scikit-learn

# Train the adversarial model
python train_adversarial_model.py

# Run inference
python advanced_inference_system.py
```

## Architecture

```
Question → Generator → Candidate Paths → Discriminator → Final Answer
    ↑         ↓                           ↓
    └── Adversarial Training Loop ←── GT Correction ←──┘
```

## Core Components

- `train_adversarial_model.py` - Main training script
- `models/path_discover/` - Generator components
- `models/path_ranker/` - Discriminator components  
- `advanced_inference_system.py` - Production inference system

## Performance

- **Query Success Rate**: 72%+ on multi-hop reasoning
- **Adversarial Detection**: Identifies generator-discriminator collusion cases
- **Path Diversity**: Generates diverse reasoning paths

---

**Note**: This repository contains only code files. Data files and trained models are stored separately due to size constraints.