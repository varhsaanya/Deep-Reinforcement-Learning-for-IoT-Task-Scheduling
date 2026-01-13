# Deep Reinforcement Learning for Edge-Cloud IoT Task Scheduling

18D Deep Q-Network implementation in MATLAB for energy-efficient task scheduling across 200+ agricultural IoT sensor nodes. Compares against 7 baselines with 25% latency reduction and 10% throughput gains.

## Project Overview

**Problem**: Dynamically assign sensor processing tasks from dynamic clusters of IoT nodes to 5 edge nodes + cloud fallback while optimizing latency, throughput, energy, and load balance.

**Key Innovation**: 
- 18-dimensional predictive state (current loads, distances, energy scores, load trends)
- Adaptive cloud offloading thresholds via K-Means clustering prediction
- Multi-objective reward balancing energy costs, load variance, and completion time

**Results**: Outperforms RRS, SJFS, LLFS, FL, EA, DRA baselines across 6 metrics.

## Core Features

- **Dynamic Clustering**: K-Means on 5D sensor features + 2D positions (silhouette-optimized)
- **18D State Space**: Cluster position, edge utilizations, distances, energy efficiency, predictive load trends
- **DQN Architecture**: [256→128→64→6] with experience replay, target network, epsilon-greedy
- **Multi-Objective Reward**: Energy costs + load balancing + completion time + predictive variance
- **Cloud Offloading**: Automatic fallback when all edges exceed adaptive thresholds
- **7 Baselines**: Round-robin, shortest-job, least-laxity, capacity-weighted, greedy, dynamic allocation

## Technical Implementation

State (18D): [clusterX, clusterY, avgUtil, loadVar, maxUtil, minUtil, avgDist, minDist,
energyScore, optimalRatio, overloadRatio, distVar, predVar, predMax, predTrend,
loadVar, minUtil, maxUtil]

Action Space: {Edge1, Edge2, Edge3, Edge4, Edge5, Cloud}

Network: Input(18) → FC(256) → ReLU → Drop(0.1) → FC(128) → ReLU → Drop(0.1) → FC(64) → ReLU → Output(6)


**Training**: 10K episodes, ε-decay from 0.1→0.01, batch=32, target sync every 100 steps

## Performance Results

| Metric | DQN | Best Baseline | Improvement |
|--------|-----|---------------|-------------|
| **Load Variance** | 0.021 | 0.089 (DRA) | **76% reduction** |
| **Throughput** | 18.4 | 16.7 (FL) | **+10%** |
| **Latency** | 0.023s | 0.031s (SJFS) | **25% reduction** |
| **Task Time** | 1.42s | 1.89s (LLFS) | **25% reduction** |
| **Resource Util.** | 0.67 | 0.59 (EA) | **+14%** |
| **Energy Eff.** | 2.84 | 2.41 (FL) | **+18%** |

**Cloud Offloads**: <5% of total tasks (only when all edges saturated)

## Tech Stack

- **MATLAB R2023a+** (Deep Learning Toolbox, Statistics Toolbox)
- **Dataset**: 200-node agricultural sensors (temp, humidity, soil moisture, light, nutrients)
- **Visualization**: MATLAB plots (training curves, topology, baseline comparison)


## Key Functions

- `initialize_dqn_model()`: Builds DQN with dropout regularization
- `get_predictive_state_15d()`: 18D state with 3D load prediction
- `calculate_multi_objective_reward()`: Energy + balance + completion time
- `dynamicClustering()`: Silhouette-optimized K-Means
- `calculateAdaptiveThreshold()`: Dynamic overload prevention

## Research Context

**Academic Contribution** (IIT Roorkee Research Internship):
- Novel 18D predictive state for proactive scheduling
- Energy-aware Q-value shaping with distance + utilization penalties
- Production-grade evaluation framework vs 7 established baselines

**Metrics tracked**:
Load Balance Variance: 0.0214 (76% better)
Max Edge Utilization: 78% (no overloads)
Cloud Offloads: 4.2% (only emergency cases)
Energy Efficiency: 2.84 tasks/unit energy

## Usage

```matlab
% Prerequisites: MATLAB Deep Learning Toolbox
addpath('src'); run('main_dqn_scheduler.m');
