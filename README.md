# DRL-AUV-3D-Navigation

Deep Reinforcement Learning for Autonomous Underwater Vehicle (AUV) 3D Navigation.

## Overview

This project uses the **TD3 (Twin Delayed DDPG)** algorithm to train an AUV for autonomous navigation in 3D underwater environments, using Velodyne LiDAR point cloud observations.

## Structure

- `algorithm/` — TD3 training and evaluation scripts, replay buffer, environment wrapper, and pre-trained models.
- `simulation/` — ROS + Gazebo simulation workspace with AUV models, underwater sensors, and world plugins.

## Quick Start

1. Set up the ROS catkin workspace in `simulation/` and build:
   ```bash
   cd simulation
   catkin_make
   source devel/setup.bash
   ```

2. Launch the Gazebo simulation with the desired world and AUV model.

3. Run TD3 training:
   ```bash
   cd algorithm/TD3
   python train_velodyne_td3.py
   ```

## Requirements

- ROS (tested on Melodic/Noetic)
- Gazebo
- PyTorch
- NumPy
- TensorBoard

## License

See `algorithm/LICENSE` for details.
