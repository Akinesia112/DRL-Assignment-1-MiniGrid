# DRL Assignment 1: Q-Learning, Policy Learning, Reward Shaping with Tabular Methods in Gym MiniGrid Environment, and Tabular Learning with PyTorch 

This repository contains the implementation of a Deep Reinforcement Learning (DRL) assignment where we develop both **Tabular Q-Learning** and **Policy Learning** algorithms using PyTorch. The project experiments with the MiniGrid environments to train agents that learn to navigate and reach their goals efficiently.

## Overview

The project consists of two main parts:

- **Tabular Q-Learning Implementation**  
  The Q-learning agent uses a tabular representation of Q-values stored as a PyTorch tensor. The update rule is based on the Bellman equation:
$
  Q(s, a) \leftarrow Q(s, a) + \alpha \Big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\Big],
$
  with the loss function defined as:
$
  \mathcal{L} = \text{MSE}\Big(Q(s, a), \text{sg}\big(r + \gamma \max_{a'} Q(s', a')\big)\Big),
$
  where the stop-gradient operation is used to prevent gradients from flowing through the target value. The agent uses an ε-greedy policy for exploration and is applied on environments like `MiniGrid-Empty-8x8-v0` and `MiniGrid-DoorKey-8x8-v0`.

- **Policy Learning Implementation**  
  The policy learning agent uses a softmax policy, with the policy represented by a logits table. The agent is updated using policy gradient methods via cross-entropy loss. A baseline is maintained to reduce variance in updates. This part demonstrates how to directly learn the policy using gradients, sampling actions from the softmax distribution.

---

## Question 1: Implement and Familiarize Yourself with a Grid World Environment

- **Setup:**  
  Configure and initialize the MiniGrid environment using OpenAI Gym.
  
- **Understanding the Gym MiniGrid Environment:**  
  Explore the features of the MiniGrid environment and understand its dynamics.
  
- **Implementing a Random Agent in MiniGrid:**  
  Develop a baseline agent that selects actions randomly.
  
- **Implementing a Rule-Based Agent in MiniGrid:**  
  Create an agent that follows predefined rules to navigate the grid world.

---

## Question 2: Reinforcement Learning with Tabular Methods

- **Value-Based Learning (Q-Learning):**  
  Implement Q-Learning for the grid world, learning optimal state-action values.
  
- **Policy-Based Learning:**  
  Explore policy-based reinforcement learning methods to directly learn action policies.
  
- **Reward Shaping:**  
  Enhance the learning process by modifying the reward function to better guide the agent's behavior.

---

## Question 3: Implementing Tabular Learning with PyTorch

- **Tabular Q-Learning using PyTorch:**  
  Transition from a NumPy-based Q-table to a PyTorch tensor-based implementation, leveraging automatic differentiation.
  
- **Softmax Policy Gradient:**  
  Understand and implement the Softmax Policy Gradient method for policy-based learning.

---


