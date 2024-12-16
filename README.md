#  Humanoid Standup Reinforcement Learning  

A deep reinforcement learning project that trains a humanoid robot to stand up using the **Deep Deterministic Policy Gradient (DDPG)** algorithm with real-time web visualization.  

---

##  Insights

- **Environment**: OpenAI Gymnasium `HumanoidStandup-v4`  
- **Algorithm**: Deep Deterministic Policy Gradient (DDPG)  
- **Visualization**: Real-time web streaming of the training process  
- **Interaction**: Web interface to start and monitor training  

---

##  Technical Specifications  

### Neural Networks  
- **Actor Network**: Used to generate actions based on current states.  
- **Critic Network**: Evaluates actions by computing the Q-value.  

### Learning Method  
- **Approach**: Model-Free, Continuous Control using Policy Gradient 
- **Training Episodes**: 20,000  
- **Exploration**: Epsilon-greedy strategy  

---

## Related Paper  

This project is inspired by and implements concepts from the following paper:  
**[Continuous Control with Deep Reinforcement Learning (DDPG)](https://arxiv.org/abs/1509.02971)** by Timothy P. Lillicrap et al.  
