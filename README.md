# Deep Reinforcement Learning for Autonomous Driving Vehicle on SMARTs platform

Reinforcement learning (RL) is a popular paradigm in machine learning in recent years. This machine
learning paradigm is useful for researchers to train an intelligent agent to take actions based on its
given environment, and has been actively developed in many sectors such as video games, robotics
and autonomous driving. As deep learning techniques develop, more and more researchers attempt
to apply artificial neural networks to the traditional reinforcement learning techniques. Mnih et al.
[2015] developed Deep Q-Network (DQN) which uses convolutional neural networks to replace the
Q-table in the traditional Q-Learning method in reinforcement learning. Another group of researchers
from OpenAI applied deep neural networks to the traditional Policy Gradient methods, and developed
a new reinforcement learning model called Proximal Policy Optimization (PPO) [Schulman et al.,
2017].
A robot vehicle that drives automatically has remained an important and long-standing goal in the
field of Artificial Intelligence. Vehicle driving needs high level of attention, skills and experience from
human. Plenty of traffic accidents are because of the lack of attention from the driver. Even though
computers never “feel tired" when driving, fully autonomous driving requires much more intelligence
than what we currently achieved by the existing AI agents [Sallab et al., 2017]. The researchers and
the companies such as Google, Tesla and Baidu are actively coping with these challenges, and our
group would also like to apply what we have learnt in this area.
In this project, we would like to apply the two aforementioned deep reinforcement learning techniques
to the area of autonomous driving. More specifically, we aim to utilize DQN and PPO models to
train a virtual vehicle (agent) to drive autonomously in the simulated road environment. Ideally, the
autonomous vehicle must not (1) crash into other vehicles on the road, (2) crash into the boundary of
roads, or (3) be too slow. Therefore, our goal in this project is to maximize the survival time of the
virtual vehicle on the road with a reasonable speed.

Below are the running steps. 
Install SMARTS environment by following the instructions on https://github.com/huawei-noah/SMARTS

Our running environment is set up on unbuntu 20.4

Source Code in SMARTS/examples folder
1.main_ppo2.py
2.ppo_test2.py 
3.main_ppo_test2.py
4.DQN_v2.py 
5.DQN_v2_test.py


How to run: 

Before running:
1. cd SMART
2. export SUMO_HOME="$PWD/sumo"
3. scl scenario build --clean scenarios/loop

For PPO
1. stay at SMARTS' folder
2. create tmp/ppo folder 
3. create 'plots' folder
3. training: run: python examples/main_ppo2.py scenarios/loop  
4. testing: set the file name as 'python examples/main_ppo_test2.py' at in supervisord.conf 

For DQN
1. stay at SMARTS' folder
2. create tmp/dqn folder  
3. create 'plots' folder
3. training: run: python examples/DQN_v2.py scenarios/loop for 
4. testing: set the file name as 'python examples/DQN_v2_test.py at in supervisord.conf  

 
