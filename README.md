# Deep Reinforcement Learning for Autonomous Driving Vehicle on SMARTs platform
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

 
