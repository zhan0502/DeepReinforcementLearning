#this is lmz's second version
import logging
import tempfile
import os

import gym
import numpy as np
import ray

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math
from examples import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from collections import namedtuple
import random
from datetime import datetime
from contextlib import redirect_stdout
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
# AGENT_ID = "Agent-007"
# AGENT_ID1= "Agent-008"
N_AGENTS=1
AGENT_IDS=['Agent %i'% i for i in range(N_AGENTS)]

BATCH_SIZE=128
GAMMA=0.999
EPS_START=0.9
EPS_END=0.05
EPS_DECAY=2000
TARGET_UPDATE=10

now = datetime.now()
current_time = now.strftime("%H:%M:%S")

def plot_learning_curve(x, scores, figure_file):
    #running_avg = np.zeros(len(scores))
    # for i in range(len(running_avg)):
    #    running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, scores)
    plt.title('DQN Testing curve')
    plt.savefig(figure_file)

class DQN(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, model_path=None):
        super(DQN, self).__init__()
        self.chkpt_dir = 'tmp/dqn'
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dims, hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims, hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims, output_dims),
        )
        # if model_path:
        #    self.model.load_state_dict(torch.load(model_path))
        # else:
        #    def init_weights(m):
        #       if type(m) == torch.nn.Linear:
        #           torch.nn.init.xavier_uniform_(m.weights)
        #           m.bias.data.fill_(0.01)
        #       self.model.apply(init_weights)

        if model_path == 'policy':
            self.checkpoint_file = os.path.join(self.chkpt_dir, 'policy_model')
        elif model_path == 'target':
            self.checkpoint_file = os.path.join(self.chkpt_dir, 'target_model')

    def forward(self, obs):
        batched_obs = np.array(obs)
        x = torch.from_numpy(batched_obs)
        y = self.model(x)
        if y.dim() == 1:
            batched_actions = y.unsqueeze(0)
        else:
            batched_actions = y

        # file=open('/home/ning/SMARTS/examples/log.txt','a+')
        # with redirect_stdout(file):
        #     print(current_time,batched_actions)
        #     file.flush()
        #     file.close()
        return batched_actions

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


 

class DQNAgent(Agent):
    def __init__(self, input_dims, hidden_dims, output_dims, policy_model_path='policy',target_model_path='target'):
        self.n_actions=output_dims
        self.policy_net=DQN( input_dims, hidden_dims, output_dims, policy_model_path)
        self.target_net=DQN( input_dims, hidden_dims, output_dims, target_model_path)

    def act(self, obs):
        batched_obs = np.array(obs)
        x = torch.from_numpy(batched_obs)
        y = self.policy_net(x)
        
        batched_actions = y.max(1)[1].view(1, 1).long()
        return batched_actions
    def random_act(self,obs):
        if obs.ndim==2:
            # return np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(-1,1)]*obs.shape[0])
            return torch.tensor([random.randrange(self.n_actions)]*obs.shape[0])        
        else:
            return torch.tensor([[random.randrange(self.n_actions)]])   
    
    def save_models(self):
        print('... saving models ...')
        self.policy_net.save_checkpoint()
        self.target_net.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.policy_net.load_checkpoint()
        self.target_net.load_checkpoint()
            # return np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(-1,1)])
    #def save(self, path):
    #    torch.save(self.target_net.state_dict(), path)


def observation_adapter(env_obs):
    ego = env_obs.ego_vehicle_state
    waypoint_paths = env_obs.waypoint_paths
    wps = [path[0] for path in waypoint_paths]

    # distance of vehicle from center of lane
    # closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))

    dist_from_centers=[]
    angle_errors=[]
    for wp in wps:
        signed_dist_from_center = wp.signed_lateral_error(ego.position)
        lane_hwidth = wp.lane_width * 0.5
        dist_from_centers.append(signed_dist_from_center / lane_hwidth)
        angle_errors.append(wp.relative_heading(ego.heading))

    neighborhood_vehicles=env_obs.neighborhood_vehicle_states
    relative_neighbor_distance=[np.array([10,10])]*3

    if neighborhood_vehicles==None or len(neighborhood_vehicles)==0:  # no neighborhood vechicle
        relative_neighbor_distance=[distance.tolist() for distance in relative_neighbor_distance]
    else:
        position_differences=np.array([math.pow(ego.position[0]-neighborhood_vehicle.position[0],2)+
            math.pow(ego.position[1]-neighborhood_vehicle.position[1],2) for neighborhood_vehicle in neighborhood_vehicles])
        
        nearest_vehicle_indexes=np.argsort(position_differences)
        for i in range(min(3,nearest_vehicle_indexes.shape[0])):
            relative_neighbor_distance[i]=np.clip((ego.position[:2]-neighborhood_vehicles[nearest_vehicle_indexes[i]].position[:2]),-10,10).tolist()   
        
    
    return np.array(
        dist_from_centers+ angle_errors+ego.position[:2].tolist()+[ego.speed, ego.steering]+[diff for diffs in relative_neighbor_distance for diff in diffs],
        dtype=np.float32,
    )


def action_adapter(model_action):
    # throttle, brake, steering = model_action
    # brake=0
    model_action=model_action.numpy()
    if model_action==0:
        return "keep_lane"
    elif model_action==1:
        return "slow_down"
    elif model_action==2:
        return "change_lane_left"
    elif model_action==3:
        return "change_lane_right"

    # return np.array([throttle, brake, steering * np.pi * 0.25])


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]
        self.position=0
    def push(self,*args):
        if len(self.memory)<self.capacity:
            self.memory.append(None)
        self.memory[self.position]=Transition(*args)
        self.position=(self.position+1) % self.capacity
    def sample(self,batch_size,agent_id):
        batch=random.sample(self.memory,batch_size)
        # for s in batch:
        #     s.state=s.state[agent_id]
        #     s.actions=s.actions[agent_id]
        #     s.next_state=s.next_state[agent_id]
        #     s.reward=s.reward[agent_id]
        # batch=[s.state[agent_id] for s in batch]
        # batch=[s.actions[agent_id] for s in batch]
        # batch.next_state=[s.next_state[agent_id] for s in batch]
        # batch.reward=[s.reward[agent_id] for s in batch]
        return batch
    def __len__(self):
        return len(self.memory)

def log(desc,info,agent_id=None):
    file=open('/home/ning/SMARTS/examples/log.txt','a+')
    with redirect_stdout(file):
        print(current_time)
        # print('y max_episode_steps:',y.max(1))
        print(agent_id,desc,info)
        print('\n')
        file.flush()
        file.close()

AGENT_ID ='Agent 0'
@ray.remote
def evaluate(training_scenarios, evaluation_scenarios, sim_name, headless, num_episodes, seed):
    agent_params = {"input_dims": 16, "hidden_dims": 256, "output_dims": 4}
    agent_spec = AgentSpec(
            interface=AgentInterface.from_type(AgentType.Laner,
                max_episode_steps=5000),
            agent_params=agent_params,
            # agent_builder=PyTorchAgent,
            agent_builder=DQNAgent,
            action_adapter=action_adapter,
            observation_adapter=observation_adapter,
    )
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=args.scenarios,
        agent_specs={AGENT_ID: agent_spec},
        # agent_specs=agent_spec,
        sim_name=args.sim_name,
        #######this is for disable envision
        #headless=True,
        #visdom=False,
        #######################
        headless=headless,
        timestep_sec=0.1,
        seed=args.seed,
    )
    n_epoch = 0
    agent  = agent_spec.build_agent()
    score_history = []
    best_score = env.reward_range[0]
    learn_iters = 0
    avg_score = 0
    n_steps=0
    agent.load_models()
    for episode in episodes(num_episodes):

        n_epoch +=1
        
        
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        score = 0
        while not dones["__all__"]:
            old_state=observations
            state=observations
            #now = datetime.now()
            #current_time = now.strftime("%H:%M:%S")
            #with open('home/ning/SMARTS/examples/log.txt','a+') as file:
             #   file.write('output observation:',current_time,observations)
            agent_obs= observations[AGENT_ID]
                
            agent_action = agent.act(agent_obs)
            n_steps +=1
                    #done = True

            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)
            score += rewards['Agent 0'] 
            
            next_state=observations
             
             
        score_history.append(score)
        avg_score = np.mean(score_history[-1000:])
        if n_epoch %10 ==0:
            print('episode', n_epoch, 'score %.1f' % score, 'avg score %.1f' % avg_score,'best score %.1f' % best_score,
                'time_steps', n_steps)
 

        x = [i+1 for i in range(len(score_history))]
        figure_file = 'plots/dqn.png'
        plot_learning_curve(x, score_history, figure_file)
         
        


    env.close()

    #print(f"Finished Evaluating Agent: {accumulated_reward:.2f}")


def main(
    training_scenarios,
    evaluation_scenarios,
    sim_name,
    headless,
    num_episodes,
    seed,
):
    ray.init()
    ray.wait(
        [
            evaluate.remote(
                training_scenarios,
                evaluation_scenarios,
                sim_name,
                headless,
                num_episodes,
                seed,
            )
        ]
    )


if __name__ == "__main__":
    parser = default_argument_parser("pytorch-example")
    parser.add_argument(
        "--evaluation-scenario",
        default="scenarios/loop",
        help="The scenario to use for evaluation.",
        type=str,
    )
    args = parser.parse_args()

    main(
        training_scenarios=args.scenarios,
        evaluation_scenarios=[args.evaluation_scenario],
        sim_name=args.sim_name,
        #headless=args.headless,
        headless=True,
        num_episodes=50,
        #num_episodes=args.episodes,
        seed=args.seed,
    )
