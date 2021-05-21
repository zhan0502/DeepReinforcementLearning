import logging
import tempfile

import gym
import numpy as np
import ray

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from examples import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from collections import namedtuple
import random
from datetime import datetime
from contextlib import redirect_stdout

import gym
import numpy as np
from ppo_test2 import PPOAgent
 

import numpy as np
import matplotlib.pyplot as plt
AGENT_ID = "Agent-007"

def plot_learning_curve(x, scores, figure_file):
    #running_avg = np.zeros(len(scores))
    #for i in range(len(running_avg)):
        #running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, scores)
    plt.title('PPO Learning curve')
    plt.savefig(figure_file)


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

    if neighborhood_vehicles==None or len(neighborhood_vehicles)==0: # no neighborhood vechicle
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

@ray.remote
def train(
    training_scenarios, evaluation_scenarios, sim_name, headless, num_episodes, seed
):
    #agent_params = {"input_dims": 20, "hidden_dims": 256, "output_dims": 4}
    agent_params = {"input_dims": 16, "hidden_dims": 256, "output_dims": 4}

    agent_spec = AgentSpec(
            interface=AgentInterface.from_type(AgentType.Laner,
                max_episode_steps=5000),
            agent_params=agent_params,
            # agent_builder=PyTorchAgent,
            agent_builder=PPOAgent,
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
    agent  = agent_spec.build_agent()

    N = 300
    score_history = []
    best_score = env.reward_range[0]
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    n_epoch = 0
    for episode in episodes(num_episodes):
        
        n_epoch +=1
        if n_epoch % 100 ==0:
            N = N+ 1
        
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        score = 0
        while not dones["__all__"]:
            agent_obs= observations[AGENT_ID]
                
            agent_action, prob, val   = agent.choose_action(agent_obs)
                #agent_action = torch.tensor([[agent_action]])
            done = False
                
                    #done = True

            observations_, rewards, dones, infos = env.step(
                    {AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)
            score += rewards[AGENT_ID]
            n_steps += 1
            agent.remember(observations[AGENT_ID], agent_action, prob, val, rewards[AGENT_ID], done)

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observations = observations_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        if n_epoch %10 ==0:
            print('episode', n_epoch, 'score %.1f' % score, 'avg score %.1f' % avg_score,'best score %.1f' % best_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
        x = [i+1 for i in range(len(score_history))]
        figure_file = 'plots/ppo2.png'
        plot_learning_curve(x, score_history, figure_file)
    env.close()
    
        
    #print("first iteration done")
    

    
        

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
            train.remote(
                training_scenarios,
                evaluation_scenarios,
                sim_name,
                headless,
                num_episodes,
                seed,
            )
        ]
    )

 
if __name__ == '__main__':

    
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
        headless=True,
        #headless=args.headless,
        num_episodes=50,
        #num_episodes=args.episodes,
        seed=args.seed,
    )


   
        



    


   

