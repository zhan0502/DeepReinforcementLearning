# this is lmz's second version
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
N_AGENTS = 1
AGENT_IDS = ['Agent %i' % i for i in range(N_AGENTS)]

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 10

now = datetime.now()
current_time = now.strftime("%H:%M:%S")


def plot_learning_curve(x, scores, figure_file):
    #running_avg = np.zeros(len(scores))
    # for i in range(len(running_avg)):
    #    running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, scores)
    plt.title('DQN Learning curve')
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
    def __init__(self, input_dims, hidden_dims, output_dims, policy_model_path='policy', target_model_path='target'):
        self.n_actions = output_dims
        self.policy_net = DQN(input_dims, hidden_dims,
                              output_dims, policy_model_path)
        self.target_net = DQN(input_dims, hidden_dims,
                              output_dims, target_model_path)

    def act(self, obs):
        batched_obs = np.array(obs)
        x = torch.from_numpy(batched_obs)
        y = self.policy_net(x)

        batched_actions = y.max(1)[1].view(1, 1).long()
        return batched_actions

    def random_act(self, obs):
        if obs.ndim == 2:
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
    # def save(self, path):
    #    torch.save(self.target_net.state_dict(), path)


def observation_adapter(env_obs):
    ego = env_obs.ego_vehicle_state
    waypoint_paths = env_obs.waypoint_paths
    wps = [path[0] for path in waypoint_paths]

    # distance of vehicle from center of lane
    # closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))

    dist_from_centers = []
    angle_errors = []
    for wp in wps:
        signed_dist_from_center = wp.signed_lateral_error(ego.position)
        lane_hwidth = wp.lane_width * 0.5
        dist_from_centers.append(signed_dist_from_center / lane_hwidth)
        angle_errors.append(wp.relative_heading(ego.heading))

    neighborhood_vehicles = env_obs.neighborhood_vehicle_states
    relative_neighbor_distance = [np.array([10, 10])]*3

    # no neighborhood vechicle
    if neighborhood_vehicles == None or len(neighborhood_vehicles) == 0:
        relative_neighbor_distance = [
            distance.tolist() for distance in relative_neighbor_distance]
    else:
        position_differences = np.array([math.pow(ego.position[0]-neighborhood_vehicle.position[0], 2) +
                                         math.pow(ego.position[1]-neighborhood_vehicle.position[1], 2) for neighborhood_vehicle in neighborhood_vehicles])

        nearest_vehicle_indexes = np.argsort(position_differences)
        for i in range(min(3, nearest_vehicle_indexes.shape[0])):
            relative_neighbor_distance[i] = np.clip(
                (ego.position[:2]-neighborhood_vehicles[nearest_vehicle_indexes[i]].position[:2]), -10, 10).tolist()

    return np.array(
        dist_from_centers + angle_errors+ego.position[:2].tolist()+[ego.speed, ego.steering]+[
            diff for diffs in relative_neighbor_distance for diff in diffs],
        dtype=np.float32,
    )


def action_adapter(model_action):
    # throttle, brake, steering = model_action
    # brake=0
    model_action = model_action.numpy()
    if model_action == 0:
        return "keep_lane"
    elif model_action == 1:
        return "slow_down"
    elif model_action == 2:
        return "change_lane_left"
    elif model_action == 3:
        return "change_lane_right"

    # return np.array([throttle, brake, steering * np.pi * 0.25])


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1) % self.capacity

    def sample(self, batch_size, agent_id):
        batch = random.sample(self.memory, batch_size)
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


def log(desc, info, agent_id=None):
    file = open('/home/ning/SMARTS/examples/log.txt', 'a+')
    with redirect_stdout(file):
        print(current_time)
        # print('y max_episode_steps:',y.max(1))
        print(agent_id, desc, info)
        print('\n')
        file.flush()
        file.close()


@ray.remote
def train(
    training_scenarios, evaluation_scenarios, sim_name, headless, num_episodes, seed
):
    # agent_params = {"input_dims": 4, "hidden_dims": 7, "output_dims": 3}
    agent_params = {"input_dims": 16, "hidden_dims": 256, "output_dims": 4}
    # agent_spec = AgentSpec(
    #     interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=5000),
    #     agent_params=agent_params,
    #     agent_builder=PyTorchAgent,
    #     observation_adapter=observation_adapter,
    # )
    # agent_spec1 = AgentSpec(
    #     interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=5000),
    #     agent_params=agent_params,
    #     agent_builder=PyTorchAgent,
    #     observation_adapter=observation_adapter,
    # )

    agent_specs = {
        agent_id: AgentSpec(
            interface=AgentInterface.from_type(AgentType.Laner,
                                               max_episode_steps=5000),
            agent_params=agent_params,
            # agent_builder=PyTorchAgent,
            agent_builder=DQNAgent,
            action_adapter=action_adapter,
            observation_adapter=observation_adapter,

        )
        for agent_id in AGENT_IDS
    }
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=training_scenarios,
        agent_specs=agent_specs,
        # agent_specs=agent_spec,
        sim_name=sim_name,
        headless=headless,
        timestep_sec=0.1,
        seed=seed,
    )
    agents = {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }

    optimizers = {agent_id: optim.RMSprop(
        agents[agent_id].policy_net.parameters()) for agent_id in AGENT_IDS}
    memory = ReplayMemory(200)
    steps = 0

    def optimize_model(agent_id, agent):
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE, agent_id)
        batch = Transition(*zip(*transitions))

        # non_final_mask=torch.tensor(tuple(map(lambda s: agent_id in s.keys() and s[agent_id] is not None,batch.next_state )),dtype=torch.bool)
        # non_final_next_states=[torch.Tensor([s[agent_id]]) for s in batch.next_state if  agent_id in s.keys() and s[agent_id] is not None]
        # if non_final_next_states!=[]:
        #     non_final_next_states=torch.cat(non_final_next_states)
        # else:
        #     return
        # # state_batch=torch.cat(batch.state)
        # # action_batch=torch.cat(batch.action )
        # # reward_batch=torch.cat(batch.reward )
        # state_batch=[torch.Tensor([s[agent_id]]) for s in batch.state if  agent_id in s.keys()]
        # if state_batch!=[]:
        #     state_batch=torch.cat( state_batch ,dim=0)
        # else:
        #     return
        # action_batch=[s[agent_id]for s in batch.action if  agent_id in s.keys()]
        # if action_batch!=[]:
        #     action_batch=torch.cat(action_batch,dim=0)
        # else:
        #     return
        # reward_batch=[torch.Tensor([s[agent_id]]) for s in batch.reward if  agent_id in s.keys()]
        # if reward_batch!=[]:
        #     reward_batch=torch.cat(reward_batch,dim=0)
        # else:
        #     return
        non_final_mask = []
        non_final_next_states = []
        state_batch = []
        action_batch = []
        reward_batch = []
        for state, action, reward, next_state in zip(batch.state, batch.action, batch.reward, batch.next_state):
            if agent_id in state.keys() and agent_id in action.keys() and agent_id in reward.keys() and agent_id in next_state.keys():
                if next_state[agent_id] is None:
                    non_final_mask.append(False)
                else:
                    non_final_mask.append(True)
                non_final_next_states.append(
                    torch.Tensor([next_state[agent_id]]))
                state_batch.append(torch.Tensor([state[agent_id]]))
                action_batch.append(action[agent_id])
                reward_batch.append(torch.Tensor([reward[agent_id]]))
        if non_final_mask == [] or non_final_next_states == [] or state_batch == [] or action_batch == [] or reward_batch == []:
            return
        non_final_mask = torch.tensor(non_final_mask, dtype=torch.bool)
        non_final_next_states = torch.cat(non_final_next_states)
        state_batch = torch.cat(state_batch, dim=0)
        action_batch = torch.cat(action_batch, dim=0)
        reward_batch = torch.cat(reward_batch, dim=0)

        state_action_values = agent.policy_net(
            state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(reward_batch.size()[0])
        next_state_values[non_final_mask] = agent.target_net(
            non_final_next_states).max(1)[0].detach()
        # log('next_state_values',next_state_values.size())
        # log('reward batch',reward_batch.size())
        expected_state_action_values = (next_state_values * GAMMA)+reward_batch

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))
        # log('loss',loss,agent_id)
        optimizers[agent_id].zero_grad()
        loss.backward()
        for param in agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizers[agent_id].step()
        return loss.detach()
    i_episode = 0
    score_history = []
    best_score = env.reward_range[0]
    avg_score = 0
    for episode in episodes(n=num_episodes):
        # agent = agent_spec.build_agent()
        steps_per_episodes = 0
        i_episode += 1
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        state = old_state = observations
        score = 0
        while not dones["__all__"]:
            # now = datetime.now()
            # current_time = now.strftime("%H:%M:%S")
            steps_per_episodes += 1

            # agent_obs = observations[AGENT_ID]
            # agent_action = agent.act(agent_obs)
            old_state = observations
            state = observations
            actions = {}
            for agent_id, agent_obs in observations.items():
                sample = random.random()
                eps_threshold = EPS_END + \
                    (EPS_START - EPS_END) * math.exp(-1. * steps/EPS_DECAY)
                if sample > eps_threshold:
                    with torch.no_grad():
                        actions[agent_id] = agents[agent_id].act(agent_obs)
                else:
                    actions[agent_id] = agents[agent_id].random_act(agent_obs)
            # actions={
            #     agent_id: agents[agent_id].act(agent_obs)
            #     for agent_id, agent_obs in observations.items()
            # }
            # log('observation',agent_id,observations)
            observations, rewards, dones, infos = env.step(actions)
            episode.record_step(observations, rewards, dones, infos)
            steps += 1

            score += rewards['Agent 0']

            next_state = observations
            memory.push(state, actions, next_state, rewards)

            for agent_id, agent in agents.items():
                loss = optimize_model(agent_id, agent)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        if i_episode % 10 == 0:
            print('episode', i_episode, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'best score %.1f' % best_score,
                  'time_steps', steps)

        log('step', episode, steps_per_episodes)
        if i_episode % TARGET_UPDATE == 0:
            for agent_id, agent in agents.items():
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            # if steps % 500 == 0:
            #     print("Evaluating agent")

            #     # We construct an evaluation agent based on the saved
            #     # state of the agent in training.
            #     for agent,agent_id in zip(agents,AGENT_IDS):
            #         model_path = tempfile.mktemp()
            #         agent.save(model_path)

            #         # eval_agent_spec = agent_spec.replace(
            #         #     agent_params=dict(agent_params, model_path=model_path)
            #         # )
            #         eval_agent_spec=agent_specs[agent_id].replace(
            #             agent_params=dict(agent_params,model_path=model_path))
            #     # Remove the call to ray.wait if you want evaluation to run
            #     # in parallel with training
            #     ray.wait(
            #         [
            #             evaluate.remote(
            #                 eval_agent_spec, evaluation_scenarios, headless, seed
            #             )
            #         ]
            #     )
        x = [i+1 for i in range(len(score_history))]
        figure_file = 'plots/dqn.png'
        plot_learning_curve(x, score_history, figure_file)

    env.close()
    # print(score_history)
    


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
        # num_episodes=args.episodes,
        seed=args.seed,
    )
