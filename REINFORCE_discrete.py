# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:02:49 2020

@author: coldhenry
####################################################
#                                                  #
# REINFORCE algorithms w/ continuous action space  #
#                                                  #
####################################################

env: modified_gym_env:ReacherPyBulletEnv-v1

"""
import gym  # open ai gym
import pybulletgym.envs
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch import optim
import torch.nn.utils as utils
from torch.autograd import Variable
from torch.distributions import MultivariateNormal

env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)

# for reproducibility
# env.seed(1)
# torch.manual_seed(1)

gamma = 0.9
batch_size = 1000
iterations = 500

state_space = 8  # there is 9 states in default, but seems to have a bug
action_space = 2  # 2 motors


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        num_hidden = 64
        num_hidden2 = 32

        self.l1 = nn.Linear(state_space, num_hidden)
        self.l2 = nn.Linear(num_hidden, num_hidden2)
        self.l3 = nn.Linear(num_hidden2, action_space)
        self.var = torch.nn.Parameter(torch.FloatTensor(torch.eye(2) * 0.1))

    def forward(self, x):
        # fully connected model
        model = torch.nn.Sequential(
            self.l1,
            nn.Tanh(),
            self.l2,
            nn.Tanh(),
            self.l3,
        )
        return model(x)


def predict(state, policy, mode=1):

    mu = policy(Variable(state))
    var = torch.abs(policy.var) + 0.001
    dist = MultivariateNormal(mu, var)
    action = dist.sample()
    log_pb = dist.log_prob(action)

    return action, mu, log_pb


def discounted_reward(rewards, gamma=0.9):

    r = []
    for t in range(1, len(rewards) + 1):
        for t_ in range(t, len(rewards) + 1):
            r.append(torch.pow(torch.tensor(gamma), (t_ - t)) * rewards[t_ - t])
    r = np.sum(r)
    return r


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    policy = Policy()
    policy.train()
    # policy = policy.to(device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    actions = np.arange(action_space)
    plot_reward = []
    for eps in range(iterations):

        total_rewards = 0
        batch_count = 0
        traj_count = 0
        total_loss = 0

        states, rewards = [], []
        s_curr = env.reset()
        done = False

        log_sum = 0
        print(policy.var)

        batch_reward = []
        batch_log_pb = []

        while batch_count != batch_size:

            # update count
            batch_count += 1

            action, _, log_pb = predict(torch.FloatTensor(s_curr), policy)
            log_sum += log_pb
            s_next, reward, done, _ = env.step(action.numpy())
            s_curr = s_next

            states.append(s_next)
            rewards.append(reward)

            if done or batch_count == batch_size:

                s_curr = env.reset()

                traj_count += 1

                # discounted reward of a trajectory
                batch_log_pb.append(log_sum)
                batch_reward.append(discounted_reward(rewards, gamma))

                total_rewards += sum(rewards)

                states, rewards = [], []

                log_sum = 0
                done = False

        # calculate the loss
        batch_reward -= np.mean(batch_reward) / (
            np.std(batch_reward) + np.finfo(np.float32).eps
        )
        loss = np.array(batch_reward) * np.array(batch_log_pb)
        loss = -np.sum(loss) / traj_count

        optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(policy.parameters(), 40)
        optimizer.step()

        mean_reward = total_rewards / traj_count
        print(
            "Episode: {} / Avg. last {}: {:.2f}, Traj: {}, loss {}".format(
                eps, batch_size, mean_reward, traj_count, loss
            )
        )
        if len(plot_reward) > 0:
            if plot_reward[-1] - mean_reward < 0:
                torch.save(policy.state_dict(), "reinforce_model_attempt2.pkl")
        plot_reward.append(mean_reward)

    #%%

    t = np.arange(0, iterations, 1)
    plt.figure(figsize=(9, 9))
    plt.plot(t, plot_reward)
    # plt.legend(["g = 0.9","g = 0.95","g = 0.99"], fontsize=15)
    plt.xlabel("Episodes", fontsize=15)
    plt.ylabel("Avg. Reward", fontsize=15)
    plt.title("REINFORCE", fontsize=20)

    plt.savefig("REINFORCE_2link_attempt2.png")
    plt.show()
