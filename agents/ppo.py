import datetime
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):  # 输出的是一个值的话，输入不应该是（s,a）吗？如果输入只是s，应该有几个动作输出几个值？？
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_actions,
                 hidden_dim,
                 log_std_min=-20,
                 log_std_max=0.5):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        # self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std = AddBias(torch.zeros(num_actions))

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mu = torch.tanh(self.mean_linear(x))
        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        zeros = torch.zeros(mu.size())
        if state.is_cuda:
            zeros = zeros.cuda()
        log_std = self.log_std(zeros)
        return mu, log_std

class Agent():
    def __init__(self,
                 name,
                 obs_shape,
                 act_space,
                 agent_num,
                 group_num,
                 agent_group_index,
                 share_parameters,
                 parameters,
                 model_path,
                 log_path,
                 create_summary_writer=False,
                 resume=False):

        self.name = name
        self.obs_shape_list = obs_shape  # env.observation_space [12,12,12]
        self.act_space_list = act_space  # env.action_space [2,2,2]
        self.agent_num = agent_num
        self.group_num = group_num
        self.agent_group_index = agent_group_index  # "agent_group_index":[0, 0, 0], #环境中每个agent对应的组编号，若3个智能体都在一队，则为[0,0,0]
        self.share_parameters = share_parameters
        self.parameters = parameters
        self.model_path = model_path
        self.log_path = log_path

        self.group_obs_shape_list = [0 for i in range(self.group_num)]
        self.group_act_shape_list = [0 for i in range(self.group_num)]
        for agent_index, group_index in enumerate(self.agent_group_index):  # ????
            if self.group_obs_shape_list[group_index] == 0:
                self.group_obs_shape_list[group_index] = self.obs_shape_list[agent_index]
            if self.group_act_shape_list[group_index] == 0:
                self.group_act_shape_list[group_index] = self.act_space_list[agent_index]

        if self.share_parameters:
            self.actors = [PolicyNetwork(num_inputs=self.group_obs_shape_list[group_index],
                                         num_actions=self.group_act_shape_list[group_index],
                                         hidden_dim=128).to(device) for group_index in range(self.group_num)]
            self.critics = [ValueNetwork(state_dim=self.group_obs_shape_list[group_index],
                                         hidden_dim=128).to(device) for group_index in range(self.group_num)]
            self.actor_optimizers = [optim.Adam(self.actors[group_index].parameters(), lr=self.parameters["A_LR"])
                                     for group_index in range(self.group_num)]
            self.critic_optimizers = [optim.Adam(self.critics[group_index].parameters(), lr=self.parameters["C_LR"])
                                     for group_index in range(self.group_num)]
        else:
            self.actors = [PolicyNetwork(num_inputs=self.obs_shape_list[agent_index],
                                         num_actions=self.act_space_list[agent_index],
                                         hidden_dim=128).to(device) for agent_index in range(self.agent_num)]

            self.critics = [ValueNetwork(state_dim=self.obs_shape_list[agent_index],
                                         hidden_dim=128).to(device) for agent_index in range(self.agent_num)]
            self.actor_optimizers = [optim.Adam(self.actors[agent_index].parameters(), lr=self.parameters["A_LR"])
                                     for agent_index in range(self.agent_num)]
            self.critic_optimizers = [optim.Adam(self.critics[agent_index].parameters(), lr=self.parameters["C_LR"])
                                     for agent_index in range(self.agent_num)]

        self.buffers = [{'state':[],
                         'action':[],
                         'reward':[],
                         'new_state':[],
                         'done':[],
                         'discounted_r':[],
                         'action_log_prob': [],
                         'state_value': [],
                         'td_error':[],
                         'gae_adv':[],
                         }
                        for _ in range(self.agent_num)]

        # 为每一个agent构建tensorboard可视化训练过程
        if resume:
            with open(self.log_path + '/log_info.txt', 'r') as load_f:
                self.log_info_json = json.load(load_f)
                load_f.close()

            if create_summary_writer:
                self.summary_writers = []
                for i in range(self.agent_num):
                    train_log_dir = self.log_path + self.log_info_json["summary_dir"] + "agent_" + str(i)
                    self.summary_writers.append(SummaryWriter(train_log_dir))
            else:
                pass
            self.load_model()

        else:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_info_json = {
                "summary_dir": '/PPO_Summary_' + str(current_time),
                "epoch": 0,
                "train_step": 0,
                "log_episode": 0
            }
            if create_summary_writer:
                self.summary_writers = []
                for i in range(self.agent_num):
                    train_log_dir = self.log_path + self.log_info_json["summary_dir"] + "agent_" + str(i)
                    self.summary_writers.append(SummaryWriter(train_log_dir))
            else:
                pass

    def action(self, obs_n, evaluation=False):  # 所有智能体obs in,所有action out
        action_n = []  # SAC中是[array([]),array([])]
        for i, obs in enumerate(obs_n):  # SAC中obs:[,,]
            obs = torch.as_tensor([obs], dtype=torch.float32, device=device)
            if self.share_parameters:
                mu, log_std = self.actors[self.agent_group_index[i]](obs)
            else:
                mu, log_std = self.actors[i](obs)

            if evaluation:
                act = mu.cpu().detach().numpy()[0]
            else:
                pi = torch.distributions.Normal(mu, torch.exp(log_std))
                act = pi.sample().cpu().numpy()[0]

            act = np.clip(act, -1, 1)
            action_n.append(act)
        return action_n

    #存储transition
    def experience(self, obs_n, act_n, rew_n, new_obs_n, done_n):
        for i in range(self.agent_num):  # buffer{list:3} ,每一个元素对应每个智能体的字典{‘state’:[array([第1时刻的12个状态])，array([第2时刻的12个状态])，。。。],'action':[],'reward':[r1,r2,r3],'new_state':[],'done':[[false,false,false]]}
            self.buffers[i]['state'].append(obs_n[i])
            self.buffers[i]['action'].append(act_n[i])
            self.buffers[i]['reward'].append([rew_n[i]])
            self.buffers[i]['new_state'].append(new_obs_n[i])
            self.buffers[i]['done'].append([float(done_n[i])])
            if self.share_parameters:
                pass
            else:
                v_s = self.critics[i](
                    torch.as_tensor([obs_n[i]], dtype=torch.float32, device=device)).detach().cpu().numpy()[0]
                v_s_ = self.critics[i](
                    torch.as_tensor([new_obs_n[i]], dtype=torch.float32, device=device)).detach().cpu().numpy()[0]
                td_error = np.asarray([rew_n[i]]) + self.parameters['gamma'] * v_s_ - v_s
                self.buffers[i]['td_error'].append(td_error)

    def save_model(self):
        if self.share_parameters:
            for group_index in range(self.group_num):
                torch.save(self.actors[group_index].state_dict(),
                           self.model_path + "/ppo_actor_group_" + str(group_index) + ".pth")
        else:
            for agent_index in range(self.agent_num):
                torch.save(self.actors[agent_index].state_dict(),
                           self.model_path + "/ppo_actor_agent_" + str(agent_index) + ".pth")

    def load_model(self):
        '''
        开始训练时加载之前的模型
        :return:
        '''
        if self.share_parameters:
            for group_index in range(self.group_num):
                if os.path.exists(self.model_path + "/ppo_actor_group_" + str(group_index) + ".pth"):
                    try:
                        self.actors[group_index].load_state_dict(
                            torch.load(self.model_path + "/ppo_actor_group_" + str(group_index) + ".pth"))
                    except RuntimeError as e:
                        print("模型不匹配，加载训练模型失败，将采用随机参数进行训练！！！")
                        break
                else:
                    print("模型不存在，加载训练模型失败，将采用随机参数进行训练！！！")
                    break
        else:
            for agent_index in range(self.agent_num):
                if os.path.exists(self.model_path + "/ppo_actor_agent_" + str(agent_index) + ".pth"):
                    try:
                        self.actors[agent_index].load_state_dict(
                            torch.load(self.model_path + "/ppo_actor_agent_" + str(agent_index) + ".pth"))
                    except RuntimeError as e:
                        print("模型不匹配，加载训练模型失败，将采用随机参数进行训练！！！")
                        break
                else:
                    print("模型不存在，加载训练模型失败，将采用随机参数进行训练！！！")
                    break

    def load_actor(self):
        if self.share_parameters:
            for group_index in range(self.group_num):
                self.actors[group_index].load_state_dict(
                    torch.load(self.model_path + "/ppo_actor_group_" + str(group_index) + ".pth"))
        else:
            for agent_index in range(self.agent_num):
                self.actors[agent_index].load_state_dict(
                    torch.load(self.model_path + "/ppo_actor_agent_" + str(agent_index) + ".pth"))

    def finish_path(self, next_state_n, done_n):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        for agent_index in range(self.agent_num):
            if done_n[agent_index]:
                v_s_ = np.asarray([0])
            else:
                if self.share_parameters:
                    v_s_ = self.critics[self.agent_group_index[agent_index]](
                        torch.Tensor([next_state_n[agent_index]]).to(device)).cpu().detach().numpy()[0]
                else:
                    v_s_ = self.critics[agent_index](
                        torch.Tensor([next_state_n[agent_index]]).to(device)).cpu().detach().numpy()[0]

            discounted_r = []
            # rewards = np.asarray(self.buffers[agent_index]['reward'])/sum(np.asarray(self.buffers[agent_index]['reward']))
            for r in self.buffers[agent_index]['reward'][::-1]:#[::-1]将列表倒转，如：【1，2，3】—》【3，2，1】
                v_s_ = r + self.parameters["gamma"] * v_s_  # no future reward if next state is terminal
                discounted_r.append(v_s_)
            discounted_r.reverse()#[[1],[2],[3]]
            self.buffers[agent_index]['discounted_r'].extend(discounted_r)

            gae_adv = []
            adv = np.asarray([0])
            for rd_e in self.buffers[agent_index]['td_error'][::-1]:
                adv = rd_e + self.parameters["gamma"] * self.parameters["lambd"] * adv
                gae_adv.append(adv)
            gae_adv.reverse()
            self.buffers[agent_index]['gae_adv'].extend(gae_adv)

            self.buffers[agent_index]['reward'].clear()
            self.buffers[agent_index]['td_error'].clear()


    def update(self, train_step):
        obs_n = []  # [[第一个智能体的states],[第二个智能体的states]，。。。，[第N个智能体的state]]
        act_n = []
        discounted_r_n = []
        advantage_n = []
        old_pi_n = []
        old_value_n = []

        for agent_index in range(self.agent_num):
            obs = torch.tensor(self.buffers[agent_index]['state'], dtype=torch.float32, device=device)
            act = torch.tensor(self.buffers[agent_index]['action'], dtype=torch.float32, device=device)
            discounted_r = torch.tensor(self.buffers[agent_index]['discounted_r'], dtype=torch.float32, device=device)
            gae_adv = torch.tensor(self.buffers[agent_index]['gae_adv'], dtype=torch.float32, device=device)
            obs_n.append(obs)  # [tensor(智能体1的batch个state),tensor(智能体2的batch个state),...]
            act_n.append(act)
            discounted_r_n.append(discounted_r)
            with torch.no_grad():
                mu, log_std = self.actors[agent_index](obs)
                pi = torch.distributions.Normal(mu, torch.exp(log_std))
                old_value = self.critics[agent_index](obs)
                adv = discounted_r - old_value
            if self.parameters['use_gae_adv']:
                advantage_n.append(gae_adv)
            else:
                advantage_n.append(adv)
            old_pi_n.append(pi)
            old_value_n.append(old_value)

        if self.share_parameters:
            summaries = {
                'LOSS/PPO_actor_loss': [0 for _ in range(self.group_num)],
                'LOSS/PPO_critic_loss': [0 for _ in range(self.group_num)],
            }
        else:
            summaries = {
                'LOSS/PPO_actor_loss': [0 for _ in range(self.agent_num)],
                'LOSS/PPO_critic_loss': [0 for _ in range(self.agent_num)],
                }

        for _ in range(self.parameters['UPDATE_STEPS']):
            summary = self.a_train(memories=(obs_n, act_n, advantage_n, old_pi_n))
            for key in summary.keys():
                summaries[key] = [i + j for i, j in zip(summaries[key], summary[key])]

        for _ in range(self.parameters['UPDATE_STEPS']):
            summary = self.c_train(memories=(obs_n, discounted_r_n, old_value_n))
            for key in summary.keys():
                summaries[key] = [i + j for i, j in zip(summaries[key], summary[key])]

        #每一轮训练结束，清空本轮所有的buffer
        for buffer in self.buffers:
            for key in buffer.keys():
                buffer[key].clear()

        for i in range(self.agent_num):
            if self.share_parameters:
                for key in summaries.keys():
                    self.summary_writers[i].add_scalar(key,
                                                       summaries[key][self.agent_group_index[i]] / self.parameters['UPDATE_STEPS'],
                                                       global_step=train_step)
            else:
                for key in summaries.keys():
                    self.summary_writers[i].add_scalar(key,
                                                       summaries[key][i] / self.parameters['UPDATE_STEPS'],
                                                       global_step=train_step)
            self.summary_writers[i].flush()

    def a_train(self, memories):
        obs_n, act_n, advantage_n, old_pi_n = memories

        if self.share_parameters:
            actors_loss = [torch.tensor(0, dtype=torch.float32, device=device) for _ in range(self.group_num)]
        else:
            actors_loss = [torch.tensor(0, dtype=torch.float32, device=device) for agent_index in range(self.agent_num)]
            for agent_index in range(self.agent_num):
                # actor_loss
                mu, log_std = self.actors[agent_index](obs_n[agent_index])
                entropy = self.gaussian_entropy(log_std)
                pi = torch.distributions.Normal(mu, torch.exp(log_std))
                ratio = torch.exp(pi.log_prob(act_n[agent_index]) - old_pi_n[agent_index].log_prob(act_n[agent_index]))
                surr = ratio * advantage_n[agent_index]
                actor_loss = -torch.mean(
                    torch.minimum(
                        surr,
                        torch.clamp(
                            ratio,
                            1. - self.parameters['epsilon'],
                            1. + self.parameters['epsilon']
                        ) * advantage_n[agent_index]
                    )
                ) - self.parameters['ent_coef'] * entropy
                self.actor_optimizers[agent_index].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[agent_index].step()
                actors_loss[agent_index] = actor_loss

        summaries = {
            'LOSS/PPO_actor_loss': actors_loss,
            }
        return summaries

    def c_train(self, memories):
        obs_n, discounted_r_n, old_value_n = memories
        if self.share_parameters:
            critics_loss = [torch.tensor(0, dtype=torch.float32, device=device) for _ in range(self.group_num)]

        else:
            critics_loss = [torch.tensor(0, dtype=torch.float32, device=device) for agent_index in range(self.agent_num)]

            for agent_index in range(self.agent_num):
                # critic_loss
                value = self.critics[agent_index](obs_n[agent_index])
                td_error = discounted_r_n[agent_index] - value

                value_clip = old_value_n[agent_index] + torch.clamp(value - old_value_n[agent_index],
                                                                    -self.parameters['epsilon'],
                                                                    self.parameters['epsilon'])
                td_error_clip = discounted_r_n[agent_index] - value_clip
                td_square = torch.maximum(torch.square(td_error), torch.square(td_error_clip))
                critic_loss = 0.5 * torch.mean(td_square)

                self.critic_optimizers[agent_index].zero_grad()
                critic_loss.backward()
                self.critic_optimizers[agent_index].step()
                critics_loss[agent_index] = critic_loss
        summaries = {
            'LOSS/PPO_critic_loss': critics_loss,
            }
        return summaries

    def gaussian_entropy(self, log_std):
        '''
        Calculating the entropy of a Gaussian distribution.
        Args:
            log_std: log standard deviation of the gaussian distribution.
        Return:
            The average entropy of a batch of data.
        '''
        return torch.mean(0.5 * (1 + torch.log(2 * np.pi * torch.exp(log_std) ** 2 + 1e-8)))