import os
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from buffers.replay_buffer import ReplayBuffer
import json
import itertools

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor_Soft_Attention(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Actor_Soft_Attention, self).__init__()
        # init layers
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.l1 = nn.Linear(input_size * 2, hidden_dim)
        self.l2 = nn.Linear(input_size * 2, hidden_dim)
        self.l3 = nn.Linear(hidden_dim + input_size * 3, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, output_size)

    def forward(self, state):
        #state: (batch_size, num_agents, obs_length)
        index = torch.LongTensor([0, 1, 2]).unsqueeze(0).unsqueeze(-1).expand(state.shape[0], -1, state.shape[-1]).to(device)
        local_inputs = torch.gather(state, dim=1, index=index)  # (batch_size, 3, obs_length)
        agent, neighbor_agents = torch.split(local_inputs,
                                             dim=1,
                                             split_size_or_sections=[1, 2])  # (batch_size, 1, obs_length), (batch_size, 2, obs_length)
        agent = torch.tile(agent, [1, 2, 1])  # (batch_size, 2, obs_length)
        agents = torch.cat([agent, neighbor_agents], dim=-1)  # (batch_size, 2, obs_length*2)

        h_ij_1 = F.relu(self.l1(agents))  # batch_size, 2, hidden_dim
        e_ij_1 = F.relu(self.l2(agents))  # batch_size, 2, hidden_dim
        a_ij_1 = F.softmax(e_ij_1, dim=1)  # batch_size, 2, hidden_dim
        h_i_1 = torch.sum(a_ij_1 * h_ij_1, dim=1)  # batch_size, hidden_dim

        xxx = torch.cat([h_i_1, torch.reshape(local_inputs, [local_inputs.shape[0], -1])], dim=-1)

        xxx = F.relu(self.l3(xxx)) # batch_size, hidden_dim
        output = torch.tanh(self.l4(xxx)) # batch_size, 2

        return output

class Critic_Soft_Attention(nn.Module):
    def __init__(self, input_size, num_agents, action_size, hidden_dim):
        super(Critic_Soft_Attention, self).__init__()
        # init layers
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.action_size = action_size
        self.l1 = nn.Linear(input_size * 2, hidden_dim)
        self.l2 = nn.Linear(input_size * 2, hidden_dim)
        self.l3 = nn.Linear(input_size * 2, hidden_dim)
        self.l4 = nn.Linear(input_size * 2, hidden_dim)
        self.l5 = nn.Linear(input_size + hidden_dim, hidden_dim)
        self.l6 = nn.Linear(input_size + hidden_dim, hidden_dim)
        self.l7 = nn.Linear(hidden_dim + num_agents * input_size + sum(action_size), hidden_dim)
        self.l8 = nn.Linear(hidden_dim, 1)

    def forward(self, state_action_n):
        # state_action_n[0]:(batch_size, num_agents, obs_length)
        index = torch.LongTensor([0]).unsqueeze(0).unsqueeze(-1).expand(state_action_n[0].shape[0], -1,
                                                                        state_action_n[0].shape[-1]).to(device)
        agent = torch.gather(state_action_n[0], dim=1, index=index)  # (batch_size, 1, obs_length)

        index = torch.LongTensor([1]).unsqueeze(0).unsqueeze(-1).expand(state_action_n[0].shape[0], -1,
                                                                        state_action_n[0].shape[-1]).to(device)
        neighbor_agent_1 = torch.gather(state_action_n[0], dim=1, index=index)  # (batch_size, 1, obs_length)

        index = torch.LongTensor([2]).unsqueeze(0).unsqueeze(-1).expand(state_action_n[0].shape[0], -1,
                                                                        state_action_n[0].shape[-1]).to(device)
        neighbor_agent_2 = torch.gather(state_action_n[0], dim=1, index=index)  # (batch_size, 1, obs_length)

        index = torch.LongTensor([3, 4]).unsqueeze(0).unsqueeze(-1).expand(state_action_n[0].shape[0], -1,
                                                                        state_action_n[0].shape[-1]).to(device)
        neighbor_agent_1_friends = torch.gather(state_action_n[0], dim=1, index=index)  # (batch_size, 2, obs_length)

        index = torch.LongTensor([5, 6, 7]).unsqueeze(0).unsqueeze(-1).expand(state_action_n[0].shape[0], -1,
                                                                        state_action_n[0].shape[-1]).to(device)
        neighbor_agent_2_friends = torch.gather(state_action_n[0], dim=1, index=index)  # (batch_size, 3, obs_length)

        #先对neighbor_agent_1进行信息整合
        neighbor_agent_1 = torch.tile(neighbor_agent_1, [1, 2, 1])# (batch_size, 2, obs_length)
        neighbor_agents_1 = torch.cat([neighbor_agent_1, neighbor_agent_1_friends],
                                      dim=-1)# (batch_size, 2, obs_length*2)
        h_ij_1 = F.relu(self.l1(neighbor_agents_1)) # batch_size, 2, hidden_dim
        e_ij_1 = F.relu(self.l2(neighbor_agents_1)) # batch_size, 2, hidden_dim
        a_ij_1 = F.softmax(e_ij_1, dim=1)  # batch_size, 2, hidden_dim
        h_i_1 = torch.sum(a_ij_1 * h_ij_1, dim=1)  # batch_size, hidden_dim

        # if agent_num ==2:
        #     h_i_1 = Dense(hidden_dim, activation='relu')(neighbor_agent_1)
        #     h_i_1 = tf.reshape(h_i_1, [-1, hidden_dim])

        # 先对neighbor_agent_2进行信息整合
        neighbor_agent_2 = torch.tile(neighbor_agent_2, [1, 3, 1])  # (batch_size, 3, obs_length)
        neighbor_agents_2 = torch.cat([neighbor_agent_2, neighbor_agent_2_friends],
                                      dim=-1)  # (batch_size, 3, obs_length*2)
        h_ij_2 = F.relu(self.l3(neighbor_agents_2)) # batch_size, 3, hidden_dim
        e_ij_2 = F.relu(self.l4(neighbor_agents_2))  # batch_size, 3, hidden_dim
        a_ij_2 = F.softmax(e_ij_2, dim=1)  # batch_size, 3, hidden_dim
        h_i_2 = torch.sum(a_ij_2 * h_ij_2, dim=1)  # batch_size, hidden_dim

        h_i_12 = torch.stack([h_i_1, h_i_2], dim=1)  # batch_size, 2, hidden_dim
        agent = torch.tile(agent, [1, 2, 1])# (batch_size, 2, obs_length)
        h_i_12 = torch.cat([agent, h_i_12], dim=-1)  # (batch_size, 2, obs_length+hidden_dim)
        h_i = F.relu(self.l5(h_i_12)) # batch_size, 2, hidden_dim
        q_i = F.relu(self.l6(h_i_12))  # batch_size, 2, hidden_dim
        a_i = F.softmax(q_i, dim=1)  # batch_size, 2, hidden_dim
        h_i_tlt = torch.sum(a_i * h_i, dim=1)  # batch_size, hidden_dim

        xxx = torch.cat([h_i_tlt, torch.reshape(state_action_n[0], [state_action_n[0].shape[0], -1])], dim=-1)
        concat = torch.cat([xxx] + state_action_n[1:], dim=-1)
        hidden = F.relu(self.l7(concat))
        output = self.l8(hidden)

        return output



# H2G_MAACAgent Class
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
        self.obs_shape_list = obs_shape
        self.act_space_list = act_space
        self.agent_num = agent_num
        self.group_num = group_num
        self.agent_group_index = agent_group_index
        self.share_parameters = share_parameters
        self.parameters = parameters
        self.model_path = model_path
        self.log_path = log_path

        self.group_obs_shape_list = [0 for i in range(self.group_num)]
        self.group_act_shape_list = [0 for i in range(self.group_num)]
        for agent_index, group_index in enumerate(self.agent_group_index):
            if self.group_obs_shape_list[group_index] == 0:
                self.group_obs_shape_list[group_index] = self.obs_shape_list[agent_index]
            if self.group_act_shape_list[group_index] == 0:
                self.group_act_shape_list[group_index] = self.act_space_list[agent_index]

        if self.share_parameters:
            self.actors = [Actor_Soft_Attention(input_size=self.group_obs_shape_list[group_index],
                                    output_size=self.group_act_shape_list[group_index],
                                    hidden_dim=128).to(device) for group_index in range(self.group_num)]
            self.critics = [Critic_Soft_Attention(input_size=self.group_obs_shape_list[group_index],
                                      action_size=self.act_space_list,
                                      hidden_dim=128, num_agents=len(self.act_space_list)).to(device) for group_index in range(self.group_num)]

            self.actor_targets = [Actor_Soft_Attention(input_size=self.group_obs_shape_list[group_index],
                                                output_size=self.group_act_shape_list[group_index],
                                                hidden_dim=128).to(device) for group_index in range(self.group_num)]
            self.critic_targets = [Critic_Soft_Attention(input_size=self.group_obs_shape_list[group_index],
                                                  action_size=self.act_space_list,
                                                  hidden_dim=128, num_agents=len(self.act_space_list)).to(device) for
                            group_index in range(self.group_num)]
            self.actor_optimizers = [optim.Adam(self.actors[group_index].parameters(), lr=self.parameters["lr_actor"])
                                     for group_index in range(self.group_num)]
            self.critic_optimizers = [
                optim.Adam(self.critics[group_index].parameters(), lr=self.parameters["lr_critic"])
                for group_index in range(self.group_num)]

        else:
            self.actors = [Actor_Soft_Attention(input_size=self.obs_shape_list[agent_index],
                                                output_size=self.act_space_list[agent_index],
                                                hidden_dim=128).to(device) for agent_index in range(self.agent_num)]
            self.critics = [Critic_Soft_Attention(input_size=self.obs_shape_list[agent_index],
                                                  action_size=self.act_space_list,
                                                  hidden_dim=128, num_agents=len(self.act_space_list)).to(device) for
                            agent_index in range(self.agent_num)]

            self.actor_targets = [Actor_Soft_Attention(input_size=self.obs_shape_list[agent_index],
                                                output_size=self.act_space_list[agent_index],
                                                hidden_dim=128).to(device) for agent_index in range(self.agent_num)]
            self.critic_targets = [Critic_Soft_Attention(input_size=self.obs_shape_list[agent_index],
                                                  action_size=self.act_space_list,
                                                  hidden_dim=128, num_agents=len(self.act_space_list)).to(device) for
                            agent_index in range(self.agent_num)]

            self.actor_optimizers = [optim.Adam(self.actors[agent_index].parameters(), lr=self.parameters["lr_actor"])
                                     for agent_index in range(self.agent_num)]
            self.critic_optimizers = [
                optim.Adam(self.critics[agent_index].parameters(), lr=self.parameters["lr_critic"])
                for agent_index in range(self.agent_num)]

        self.action_noises = [OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.act_space_list[agent_index]),
                                                           sigma=self.parameters['sigma'])
                              for agent_index in range(self.agent_num)]

        self.update_target_weights(tau=1)

        # Create experience buffer
        self.replay_buffers = [ReplayBuffer(self.parameters["buffer_size"]) for agent_index in range(self.agent_num)]
        self.max_replay_buffer_len = self.parameters['max_replay_buffer_len']

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
                "summary_dir": '/H2G_MAAC_Summary_' + str(current_time),
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


    # update network parameters
    def update_target_weights(self, tau=1):
        # update target networks
        for actor, actor_target, critic, critic_target in zip(self.actors, self.actor_targets, self.critics,
                                                              self.critic_targets):
            for eval_param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(tau * eval_param + (1 - tau) * target_param)
            for eval_param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(tau * eval_param + (1 - tau) * target_param)

    def action(self, obs_n, evaluation=False):
        action_n = []
        obs_n = self.obs_process(obs_n)
        for i, obs in enumerate(obs_n):
            obs = torch.as_tensor([obs], dtype=torch.float32, device=device)
            if self.share_parameters:
                mu = self.actors[self.agent_group_index[i]](obs).detach().cpu().data.numpy()
            else:
                mu = self.actors[i](obs).detach().cpu().data.numpy()
            noise = np.asarray([self.action_noises[i]() for j in range(mu.shape[0])])
            # print(noise)
            pi = np.clip(mu + noise, -1, 1)
            a = mu if evaluation else pi
            action_n.append(np.array(a[0]))
        return action_n

    def obs_process(self, obs_n):
        '''
        obs_n包含每个智能体的obs，obs是由一个列表组成，列表的长度为2：
        第一维度为智能体的观测值
        第二维度为智能体的relation graph
        # relation graph通过list来表示，list中的值为agent的index，主要分为三层:
            #example: [[0], [2, 7], [[1, 3], [4, 5, 6]]]]
            # 第一层为当前agent的index
            # 第二层为，每个agent类别中，与当前agent最近的agent的index
            # 第三层为，与第二层中agent属于同一类别的agent的index
        '''
        observations_info = []
        relation_graphs = []
        for i, obs in enumerate(obs_n):
            observations_info.append(obs[0])
            relation_graphs.append(obs[1])

        obs_has_relation_info = []
        for relation_graph in relation_graphs:
            obs_i = []
            for index in list(itertools.chain.from_iterable(relation_graph)):
                obs_i.append(observations_info[index])
            obs_has_relation_info.append(np.asarray(obs_i, dtype=np.float32))

        return obs_has_relation_info

    def experience(self, obs_n, act_n, rew_n, new_obs_n, done_n):
        obs_n = self.obs_process(obs_n)
        new_obs_n = self.obs_process(new_obs_n)
        # Store transition in the replay buffer.
        for i in range(self.agent_num):
            self.replay_buffers[i].add(obs_n[i], act_n[i], [rew_n[i]], new_obs_n[i], [float(done_n[i])])

    # save_model("models/maddpg_actor_agent_", "models/maddpg_critic_agent_")
    def save_model(self):
        if self.share_parameters:
            for group_index in range(self.group_num):
                torch.save(self.actors[group_index].state_dict(),
                           self.model_path + "/H2G_MAAC_actor_group_" + str(group_index) + ".pth")
                torch.save(self.critics[group_index].state_dict(),
                           self.model_path + "/H2G_MAAC_critic_group_" + str(group_index) + ".pth")
        else:
            for agent_index in range(self.agent_num):
                torch.save(self.actors[agent_index].state_dict(),
                           self.model_path + "/H2G_MAAC_actor_agent_" + str(agent_index) + ".pth")
                torch.save(self.critics[agent_index].state_dict(),
                           self.model_path + "/H2G_MAAC_critic_agent_" + str(agent_index) + ".pth")

    def load_model(self):
        '''
        开始训练时加载之前的模型
        :return:
        '''
        if self.share_parameters:
            for group_index in range(self.group_num):
                if os.path.exists(
                        self.model_path + "/H2G_MAAC_actor_group_" + str(group_index) + ".pth") and os.path.exists(
                    self.model_path + "/H2G_MAAC_critic_group_" + str(group_index) + ".pth"):
                    try:
                        self.actors[group_index].load_state_dict(
                            torch.load(self.model_path + "/H2G_MAAC_actor_group_" + str(group_index) + ".pth"))
                        self.critics[group_index].load_state_dict(
                            torch.load(self.model_path + "/H2G_MAAC_critic_group_" + str(group_index) + ".pth"))
                    except RuntimeError as e:
                        print("模型不匹配，加载训练模型失败，将采用随机参数进行训练！！！")
                        break
                else:
                    print("模型不存在，加载训练模型失败，将采用随机参数进行训练！！！")
                    break
        else:
            for agent_index in range(self.agent_num):
                if os.path.exists(
                        self.model_path + "/H2G_MAAC_actor_agent_" + str(agent_index) + ".pth") and os.path.exists(
                    self.model_path + "/H2G_MAAC_critic_agent_" + str(agent_index) + ".pth"):
                    try:
                        self.actors[agent_index].load_state_dict(
                            torch.load(self.model_path + "/H2G_MAAC_actor_agent_" + str(agent_index) + ".pth"))
                        self.critics[agent_index].load_state_dict(
                            torch.load(self.model_path + "/H2G_MAAC_critic_agent_" + str(agent_index) + ".pth"))
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
                    torch.load(self.model_path + "/H2G_MAAC_actor_group_" + str(group_index) + ".pth"))
        else:
            for agent_index in range(self.agent_num):
                self.actors[agent_index].load_state_dict(
                    torch.load(self.model_path + "/H2G_MAAC_actor_agent_" + str(agent_index) + ".pth"))

    def can_update(self):
        can_up = []
        for i in range(self.agent_num):
            if len(self.replay_buffers[i]) > self.max_replay_buffer_len:
                can_up.append(True)
            else:
                can_up.append(False)
        return all(can_up)

    def update(self, train_step):
        replay_sample_index = self.replay_buffers[0].make_index(self.parameters['batch_size'])
        # collect replay sample from all agents
        obs_n = []
        act_n = []
        obs_next_n = []
        rew_n = []
        done_n = []
        act_next_n = []

        for i in range(self.agent_num):
            obs, act, rew, obs_next, done = self.replay_buffers[i].sample_index(replay_sample_index)
            obs_n.append(torch.tensor(obs, dtype=torch.float32, device=device))
            obs_next_n.append(torch.tensor(obs_next, dtype=torch.float32, device=device))
            act_n.append(torch.tensor(act, dtype=torch.float32, device=device))
            done_n.append(torch.tensor(done, dtype=torch.float32, device=device))
            rew_n.append(torch.tensor(rew, dtype=torch.float32, device=device))

        for i, obs_next in enumerate(obs_next_n):
            if self.share_parameters:
                target_mu = self.actor_targets[self.agent_group_index[i]](obs_next)
            else:
                target_mu = self.actor_targets[i](obs_next)
                # action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
            act_next_n.append(target_mu)

        summaries = self.train((obs_n, act_n, rew_n, obs_next_n, done_n, act_next_n))

        # if train_step % 10 == 0:  # only update every 100 steps
        self.update_target_weights(tau=self.parameters["tau"])

        for i in range(self.agent_num):
            if self.share_parameters:
                for key in summaries.keys():
                    self.summary_writers[i].add_scalar(key, summaries[key][self.agent_group_index[i]],
                                                       global_step=train_step)
            else:
                for key in summaries.keys():
                    self.summary_writers[i].add_scalar(key, summaries[key][i], global_step=train_step)
            self.summary_writers[i].flush()

    def train(self, memories):
        obs_n, act_n, rew_n, obs_next_n, done_n, act_next_n = memories

        if self.share_parameters:
            q_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.group_num)]
            actor_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.group_num)]
            for agent_index in range(self.agent_num):
                # critic_loss
                q_target = self.critic_targets[self.agent_group_index[agent_index]]([obs_next_n[agent_index]] + act_next_n)
                q_target = rew_n[agent_index] + self.parameters['gamma'] * q_target * (
                        1 - done_n[agent_index])
                q_eval = self.critics[self.agent_group_index[agent_index]]([obs_n[agent_index]] + act_n)
                q_loss[self.agent_group_index[agent_index]] += nn.MSELoss()(q_eval, q_target.detach())

                # actor_loss
                mu = self.actors[self.agent_group_index[agent_index]](obs_n[agent_index])
                temp_act_n = [mu if i == agent_index else act for i, act in enumerate(act_n)]
                actor_loss[self.agent_group_index[agent_index]] += -torch.mean(
                    self.critics[self.agent_group_index[agent_index]]([obs_n[agent_index]] + temp_act_n))

            for group_index in range(self.group_num):
                # optimize actor
                self.actor_optimizers[group_index].zero_grad()
                actor_loss[group_index].backward()
                self.actor_optimizers[group_index].step()

                # optimize critic
                self.critic_optimizers[group_index].zero_grad()
                q_loss[group_index].backward()
                self.critic_optimizers[group_index].step()
        else:
            q_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
            actor_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]

            for agent_index in range(self.agent_num):
                # critic_loss
                q_target = self.critic_targets[agent_index]([obs_next_n[agent_index]] + act_next_n)
                q_target = rew_n[agent_index] + self.parameters['gamma'] * q_target * (
                        1 - done_n[agent_index])
                q_eval = self.critics[agent_index]([obs_n[agent_index]] + act_n)
                q_l = nn.MSELoss()(q_eval, q_target.detach())

                self.critic_optimizers[agent_index].zero_grad()
                q_l.backward()
                self.critic_optimizers[agent_index].step()
                q_loss[agent_index] = q_l

                # actor_loss
                mu = self.actors[agent_index](obs_n[agent_index])
                tmp_act_n = [mu if i == agent_index else act for i, act in enumerate(act_n)]
                a_l = -torch.mean(self.critics[agent_index]([obs_n[agent_index]] + tmp_act_n))

                self.actor_optimizers[agent_index].zero_grad()
                a_l.backward(retain_graph=True)
                self.actor_optimizers[agent_index].step()
                actor_loss[agent_index] = a_l

        summaries = dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/q_loss', q_loss],
        ])
        return summaries


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
