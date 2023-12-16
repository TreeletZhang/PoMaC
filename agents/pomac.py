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
import copy
import random

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# treelet 
class PfNet(nn.Module):
    def __init__(self, obs_size, target_loc_size, mid_size):
        super(PfNet, self).__init__()
        self.linear1 = nn.Linear(obs_size + target_loc_size, mid_size)
        # self.linear3 = nn.Linear(mid_size, mid_size)
        self.linear4 = nn.Linear(mid_size, 2)

    def forward(self, obs_and_target):
        x = torch.cat(obs_and_target, dim=-1)
        x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        out = F.softmax(self.linear4(x))
        return out

# message encode network
class MsgEncoder(nn.Module):
    def __init__(self, message_size, output_size, mid_size):
        super(MsgEncoder, self).__init__()
        self.linear1 = nn.Linear(message_size, mid_size)
        self.linear2 = nn.Linear(mid_size, output_size)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x

# Actor network
class MLPActor(nn.Module):
    def __init__(self, obs_size, msg_size, output_size, mid_size):
        super(MLPActor, self).__init__()
        self.linear1 = nn.Linear(obs_size+msg_size, mid_size)
        self.linear2 = nn.Linear(mid_size, mid_size)
        self.linear3 = nn.Linear(mid_size, mid_size)
        self.linear4 = nn.Linear(mid_size, output_size)

    def forward(self, obs_msg):
        x = torch.cat(obs_msg, dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = torch.tanh(self.linear4(x))
        return x


# Critic Model
class MLPCritic(nn.Module):
    def __init__(self, state_sizes, action_sizes, mid_size):
        super(MLPCritic, self).__init__()
        self.linear1 = nn.Linear(sum(state_sizes) + sum(action_sizes), mid_size) 
        self.linear2 = nn.Linear(mid_size, mid_size)
        self.linear3 = nn.Linear(mid_size, mid_size)
        self.linear4 = nn.Linear(mid_size, 1)

    def forward(self, states_actions):
        x = torch.cat(states_actions, dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


# POMAC Agent Class
class Agent():
    def __init__(self,
                 name,
                 obs_shape,
                 message_shape,
                 target_loc_space_n,
                 n_agents_obs,
                 prior_buffer,
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
        # treelet
        self.message_shape_list = message_shape
        self.target_loc_shape_list = target_loc_space_n
        self.num_agents_obs = n_agents_obs
        self.prior_buffer = prior_buffer
        # treelet
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
        # treelet
        self.group_message_shape_list = [0 for i in range(self.group_num)]  
        self.group_target_loc_list = [0 for i in range(self.group_num)]
        # treelet
        for agent_index, group_index in enumerate(self.agent_group_index):
            if self.group_obs_shape_list[group_index] == 0:
                self.group_obs_shape_list[group_index] = self.obs_shape_list[agent_index]  
            if self.group_act_shape_list[group_index] == 0:
                self.group_act_shape_list[group_index] = self.act_space_list[agent_index]
            # treelet
            if self.group_message_shape_list[group_index] == 0:
                self.group_message_shape_list[group_index] = self.message_shape_list[agent_index]  
            if self.group_target_loc_list[group_index] == 0:
                self.group_target_loc_list[group_index] = self.target_loc_shape_list[agent_index]
            # treelet

        if self.share_parameters:
            self.actors = [MLPActor(obs_size=self.group_obs_shape_list[group_index],
                                    msg_size=32,  
                                    output_size=self.group_act_shape_list[group_index],
                                    mid_size=128).to(device) for group_index in range(self.group_num)]
            self.critics = [MLPCritic(state_sizes=self.obs_shape_list,
                                      action_sizes=self.act_space_list,
                                      mid_size=128).to(device) for group_index in range(self.group_num)]

            self.actor_targets = [MLPActor(obs_size=self.group_obs_shape_list[group_index],
                                           msg_size=32,  
                                           output_size=self.group_act_shape_list[group_index],
                                           mid_size=128).to(device) for group_index in range(self.group_num)]
            self.critic_targets = [MLPCritic(state_sizes=self.obs_shape_list,
                                             action_sizes=self.act_space_list,
                                             mid_size=128).to(device) for group_index in range(self.group_num)]

            self.actor_optimizers = [optim.Adam(self.actors[group_index].parameters(), lr=self.parameters["lr_actor"])
                                     for group_index in range(self.group_num)]
            self.critic_optimizers = [
                optim.Adam(self.critics[group_index].parameters(), lr=self.parameters["lr_critic"])
                for group_index in range(self.group_num)]
            # treelet
            self.pfNets = [PfNet(obs_size=self.group_obs_shape_list[group_index],
                                  target_loc_size=self.group_target_loc_list[group_index][0],
                                  mid_size=128).to(device) for group_index in range(self.group_num)]
            self.pfNets_target = [PfNet(obs_size=self.group_obs_shape_list[group_index],
                                  target_loc_size=self.group_target_loc_list[group_index][0],
                                  mid_size=128).to(device) for group_index in range(self.group_num)]
            self.pfNets_optimizers = [
                optim.Adam(self.pfNets[group_index].parameters(), lr=self.parameters["lr_pfNet"])
                for group_index in range(self.group_num)]
            self.msg_encoder = [MsgEncoder(message_size=self.num_agents_obs * self.obs_shape_list[0],
                                           output_size=32,  
                                           mid_size=128).to(device) for group_index in range(self.group_num)]
            # treelet
        else:
            self.actors = [MLPActor(obs_size=self.obs_shape_list[agent_index],
                                    msg_size=32,  
                                    output_size=self.act_space_list[agent_index],
                                    mid_size=128).to(device) for agent_index in range(self.agent_num)]
            self.critics = [MLPCritic(state_sizes=self.obs_shape_list,
                                      action_sizes=self.act_space_list,
                                      mid_size=128).to(device) for agent_index in range(self.agent_num)]

            self.actor_targets = [MLPActor(obs_size=self.obs_shape_list[agent_index],
                                           msg_size=32,  
                                           output_size=self.act_space_list[agent_index],
                                           mid_size=128).to(device) for agent_index in range(self.agent_num)]
            self.critic_targets = [MLPCritic(state_sizes=self.obs_shape_list,
                                             action_sizes=self.act_space_list,
                                             mid_size=128).to(device) for agent_index in range(self.agent_num)]

            self.actor_optimizers = [optim.Adam(self.actors[agent_index].parameters(), lr=self.parameters["lr_actor"])
                                     for agent_index in range(self.agent_num)]
            self.critic_optimizers = [
                optim.Adam(self.critics[agent_index].parameters(), lr=self.parameters["lr_critic"])
                for agent_index in range(self.agent_num)]
            # treelet
            self.pfNets = [PfNet(obs_size=self.obs_shape_list[agent_index],
                                 target_loc_size=self.target_loc_shape_list[agent_index][0],
                                 mid_size=128).to(device) for agent_index in range(self.agent_num)]
            self.pfNets_target = [PfNet(obs_size=self.obs_shape_list[agent_index],
                                 target_loc_size=self.target_loc_shape_list[agent_index][0],
                                 mid_size=128).to(device) for agent_index in range(self.agent_num)]
            self.pfNets_optimizers = [
                optim.Adam(self.pfNets[agent_index].parameters(), lr=self.parameters["lr_pfNet"])
                for agent_index in range(self.agent_num)]

            self.msg_encoder = [MsgEncoder(message_size=self.num_agents_obs*self.obs_shape_list[0],
                                           output_size=32,
                                           mid_size=128).to(device) for agent_index in range(self.agent_num)]
            # treelet

        self.action_noises = [OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.act_space_list[agent_index]),
                                                           sigma=self.parameters['sigma'])
                              for agent_index in range(self.agent_num)]

        self.update_target_weights(tau=1)

        # Create experience buffer
        self.replay_buffers = [ReplayBuffer(self.parameters["buffer_size"]) for agent_index in range(self.agent_num)]
        self.max_replay_buffer_len = self.parameters['max_replay_buffer_len']
        self.prior_buffer_size = self.parameters['prior_buffer_size']

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
                "summary_dir": '/POMAC_Summary_' + str(current_time),
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

    def action(self, obs_n, message_n, evaluation=False):
        action_n = []
        message_n_embedding = []
        for i, message in enumerate(message_n):
            message_tensor = torch.as_tensor(np.array([message]), dtype=torch.float32, device=device).reshape(-1)  # 转换为张量后摊平（3，16）--》48
            message_tensor = message_tensor.unsqueeze(0)
            if self.share_parameters:
                message_embedding = self.msg_encoder[self.agent_group_index[i]](message_tensor)
            else:
                message_embedding = self.msg_encoder[i](message_tensor)
            message_n_embedding.append(message_embedding)
        for i, obs in enumerate(obs_n):
            obs = torch.as_tensor(np.array([obs]), dtype=torch.float32, device=device)
            if self.share_parameters:
                mu = self.actors[self.agent_group_index[i]]([obs]+[message_n_embedding[i]]).detach().cpu().data.numpy()
            else:
                mu = self.actors[i]([obs]+[message_n_embedding[i]]).detach().cpu().data.numpy()  
            noise = np.asarray([self.action_noises[i]() for j in range(mu.shape[0])])
            
            pi = np.clip(mu + noise, -1, 1)
            a = mu if evaluation else pi
            action_n.append(np.array(a[0]))
        return action_n

    def experience(self, obs_n, other_loc_n, other_idx_n, message_n, new_message_n, act_n, rew_n, new_obs_n, new_other_loc_n, new_other_idx_n, done_n):
        # Store transition in the replay buffer.

        for i in range(self.agent_num):
            self.replay_buffers[i].add_i2c(obs_n[i], other_loc_n[i], other_idx_n[i], message_n[i], new_message_n[i], act_n[i], [rew_n[i]], new_obs_n[i], new_other_loc_n[i], new_other_idx_n[i], [float(done_n[i])])

    
    def save_model(self):
        if self.share_parameters:
            for group_index in range(self.group_num):
                torch.save(self.actors[group_index].state_dict(),
                           self.model_path + "/pomac_actor_group_" + str(group_index) + ".pth")
                torch.save(self.critics[group_index].state_dict(),
                           self.model_path + "/pomac_critic_group_" + str(group_index) + ".pth")
        else:
            for agent_index in range(self.agent_num):
                torch.save(self.actors[agent_index].state_dict(),
                           self.model_path + "/pomac_actor_agent_" + str(agent_index) + ".pth")
                torch.save(self.critics[agent_index].state_dict(),
                           self.model_path + "/pomac_critic_agent_" + str(agent_index) + ".pth")

    def load_model(self):
        '''
        开始训练时加载之前的模型
        :return:
        '''
        if self.share_parameters:
            for group_index in range(self.group_num):
                if os.path.exists(
                        self.model_path + "/pomac_actor_group_" + str(group_index) + ".pth") and os.path.exists(
                    self.model_path + "/pomac_critic_group_" + str(group_index) + ".pth"):
                    try:
                        self.actors[group_index].load_state_dict(
                            torch.load(self.model_path + "/pomac_actor_group_" + str(group_index) + ".pth"))
                        self.critics[group_index].load_state_dict(
                            torch.load(self.model_path + "/pomac_critic_group_" + str(group_index) + ".pth"))
                    except RuntimeError as e:
                        print("模型不匹配，加载训练模型失败，将采用随机参数进行训练！！！")
                        break
                else:
                    print("模型不存在，加载训练模型失败，将采用随机参数进行训练！！！")
                    break
        else:
            for agent_index in range(self.agent_num):
                if os.path.exists(
                        self.model_path + "/pomac_actor_agent_" + str(agent_index) + ".pth") and os.path.exists(
                    self.model_path + "/pomac_critic_agent_" + str(agent_index) + ".pth"):
                    try:
                        self.actors[agent_index].load_state_dict(
                            torch.load(self.model_path + "/pomac_actor_agent_" + str(agent_index) + ".pth"))
                        self.critics[agent_index].load_state_dict(
                            torch.load(self.model_path + "/pomac_critic_agent_" + str(agent_index) + ".pth"))
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
                    torch.load(self.model_path + "/pomac_actor_group_" + str(group_index) + ".pth"))
        else:
            for agent_index in range(self.agent_num):
                self.actors[agent_index].load_state_dict(
                    torch.load(self.model_path + "/pomac_actor_agent_" + str(agent_index) + ".pth"))

    def can_update(self):  
        can_up = []
        for i in range(self.agent_num):
            if len(self.replay_buffers[i]) > self.max_replay_buffer_len:
                can_up.append(True)
            else:
                can_up.append(False)
        return all(can_up) 

    def update(self, train_step, PfNet):
        pfNet_summaries_flag = False
        replay_sample_index = self.replay_buffers[0].make_index(self.parameters['batch_size'])
        
        obs_n = [] 
        act_n = [] 
        obs_next_n = [] 
        rew_n = []
        done_n = []
        act_next_n = []
        # treelet
        message_n = []
        message_n_embedding = []
        message_next_n_embedding = []

        obs_n_for_prior = []
        act_n_for_prior = []
        target_loc_n = []
        target_idx_n = []
        # treelet

        for i in range(self.agent_num):
            obs, other_loc, other_idx, message, new_message, act, rew, new_obs, new_other_loc, new_other_idx, done = self.replay_buffers[i].sample_index(replay_sample_index)
            obs_n_for_prior.append(obs)
            act_n_for_prior.append(act)
            obs_n.append(torch.tensor(obs, dtype=torch.float32, device=device)) 
            obs_next_n.append(torch.tensor(new_obs, dtype=torch.float32, device=device))  
            act_n.append(torch.tensor(act, dtype=torch.float32, device=device))  
            done_n.append(torch.tensor(done, dtype=torch.float32, device=device))  
            rew_n.append(torch.tensor(rew, dtype=torch.float32, device=device))  
            # treelet
            target_loc_n.append(other_loc)
            target_idx_n.append(other_idx)
            
            message_tensor = torch.as_tensor(np.array([message]), dtype=torch.float32, device=device).reshape(
                self.parameters['batch_size'], -1)  
            if self.share_parameters:
                message_embedding = self.msg_encoder[self.agent_group_index[i]](message_tensor)
            else:
                message_embedding = self.msg_encoder[i](message_tensor)
            message_n_embedding.append(message_embedding)

            new_message_tensor = torch.as_tensor(np.array([new_message]), dtype=torch.float32, device=device).reshape(
                self.parameters['batch_size'], -1)  
            new_message_tensor = new_message_tensor
            if self.share_parameters:
                new_message_embedding = self.msg_encoder[self.agent_group_index[i]](new_message_tensor)
            else:
                new_message_embedding = self.msg_encoder[i](new_message_tensor)
            message_next_n_embedding.append(new_message_embedding)
            # treelet

        for i, obs_next in enumerate(obs_next_n):
            if self.share_parameters:
                target_mu = self.actor_targets[self.agent_group_index[i]]([obs_next] + [message_next_n_embedding[i]])
            else:
                target_mu = self.actor_targets[i]([obs_next] + [message_next_n_embedding[i]])
                
            act_next_n.append(target_mu)
        summaries = self.train((obs_n, act_n, rew_n, obs_next_n, done_n, act_next_n, message_n_embedding))

        if PfNet and len(self.replay_buffers[0]) > self.prior_buffer_size:
            
            pfNet_summaries = self.prior_train(self.parameters['prior_batch_size'],(obs_n_for_prior, target_loc_n, target_idx_n, act_n_for_prior))
            pfNet_summaries_flag = True
       

        # if train_step % 10 == 0:  # only update every 100 steps
        self.update_target_weights(tau=self.parameters["tau"])

        for i in range(self.agent_num):
            if self.share_parameters:
                for key in summaries.keys():
                    self.summary_writers[i].add_scalar(key, summaries[key][self.agent_group_index[i]],
                                                       global_step=train_step)
                if PfNet and pfNet_summaries_flag:
                    for key in pfNet_summaries.keys():
                        self.summary_writers[i].add_scalar(key, pfNet_summaries[key][self.agent_group_index[i]],
                                                           global_step=train_step)
            else:
                for key in summaries.keys():
                    self.summary_writers[i].add_scalar(key, summaries[key][i], global_step=train_step)
                if PfNet and pfNet_summaries_flag:
                    for key in pfNet_summaries.keys():
                        self.summary_writers[i].add_scalar(key, pfNet_summaries[key][i], global_step=train_step)
            self.summary_writers[i].flush()

    def train(self, memories):
        obs_n, act_n, rew_n, obs_next_n, done_n, act_next_n, message_n_embedding = memories

        if self.share_parameters:
            q_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.group_num)]
            actor_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.group_num)]
            for agent_index in range(self.agent_num):
                # critic_loss
                q_target = self.critic_targets[self.agent_group_index[agent_index]](obs_next_n + act_next_n)
                q_target = rew_n[agent_index] + self.parameters['gamma'] * q_target * (
                        1 - done_n[agent_index])
                q_eval = self.critics[self.agent_group_index[agent_index]](obs_n + act_n)
                q_loss[self.agent_group_index[agent_index]] += nn.MSELoss()(q_eval, q_target.detach())

                # actor_loss
                mu = self.actors[self.agent_group_index[agent_index]]([obs_n[agent_index]] + [message_n_embedding[agent_index]])
                temp_act_n = [mu if i == agent_index else act for i, act in enumerate(act_n)]
                actor_loss[self.agent_group_index[agent_index]] += -torch.mean(
                    self.critics[self.agent_group_index[agent_index]](obs_n + temp_act_n))

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
                q_target = self.critic_targets[agent_index](obs_next_n + act_next_n)
                q_target = rew_n[agent_index] + self.parameters['gamma'] * q_target * (
                        1 - done_n[agent_index])
                q_eval = self.critics[agent_index](obs_n + act_n)
                q_l = nn.MSELoss()(q_eval, q_target.detach())

                self.critic_optimizers[agent_index].zero_grad()
                q_l.backward()
                self.critic_optimizers[agent_index].step()
                q_loss[agent_index] = q_l

                mu = self.actors[agent_index]([obs_n[agent_index]] + [message_n_embedding[agent_index]])
                tmp_act_n = [mu if i == agent_index else act for i, act in enumerate(act_n)]
                a_l = -torch.mean(self.critics[agent_index](obs_n + tmp_act_n))

                self.actor_optimizers[agent_index].zero_grad()
                a_l.backward(retain_graph=True)
                self.actor_optimizers[agent_index].step()
                actor_loss[agent_index] = a_l



        summaries = dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/q_loss', q_loss],
        ])
        return summaries


    # treelet
    def target_comm(self, obs_n, other_loc_n, other_idx_n, train_parameters):
        obs_n_tensor = []
        other_loc_n_tensor = copy.deepcopy(other_loc_n)  
        num_comm = 0
        for obs in obs_n:
            obs_n_tensor.append(torch.tensor(np.array([obs]), dtype=torch.float32, device=device))  
        for agent_index in range(self.agent_num):
            for loc_index in range(len(other_loc_n[agent_index])):
                other_loc_n_tensor[agent_index][loc_index] = torch.tensor(np.array([other_loc_n_tensor[agent_index][loc_index]]), dtype=torch.float32, device=device)
        target_index_n = []
        c_pred = []
        for agent_index in range(self.agent_num):
            other_loc = other_loc_n_tensor[agent_index]  
            other_idx = other_idx_n[agent_index]  
            target_index = []
            c_pred_each_agent = []
            for j in range(len(other_loc)):
                if self.share_parameters:
                    comm_pred = self.pfNets[self.agent_group_index[agent_index]]([obs_n_tensor[agent_index]] + [other_loc[j]])
                else:
                    comm_pred = self.pfNets[agent_index]([obs_n_tensor[agent_index]] + [other_loc[j]])
                c_pred_each_agent.append(comm_pred.detach().cpu().data.numpy())
                comm_flag = torch.greater(comm_pred[:,0], 0.5)  
                
                if comm_flag:
                    target_index.append(other_idx[j])
            num_comm += len(target_index)
            target_index_n.append(target_index)
            c_pred.append(c_pred_each_agent)
       
        return target_index_n, num_comm, c_pred_each_agent
    # treelet

    def target_comm_rule(self, other_idx_n):
        num_comm = len(other_idx_n)
        target_index_n = other_idx_n
        return target_index_n, num_comm

    def target_comm_ruleAndnet(self, distWithagent, lmDist, numLm, velocity, obs_n, other_loc_n, other_idx_n, rew_n):
        obs_n_tensor = []
        other_loc_n_tensor = copy.deepcopy(other_loc_n)  
        num_comm = 0
        for obs in obs_n:
            obs_n_tensor.append(
                torch.tensor(np.array([obs]), dtype=torch.float32, device=device)) 
        for agent_index in range(self.agent_num):
            for loc_index in range(len(other_loc_n[agent_index])):
                other_loc_n_tensor[agent_index][loc_index] = torch.tensor(
                    np.array([other_loc_n_tensor[agent_index][loc_index]]), dtype=torch.float32, device=device)
        target_index_n = []
        c_pred = []
        for agent_index in range(self.agent_num):
            other_loc = other_loc_n_tensor[agent_index]  
            other_idx = other_idx_n[agent_index]  
            target_index = []
            c_pred_each_agent = []
            c_pred_r_each_agent = []
            for j in range(len(other_loc)):
                if self.share_parameters:
                    comm_pred_net = self.pfNets[self.agent_group_index[agent_index]](
                        [obs_n_tensor[agent_index]] + [other_loc[j]])
                else:
                    comm_pred_net = self.pfNets[agent_index]([obs_n_tensor[agent_index]] + [other_loc[j]])
                c_pred_each_agent.append(comm_pred_net.detach().cpu().data.numpy())

                
                comm_pred_r = self.fuzzy_rules(distWithagent, lmDist, numLm, velocity)

                if rew_n[0] <= -2.0:
                    alpha = 0.3
                    comm_pred_final = alpha * comm_pred_net + (1-alpha) * comm_pred_r
                if -2.0 < rew_n[0] < -1.2:
                    alpha = 0.6
                    comm_pred_final = alpha * comm_pred_net + (1 - alpha) * comm_pred_r
                if rew_n[0] >= -1.2:
                    alpha = 0.85
                    comm_pred_final = alpha * comm_pred_net + (1 - alpha) * comm_pred_r

                comm_flag = torch.greater(comm_pred_final[:, 0], 0.5)  
                
                if comm_flag:
                    target_index.append(other_idx[j])
            num_comm += len(target_index)
            target_index_n.append(target_index)
            c_pred.append(c_pred_each_agent)


    def fuzzy_rules(self, distWithagent, lmDist, numLm, velocity):
        mu_NE = 2 - 0.2 * distWithagent  # membership of NE
        T_NE = np.clip(mu_NE, 0, 1)  # truth value T of 'DistBetweenAgent is NE' precondition
        mu_FA = 0.2 * distWithagent - 2
        T_FA = np.clip(mu_FA, 0, 1)

        mu_NE_ = 2 - 0.2 * lmDist  # membership of NE'
        T_NE_ = np.clip(mu_NE_, 0, 1)  # truth value T of 'DistBetweenAgent is NE' precondition
        mu_FA_ = 0.2 * lmDist - 1
        T_FA_ = np.clip(mu_FA_, 0, 1)

        mu_LA = 0.5 * numLm - 1  # membership of LA
        T_LA = np.clip(mu_LA, 0, 1)  # truth value T of 'DistBetweenAgent is LA precondition
        mu_SM = 2 - 0.5 * distWithagent
        T_SM = np.clip(mu_SM, 0, 1)

        mu_SL = 2 - 4 * velocity  # membership of SL
        T_SL = np.clip(mu_SL, 0, 1)  # truth value T of 'DistBetweenAgent is SL precondition
        mu_QU = 4 * velocity - 2
        T_QU = np.clip(mu_QU, 0, 1)

        # R1 'IF DistBetweenAgents is NE and Velocity is SL THEN p is HG'
        comm_p_R1 = np.mean(np.array([T_NE, T_SL]))

        # R3 'IF NumLandmark is LA and LandmarkDist is NE' THEN p is HG'
        comm_p_R3 = np.mean(np.array([T_NE, T_SL]))

        comm_p_r = np.mean(np.array(comm_p_R1, comm_p_R3))

        return comm_p_r

    # treelet
    def get_message(self, obs_n, target_idx_n):
        message_n = [np.zeros((self.num_agents_obs, self.obs_shape_list[0]), dtype=np.float32) for _ in range(self.agent_num)]
        for agent_index in range(self.agent_num):
            for target_index in range(len(target_idx_n[agent_index])):
                message_n[agent_index][target_index, :] = obs_n[target_idx_n[agent_index][target_index]]
        return message_n  

    def get_samples_for_pfNet_2(self, agent_idx, memory):
        obs_n, other_loc_n, other_idx_n, act_n = memory
        target_loc = other_loc_n[agent_idx]
        target_idx = other_idx_n[agent_idx]
        obs_inputs, target_loc_inputs, KL_values = self.get_KL_value(obs_n, act_n, target_loc, target_idx, agent_idx)
        is_full = self.prior_buffer.insert(len(obs_inputs), obs_inputs, target_loc_inputs, KL_values)
        return is_full


    # treelet
    def get_samples_for_pfNet(self, agent_idx):
        pfNet_replay_sample_index = self.replay_buffers[agent_idx].make_index(self.parameters['prior_buffer_size'])
        obs_n = []
        obs_next_n = []
        act_n = []
        message_n = []
        target_loc_next_n = []
        target_idx_next_n = []
        for i in range(self.agent_num):
            sample_index_start = time.time()
            obs, other_loc, other_idx, message, new_message, act, rew, new_obs, new_other_loc, new_other_idx, done = self.replay_buffers[i].sample_index(pfNet_replay_sample_index)
            obs_n.append(obs)  #
            obs_next_n.append(new_obs)
            message_n.append(message)
            act_n.append(act)
            target_loc_next_n.append(new_other_loc)
            target_idx_next_n.append(new_other_idx)
        obs, target_loc, target_idx, message, new_message, act, rew, obs_next, target_loc_next, target_loc_idx_next, done = self.replay_buffers[agent_idx].sample_index(pfNet_replay_sample_index)  
        obs_inputs, target_loc_inputs, KL_values = self.get_KL_value(obs_n, act_n, target_loc, target_idx, agent_idx) 
        is_full = self.prior_buffer.insert(len(obs_inputs), obs_inputs, target_loc_inputs, KL_values)
        return is_full

    # treelet
    def get_KL_value(self, obs_n, act_n, target_loc, target_idx, agent_idx):
        act_dim_self = self.act_space_list[0]
        sample_size = len(obs_n[0])
        KL_values = []
        target_loc_input_n = [[] for _ in range(self.agent_num)]  
        obs_act_idx = [[] for _ in range(self.agent_num)]  
        for i in range(sample_size):
            for j in range(self.num_agents_obs):
                idx_tmp = target_idx[i, j]
                target_loc_input_n[idx_tmp].append(target_loc[i,j,:])  
                obs_act_idx[idx_tmp].append(i)  
        obs_input_n = [[obs_n[n][obs_act_idx[k], :] for n in range(self.agent_num)] for k in range(self.agent_num)]  
        act_input_n = [[act_n[n][obs_act_idx[k], :] for n in range(self.agent_num)] for k in range(self.agent_num)]  
        for i in range(self.agent_num): 
            if i == agent_idx or len(obs_act_idx[i]) == 0: continue  
            act_dim_other = len(act_n[i][0])
            obs_input = obs_input_n[i][:]
            obs_input_tensor = []
            for obs_input_i in range(len(obs_input)):
                obs_input_tensor.append(torch.tensor(obs_input[obs_input_i],dtype=torch.float32, device=device))
            act_input = act_input_n[i][:]
            act_target = act_input[i][:, :].copy()
            Q_s = []
            Q_s_t = []
            for k in range(act_dim_self):
                one_hot = [0] * act_dim_self
                one_hot[k] = 1
                act_input[agent_idx][:, :] = one_hot[:]
                act_input_tensor = []
                for act_input_i in range(len(act_input)):
                    act_input_tensor.append(torch.tensor(act_input[act_input_i], dtype=torch.float32, device=device))
                if self.share_parameters:
                    q_s_print = self.critics[self.agent_group_index[agent_idx]](obs_input_tensor + act_input_tensor)
                else:
                    q_s_print = self.critics[agent_idx](obs_input_tensor + act_input_tensor)
                q_s_print = q_s_print.detach().cpu().data.numpy()
                q_s_print = q_s_print.flatten()
                Q_s.append(np.exp(q_s_print + 1e-8))
                Q_tmp = []
                for m in range(act_dim_other):
                    one_hot = [0] * act_dim_other
                    one_hot[m] = 1
                    act_input[i][:, :] = one_hot[:]
                    act_input_tensor = []
                    for act_input_ii in range(len(act_input)):
                        act_input_tensor.append(torch.tensor(act_input[act_input_ii], dtype=torch.float32, device=device))
                    if self.share_parameters:
                        Q_tmp.append(np.exp(self.critics[self.agent_group_index[agent_idx]](obs_input_tensor + act_input_tensor).detach().cpu().data.numpy().flatten() + 1e-8))
                    else:
                        Q_tmp.append(np.exp(self.critics[agent_idx](obs_input_tensor + act_input_tensor).detach().cpu().data.numpy().flatten() + 1e-8))  
                act_input[i][:, :] = act_target[:, :]
                Q_s_t.append(Q_tmp)
            Q_t_sum = [sum(Q_s_t[ii]) for ii in range(act_dim_self)]
            prob_s_marg = np.array(Q_t_sum / sum(Q_t_sum))
            prob_s_cond_t = np.array(Q_s / sum(Q_s))
            KL_value = np.sum(prob_s_marg * np.log(prob_s_marg / prob_s_cond_t), 0)
            KL_values.append(KL_value)
        KL_values = np.concatenate(KL_values, 0)
        obs_inputs = np.concatenate([obs_input_n[ii][agent_idx] for ii in range(self.agent_num)], 0)
        while [] in target_loc_input_n:
            target_loc_input_n.remove([])
        target_loc_inputs = np.concatenate(target_loc_input_n, 0)
        return obs_inputs, target_loc_inputs, KL_values

    def prior_train(self, batch_size, memory):
        if self.share_parameters:
            w_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
            agent_idx = random.randint(0, self.agent_num-1)

            is_full = self.get_samples_for_pfNet(agent_idx)  
            if is_full:
                
                for _ in range(self.parameters['prior_buffer_size']//batch_size):  
                    obs_inputs, obs_loc_inputs, labels = self.prior_buffer.get_samples(
                        batch_size)  
                    obs_inputs_tensor = torch.tensor(obs_inputs, dtype=torch.float32, device=device)
                    obs_loc_inputs_tensor = torch.tensor(obs_loc_inputs, dtype=torch.float32, device=device)
                    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)  
                    c_pred = self.pfNets[self.agent_group_index[agent_idx]]([obs_inputs_tensor] + [obs_loc_inputs_tensor])
                    w_loss[self.agent_group_index[agent_idx]] += nn.CrossEntropyLoss()(c_pred, labels_tensor) 
            for group_index in range(self.group_num):
                self.pfNets_optimizers[group_index].zero_grad()
                w_loss[group_index].backward()
                self.pfNets_optimizers[group_index].step()
        else:
            w_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
            agent_idx = random.randint(0, self.agent_num - 1)
            is_full = self.get_samples_for_pfNet(agent_idx)
            if is_full:
                for _ in range(self.parameters['prior_buffer_size']//batch_size):  
                    obs_inputs, obs_loc_inputs, labels = self.prior_buffer.get_samples(batch_size)  
                    obs_inputs_tensor = torch.tensor(obs_inputs, dtype=torch.float32, device=device)
                    obs_loc_inputs_tensor = torch.tensor(obs_loc_inputs, dtype=torch.float32, device=device)
                    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
                    c_pred = self.pfNets[agent_idx]([obs_inputs_tensor] + [obs_loc_inputs_tensor])  
                    w_l = nn.CrossEntropyLoss()(c_pred, labels_tensor)  
                    self.pfNets_optimizers[agent_idx].zero_grad()
                    w_l.backward()
                    self.pfNets_optimizers[agent_idx].step()
                    w_loss[agent_idx] = w_l
        pfNet_summaries = dict([['LOSS/pfNet_loss', w_loss]])
        return pfNet_summaries


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
