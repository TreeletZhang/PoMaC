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
from buffers.replay_buffer import ReplayBuffer

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SoftQNetwork(nn.Module):
    def __init__(self, state_size, action_size, mid_size):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(state_size + action_size, mid_size)
        self.linear2 = nn.Linear(mid_size, mid_size)
        self.linear3 = nn.Linear(mid_size, mid_size)
        self.linear4 = nn.Linear(mid_size, 1)

    def forward(self, state_action):
        x = torch.cat(state_action, dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 mid_size,
                 action_range=1.,
                 log_std_min=-20,
                 log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(state_size, mid_size)
        self.linear2 = nn.Linear(mid_size, mid_size)
        self.linear3 = nn.Linear(mid_size, mid_size)
        self.mean_linear = nn.Linear(mid_size, action_size)
        self.log_std_linear = nn.Linear(mid_size, action_size)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_actions = action_size

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        #gaussian_rsample
        std = log_std.exp()  # no clip in evaluation, clip affects gradients flow
        z = torch.distributions.Normal(0, 1).sample(mean.shape).to(device)
        action = torch.tanh(mean + std * z)  # TanhNormal distribution as actions; reparameterization trick
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(
            1. - action.pow(2) + epsilon)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        #gaussian_rsample
        std = log_std.exp()  # no clip in evaluation, clip affects gradients flow
        z = torch.distributions.Normal(0, 1).sample(mean.shape).to(device)
        action = torch.tanh(mean + std * z)

        action = torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else \
        action.detach().cpu().numpy()[0]
        return action

    def sample_action(self, ):
        a = torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return a.numpy()


class Agent():
    '''
    Soft Actor-Critic version 2
    using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
    add alpha loss compared with version 1
    paper: https://arxiv.org/pdf/1812.05905.pdf
    '''
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
            self.soft_q_net1s = [SoftQNetwork(state_size=self.group_obs_shape_list[group_index],
                                              action_size=self.group_act_shape_list[group_index],
                                              mid_size=128).to(device) for group_index in range(self.group_num)]
            self.soft_q_net2s = [SoftQNetwork(state_size=self.group_obs_shape_list[group_index],
                                              action_size=self.group_act_shape_list[group_index],
                                              mid_size=128).to(device) for group_index in range(self.group_num)]
            self.target_soft_q_net1s = [SoftQNetwork(state_size=self.group_obs_shape_list[group_index],
                                              action_size=self.group_act_shape_list[group_index],
                                              mid_size=128).to(device) for group_index in range(self.group_num)]
            self.target_soft_q_net2s = [SoftQNetwork(state_size=self.group_obs_shape_list[group_index],
                                              action_size=self.group_act_shape_list[group_index],
                                              mid_size=128).to(device) for group_index in range(self.group_num)]
            self.policy_nets = [PolicyNetwork(state_size=self.group_obs_shape_list[group_index],
                                              action_size=self.group_act_shape_list[group_index],
                                              mid_size=128).to(device) for group_index in range(self.group_num)]
            self.log_alphas = [torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device) for group_index in
                               range(self.group_num)]

            # entropy = -log(1/|A|) = log |A|
            self.target_entropys = [0.98 * (-self.group_act_shape_list[group_index]) for group_index in
                                    range(self.group_num)]
            self.alphas = [self.parameters["alpha"] for group_index in range(self.group_num)]

            self.soft_q_optimizer1s = [
                optim.Adam(self.soft_q_net1s[group_index].parameters(), lr=self.parameters["soft_q_lr"]) for group_index
                in range(self.group_num)]
            self.soft_q_optimizer2s = [
                optim.Adam(self.soft_q_net2s[group_index].parameters(), lr=self.parameters["soft_q_lr"]) for group_index
                in range(self.group_num)]
            self.policy_optimizers = [
                optim.Adam(self.policy_nets[group_index].parameters(), lr=self.parameters["policy_lr"]) for group_index
                in range(self.group_num)]
            self.alpha_optimizers = [optim.Adam([self.log_alphas[group_index]], lr=self.parameters["alpha_lr"]) for
                                     group_index in range(self.group_num)]

        else:
            self.soft_q_net1s = [SoftQNetwork(state_size=self.obs_shape_list[agent_index],
                                              action_size=self.act_space_list[agent_index],
                                              mid_size=128).to(device) for agent_index in range(self.agent_num)]
            self.soft_q_net2s = [SoftQNetwork(state_size=self.obs_shape_list[agent_index],
                                              action_size=self.act_space_list[agent_index],
                                              mid_size=128).to(device) for agent_index in range(self.agent_num)]
            self.target_soft_q_net1s = [SoftQNetwork(state_size=self.obs_shape_list[agent_index],
                                              action_size=self.act_space_list[agent_index],
                                              mid_size=128).to(device) for agent_index in range(self.agent_num)]
            self.target_soft_q_net2s = [SoftQNetwork(state_size=self.obs_shape_list[agent_index],
                                              action_size=self.act_space_list[agent_index],
                                              mid_size=128).to(device) for agent_index in range(self.agent_num)]
            self.policy_nets = [PolicyNetwork(state_size=self.obs_shape_list[agent_index],
                                              action_size=self.act_space_list[agent_index],
                                              mid_size=128).to(device) for agent_index in range(self.agent_num)]
            self.log_alphas = [torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device) for agent_index in
                               range(self.agent_num)]

            # entropy = -log(1/|A|) = log |A|
            self.target_entropys = [0.98 * (-self.act_space_list[agent_index]) for agent_index in
                                    range(self.agent_num)]
            self.alphas = [self.parameters["alpha"] for agent_index in range(self.agent_num)]

            self.soft_q_optimizer1s = [
                optim.Adam(self.soft_q_net1s[agent_index].parameters(), lr=self.parameters["soft_q_lr"]) for agent_index
                in range(self.agent_num)]
            self.soft_q_optimizer2s = [
                optim.Adam(self.soft_q_net2s[agent_index].parameters(), lr=self.parameters["soft_q_lr"]) for agent_index
                in range(self.agent_num)]
            self.policy_optimizers = [
                optim.Adam(self.policy_nets[agent_index].parameters(), lr=self.parameters["policy_lr"]) for agent_index
                in range(self.agent_num)]
            self.alpha_optimizers = [optim.Adam([self.log_alphas[agent_index]], lr=self.parameters["alpha_lr"]) for
                                     agent_index in range(self.agent_num)]

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
                "summary_dir": '/SAC_Summary_' + str(current_time),
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
        for soft_q_net1, target_soft_q_net1, soft_q_net2, target_soft_q_net2 in zip(self.soft_q_net1s,
                                                                                    self.target_soft_q_net1s,
                                                                                    self.soft_q_net2s,
                                                                                    self.target_soft_q_net2s):
            for eval_param, target_param in zip(soft_q_net1.parameters(), target_soft_q_net1.parameters()):
                target_param.data.copy_(tau * eval_param + (1 - tau) * target_param)
            for eval_param, target_param in zip(soft_q_net2.parameters(), target_soft_q_net2.parameters()):
                target_param.data.copy_(tau * eval_param + (1 - tau) * target_param)

    def action(self, obs_n, evaluation=False):
        action_n = []
        for i, obs in enumerate(obs_n):
            obs = torch.as_tensor([obs], dtype=torch.float32, device=device)
            if self.share_parameters:
                mu = self.policy_nets[self.agent_group_index[i]].get_action(obs, deterministic=evaluation)
            else:
                mu = self.policy_nets[i].get_action(obs, deterministic=evaluation)
            action_n.append(mu)
        return action_n

    def experience(self, obs_n, act_n, rew_n, new_obs_n, done_n):
        # Store transition in the replay buffer.
        for i in range(self.agent_num):
            self.replay_buffers[i].add(obs_n[i], act_n[i], [rew_n[i]], new_obs_n[i], [float(done_n[i])])

    # save_model("models/maddpg_actor_agent_", "models/maddpg_critic_agent_")
    def save_model(self):
        if self.share_parameters:
            for group_index in range(self.group_num):
                torch.save(self.policy_nets[group_index].state_dict(),
                           self.model_path + "/sac_actor_group_" + str(group_index) + ".pth")

                torch.save(self.soft_q_net1s[group_index].state_dict(),
                           self.model_path + "/sac_soft_q_1_group_" + str(group_index) + ".pth")

                torch.save(self.soft_q_net2s[group_index].state_dict(),
                           self.model_path + "/sac_soft_q_2_group_" + str(group_index) + ".pth")
        else:
            for agent_index in range(self.agent_num):
                torch.save(self.policy_nets[agent_index].state_dict(),
                           self.model_path + "/sac_actor_agent_" + str(agent_index) + ".pth")

                torch.save(self.policy_nets[agent_index].state_dict(),
                           self.model_path + "/sac_soft_q_1_agent_" + str(agent_index) + ".pth")

                torch.save(self.policy_nets[agent_index].state_dict(),
                           self.model_path + "/sac_soft_q_2_agent_" + str(agent_index) + ".pth")

    def load_model(self):
        '''
        开始训练时加载之前的模型
        :return:
        '''
        if self.share_parameters:
            for group_index in range(self.group_num):
                if os.path.exists(self.model_path + "/sac_actor_group_" + str(group_index) + ".pth"):
                    try:
                        self.policy_nets[group_index].load_state_dict(
                            torch.load(self.model_path + "/sac_actor_group_" + str(group_index) + ".pth"))
                        self.soft_q_net1s[group_index].load_state_dict(
                            torch.load(self.model_path + "/sac_soft_q_1_group_" + str(group_index) + ".pth"))
                        self.soft_q_net2s[group_index].load_state_dict(
                            torch.load(self.model_path + "/sac_soft_q_2_group_" + str(group_index) + ".pth"))
                    except RuntimeError as e:
                        print("模型不匹配，加载训练模型失败，将采用随机参数进行训练！！！")
                        break
                else:
                    print("模型不存在，加载训练模型失败，将采用随机参数进行训练！！！")
                    break
        else:
            for agent_index in range(self.agent_num):
                if os.path.exists(self.model_path + "/sac_actor_agent_" + str(agent_index) + ".pth"):
                    try:
                        self.policy_nets[agent_index].load_state_dict(
                            torch.load(self.model_path + "/sac_actor_agent_" + str(agent_index) + ".pth"))
                        self.soft_q_net1s[agent_index].load_state_dict(
                            torch.load(self.model_path + "/sac_soft_q_1_agent_" + str(agent_index) + ".pth"))
                        self.soft_q_net2s[agent_index].load_state_dict(
                            torch.load(self.model_path + "/sac_soft_q_2_agent_" + str(agent_index) + ".pth"))
                    except RuntimeError as e:
                        print("模型不匹配，加载训练模型失败，将采用随机参数进行训练！！！")
                        break
                else:
                    print("模型不存在，加载训练模型失败，将采用随机参数进行训练！！！")
                    break


    def load_actor(self):
        if self.share_parameters:
            for group_index in range(self.group_num):
                self.policy_nets[group_index].load_state_dict(
                    torch.load(self.model_path + "/sac_actor_group_" + str(group_index) + ".pth"))
        else:
            for agent_index in range(self.agent_num):
                self.policy_nets[agent_index].load_state_dict(
                    torch.load(self.model_path + "/sac_actor_agent_" + str(agent_index) + ".pth"))

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

        for i in range(self.agent_num):
            obs, act, rew, obs_next, done = self.replay_buffers[i].sample_index(replay_sample_index)
            obs_n.append(torch.tensor(obs, dtype=torch.float32, device=device))
            obs_next_n.append(torch.tensor(obs_next, dtype=torch.float32, device=device))
            act_n.append(torch.tensor(act, dtype=torch.float32, device=device))
            done_n.append(torch.tensor(done, dtype=torch.float32, device=device))
            rew_n.append(torch.tensor(rew, dtype=torch.float32, device=device))

        summaries = self.train(memories=(obs_n, act_n, rew_n, obs_next_n, done_n))

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

        obs_n, act_n, rew_n, obs_next_n, done_n = memories

        if self.share_parameters:
            q1_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
            q2_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
            policy_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
            alpha_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]

        else:
            q1_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
            q2_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
            policy_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
            alpha_loss = [torch.tensor(0, dtype=torch.float32, device=device) for j in range(self.agent_num)]
            for agent_index in range(self.agent_num):
                new_action, log_prob, z, mean, log_std = self.policy_nets[agent_index].evaluate(obs_n[agent_index])
                new_next_action, next_log_prob, _, _, _ = self.policy_nets[agent_index].evaluate(obs_next_n[agent_index])

                # Updating alpha wrt entropy
                # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
                alpha_l = -(self.log_alphas[agent_index] * (log_prob + self.target_entropys[agent_index]).detach()).mean()
                # print('alpha loss: ',alpha_loss)
                self.alpha_optimizers[agent_index].zero_grad()
                alpha_l.backward()
                self.alpha_optimizers[agent_index].step()
                self.alphas[agent_index] = self.log_alphas[agent_index].exp()
                alpha_loss[agent_index] = alpha_l

                # Training Q Function
                predicted_q_value1 = self.soft_q_net1s[agent_index]([obs_n[agent_index], act_n[agent_index]])
                predicted_q_value2 = self.soft_q_net2s[agent_index]([obs_n[agent_index], act_n[agent_index]])
                target_q_min = torch.min(
                    self.target_soft_q_net1s[agent_index]([obs_next_n[agent_index], new_next_action]),
                    self.target_soft_q_net2s[agent_index]([obs_next_n[agent_index],
                                                          new_next_action])) - self.alphas[agent_index] * next_log_prob
                target_q_value = rew_n[agent_index] + (1 - done_n[agent_index]) * self.parameters['gamma'] * target_q_min  # if done==1, only reward
                q_value_loss1 = nn.MSELoss()(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
                q_value_loss2 = nn.MSELoss()(predicted_q_value2, target_q_value.detach())

                self.soft_q_optimizer1s[agent_index].zero_grad()
                q_value_loss1.backward()
                self.soft_q_optimizer1s[agent_index].step()
                self.soft_q_optimizer2s[agent_index].zero_grad()
                q_value_loss2.backward()
                self.soft_q_optimizer2s[agent_index].step()
                q1_loss[agent_index] = q_value_loss1
                q2_loss[agent_index] = q_value_loss2

                # Training Policy Function
                predicted_new_q_value = torch.min(self.soft_q_net1s[agent_index]([obs_n[agent_index], new_action]),
                                                  self.soft_q_net2s[agent_index]([obs_n[agent_index], new_action]))
                policy_l = -(predicted_new_q_value - self.alphas[agent_index] * log_prob).mean()

                self.policy_optimizers[agent_index].zero_grad()
                policy_l.backward()
                self.policy_optimizers[agent_index].step()
                policy_loss[agent_index] = policy_l

        summaries = dict([
            ['LOSS/policy_loss', policy_loss],
            ['LOSS/q1_loss', q1_loss],
            ['LOSS/q2_loss', q2_loss],
            ['LOSS/alpha_loss', alpha_loss],
        ])
        return summaries