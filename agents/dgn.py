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
from agents.attention import MultiHeadAttention
from buffers.replay_buffer import ReplayBuffer


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionQ(nn.Module):
    def __init__(self, sa_sizes, hidden_dim=64, attend_heads=4):
        super(AttentionQ, self).__init__()
        # init layers
        self.inputs_embeding_layers = nn.ModuleList()
        self.Q_compute_layers = nn.ModuleList()
        for sdim, adim in sa_sizes:
            self.inputs_embeding_layers.append(nn.Linear(sdim, hidden_dim))
            self.Q_compute_layers.append(nn.Linear(hidden_dim*3, adim))

        self.multi_head_attention_layer1 = MultiHeadAttention(input_size=hidden_dim, d_model=hidden_dim, num_heads=attend_heads)
        self.multi_head_attention_layer2 = MultiHeadAttention(input_size=hidden_dim, d_model=hidden_dim, num_heads=attend_heads)


    def forward(self, state_n):#[]
        # extract state-action encoding for each agent
        s_encodings = []
        for i, state in enumerate(state_n):
            l1 = F.relu(self.inputs_embeding_layers[i](state))
            s_encodings.append(l1)

        #(batch_size, num_agent, hidden_dim)
        s_encodings = torch.transpose(torch.stack(s_encodings, dim=0), dim0=0, dim1=1)

        # (batch_size, num_agent, hidden_dim)
        h_1, attend_weights_1 = self.multi_head_attention_layer1(s_encodings, s_encodings, s_encodings)

        # (batch_size, num_agent, hidden_dim)
        h_2, attend_weights_2 = self.multi_head_attention_layer2(h_1, h_1, h_1)

        # (batch_size, num_agent, hidden_dim*3)
        Q_input = torch.cat([s_encodings, h_1, h_2], dim=-1)
        # (num_agent, batch_size, hidden_dim*3)
        Q_input = torch.transpose(Q_input, dim0=0, dim1=1)

        #list: num_agent X (batch_size, hidden_dim*3)
        Q_n = []
        for i in range(Q_input.shape[0]):
            Q_1 = self.Q_compute_layers[i](Q_input[i])
            Q_n.append(Q_1)
        return Q_n

# DGNAgent Class
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
        self.epsilon = parameters['epsilon']
        self.epsilon_decay = parameters['epsilon_decay']
        self.epsilon_min = parameters['epsilon_min']

        self.group_obs_shape_list = [0 for i in range(self.group_num)]
        self.group_act_shape_list = [0 for i in range(self.group_num)]
        for agent_index, group_index in enumerate(self.agent_group_index):
            if self.group_obs_shape_list[group_index] == 0:
                self.group_obs_shape_list[group_index] = self.obs_shape_list[agent_index]
            if self.group_act_shape_list[group_index] == 0:
                self.group_act_shape_list[group_index] = self.act_space_list[agent_index]

        if self.share_parameters:
            pass
        else:
            self.Q = AttentionQ(sa_sizes=[(obs, act) for obs, act in zip(obs_shape, act_space)]).to(device)
            self.Q_target = AttentionQ(sa_sizes=[(obs, act) for obs, act in zip(obs_shape, act_space)]).to(device)
            self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=self.parameters["lr"])

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
                "summary_dir": '/DGN_Summary_' + str(current_time),
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
        for eval_param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(tau * eval_param + (1 - tau) * target_param)

    def action(self, obs_n, evaluation=False):
        o_n = []
        for i, obs in enumerate(obs_n):
            o_n.append(torch.tensor([obs], dtype=torch.float32, device=device))

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        agents_Q = self.Q(o_n)

        action_n = []
        for i, agent_Q in enumerate(agents_Q):
            action = np.zeros(self.act_space_list[i])
            if evaluation:
                action[torch.argmax(agent_Q[0])] = 1
                action_n.append(action)
            else:
                if np.random.random() < self.epsilon:
                    action[np.argmax(np.random.random(self.act_space_list[i]))] = 1
                    action_n.append(action)
                else:
                    action[torch.argmax(agent_Q[0])] = 1
                    action_n.append(action)
        return action_n

    def experience(self, obs_n, act_n, rew_n, new_obs_n, done_n):
        # Store transition in the replay buffer.
        for i in range(self.agent_num):
            self.replay_buffers[i].add(obs_n[i], act_n[i], [rew_n[i]], new_obs_n[i], [float(done_n[i])])

    # save_model("models/maddpg_actor_agent_", "models/maddpg_critic_agent_")
    def save_model(self):
        if self.share_parameters:
            pass
        else:
            torch.save(self.Q.state_dict(),
                       self.model_path + "/dgn_Q.pth")
            torch.save(self.Q_target.state_dict(),
                       self.model_path + "/dgn_Q_target.pth")

    def load_model(self):
        '''
        开始训练时加载之前的模型
        :return:
        '''
        if self.share_parameters:
            pass
        else:
            if os.path.exists(self.model_path + "/dgn_Q.pth"):
                try:
                    self.Q.load_state_dict(torch.load(self.model_path + "/dgn_Q.pth"))
                except RuntimeError as e:
                    print("模型不匹配，加载训练模型失败，将采用随机参数进行训练！！！")
            else:
                print("模型不存在，加载训练模型失败，将采用随机参数进行训练！！！")

    def load_actor(self):
        if self.share_parameters:
            pass
        else:
            self.Q.load_state_dict(torch.load(self.model_path + "/dgn_Q.pth"))

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

        summaries = self.train((obs_n, act_n, rew_n, obs_next_n, done_n))

        # if train_step % 10 == 0:  # only update every 100 steps
        self.update_target_weights(tau=self.parameters["tau"])

        for i in range(self.agent_num):
            if self.share_parameters:
                pass
            else:
                for key in summaries.keys():
                    self.summary_writers[i].add_scalar(key, summaries[key], global_step=train_step)
            self.summary_writers[i].flush()

    def train(self, memories):
        obs_n, act_n, rew_n, obs_next_n, done_n = memories

        if self.share_parameters:
            pass
        else:
            agents_Q_s = self.Q(obs_n)
            next_agents_Q_s = self.Q_target(obs_next_n)

            agents_Q = [torch.sum(torch.multiply(agent_Q_s, a), dim=-1, keepdim=True) for agent_Q_s, a in
                        zip(agents_Q_s, act_n)]
            next_agents_Q = [torch.max(next_agent_Q_s, dim=-1, keepdim=True).values for next_agent_Q_s in
                             next_agents_Q_s]

            agents_y = [rew + self.parameters['gamma'] * next_Q * (1 - done) for rew, next_Q, done in
                        zip(rew_n, next_agents_Q, done_n)]

            q_loss = 0
            for i in range(self.agent_num):
                critic_loss_1 = torch.mean(torch.square(agents_y[i] - agents_Q[i]))
                q_loss += critic_loss_1
            q_loss = q_loss / self.agent_num

            self.Q_optimizer.zero_grad()
            q_loss.backward()
            self.Q_optimizer.step()

            summaries = dict([
                ['LOSS/q_loss', q_loss],
            ])
        return summaries