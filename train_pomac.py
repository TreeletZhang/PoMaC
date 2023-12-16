import argparse
import json
import logging
import numpy as np
import time, os
import agents
from mlagents_envs.Unity_Env import Unity_Env
from multiagent_i2c.make_env import make_env
from model_parameters import parameters
from buffers.storage import Buffer
import torch

# treelet # 获取每个智能体观测到的其他智能体的相对位置和index
def get_comm_pairs(obs_n, num_agents_obs, num_others):
    num_agents = len(obs_n)
    target_locs_n = []  # 所有智能体观测到的其他智能体的相对位置
    target_idxes_n = []  # 所有智能体观测到的其他智能体的index
    # target_idx = None
    real_loc_n = []  # 所有智能体的真实位置信息
    # 获取每个智能体的位置信息，在obs中的第三第四元素
    for i in range(num_agents):
        real_loc_n.append(obs_n[i][2:4])
    for i in range(num_agents):
        # 去除智能体的真实位置信息和速度信息，保留相对位置
        obs_tmp = obs_n[i][4:].copy()  # 提取智能体观测中地标的相对位置信息和其他智能体的相对位置信息
        obs_tmp[0::2] = obs_tmp[0::2]+real_loc_n[i][0]  # x轴相加
        obs_tmp[1::2] = obs_tmp[1::2]+real_loc_n[i][1]  # y轴相加，此时obs_tmp中是智能体观测中地标的真实位置信息和其他智能体的真实位置信息
        target_locations = []  # 每个智能体观测到的其他智能体的相对位置
        target_indexes = []  # 每个智能体观测到的其他智能体的index
        for j in range(num_agents_obs):  # 遍历观测到的智能体数量
            target_loc = obs_tmp[int((num_others+j)*2):int((num_others+j)*2+2)]  # 观测到的每个智能体的真实位置
            for ii in range(len(real_loc_n)):  # 遍历所有智能体的真实位置信息
                if (abs(real_loc_n[ii][0]-target_loc[0])<1e-5) and abs(real_loc_n[ii][1]-target_loc[1])<1e-5:  # 判断观测到的智能体是环境中的智能体几
                    target_idx = ii  # 得到观测到的智能体的index
            target_locations.append(real_loc_n[i]-target_loc)
            target_indexes.append(target_idx)
        target_locs_n.append(target_locations)
        target_idxes_n.append(target_indexes)
    return target_locs_n, target_idxes_n

# 初始化环境（包括选择环境类型MAPE或UNITY,选择具体场景等）
# 初始化多智能体具体算法中的Agent类
# for episode:
#    初始化环境得到obs_n
#    根据不同算法选择是否添加动作噪声
#    for steps:
#       具体算法输出动作
#       执行动作，得到o',r
#       收集样本
#       判断不同方法是否满足更新条件，满足则更新
#    记录训练数据

def train(train_parameters, model_parameters, model_path, log_path):
    '''
    :param train_parameters:
    :param model_parameters:
    :param model_path:
    :param log_path:
    :return:
    '''

    # 初始化环境
    if train_parameters["env_type"] == 'mape':
        if train_parameters["discrete_action"]:
            env = make_env(scenario_name=train_parameters['env_name'], graph_obs=train_parameters["graph_obs"], discrete_action_space=True,
                           discrete_action_input=False)

            print("obs_space:", env.observation_space)  # [Box(16,),Box(16,),Box(16,)...]
            print("act_space:", env.action_space)  # [111...]
            # treelet
            obs_shape_n_i2c = [env.observation_space[0].shape for _ in range(env.n)]  # [(16,),(16,)...]
            # treelet
            env.observation_space = [o.shape[0] for o in env.observation_space]  # [16, 16, 16, 16, 16, 16, 16]
            env.action_space = [a.n for a in env.action_space]  # [5, 5, 5, 5, 5, 5, 5]
        else:
            env = make_env(scenario_name=train_parameters['env_name'], graph_obs=train_parameters["graph_obs"], discrete_action_space=False,
                           discrete_action_input=False)
            env.observation_space = [o.shape[0] for o in env.observation_space]
            env.action_space = [a.shape[0] for a in env.action_space]
    else:
        if train_parameters["env_run_type"] == 'exe':
            env = Unity_Env(file_name="./Unity_Envs/" +
                                      train_parameters["env_name"] + "/" +
                                      train_parameters["env_name"],
                            no_graphics=train_parameters["no_graphics"],
                            worker_id=train_parameters["env_worker_id"],
                            time_scale=20)
        else:
            logging.basicConfig(level=logging.INFO)
            env = Unity_Env(time_scale=20)

    # treelet 实例化buffer
    if train_parameters["train_algorithm"] == "I2C":
        obs_n = env.reset()
        print("obs_n", obs_n)
        num_others = env.n_landmarks_obs if train_parameters['env_name'] == 'cn' else env.n_preys_obs
        other_loc_n, other_idx_n = get_comm_pairs(obs_n, env.n_agents_obs, num_others)  # 获取在交互范围内的智能体信息
        prior_buffer = Buffer(model_parameters, env.observation_space[0], len(other_loc_n[0][0]))
        num_adversaries = min(env.n, train_parameters['num_adversaries'])
        message_shape_n = [(env.n_agents_obs,) + obs_shape_n_i2c[0] for _ in range(env.n)]  # 每个智能体要传递的交互信息维度[(交互的智能体数量,16),(交互的智能体数量,16),...]
        target_loc_space_n = [(len(other_loc_n[0][0]),) for _ in range(env.n)]  # 可交互智能体的location空间维度[(2,),(2,)...]
        train_agents = agents.load(train_parameters['train_algorithm'] + ".py").Agent(
            name=train_parameters['env_name'],
            obs_shape=env.observation_space,  # add [16,16,...]
            message_shape=message_shape_n,  # add [(3,16),(3,16),...]
            target_loc_space_n=target_loc_space_n,  # add [(2,),(2,),...]
            n_agents_obs=env.n_agents_obs,  # add
            prior_buffer=prior_buffer,  # add
            act_space=env.action_space,  # [5,5,...]
            agent_num=env.n,
            group_num=train_parameters['group_num'],
            agent_group_index=train_parameters['agent_group_index'],
            share_parameters=train_parameters['share_parameters'],
            parameters=model_parameters,
            model_path=model_path,
            log_path=log_path,
            create_summary_writer=True,
            resume=train_parameters['resume'])
    else:
        # 初始化Multi-Agents model
        train_agents = agents.load(train_parameters['train_algorithm'] + ".py").Agent(
            name=train_parameters['env_name'],
            obs_shape=env.observation_space,
            act_space=env.action_space,
            agent_num=env.n,
            group_num=train_parameters['group_num'],
            agent_group_index=train_parameters['agent_group_index'],
            share_parameters=train_parameters['share_parameters'],
            parameters=model_parameters,
            model_path=model_path,
            log_path=log_path,
            create_summary_writer=True,
            resume=train_parameters['resume'])

    # treelet  action_noises,action,experience,buffers,finish_path,update,can_update

    print('Starting training...')
    episode = 0
    epoch = train_agents.log_info_json["epoch"]
    train_step = train_agents.log_info_json["train_step"]
    log_episode = train_agents.log_info_json["log_episode"]
    print('******************本次训练从epoch=' + str(epoch) + " train_step="+ str(train_step) + "处开始***************************")
    t_start = time.time()

    try:
        while episode < train_parameters["num_episodes"]:
            obs_n = env.reset()
            episode_steps = 0
            episode_rewards = [0.0]  # sum of rewards for all agents
            group_rewards = [0.0 for i in range(train_parameters["group_num"])]
            agent_rewards = [0.0 for _ in range(env.n)]  # individual agent reward
            # treelet
            num_comm_n = []  # 记录每一步的交互数量
            # treelet

            if train_parameters["train_algorithm"] == "SAC" or \
                    train_parameters["train_algorithm"] == "PPO" or \
                    train_parameters["train_algorithm"] == "DGN":
                pass
            else:
                for action_noise in train_agents.action_noises:
                    action_noise.reset()

            new_obs_n, done_n = [], []
            while episode_steps < train_parameters["max_episode_len"]:
                # get action
                # treelet
                if train_parameters["train_algorithm"] == "I2C" and train_parameters["WhoNet"]:
                    #  get comm target
                    other_loc_n, other_idx_n = get_comm_pairs(obs_n, env.n_agents_obs, num_others)  # 获取在交互范围内的智能体信息
                    # 在训练前期用先验规则选择要交互的智能体的index
                    if train_parameters["train_algorithm"] == "Rule":
                        target_index_n, num_comm, c_pred_each_agent = train_agents.target_comm_rule(other_idx_n)
                    else:
                        target_index_n, num_comm, c_pred_each_agent = train_agents.target_comm(obs_n, other_loc_n,
                                                                        other_idx_n, train_parameters)  # 得到N个智能体要交互的智能体的index
                    num_comm_n.append(num_comm)
                    #  get message
                    message_n = train_agents.get_message(obs_n, target_index_n)
                    action_n = train_agents.action(obs_n, message_n, evaluation=False)
                elif train_parameters["train_algorithm"] == "I2C":  # 广播交互无自主选择
                    #  get comm target
                    other_loc_n, other_idx_n = get_comm_pairs(obs_n, env.n_agents_obs, num_others)  # 获取在交互范围内的智能体信息
                    #  get message
                    message_n = train_agents.get_message(obs_n, other_idx_n)
                    action_n = train_agents.action(obs_n, message_n, evaluation=False)
                else:  # 无交互无消息编码
                    action_n = train_agents.action(obs_n, evaluation=False)
                # environment step
                # print(action_n)
                new_obs_n, rew_n, done_n, _ = env.step(action_n)

                # print("环境反馈的rew:", rew_n)
                done = all(done_n)
                # treelet  得到新的状态下的mess_n并存储，用于更新时a_new的计算
                if train_parameters["train_algorithm"] == "I2C" and train_parameters["WhoNet"]:
                    new_other_loc_n, new_other_idx_n = get_comm_pairs(new_obs_n, env.n_agents_obs, num_others)  # 获取在交互范围内的智能体信息
                    new_target_index_n, new_num_com = train_agents.target_comm(new_obs_n, new_other_loc_n, new_other_idx_n, train_parameters)
                    new_message_n = train_agents.get_message(new_obs_n, new_target_index_n)
                elif train_parameters["train_algorithm"] == "I2C":
                    new_other_loc_n, new_other_idx_n = get_comm_pairs(new_obs_n, env.n_agents_obs,
                                                                      num_others)  # 获取在交互范围内的智能体信息
                    new_message_n = train_agents.get_message(new_obs_n, new_other_idx_n)
                # treelet
                # collect experience
                if train_parameters["train_algorithm"] == "I2C":
                    train_agents.experience(obs_n, other_loc_n, other_idx_n, message_n, new_message_n, action_n, rew_n, new_obs_n, new_other_loc_n, new_other_idx_n, done_n)


                else:
                    train_agents.experience(obs_n, action_n, rew_n, new_obs_n, done_n)

                # 记录reward数据
                # for i, rew in enumerate(rew_n):
                #     episode_rewards[-1] += rew
                #     agent_rewards[i] += rew
                # treelet
                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew
                    agent_rewards[i] += rew  # 每个智能体一个episode的总reward
                # treelet

                # 记录每个组的数据
                for agent_index, group_index in enumerate(train_parameters["agent_group_index"]):
                    group_rewards[group_index] += rew_n[agent_index]

                if train_parameters["train_algorithm"] == "PPO":
                    if len(train_agents.buffers[0]['state']) == model_parameters['batch_size']:
                        train_agents.finish_path(next_state_n=new_obs_n, done_n=done_n)
                        train_agents.update(train_step)
                        train_step += 1
                else:
                    # update all trainers
                    if epoch % train_parameters["train_frequency"] == 0:
                        if train_agents.can_update():
                            if train_parameters["train_algorithm"] == "I2C":
                                train_agents.update(train_step, train_parameters["WhoNet"])  # 包含actor,critic和whonet的更新
                                train_step += 1
                            else:
                                train_agents.update(train_step)  # 包含actor,critic和whonet的更新
                                train_step += 1

                if done:
                    break

                obs_n = new_obs_n
                episode_steps += 1
                epoch += 1

            if train_parameters["train_algorithm"] == "PPO":
                train_agents.finish_path(next_state_n=new_obs_n, done_n=done_n)

            if episode % train_parameters["print_frequency"] == 0:

                print(
                    f"Episode: {log_episode + episode:3d}\t"
                    f"Episode Steps: {episode_steps: 2d}\t"
                    f"Epoch: {epoch: 3d}\t"
                    f"Train Steps: {train_step: 3d}\t"
                    f"Time: {time.time() - t_start: 6.3f}\t"
                    f"Reward: {agent_rewards}"
                )
                t_start = time.time()

            episode += 1

            for i, summary_writer in enumerate(train_agents.summary_writers):
                #summary_writer.add_scalar('A_Main/total_reward', episode_rewards[-1], log_episode + episode)
                # treelet
                summary_writer.add_scalar('A_Main/total_reward', episode_rewards[-1], log_episode + episode)
                # treelet
                summary_writer.add_scalar('A_Main/Agent_reward', agent_rewards[i]/100.0, log_episode + episode)
                summary_writer.add_scalar('A_Main/group_reward', group_rewards[train_parameters["agent_group_index"][i]],
                                          log_episode + episode)
                summary_writer.add_scalar('A_Main/episode_steps', episode_steps, log_episode + episode)
                summary_writer.flush()

            if episode != 0 and episode % train_parameters["save_frequency"] == 0:
                # 保存模型参数
                train_agents.save_model()

    except KeyboardInterrupt:
        print("人为取消训练。。。。。")
        #保存log_info_json信息
        train_agents.log_info_json["epoch"] = epoch
        train_agents.log_info_json["train_step"] = train_step
        train_agents.log_info_json["log_episode"] = log_episode + episode
        print("保存断电中。。。。。")
        time.sleep(0.5)
        with open(log_path + '/log_info.txt', "w") as fp:
            fp.write(json.dumps(train_agents.log_info_json))
            fp.close()
        # 保存模型参数
        print("保存模型中。。。。。")
        time.sleep(0.5)
        train_agents.save_model()

        # 关闭summary，回收资源
        print("关闭summary中。。。。。")
        time.sleep(0.5)
        for summary_writer in train_agents.summary_writers:
            summary_writer.close()
        env.close()
        print("关闭程序成功！！！")
        exit()

    # 保存log_info_json信息
    train_agents.log_info_json["epoch"] = epoch
    train_agents.log_info_json["train_step"] = train_step
    with open(log_path + '/log_info.txt', "w") as fp:
        fp.write(json.dumps(train_agents.log_info_json))
        fp.close()
    # 保存模型参数
    train_agents.save_model()

    # 关闭summary，回收资源
    for summary_writer in train_agents.summary_writers:
        summary_writer.close()
    env.close()


def inference(train_parameters, model_parameters, model_path):
    '''
    :param train_parameters:
    :param model_parameters:
    :param model_path:
    :return:
    '''

    # 初始化环境
    if train_parameters["env_type"] == 'mape':
        if train_parameters["discrete_action"]:
            env = make_env(scenario_name=train_parameters['env_name'], graph_obs=train_parameters["graph_obs"], discrete_action_space=True,
                           discrete_action_input = False)
            # treelet
            obs_shape_n_i2c = [env.observation_space[0].shape for _ in range(env.n)]  # [(16,),(16,)...]
            # treelet
            env.observation_space = [o.shape[0] for o in env.observation_space]
            env.action_space = [a.n for a in env.action_space]
        else:
            env = make_env(scenario_name=train_parameters['env_name'], graph_obs=train_parameters["graph_obs"], discrete_action_space=False,
                           discrete_action_input=False)
            env.observation_space = [o.shape[0] for o in env.observation_space]
            env.action_space = [a.shape[0] for a in env.action_space]
    else:
        if train_parameters["env_run_type"] == 'exe':
            env = Unity_Env(file_name="./Unity_Envs/" +
                                      train_parameters["env_name"] + "/" +
                                      train_parameters["env_name"],
                            no_graphics=False,
                            worker_id=train_parameters["env_worker_id"],
                            time_scale=3)
        else:
            logging.basicConfig(level=logging.INFO)
            env = Unity_Env(time_scale=3)
    print('******************环境加载成功**********************')
    # 初始化MADDPGAgent
    # treelet
    if train_parameters["train_algorithm"] == "I2C":
        obs_n = env.reset()

        num_others = env.n_landmarks_obs if train_parameters['env_name'] == 'cn' else env.n_preys_obs
        other_loc_n, other_idx_n = get_comm_pairs(obs_n, env.n_agents_obs, num_others)  # 获取在交互范围内的智能体信息
        prior_buffer = Buffer(model_parameters, env.observation_space[0], len(other_loc_n[0][0]))
        num_adversaries = min(env.n, train_parameters['num_adversaries'])
        message_shape_n = [(env.n_agents_obs,) + obs_shape_n_i2c[0] for _ in range(env.n)]  # 每个智能体要传递的交互信息维度[(交互的智能体数量,16),(交互的智能体数量,16),...]
        target_loc_space_n = [(len(other_loc_n[0][0]),) for _ in range(env.n)]  # 可交互智能体的location空间维度[(2,),(2,)...]
        inference_agents = agents.load(train_parameters['train_algorithm'] + ".py").Agent(
            name=train_parameters['env_name'],
            obs_shape=env.observation_space,  # add
            message_shape=message_shape_n,  # add
            target_loc_space_n=target_loc_space_n,  # add
            n_agents_obs=env.n_agents_obs,  # add
            prior_buffer=prior_buffer,  # add
            act_space=env.action_space,
            agent_num=env.n,
            group_num=train_parameters['group_num'],
            agent_group_index=train_parameters['agent_group_index'],
            share_parameters=train_parameters['share_parameters'],
            parameters=model_parameters,
            model_path=model_path,
            log_path=log_path,
            create_summary_writer=True)
    # treelet
    else:
        inference_agents = agents.load(train_parameters['train_algorithm'] + ".py").Agent(
            name=train_parameters['env_name'],
            obs_shape=env.observation_space,
            act_space=env.action_space,
            agent_num=env.n,
            group_num=train_parameters['group_num'],
            agent_group_index=train_parameters['agent_group_index'],
            share_parameters=train_parameters['share_parameters'],
            parameters=model_parameters,
            model_path=model_path,
            log_path=log_path,
            create_summary_writer=False)
    inference_agents.load_actor()
    print('******************模型加载成功******************')
    print('******************Starting inference...******************')
    episode = 0
    while episode < train_parameters["num_episodes"]:
        rewards = np.zeros(env.n, dtype=np.float32)
        cur_state = env.reset()

        step = 0
        # treelet
        num_comm_n = []  # 记录每一步的交互数量
        # treelet
        while step < train_parameters["max_episode_len"]:
            # get action
            # treelet
            if train_parameters["train_algorithm"] == "I2C" and train_parameters["WhoNet"]:
                #  get comm terget
                other_loc_n, other_idx_n = get_comm_pairs(cur_state, env.n_agents_obs, num_others)  # 获取在交互范围内的智能体信息
                target_index_n, num_comm, c_pred_each_agent = inference_agents.target_comm(cur_state, other_loc_n,
                                                                    other_idx_n, train_parameters)  # 得到N个智能体要交互的智能体的index
                print("c_pred_each_agent:", c_pred_each_agent)
                num_comm_n.append(num_comm)
                #  get message
                message_n = inference_agents.get_message(cur_state, target_index_n)
                action_n = inference_agents.action(cur_state, message_n, evaluation=False)
            elif train_parameters["train_algorithm"] == "I2C":  # 广播交互无自主选择
                #  get comm target
                other_loc_n, other_idx_n = get_comm_pairs(cur_state, env.n_agents_obs, num_others)  # 获取在交互范围内的智能体信息
                #  get message
                message_n = inference_agents.get_message(cur_state, other_idx_n)
                action_n = inference_agents.action(cur_state, message_n, evaluation=False)
            # treelet
            else:
                action_n = inference_agents.action(cur_state, evaluation=True)
            # print(action_n)
            # environment step
            next_state, reward, done, _ = env.step(action_n)

            done = all(done)
            if train_parameters["env_type"] == "mape":
                env.render()
                time.sleep(0.07)
            cur_state = next_state
            rewards += np.asarray(reward, dtype=np.float32)
            step += 1
            if done:
                break
        episode += 1

        print(
            f"Episode {episode:6d} - "
            f"Episode Step {step: 5d} - "
            f"Total Reward {rewards} - "
            f"Comm Num in every step {num_comm_n}"
        )

    env.close()


if __name__ == '__main__':
    print(torch.cuda.is_available())
    # *******************************************1、训练参数设置*********************************************************
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement learning")
    parser.add_argument('--env-type', type=str, default='mape', help='训练环境的类型，\"unity\" 或者 \"mape\"')
    parser.add_argument('--env-name', type=str, default='cn', help='unity或mape中环境的名称')
    parser.add_argument('--env-run-type', type=str, default='exe', help='unity训练环境客户端训练还是可执行程序训练，\"exe\" or '
                                                                        '\"client\", 对于mape环境没有用')
    parser.add_argument('--env-worker-id', type=int, default=0, help='\"exe\"环境的worker_id, 可进行多环境同时训练, 默认0, '
                                                                     '使用client是必须设为0!')
    # treelet
    parser.add_argument("--WhoNet", action="store_true", help="选择使用交互对象的自主选择")  # 原设置store_true
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")

    parser.add_argument('--no-graphics', action="store_true", help='使用unity训练时是否打开界面')
    parser.add_argument('--group-num', type=int, default=1, help='环境中智能体 组/类别 的数量')
    parser.add_argument('--agent-group-index', nargs='+', type=int, default=[0, 0, 0, 0], help='环境中每个agent对应的组编号')
    parser.add_argument('--share-parameters', action="store_true", help='环境中每组智能体是否组间共享网络')  # 原设置store_true
    parser.add_argument('--discrete-action', action="store_false", help='环境中是否是离散的动作')
    parser.add_argument('--graph-obs', action="store_true", help='环境中是否是图形式的观测')
    parser.add_argument('--train-algorithm', type=str, default='I2C',
                        help='训练算法: 目前支持DDPG, MADDPG, SAC, PPO,且名称都需要大写')
    parser.add_argument('--train', action="store_true", help='是否训练')  # 原设置store_true
    parser.add_argument('--inference', action="store_true", help='是否推断') # 原设置store_true
    parser.add_argument('--max-episode-len', type=int, default=40, help='每个episode的最大step数')
    parser.add_argument('--num-episodes', type=int, default=100000, help='episode数量')
    parser.add_argument('--save-frequency', type=int, default=100, help='模型保存频率')
    parser.add_argument('--train-frequency', type=int, default=50, help='模型训练频率, 1表示每个step调用训练函数一次')
    parser.add_argument('--print-frequency', type=int, default=50, help='训练数据打印输出频率, 100表示每100轮打印一次')
    parser.add_argument('--resume', action="store_true", help='是否按照上一次的训练结果，继续训练')
    # treelet
    parser.add_argument('--save-folder', type=str, default='defult/defult', help='存储模型和log的文件夹名称,一般设置为场景名+算法名')  # 例如--save-folder=simple_tag2/i2c_noWhoNet
    parser.add_argument('--Rule', action="store_true", help='训练的时候使用规则选择交互对象')
    args = parser.parse_args()
    train_parameters = vars(args)

    # *******************************************2、打印Logo*********************************************************
    print(
        """
                                                            *▓▓▓▓ ▓▓▓*▓▓▓
                                                          ▓▓*▓▓▓▓▓*▓▓▓▓▓*▓▓▓*
                                                          ▓▓▓▓         **▓▓▓
                                                        *▓▓▓▓             *▓▓▓
                                                        ▓▓▓▓*              *▓▓▓
                                                       ▓▓▓▓▓▓▓              ▓▓▓▓
                                                      ▓▓      *             *▓▓▓▓▓
                                        *             ▓▓      ▓             ▓▓▓▓▓▓*
                                 *▓▓▓▓**  ***▓▓▓▓▓********▓▓▓▓▓▓           ▓*     ▓▓
                               *▓▓▓*                        **▓▓▓          ▓      *▓▓
                              ▓*   *▓▓▓*  ****            **   *▓▓▓▓▓▓▓▓▓▓*▓        ▓▓▓▓▓▓▓▓***
                              *  ▓▓** *   ▓▓▓▓▓▓▓▓▓*  **▓▓▓▓    ▓ **       ▓         *▓▓▓▓▓▓▓▓▓▓▓▓▓**
                            *▓▓▓▓   *▓  ▓▓       ▓   ▓   **     ▓          ▓                        *▓▓▓▓
                          *▓▓    *▓▓▓  ▓▓▓*     ▓▓  *▓▓▓**      **       *▓*                    **▓▓▓▓▓▓*
                        *▓                     *                 ▓▓▓**                 **▓▓▓▓▓**      ▓▓
                      ▓**                                                         **▓▓▓▓▓▓*          *▓*
                   *▓                                                 *▓▓▓▓▓▓▓▓▓▓**           *▓▓▓▓▓▓▓
                   *▓                                            ***▓▓▓▓▓▓▓***            *▓▓▓▓▓▓▓*
                    *▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓****▓▓▓▓▓▓▓▓▓▓▓▓▓▓***                ***▓▓▓▓▓▓▓▓*
                      ▓****▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓*******                 *▓▓▓▓▓▓▓▓▓▓▓*
                      ▓*                                              **▓▓▓▓▓▓▓▓▓*
                      ▓▓                                          ***▓▓▓▓▓▓▓▓▓
                      ▓▓*                                  *▓▓▓▓▓▓▓▓▓▓▓*
                      *▓▓                            *▓▓▓▓▓▓▓▓▓▓▓▓**
                       ▓▓*                    **▓▓▓▓▓▓▓▓▓▓▓**
                       ▓▓▓                ▓▓▓▓▓▓▓▓▓▓▓▓▓▓*
                        ▓▓▓▓**▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓**
                        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓*
                         *▓▓▓▓▓▓▓▓▓▓**
                          *▓▓▓***
                                                                      *
    * ▓▓▓▓▓    *▓▓    *▓▓             *▓▓▓    *▓▓▓▓▓▓▓**            *▓▓▓          ****
  *▓▓▓▓▓▓▓▓   ▓▓▓▓    ▓▓▓*            ▓▓▓▓ *▓▓▓▓▓▓▓▓▓▓▓▓▓*         ▓▓▓▓*      **▓▓▓▓▓▓▓▓
  ▓▓▓▓*  ** *▓▓▓▓▓  ▓▓▓▓▓          *▓*▓▓▓ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓        ▓▓ ▓▓     ▓▓▓▓▓▓▓▓▓▓▓▓
 *▓▓▓▓   ▓  ▓▓*▓▓* ▓▓▓▓▓          ▓▓* ▓▓ *▓▓▓▓▓**▓     *▓▓       *▓ *▓     ▓▓▓▓▓▓▓▓*▓▓▓*
 **     ▓▓▓▓* ▓▓**▓*▓▓          *▓   ▓▓   ***  *▓*      ▓▓      ▓▓ ▓▓▓    ▓▓▓▓     ▓*
       *▓▓▓*  ▓▓ ▓**▓▓          ▓*  *▓*       *▓▓     *▓▓▓     *▓**▓▓▓    ▓▓▓                                        *
      *▓▓▓   ▓▓▓▓*▓▓▓     *▓▓*▓▓▓▓▓▓▓▓▓▓      ▓***▓▓▓▓▓▓      ▓▓  ▓▓▓     ▓▓▓▓▓*                 ▓  ***    *         ▓
      ▓▓▓▓  ▓▓▓▓* ▓▓      ▓▓▓▓▓▓▓▓▓▓▓▓▓*   **▓▓▓▓▓▓▓▓▓        ▓*  ▓*       *▓▓▓▓▓▓              ▓**▓▓▓▓*  ▓▓* *▓▓▓   ▓
     ▓▓▓*  ▓▓▓▓  ▓▓       ▓▓▓▓▓▓▓▓▓▓▓▓    ▓▓▓▓*              ▓                  **▓▓*          ▓     ** ▓▓  ▓       *▓
    *▓▓▓  ▓▓▓▓  ▓▓*        ▓▓     ▓▓      ▓▓▓▓▓▓*           ▓▓                    *▓▓         *▓    *▓ *▓  *▓*▓▓▓   ▓▓
   ▓▓▓*  *▓▓▓  *▓▓    ▓***▓      ▓▓       ▓▓ *▓▓▓▓▓▓*   ▓▓▓▓▓▓▓▓▓*       *        ▓▓▓*        ▓    ▓▓ *▓  *▓    ** *▓
   ▓▓▓   ▓▓▓*  *▓▓  *▓**▓▓      *▓▓ *▓    ▓*  *▓▓▓▓▓▓▓ ▓▓▓▓▓▓▓▓▓▓▓▓▓** *▓▓    ***▓▓▓▓         ▓*  *▓  ▓*  ▓*   *▓  ▓*
    ▓*   *▓    *▓▓▓▓▓ ▓▓▓       *▓▓*    *▓        ▓▓▓▓▓▓▓* **▓▓▓▓▓▓▓▓ *▓▓▓▓▓▓▓▓▓▓▓▓▓          *▓  ▓▓   ▓▓▓ *▓▓▓*  ▓*
                ▓▓▓*  ▓▓         ▓      ▓▓         *▓▓▓*       **▓▓▓  *▓▓▓▓▓▓▓▓▓▓▓*                         **   
    """
    )

    # *****************************************3、获得模型参数和模型/log路径**********************************************
    model_parameters = parameters[train_parameters['train_algorithm']]
    # 创建相关的log和model的保存路径
    # model_path = "./models/" + \
    #              train_parameters['env_type'] + "/" + \
    #              train_parameters['env_name'] + "4v2_3" + "/" + \
    #              train_parameters['train_algorithm']
    # log_path = "./logs/" + \
    #            train_parameters['env_type'] + "/" + \
    #            train_parameters['env_name'] + "4v2_3" + "/" + \
    #            train_parameters['train_algorithm']
    # treelet
    model_path = "./models/" + \
                 train_parameters['env_type'] + "/" + \
                 train_parameters['save_folder']
    log_path = "./logs/" + \
               train_parameters['env_type'] + "/" + \
               train_parameters['save_folder']
    if os.path.exists(model_path):
        pass
    else:
        os.makedirs(model_path)
    if os.path.exists(log_path):
        pass
    else:
        os.makedirs(log_path)

    # *****************************************4、打印训练参数、模型参数、相关路径*****************************************
    # （1）打印训练参数
    print("++++++++++++++++++++++++训练参数+++++++++++++++++++++++++++")
    for key in train_parameters.keys():
        print("\t{0:^20}\t{1:^20}".format(key + ":", str(train_parameters[key])))
    print("\n")
    # （2）打印模型参数
    print("++++++++++++++++++++++++模型参数+++++++++++++++++++++++++++")
    for key in model_parameters.keys():
        print("\t{0:^20}\t{1:^20}".format(key + ":", str(model_parameters[key])))
    print("\n")
    # （3）打印相关路径
    print("++++++++++++++++++++++++相关路径+++++++++++++++++++++++++++")
    print("\t" + "模型保存路径" + ":" + "\t" + str(model_path))
    print("\t" + "log保存路径" + ":" + "\t" + str(log_path))
    print("\n")

    # *****************************************5、开始训练或者推断******************************************************
    if train_parameters["train"]:
        train(train_parameters, model_parameters, model_path, log_path)
    elif train_parameters["inference"]:
        inference(train_parameters, model_parameters, model_path)
