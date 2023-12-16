from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
import logging

class Unity_Env(object):
    """
    Unity_Env类基于Unity官方开源的mlagents项目，形成多智能体环境封装类。
    该类目前支持情况:
    - 支持mlagents版本:17
    - 环境中每个智能体的behavier_name需要不一样，或者team_id不一样
    - 只支持连续动作
    - 只支持向量化observation
    """
    def __init__(self, file_name="", no_graphics=False, time_scale=1, worker_id=0):
        """
        :param file_name:
        :param no_graphics:
        :param time_scale:
        :param worker_id:
        """
        self.engine_configuration_channel = EngineConfigurationChannel()
        if file_name == "":
            self.env = UnityEnvironment(worker_id=worker_id,
                                        side_channels=[self.engine_configuration_channel])
        else:
            self.env = UnityEnvironment(file_name=file_name,
                                        worker_id=worker_id,
                                        no_graphics=no_graphics,
                                        side_channels=[self.engine_configuration_channel])

        self.engine_configuration_channel.set_configuration_parameters(
            width = 900,
            height = 700,
            # quality_level = 5, #1-5
            time_scale = time_scale  # 1-100, 10执行一轮的时间约为10秒，20执行一轮的时间约为5秒。
            # target_frame_rate = 60, #1-60
            # capture_frame_rate = 60 #default 60
        )
        self.env.reset()

        # 获得所有的behavior_name,
        # 名称结构如：DefenderCarAgent?team=0，
        # 因此在制作图的时候尽量保持每个智能体对应一个名称
        self.agent_names = self.env.behavior_specs.keys()
        self.n = len(self.agent_names)

        #state(observation_specs)主要分为三个部分:【图像、雷达、向量】
        self.observation_space = [self.env.behavior_specs.get(behavior_name).observation_specs[0].shape[0] for behavior_name in
                             self.agent_names]
        self.action_space = [self.env.behavior_specs.get(behavior_name).action_spec.continuous_size for behavior_name in
                             self.agent_names]
        #是否是连续动作
        self.action_type = [self.env.behavior_specs.get(behavior_name).action_spec.is_continuous() for behavior_name in
                             self.agent_names]

    def reset(self):
        self.env.reset()
        cur_state = []
        for behavior_name in self.agent_names:
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_name)
            cur_state.append(DecisionSteps.obs[0][0])
        return cur_state #team_number x (agent_number_for_each_team, obs_length)

    def step(self, actions):
        next_state = []
        reward = []
        done = []
        info = []
        for behavior_name_index, behavior_name in enumerate(self.agent_names):
            action = ActionTuple()
            action.add_continuous(np.asarray(actions[behavior_name_index]).reshape((1, -1)))
            self.env.set_actions(behavior_name=behavior_name, action=action)
        self.env.step()

        for i, behavior_name in enumerate(self.agent_names):
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_name)
            if len(TerminalSteps.reward) == 0:
                next_state.append(DecisionSteps.obs[0][0])
                reward.append(DecisionSteps.reward[0] + DecisionSteps.group_reward[0])
                done.append(False)
            else:
                next_state.append(TerminalSteps.obs[0][0])
                reward.append(TerminalSteps.reward[0] + TerminalSteps.group_reward[0])
                done.append(True)

        return next_state, reward, done, info

    def close(self):
        self.env.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # env = Unity_Env(file_name=r"C:\Users\Abluc\Desktop\MARLS_Unity\Unity_Envs\GoToGoal\GoToGoal.exe", no_graphics=False, time_scale=1)
    env = Unity_Env(time_scale=1)
    print(list(env.agent_names))
    print(env.n)
    print(env.observation_space)
    print(env.action_space)
    print(env.action_type)

    for j in range(10):
        cur_sate = env.reset()
        for i in range(1000):
            actions = []
            for action_dim in env.action_space:
                actions.append(np.random.uniform(-1, 1, action_dim))
            next_state, reward, done, _ = env.step(actions)
            print("step:{}, reward:{}, done:{}".format(i, reward, done))
            if all(done):
                break

    env.close()


