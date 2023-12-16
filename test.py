import time
from multiagent_i2c.make_env import make_env
import numpy as np

#:>python train.py --env-type=unity --env-name=USVAssetGuarding --train-algorithm=PPO --env-worker-id=1 --group-num=1 --agent-group-index 0 0 0 0 --max-episode-len 500 --print-frequency 5 --inference --no-graphics
#:>python train.py --env-type=mape --env-name=simple_push --train-algorithm=MADDPG --group-num=2 --agent-group-index 0 0 0 1 1 --max-episode-len 50 --print-frequency 50 --inference --no-graphics
#:>mlagents-learn config/sac/USVAssetGuarding.yaml --env=envs/USVAssetGuarding/USVAssetGuarding --run-id=USVAssetGuarding --base-port=10  --no-graphics --resume

env = make_env(scenario_name="cn-for-i2c", graph_obs=False, discrete_action_space=True, discrete_action_input = False)
env.observation_space = [o.shape[0] for o in env.observation_space]
env.action_space = [a.n for a in env.action_space]
print("env_observation_space:", str(env.observation_space))
print("env_:action_space", str(env.action_space))

for j in range(10):
    cur_sate = env.reset()
    for i in range(1000):
        actions = []
        for action_dim in env.action_space:
            actions.append(np.random.uniform(-1, 1, action_dim))
        next_state, reward, done, _ = env.step(actions)
        env.render()
        time.sleep(0.1)
        print("step:{},action:{}, reward:{}, done:{}".format(i, actions, reward, done))
        if all(done):
            break

env.close()