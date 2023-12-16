
parameters ={
    "DDPG":{
        "buffer_size":1000000,
        "lr_actor":1.0e-3,
        "lr_critic":1.0e-3,
        "sigma":0.15,
        "gamma":0.99,
        "batch_size":1024,
        "max_replay_buffer_len": 10240,
        "tau":0.01
    },
    "DGN":{
        "buffer_size":100000,
        "lr":1.0e-2,
        "epsilon":1.0,
        "epsilon_decay":0.995,
        "epsilon_min":0.01,
        "gamma":0.95,
        "batch_size":1024,
        "max_replay_buffer_len": 10240,
        "tau":0.01
    },
    "MADDPG":{
        "buffer_size":1000000,
        "lr_actor":1.0e-5,  # 原设置1.0e-3
        "lr_critic":1.0e-5,  # 原设置1.0e-3
        "sigma":0.15,
        "gamma":0.99,
        "batch_size":1024,
        "max_replay_buffer_len": 10240,
        "tau":0.01
    },
    "H2G_MAAC": {
        "buffer_size": 1000000,
        "lr_actor": 1.0e-3,
        "lr_critic": 1.0e-3,
        "sigma": 0.15,
        "gamma": 0.99,
        "batch_size": 1024,
        "max_replay_buffer_len": 10240,
        "tau": 0.01
    },
    "SAC": {
        "buffer_size": 1000000,
        "soft_q_lr": 1.0e-3,
        "policy_lr": 5.0e-4,
        "alpha_lr": 5.0e-4,
        "gamma": 0.99,
        "alpha":0.2,
        "batch_size": 1024,
        "max_replay_buffer_len": 10240,
        "tau": 0.01
    },
    "PPO": {
        "batch_size": 64,  # 32
        "UPDATE_STEPS": 5,  # 10
        "epsilon": 0.2,
        "gamma": 0.95,
        "A_LR": 3.0e-4,  # 0.00001
        "C_LR": 1.0e-3,  # 0.00002
        "EPS": 1e-6,  # numerical residual
        "ent_coef": 0.005,
        'lambd':0.95,
        'use_gae_adv':True
    },
    "I2C": {
        #"good_policy":'maddpg',  # policy for good agents
        #"adv_policy":'maddpg',  # policy for adversaries
        # MADDPG
        "buffer_size":1000000,  # 原设置1000000
        "lr_actor":1.0e-3,  # 原设置1.0e-3
        "lr_critic":1.0e-3,
        "lr_whoNet":1.0e-4,
        "sigma":0.15,
        "gamma":0.99,
        "batch_size":1024,  # 原设置1024
        "max_replay_buffer_len": 10240,  # 原来是10240,更新判别条件
        "tau":0.01,
        # prior net buffer
        "prior_buffer_size":10000,  # 原设置400000 取一批数据作为样本填满prior buffer
        "prior_training_percentile":80,  # 原设置80
        "prior_training_rate":20000,  # 原设置20000
        "prior_num_iter":2,  # prior_num_iter*prior_batch_size=prior_buffer_size
        "prior_batch_size":2000,  # 原设置2000  取一批数据更新
    }

}

