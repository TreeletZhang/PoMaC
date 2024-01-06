## Anaconda
```bash
\MARLS_Unity>conda create -n MARLS_Unity python=3.7
\MARLS_Unity>conda activate MARLS_Unity
(MARLS_Unity) \MARLS_Unity>conda activate MARLS_Unity
(MARLS_Unity) \MARLS_Unity>pip install -r requirements.txt
```
## Training Parameter
```bash
(MARLS_Unity) \MARLS_Unity>python train.py -h
usage: train.py [-h] [--env-type ENV_TYPE] [--env-name ENV_NAME]
                [--env-run-type ENV_RUN_TYPE] [--env-worker-id ENV_WORKER_ID]
                [--no-graphics] [--group-num GROUP_NUM]
                [--agent-group-index AGENT_GROUP_INDEX [AGENT_GROUP_INDEX ...]]
                [--share-parameters] [--train-algorithm TRAIN_ALGORITHM]
                [--train] [--inference] [--max-episode-len MAX_EPISODE_LEN]
                [--num-episodes NUM_EPISODES]
                [--save-frequency SAVE_FREQUENCY]
                [--train-frequency TRAIN_FREQUENCY]
                [--print-frequency PRINT_FREQUENCY] [--resume]
Multi-Agent Reinforcement learning
optional arguments:
  -h, --help             # show this help message and exit
  --env-type             # environment type，"unity" or "mape"
  --env-name             # name of training environment
  --env-run-type         # "exe" or "client"
  --env-worker-id        # worker_id of "exe" environment
  --no-graphics          # open training interface or not
  --group-num            # the number of agent groups / categories in the environment
  --agent-group-index    # the group number corresponding to each agent in the environment
  --share-parameters     # share network or not
  --train-algorithm      # name of training algorithm
  --train                # training or not
  --inference            # inference or not
  --max-episode-len      # max steps in each episode
  --num-episodes         # training episode number
  --save-frequency       # model saving frequency
  --train-frequency      # model training frequency, where 1 represents calling the training function once per step
  --print-frequency      # the frequency of printing training data, where 100 represents printing every 100 rounds
  --resume               # continue training according to the previous training results
```

###MAPE ENVIRONMENT：
```bash
#TRAINING
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=mape --env-name=simple_tag --train-algorithm=MADDPG --group-num=2 --agent-group-index 0 0 0 1 --train

#CONTINUE TRAINING
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=mape --env-name=simple_tag --train-algorithm=MADDPG --group-num=2 --agent-group-index 0 0 0 1 --train --resume

#MODEL INFERENCE
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=mape --env-name=simple_tag --train-algorithm=MADDPG --group-num=2 --agent-group-index 0 0 0 1 --inference
```

###Example with PoMaC algorithm：
```bash
#mape environment
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=mape --env-name=simple_spread --train-algorithm=POMAC --group-num=1 --agent-group-index 0 0 0 --discrete-action --max-episode-len=50 --train-frequency=50 --print-frequency=5 --train 
```
