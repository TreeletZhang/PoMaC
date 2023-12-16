## 环境配置
```bash
\MARLS_Unity>conda create -n MARLS_Unity python=3.7
\MARLS_Unity>conda activate MARLS_Unity
(MARLS_Unity) \MARLS_Unity>conda activate MARLS_Unity
(MARLS_Unity) \MARLS_Unity>pip install -r requirements.txt
```

###mape环境：
```bash
#mape环境训练
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=mape --env-name=simple_tag --train-algorithm=MADDPG --group-num=2 --agent-group-index 0 0 0 1 --train

#mape环境继续训练
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=mape --env-name=simple_tag --train-algorithm=MADDPG --group-num=2 --agent-group-index 0 0 0 1 --train --resume

#mape环境推断
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=mape --env-name=simple_tag --train-algorithm=MADDPG --group-num=2 --agent-group-index 0 0 0 1 --inference
```

###PoMaC算法命令行调用示例：
```bash
#mape环境训练
(MARLS_Unity) \MARLS_Unity>python train.py --env-type=mape --env-name=simple_spread --train-algorithm=POMAC --group-num=1 --agent-group-index 0 0 0 --discrete-action --max-episode-len=50 --train-frequency=50 --print-frequency=5 --train 
```
