# Introduction

The best practice of pytorch DDP (DistributedDataParallel)

# Usage

1.install pip requirements
```shell script
pip install -r requirements
```
2.edit the config in yamls/train.yaml

3.mkdir
```shell script
mkdir checkpoints figs logs
```

4.run the python script. the experiment name is what you edit in train.yaml.
```shell script
# linux
nohup python train --file yamls/train.yaml &
tail -f logs/out${experiment_name}.log
```