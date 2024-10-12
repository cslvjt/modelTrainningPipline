# ImageRestoration
使用accelerate训练图像视频复原相关的代码

# Train
```
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 1233 trainer.py --name SIMODeblur_SE --config config.yaml
```

# 如何使用该仓库
当需要训练模型时，仅仅需要对dataloader，trainer，option/train_config.yaml三个文件进行修改

## dataloader
这里指定的是数据加载流程。在该代码中，dataloader会根据在option/train_config.yaml中的指定参数读取数据
## trainer
这里主要分为两个部分，一个是模型结构，一个是单步训练loss计算过程
+ 模型结构在Net类中
+ 单步loss计算在Trainer.train_step
# option/train_config.yaml
指定训练参数，各种参数代表含义已经在option/train_config.yaml文件中指明