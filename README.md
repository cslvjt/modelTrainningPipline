# 仓库简介
使用accelerate训练图像视频复原相关的代码
# 环境配置
"""
pip install -r requirements.txt
python install -e ./ModelTrainPipeline
"""

# 训练命令
```
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 1233 trainer.py --name SIMODeblur_SE --config config.yaml
```
#  如何使用该仓库
当需要训练模型时，仅仅需要对ModelTrainPipeline/dataloader中建立数据读取函数，trainer，option/train_config.yaml三个文件进行修改。
##  ModelTrainPipeline
文件夹下是所有通用的代码，包含数据读取，模型训练设置等
在ModelTrainPipeline/documents.md中有详细介绍。
## ModelZoo
存放的是用本框架训练的模型，具体模型及相应的论文在ModelZoo/modelList.md
## option
存放的是训练参数,各种参数代表含义已经在option/train_config.yaml文件中指明
+ dataset：训练数据集路径，模型增强参数，batchsize等
+ net：模型超参数，例如模型结构
+ train：训练参数，例如学习率，优化器，损失函数，调度器
+ val：验证参数，例如验证间隔，模型保存间隔等
### trainer.py
这里主要分为两个部分，一个是模型结构，一个是单步训练loss计算过程
+ 模型结构在Net类中
+ 单步loss计算在trainer.train_step

# 感谢
本仓库的搭建思路借鉴于[BasicSR](https://github.com/XPixelGroup/BasicSR)