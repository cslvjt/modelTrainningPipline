# abstract
介绍整个仓库的每一个文件作用

# utils
存放的是一些工具函数，例如数据增强，数据读取等
## img_utils
存放的是图像处理的工具函数，例如图像增强，图像读取等
## logger
生成logger文件
## losses
存放各种损失函数
## metrics
存放各种评估指标

# BaseTrainer
模型训练公用的函数，里面存放的是调度器，优化器，损失函数的定义
# dataloader
数据集加载器,在该代码中，dataloader会根据在option/train_config.yaml中的指定参数读取数据
