# abstract
介绍整个仓库的每一个文件作用
# option
存放的是模型训练配置文件，文件中包含以下设置
+ dataset：训练数据集路径，模型增强参数，batchsize等
+ net：模型超参数，例如模型结构
+ train：训练参数，例如学习率，优化器，损失函数，调度器
+ val：验证参数，例如验证间隔，模型保存间隔等

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
模型训练公用的函数
# dataloader
数据集加载器
# trainer
模型训练文件，包含模型结构，单步学习的函数
