# ImageRestoration
使用accelerate训练图像视频复原相关的代码

# Train
```
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 1233 trainer.py --name SIMODeblur_SE --config config.yaml
```
