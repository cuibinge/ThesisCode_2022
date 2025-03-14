# WS-RTNet & HSF-RTNet Chenruipeng of SDUST 2025硕士学位论文
# 基于原型学习和小波变换的两阶段弱监督赤潮遥感监测方法研究

## 数据准备

弱监督赤潮数据集路径: **浪潮服务器/chenruipeng/weakly/WRTNet/WRTNet/data**

resnet预训练权重: **浪潮服务器/chenruipeng/weakly/WRTNet/WRTNet/resnet.pth**

数据集结构:

```
data
    ├── eval
    │   ├── img
    │   └── mask
    ├── test.txt
    ├── train
    │   ├── img
    │   ├── superpixel
    └── train.txt
```
## 环境依赖

    --使用以下命令创建conda依赖环境:
    ```bash
    conda env create -f env.yaml
    ```

## 开始

### Step 1: 训练WS-RTNet并生成伪标签

- Please specify a workspace to save the model and logs.

    ```bash
    CUDA_VISIBLE_DEVICES=0 python run_sample.py --data_root ./data --work_space ./data/workspace --log_name sample_train --train_cam_pass True --train_procam_pass True --make_procam_pass True
    ```

- (Optional) To generate pseudo masks on the test set, change ```--img_list```:

    ```bash
    CUDA_VISIBLE_DEVICES=0 python run_sample.py --data_root ./data --work_space ./data/workspace --log_name sample_eval --make_procam_pass False --eval_cam_pass True  --img_list test.txt
    ```

### Step 2: 训练语义分割网络(论文工作二 HSF-RTNet)


- 进入文件夹 ```<segmentation>```:

    ```bash
    cd segmentation
    ```

- 为训练集创建标签文件夹，并将伪标签放入其中:

    ```
    cp -r ../data/workspace/procam_mask ../data/train/mask
    ```

- 训练 HSF-RTNet(注意更改数据集路径并选择HSF-RTNet模型):

    ```bash
    python train.py
    ```
### Step 3: 模型测试和精度评估(测试前请准备赤潮测试集)

    在segmentation文件夹下
    ```bash
    python predict.py
    ```
