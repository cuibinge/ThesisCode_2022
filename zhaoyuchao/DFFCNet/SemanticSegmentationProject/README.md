# DFFCNet: 动态频域滤波的高分辨率海岸线提取网络

## 网络特点

1. **小波引导的动态频域融合模块（WDF²M）**：通过Haar小波变换将特征分解为低频（LL）与高频分量（LH/HL/HH），并引入可学习频率权重Token来动态调节目标敏感频段，有效提升了网络对复杂场景中高频细节特征的捕捉能力。

2. **全局-局部特征交互模块（GLFI）**：构建了双向门控融合机制，使Swin Transformer捕获的全局上下文信息通过空间注意力机制引导EfficientNet提取的局部细节特征进行增强，同时局部特征反向修正Transformer的位置编码偏移。

3. **三重频空融合优化（TFS-Fusion）**：在频域、空域和语义域三个层面构建联合优化目标。

## 数据准备

请将您的数据集按照以下结构组织：

```
data/
  ├── train/
  │     ├── images/
  │     └── mask/
  └── val/
        ├── images/
        └── mask/
  └── test/
        ├── images/
        └── mask/
```

## 使用代码

代码在 Python 3.10.0 及以上版本、CUDA 11.0 及以上版本下稳定运行。

### 克隆仓库

```bash
git clone https://github.com/cuibinge/ThesisCode_2022/tree/main/zhaoyuchao/DFFCNet.git
cd DFFCNet
```

### 安装依赖

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install timm opencv-python numpy tqdm pillow
```

或者使用 pip 安装所有依赖：

```bash
pip install torch timm opencv-python numpy tqdm pillow
```

## 训练和测试

1. **训练模型**

```bash
python train.py
```

2. **评估模型**

```bash
python accuracy_evaluation.py
```

对于多分类评估，可以使用：

```bash
python accuracy_multi_class.py
```

