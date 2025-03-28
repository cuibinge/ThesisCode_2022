# 项目名称

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)

## 📌 项目简介
###基于与视觉提示与边缘感知的冰川遥感变化检测方法研究
####项目包含两个网络模型：VPGCD-Net and EACD-Net
#####VPGCD-Net： 基于视觉提示工程的冰川遥感影像变化检测方法(A Visual Prompt Driven Network for Glacier Change Detection in Remote Sensing Imagery，VPGCD-Net)，这是一种基于transformer的网络模型，利用视觉提示引导实现精确高效的冰川变化检测，降低由阴影导致的误检。 VPGCD-Net 采用双支结构来处理位时图像。一个分支包含视觉提示模块，该模块结合了阈值分割、连接和特征减法，以突出重要的变化区域，这些区域可通过视觉提示transformer (VPT)，为变化检测过程提供高级指导。另一个分支的特点是基于transformer的变化检测（TCD) 架构，该架构以 ResNet18 为骨干网络提取位时空间特征。一组transformer blocks用于捕捉全局上下文依赖关系，从而能够对空间和时间关系有更深入的了解。随后，特征线性调制（FiLM）模块在视觉提示的指导下，自适应地完善标记以有效代表真正的冰川变化区域。 通过利用多头自注意力机制，transformer解码器可精确识别冰川变化区域，捕捉时间差异和语义变化，同时抑制伪变化和背景噪音。
#####EACD-Net： 基于边缘感知的冰川变化检测网络（Edge-Aware Change Detection Network, EACD-Net），旨在提高冰川变化区域的检测精度，特别是针对复杂光照环境、模糊边界和小尺度变化区域的挑战。EACD-Net采用ResNet-50作为骨干网络，通过空洞卷积扩展感受野，确保双时相特征对齐的同时，提升局部细节与全局语义的建模能力。引入边缘感知模块（Edge-Aware Module, EAM），通过Sobel算子和跨层级边缘注意力机制建模多尺度边缘信息，从而增强模型对变化区域边界的敏感性，提高检测结果的边界完整性。同时，变化检测解码器（Change Decoder）结合多尺度特征聚合（Multi-Level Aggregation Block, MAB）和混合特征交互模块（Mix Block, MB），并利用变化注意力机制（Change Attention Mechanism, CAM）增强时序特征对比，以提升变化区域的对比度和精准度。


## 📁 目录结构
```bash
├── data/           # 数据集目录
├── result/         # 存放模型训练和推理的结果
├── src/            # 源代码
│   ├── config      # 模型相关配置
│   ├── models      # 模型定义
│   ├── utils       # 辅助工具函数
├── weight/         # 存放训练好的模型权重
├── environment.yml  # 依赖包yml格式（使用conda安装）
├── requirements.txt # 依赖包txt格式（）
├── README.md       # 项目说明（使用pip安装）
├── test.py         # 测试代码
├── train.py        # 训练代码
```

## 🛠️ 安装依赖
```bash
pip install -r requirements.txt
```
或者
```bash
conda env create -f environment.yml
```
## 🎯 训练与测试
### 训练模型（修改“--net_G”为EACD-Net可更换网络）
```bash
python train.py --project_name train01 --batch_size 16 --max_epochs 150 --net_G VPGCD-Net
```

### 测试模型（修改“--net_G”为EACD-Net可更换网络）
```bash
python test.py --project_name train01 --net_G VPGCD-Net
```

## 📊 结果展示
无

## 📝 许可证
本项目遵循 MIT 许可证，详情请查看 [LICENSE](LICENSE)。

## 👤 作者信息
- **姓名**: Zhishen Shi
- **邮箱**: 202282060078@sdust.edu.cn
- **GitHub**: [GitHub Profile](https://github.com/cuibinge/ThesisCode_2022/tree/main/shizhishen)

欢迎 Star ⭐，贡献代码 💡 或提交 Issues 🛠️！
