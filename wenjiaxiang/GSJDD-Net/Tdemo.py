import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import tifffile as tiff
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.msgdan import MSGSAN



# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DataSplitter:
    def __init__(self, labels, valid_indices, class_count):
        self.labels = labels
        self.valid_indices = valid_indices
        self.class_count = class_count

    def stratified_split(self, train_ratio=0.02, val_ratio=0.01):
        """优化后的分层划分方法，返回训练、验证、测试索引"""
        train_idx, test_idx = train_test_split(
            self.valid_indices,
            train_size=train_ratio,
            stratify=self.labels[self.valid_indices],
            random_state=42
        )

        train_idx, val_idx = train_test_split(
            train_idx,
            test_size= val_ratio,
            stratify=self.labels[train_idx],
            random_state=42
        )
        print(len(train_idx), len(val_idx), len(test_idx))
        return train_idx, val_idx, test_idx


class HSI_LiDAR_Dataset:
    def __init__(self, dataset_name, nodata_values=(65536, 0)):
        self.dataset_name = dataset_name
        self.hsi_nodata = nodata_values[1]
        self.lidar_nodata = nodata_values[0]

        # 加载原始数据
        self.data_HSI = tiff.imread("dataset/Hospp.tif")  # (H, W, C)
        self.data_LiDAR = tiff.imread("dataset/Lospp.tif")
        self.gt = tiff.imread("dataset/labels.tif")

        # 调整维度
        if len(self.data_LiDAR.shape) == 2:
            self.data_LiDAR = np.expand_dims(self.data_LiDAR, axis=-1)

        # 数据标准化
        self.valid_mask = self.create_valid_mask()
        self.class_count = 3
        self.standardize()

    def create_valid_mask(self):
        hsi_mask = np.all(self.data_HSI != self.hsi_nodata, axis=-1, keepdims=True)
        lidar_mask = (self.data_LiDAR != self.lidar_nodata).all(axis=-1, keepdims=True)
        return np.squeeze(hsi_mask & lidar_mask)

    def standardize(self):
        hsi_valid = self.data_HSI[self.valid_mask]
        self.hsi_scaler = preprocessing.StandardScaler().fit(hsi_valid)
        self.data_HSI = self._apply_scaling(self.data_HSI, self.hsi_scaler)

        lidar_valid = self.data_LiDAR[self.valid_mask]
        self.lidar_scaler = preprocessing.StandardScaler().fit(lidar_valid)
        self.data_LiDAR = self._apply_scaling(self.data_LiDAR, self.lidar_scaler)

    def _apply_scaling(self, data, scaler):
        original_shape = data.shape
        data_flat = data.reshape(-1, original_shape[-1])
        valid_flat = self.valid_mask.reshape(-1)
        data_flat[valid_flat] = scaler.transform(data_flat[valid_flat])
        return data_flat.reshape(original_shape)

    def generate_patches(self, patch_size=13):
        pad_width = patch_size // 2

        hsi_pad = np.pad(self.data_HSI,
                         [(pad_width, pad_width), (pad_width, pad_width), (0, 0)],
                         mode='constant')
        lidar_pad = np.pad(self.data_LiDAR,
                           [(pad_width, pad_width), (pad_width, pad_width), (0, 0)],
                           mode='constant')
        valid_pad = np.pad(self.valid_mask,
                           pad_width,
                           mode='constant')

        rows, cols = np.where((self.gt != 0) & self.valid_mask)
        valid_patches = []

        for r, c in zip(rows, cols):
            r_pad = r + pad_width
            c_pad = c + pad_width
            if not valid_pad[r_pad - pad_width:r_pad + pad_width + 1,
                   c_pad - pad_width:c_pad + pad_width + 1].all():
                continue

            hsi_patch = hsi_pad[r_pad - pad_width:r_pad + pad_width + 1,
                        c_pad - pad_width:c_pad + pad_width + 1, :]
            lidar_patch = lidar_pad[r_pad - pad_width:r_pad + pad_width + 1,
                          c_pad - pad_width:c_pad + pad_width + 1, :]

            hsi_patch = np.transpose(hsi_patch, (2, 0, 1))  # [C, H, W]
            lidar_patch = np.transpose(lidar_patch, (2, 0, 1))

            valid_patches.append((
                hsi_patch.astype(np.float32),
                lidar_patch.astype(np.float32),
                self.gt[r, c]
            ))

        hsi_tensor = torch.stack([torch.from_numpy(x[0]) for x in valid_patches])
        lidar_tensor = torch.stack([torch.from_numpy(x[1]) for x in valid_patches])
        labels = torch.LongTensor([x[2] for x in valid_patches])
        return hsi_tensor, lidar_tensor, labels


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, num_classes):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # 添加权重衰减
        self.best_acc = 0.0
        self.scaler = torch.cuda.amp.GradScaler()  # 新增

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0

        with tqdm(self.train_loader,
                  desc=f"Epoch {epoch + 1:03d}",
                  bar_format="{l_bar}{bar:20}{r_bar}",
                  colour='green') as pbar:
            for batch_idx, (hsi, lidar, labels) in enumerate(pbar):
                self.optimizer.zero_grad()

                # 混合精度前向
                with autocast():
                    outputs = self.model(hsi.cuda(), lidar.cuda())
                    loss = self.criterion(outputs, labels.cuda() - 1)

                # 混合精度反向
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == (labels.cuda() - 1)).sum().item()

                # 实时更新进度条信息
                avg_loss = total_loss / (batch_idx + 1)
                current_acc = correct / ((batch_idx + 1) * hsi.size(0))
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{current_acc:.2%}'
                })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / len(self.train_loader.dataset)
        return avg_loss, accuracy

    def fast_validate(self):
        """优化后的快速验证方法"""
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad(), tqdm(self.val_loader,
                                   desc="Validating",
                                   bar_format="{l_bar}{bar:20}{r_bar}",
                                   colour='yellow') as pbar:
            for hsi, lidar, labels in pbar:
                # 将数据移动到GPU
                hsi = hsi.cuda(non_blocking=True)
                lidar = lidar.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                # 前向传播
                outputs = self.model(hsi, lidar)
                preds = torch.argmax(outputs, dim=1)

                # 更新正确预测的数量和总样本数
                total_correct += (preds == (labels - 1)).sum().item()
                total_samples += labels.size(0)

                # 实时更新进度条信息
                current_acc = total_correct / total_samples
                pbar.set_postfix({'acc': f'{current_acc:.2%}'})

        accuracy = total_correct / total_samples
        return accuracy

    def final_test(self):
        """最终测试方法"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad(), tqdm(self.test_loader,
                                   desc="Testing",
                                   bar_format="{l_bar}{bar:20}{r_bar}",
                                   colour='red') as pbar:
            for hsi, lidar, labels in pbar:
                outputs = self.model(hsi.cuda(), lidar.cuda())
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend((labels - 1).numpy())

        cm = confusion_matrix(all_labels, all_preds)
        accuracy = np.trace(cm) / np.sum(cm)
        aa = np.mean(cm.diagonal() / cm.sum(axis=1))
        return accuracy, aa, cm


def main():
    set_seed(42)

    # 初始化数据集
    dataset = HSI_LiDAR_Dataset("chengliu")

    # 生成图像块
    hsi, lidar, labels = dataset.generate_patches(patch_size=13)
    print("HSI输入维度:", hsi.shape)
    print("LiDAR输入维度:", lidar.shape)

    # 数据划分（训练2%，验证1%，测试97%）
    splitter = DataSplitter(labels.numpy(), np.arange(len(labels)), dataset.class_count)
    train_idx, val_idx, test_idx = splitter.stratified_split(train_ratio=0.02, val_ratio=0.2)

    # 创建数据加载器（优化参数提升速度）
    train_loader = DataLoader(
        data.TensorDataset(hsi[train_idx], lidar[train_idx], labels[train_idx]),
        batch_size=128,  # 增大batch size
        shuffle=True,
        num_workers=8,  # 增加workers数
        pin_memory=True,
        persistent_workers=True  # 减少重复初始化
    )
    val_loader = DataLoader(
        data.TensorDataset(hsi[val_idx], lidar[val_idx], labels[val_idx]),
        batch_size=12, shuffle=False)
    test_loader = DataLoader(
        data.TensorDataset(hsi[test_idx], lidar[test_idx], labels[test_idx]),
        batch_size=512, shuffle=False)

    # 初始化模型
    model = MSGSAN(num_classes=3).cuda()
    torch.backends.cudnn.benchmark = True
    # 训练循环
    trainer = Trainer(model, train_loader, val_loader, test_loader, dataset.class_count)
    best_acc = 0.0

    with tqdm(range(50),
              desc="Total Progress",
              bar_format="{l_bar}{bar:20}{r_bar}",
              colour='blue') as epoch_pbar:

        for epoch in epoch_pbar:
            train_loss, train_acc = trainer.train_epoch(epoch)
            val_acc = trainer.fast_validate()

            # 更新外层进度条信息
            epoch_pbar.set_postfix({
                'train_acc': f'{train_acc:.2%}',
                'val_acc': f'{val_acc:.2%}',
                'best_acc': f'{best_acc:.2%}'
            })

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")

            # 打印验证结果
            tqdm.write(
                f"Epoch {epoch + 1:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2%} | "
                f"Val Acc: {val_acc:.2%}"
            )

    # 最终测试
    model.load_state_dict(torch.load("best_model.pth"))
    final_acc, final_aa, cm = trainer.final_test()
    print(f"\n最终测试结果:")
    print(f"总体精度: {final_acc:.4%}")
    print(f"平均精度 (AA): {final_aa:.4%}")
    print("混淆矩阵:\n", cm)


if __name__ == "__main__":

    main()