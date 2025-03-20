import time

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
import os
import seaborn as sns
from tqdm import tqdm
from net import MSGSAN
# 设置随机种子
torch.manual_seed(345)
np.random.seed(345)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trento_data():
    hsi_data = sio.loadmat('data/Trento/HSI_Trento.mat')
    lidar_data = sio.loadmat('data/Trento/Lidar_Trento.mat')
    gt_data = sio.loadmat('data/Trento/GT_Trento.mat')
    return hsi_data['HSI_Trento'], lidar_data['Lidar_Trento'], gt_data['GT_Trento']


# 2. 修改后的数据预处理
class TrentoDataset(Dataset):
    def __init__(self, hsi, lidar, gt, patch_size=11):
        self.patch_size = patch_size
        self.pad_size = patch_size // 2

        # 归一化处理
        self.hsi = self.normalize_hsi(hsi)
        self.lidar = self.normalize_lidar(lidar)
        self.gt = gt

        # 获取有效样本索引
        self.indices = np.stack(np.where(gt > 0)).T
        self.classes = np.unique(gt[gt > 0]) - 1  # 0-based

        # 填充数据
        self.padded_hsi = np.pad(self.hsi,
                                 ((self.pad_size, self.pad_size),
                                  (self.pad_size, self.pad_size),
                                  (0, 0)), mode='reflect')
        self.padded_lidar = np.pad(self.lidar,
                                   ((self.pad_size, self.pad_size),
                                    (self.pad_size, self.pad_size)), mode='reflect')

    @staticmethod
    def normalize_hsi(data):
        data = data.astype(np.float32)
        for i in range(data.shape[-1]):
            band = data[..., i]
            data[..., i] = (band - band.min()) / (band.max() - band.min())
        return data

    @staticmethod
    def normalize_lidar(data):
        data = data.astype(np.float32)
        return (data - data.min()) / (data.max() - data.min())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row, col = self.indices[idx]
        label = self.gt[row, col] - 1  # 转换为0-based

        # 提取patch
        hsi_patch = self.padded_hsi[
                    row:row + self.patch_size,
                    col:col + self.patch_size
                    ].transpose(2, 0, 1)  # [C, H, W]

        lidar_patch = self.padded_lidar[
                      row:row + self.patch_size,
                      col:col + self.patch_size
                      ][np.newaxis, ...]  # [1, H, W]

        return (
            torch.from_numpy(hsi_patch).float(),
            torch.from_numpy(lidar_patch).float(),
            torch.tensor(label, dtype=torch.long)
        )


# 3. 修改后的训练流程
class Trainer:
    def __init__(self, model, num_classes):
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=5)
        self.scaler = GradScaler()
        self.best_acc = 0.0
        self.num_classes = num_classes

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        progress = tqdm(loader, desc="Training")

        for hsi, lidar, labels in progress:
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(hsi, lidar)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for hsi, lidar, labels in tqdm(loader, desc="Evaluating"):
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            labels = labels.to(device)

            outputs = self.model(hsi, lidar)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * hsi.size(0)
            all_preds.append(torch.argmax(outputs, dim=1).cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        return {
            'loss': total_loss / len(loader.dataset),
            'accuracy': accuracy_score(all_labels, all_preds),
            'kappa': cohen_kappa_score(all_labels, all_preds),
            'confusion': confusion_matrix(all_labels, all_preds),
            'class_acc': confusion_matrix(all_labels, all_preds).diagonal() /
                         confusion_matrix(all_labels, all_preds).sum(axis=1)
        }

    def save_model(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_acc': self.best_acc
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.best_acc = checkpoint['best_acc']


def predict_entire_image(model, hsi, lidar, patch_size=11, batch_size=64):
    model.eval()
    pad_size = patch_size // 2
    device = next(model.parameters()).device

    # 预处理
    hsi = TrentoDataset.normalize_hsi(hsi)
    lidar = TrentoDataset.normalize_lidar(lidar)

    # 填充
    hsi_padded = np.pad(hsi,
                        ((pad_size, pad_size),
                         (pad_size, pad_size),
                         (0, 0)), mode='reflect')
    lidar_padded = np.pad(lidar,
                          ((pad_size, pad_size),
                           (pad_size, pad_size)), mode='reflect')

    # 生成预测结果矩阵
    pred_map = np.zeros(hsi.shape[:2], dtype=np.uint8)

    # 分块处理避免内存不足
    for row in tqdm(range(hsi.shape[0]), desc="Generating full prediction"):
        for col in range(hsi.shape[1]):
            # 提取patch
            hsi_patch = hsi_padded[row:row + patch_size, col:col + patch_size].transpose(2, 0, 1)
            lidar_patch = lidar_padded[row:row + patch_size, col:col + patch_size][np.newaxis, ...]

            # 转换为Tensor
            hsi_tensor = torch.from_numpy(hsi_patch).unsqueeze(0).float().to(device)
            lidar_tensor = torch.from_numpy(lidar_patch).unsqueeze(0).float().to(device)

            # 预测
            with torch.no_grad():
                output = model(hsi_tensor, lidar_tensor)
                pred = output.argmax(dim=1).item()

            pred_map[row, col] = pred + 1  # 恢复1-based标签

    return pred_map

# 4. 修改后的主流程
def main():
    # 加载数据
    hsi, lidar, gt = load_trento_data()
    train_size = 0.028
    epoch = 100
    Training = False


    # 创建数据集
    full_dataset = TrentoDataset(hsi, lidar, gt)

    # 划分数据集
    train_idx, test_idx = train_test_split(
        np.arange(len(full_dataset)),
        test_size=1 - train_size,
        stratify=full_dataset.gt[full_dataset.gt > 0] - 1,
        random_state=42
    )
    # train_idx, val_idx = train_test_split(
    #     train_idx,
    #     test_size=0.25,
    #     stratify=full_dataset.gt[full_dataset.indices[train_idx][:, 0],
    #     full_dataset.indices[train_idx][:, 1]] - 1,
    #     random_state=42
    # )

    # 创建数据加载器
    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    # val_loader = DataLoader(
    #     Subset(full_dataset, val_idx),
    #     batch_size=64,
    #     num_workers=4,
    #     pin_memory=True
    # )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=64,
        num_workers=4,
        pin_memory=True
    )

    # 初始化模型
    model = MSGSAN(num_classes=len(full_dataset.classes))
    trainer = Trainer(model, num_classes=len(full_dataset.classes))


    if Training:
        # 训练参数
        epochs = epoch
        best_accuracy = math.inf
        os.makedirs("checkpoints", exist_ok=True)

        # 训练循环
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss = trainer.train_epoch(train_loader)
            val_metrics = trainer.evaluate(train_loader)

            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f} | "
                  f"Kappa: {val_metrics['kappa']:.4f}")

            # 保存最佳模型
            if val_metrics['loss'] < best_accuracy:
                best_accuracy = val_metrics['loss']
                trainer.save_model(f"checkpoints/best_model1.pth")
                print("Saved best model!")

            # 调整学习率
            trainer.scheduler.step(val_metrics['accuracy'])

    # 测试最佳模型
    trainer.load_model("checkpoints/best_model1.pth")
    test_metrics = trainer.evaluate(test_loader)
    aa = np.mean(test_metrics['class_acc'])

    # 控制台输出
    print("\nFinal Test Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Kappa: {test_metrics['kappa']:.4f}")
    print(f"AA: {aa:.4f}")
    print("Class-wise Accuracy:")
    for i, acc in enumerate(test_metrics['class_acc']):
        print(f"Class {i + 1}: {acc:.4f}")

        # 生成文本报告
    report_content = f"""
                Evaluation Report
                ==================

                Overall Accuracy (OA): {test_metrics['accuracy']:.4f}
                Average Accuracy (AA): {aa:.4f}
                Kappa Coefficient: {test_metrics['kappa']:.4f}

                Class-wise Accuracy:
                """ + "\n".join([f"Class {i + 1}: {acc:.4f}"
                                 for i, acc in enumerate(test_metrics['class_acc'])]) + "\n\n"

    # 添加混淆矩阵文本
    report_content += "Confusion Matrix:\n"
    confusion_str = np.array2string(test_metrics['confusion'],
                                    formatter={'int': lambda x: f"{x:4d}"})
    report_content += confusion_str

    # 保存到文件
    with open("report" + str(time.perf_counter()) + ".txt", "w") as f:
        f.write(report_content)

        # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_metrics['confusion'], annot=True, fmt='d',
                xticklabels=full_dataset.classes + 1,
                yticklabels=full_dataset.classes + 1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix' + str(time.perf_counter()) + '.png')
    plt.close()

    print("\nReport saved to report.txt")

    print("\nGenerating full classification map...")
    hsi_full, lidar_full, gt_full = load_trento_data()
    prediction_map = predict_entire_image(trainer.model, hsi_full, lidar_full)

    # 保存结果图
    plt.figure(figsize=(12, 10))
    plt.imshow(prediction_map, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('results/full_classification_map' + str(time.perf_counter()) + '.png', bbox_inches='tight')
    plt.close()





if __name__ == "__main__":
    main()