import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io as sio
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import get_cls_map
import time
# from patchbasedvit import MyNet_patch
from SSANet_parse import args
import random
from einops import rearrange
# from MADANet import SSAnet
from models.net import MoEProtoNet
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import defaultdict  # 导入 defaultdict
import torch
import time
from PIL import Image

import tifffile as tiff
# 你的其他代码


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# setup_seed(620)
# setup_seed(620)
# setup_seed(4403)
import h5py as hp


def loadData():
    # 读入数据
    # ip 与 salinas数据集同一传感器AVIRIS
    # pu与pua属于一个传感器ROSIS
    # 珠海一号OHS CMOS传感器
    print(args.dataset_name)
    if args.dataset_name == "IP":
        data = sio.loadmat('data/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat('data/Indian_pines_gt.mat')['indian_pines_gt']
    elif args.dataset_name == "Honghu":
        data = sio.loadmat('data/WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
        labels = sio.loadmat('data/WHU_Hi_HongHu_gt.mat')['WHU_Hi_HongHu_gt']
    elif args.dataset_name == "Salinas":
        data = sio.loadmat('data/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat('data/Salinas_gt.mat')['salinas_gt']
    elif args.dataset_name == "PC":
        data = sio.loadmat('data/Pavia.mat')['pavia']
        labels = sio.loadmat('data/Pavia_gt.mat')['pavia_gt']
    elif args.dataset_name == "KSC":
        data = sio.loadmat('data/KSC.mat')['KSC']
        labels = sio.loadmat('data/KSC_gt.mat')['KSC_gt']
    elif args.dataset_name == "Coast":
        data = np.array(hp.File('data/NC16.mat')['HSI'])
        data = rearrange(data, "c h w -> h w c")
        labels = np.array(hp.File('data/NC16.mat')['GT'])
    elif args.dataset_name == "yellow":
        data = tiff.imread('data/newtest.tif')
        data = rearrange(data, "c h w -> h w c")
        labels = tiff.imread('data/newtest_label.tif')

    elif args.dataset_name == "NC13":
        data = np.array(hp.File('data/NC13.mat')['HSI'])
        data = rearrange(data, "c h w -> h w c")
        labels = np.array(hp.File('data/NC13.mat')['GT'])

    return data, labels


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


BATCH_SIZE_TRAIN = args.batch


def create_data_loader():

    # 读入数据
    X, y = loadData()
    print(X.shape)
    print(y.shape)
    # 用于测试样本的比例
    test_ratio = args.test_ratio
    # 每个像素周围提取 patch 的尺寸
    patch_size = args.patch
    # 使用 PCA 降维，得到主成分的数量
    pca_components = args.pca
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)
    #
    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    # print('Data shape after PCA: ', X_pca.shape)
    #
    # print('\n... ... create data cubes ... ...')
    # X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    # print('Data cube X shape: ', X_pca.shape)
    # print('Data cube y shape: ', y.shape)
    #
    # print('\n... ... create train & test data ... ...')
    X_train_val, Xtest, y_train_val, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', X_train_val.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components)
    Xtrain = X_train_val.reshape(-1, patch_size, patch_size, pca_components)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)
    #
    # # 为了适应 pytorch 结构，数据要做 transpose
    X = X.transpose(0, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 3, 1, 2)
    Xtest = Xtest.transpose(0, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_loader和 test_loader
    X = TestDS(X, y_all)

    testset = TestDS(Xtest, ytest)

    trainset = TrainDS(Xtrain, y_train_val)
    # 创建验证集Dataset和DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0,
                                               )

    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=BATCH_SIZE_TRAIN,
                                              shuffle=False,
                                              num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                  batch_size=BATCH_SIZE_TRAIN,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  )
    print("starting Training test_ratio is {}.".format(args.test_ratio))

    # return train_loader, val_loader, test_loader, all_data_loader, y, Xtest
    return train_loader, test_loader, all_data_loader, y, Xtest


""" Training dataset"""


class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Testing dataset"""


class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(train_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型
    model = MoEProtoNet(
        spectral_bands=args.pca,
        num_classes=args.cls
    ).to(device)
    # 优化器配置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()



    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        total_loss = 0.0
        model.train()
        for i, (data, target) in enumerate(
                tqdm(train_loader, desc='Epoch {}'.format(epoch + 1), unit='batch', leave=False, dynamic_ncols=True)):
            data, target = data.to(device), target.to(device)
            # 前向传播
            outputs = model(data, target)
            # 计算损失
            ce_loss = criterion(outputs['logits'], target)
            gate_weights = outputs['gate_weights']
            entropy_loss = -torch.mean(torch.sum(gate_weights * torch.log(gate_weights + 1e-10), dim=1))
            loss = ce_loss + 0.1 * entropy_loss
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累积损失
            total_loss += loss.item()

            # 调整学习率
            scheduler.step()

        print(
            '\n[Epoch: %d]   [loss avg: %.4f]' % (
            epoch,
            total_loss / (epoch)))
        torch.save(model.state_dict(), args.model_pth)

    print('Finished Training')

    return model, device


def ytest(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()

    targets = []
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        preds = outputs['logits']
        preds = np.argmax(preds.detach().cpu().numpy(), axis=1)

        if count == 0:
            y_pred_test = preds
            y_test = labels
            count = 1
            targets.append(labels)
        else:
            y_pred_test = np.concatenate((y_pred_test, preds))
            y_test = np.concatenate((y_test, labels))
    return y_pred_test, y_test


from matplotlib.colors import ListedColormap, BoundaryNorm


# 自定义颜色映射函数
def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 1:
            y[index] = np.array([147, 67, 46]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 100, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 123]) / 255.
        if item == 5:
            y[index] = np.array([164, 75, 155]) / 255.
        if item == 6:
            y[index] = np.array([101, 174, 255]) / 255.
        if item == 7:
            y[index] = np.array([118, 254, 172]) / 255.
        if item == 8:
            y[index] = np.array([60, 91, 112]) / 255.
        if item == 9:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 10:
            y[index] = np.array([255, 255, 125]) / 255.
        if item == 11:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 12:
            y[index] = np.array([100, 0, 255]) / 255.
        if item == 13:
            y[index] = np.array([0, 172, 254]) / 255.
        if item == 14:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 15:
            y[index] = np.array([171, 175, 80]) / 255.
        if item == 16:
            y[index] = np.array([101, 193, 60]) / 255.
        if item == 17:
            y[index] = np.array([255, 105, 180]) / 255.  # 粉色
        if item == 18:
            y[index] = np.array([0, 128, 128]) / 255.  # 青色
        if item == 19:
            y[index] = np.array([128, 0, 128]) / 255.  # 紫色
        if item == 20:
            y[index] = np.array([210, 105, 30]) / 255.  # 巧克力色
        if item == 21:
            y[index] = np.array([32, 178, 170]) / 255.  # 淡青色
        if item == 22:
            y[index] = np.array([0, 255, 255]) / 255.  # 青色
    return y


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    target_names = args.target_names
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100


if __name__ == '__main__':
    train_loader, test_loader, all_data_loader, y_all, Xtest = create_data_loader()
    # train_loader, val_loader, test_loader, all_data_loader, y_all, Xtest = create_data_loader()
    tic1 = time.perf_counter()
    net, device = train(train_loader, epochs=args.epochs)
    # net, device = train(train_loader, val_loader, epochs=args.epochs)

    # 只保存模型参数
    # torch.save(net.state_dict(), args.model_pth)
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    net = MoEProtoNet(args.pca, args.cls).to(device)
    net.load_state_dict(torch.load(args.model_pth))
    # net.load_state_dict(torch.load('best_model.pth'))
    y_pred_test, y_test = ytest(device, net, test_loader)
    toc2 = time.perf_counter()
    # 评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    file_name = args.cls_report + args.dataset_name + ".txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

    print(file_name)
    get_cls_map.get_cls_map(net, device, all_data_loader, y_all)

    print("Training Done")
    plt.clf()