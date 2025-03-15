import os
import random
import time

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, precision_score, recall_score
from torch.utils.data import TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.deterministic = True
    torch.backends.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    dgl.random.seed(seed)
    dgl.seed(seed)
    torch.use_deterministic_algorithms(True)


def dataset(HSI, LiDAR, label):
    label_index = torch.where(label != 0)[-1]
    unlabel_index = torch.where(label == 0)[-1]
    label = label.long()
    label_HSI = HSI[label_index]
    label_LiDAR = LiDAR[label_index]
    labeled_label = label[label_index]
    # if uncertain_index.shape[0] == 0:
    unlabel_HSI = HSI[unlabel_index]
    unlabel_LiDAR = LiDAR[unlabel_index]

    len = label_index.shape[0]

    rand_list = [i for i in range(unlabel_HSI.shape[0])]  # 用于随机的列表
    rand_idx = random.sample(rand_list, np.ceil(len).astype('int32'))
    unlabel_HSI = unlabel_HSI[rand_idx]
    unlabel_LiDAR = unlabel_LiDAR[rand_idx]

    dataset = TensorDataset(label_HSI, label_LiDAR, labeled_label, unlabel_HSI, unlabel_LiDAR)
    return dataset


def SAM(X, Y):
    # 对于两个特征，它们的余弦相似度就是两个特征在经过L2归一化之后的矩阵内积
    feature1 = X
    feature2 = Y
    feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
    feature2 = F.normalize(feature2)
    distance = feature1.mm(feature2.t())  # 计算余弦相似度
    # 将定义域限制在（-1，1）
    distance = torch.clamp(distance, -1, 1)
    # SAM_value = torch.acos(distance)  # 余弦相似度转化为角度
    SAM_value = torch.sqrt(2 * (1 - distance))  # 余弦相似度转化为欧氏距离
    SAM_value = SAM_value.cpu().detach().numpy()
    SAM_value[np.isnan(SAM_value)] = 0
    return SAM_value


def EuclideanDistances(x, y):
    m, n = x.shape[0], y.shape[0]
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist = dist - 2 * torch.mm(x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    dist = dist.cpu().detach().numpy()
    return dist


def get_A_k(data, k):
    n, b = data.shape
    # 代表与每个节点最相似的k个节点，相似则为1
    A_k_data = np.zeros([3, n * k])

    # 计算每个pixel与其他pixel的关系，此处为superpixel
    if b == 1:
        score = EuclideanDistances
    else:
        score = SAM
    rel = score(data, data)

    # 求HSI中关联性最大的pixel索引
    index = np.argsort(rel)  # argsort将元素从小到大排列，提取其index
    for i in range(n):
        for j in range(k):
            A_k_data[0, k * i + j] = i
            A_k_data[1, k * i + j] = index[i, j + 1]
            A_k_data[2, k * i + j] = rel[i, index[i, j + 1]]

    return A_k_data


def gen_A_coo(data, k):
    _, b = data.shape
    if (b == 1):
        score = EuclideanDistances
    else:
        score = SAM
    data_list = []
    all_index_value = np.zeros((data.shape[0], 2, k))  # 包含所有点最终index和value的三维矩阵
    ave_num = 15000
    number = data.shape[0] // ave_num

    time_start = time.time()

    # 先把data拆成很多组数据，放在一个list里
    i = 0
    for i in range(number):
        temp = data[i * ave_num:(i + 1) * ave_num, :]
        data_list.append(temp)
        del temp
    if (i + 1) * ave_num < data.shape[0]:
        temp = data[(i + 1) * ave_num: data.shape[0], :]
        data_list.append(temp)
        del temp

    # 对于每一组数据，每一个都与全部组做SAM，拿出最大的k个值做邻居，得到图结构
    for i in range(len(data_list)):
        # 拿出第i个组，与所有的组做SAM
        b = time.time()
        data_1 = data_list[i]
        index_and_value = np.array([])
        for j in range(len(data_list)):
            a = time.time()
            data_2 = data_list[j]
            relation = score(data_1, data_2)
            index_and_value_batch = np.zeros((data_1.shape[0], 2, k))  # 建一个三维矩阵存第i批中每一个点与第j批所有点的关系
            if i == j:
                for m in range(data_1.shape[0]):
                    relation_m = relation[m, :]
                    temp_index = np.argpartition(relation_m, kth=k + 1)[:k + 1]  # argsort将元素从小到大排列，提取其index
                    relation_m = relation_m[temp_index]
                    relation_m_index = np.argsort(relation_m)[1:1 + k]
                    temp_index = temp_index[relation_m_index]
                    index = temp_index  # 特征矢量与自己的SAM为0，肯定最小，将其去掉, 得到与当前点最相似的k个点
                    # 当与其它批次的特征矢量求余弦相似度时，最小值并不为0，所以不需要去掉。
                    value = relation_m[relation_m_index]  # 把这些点对应的SAM值拿出来
                    index = index + ave_num * j

                    temp_result = np.zeros((2, k))  # 把index和value合并成一个2×k的数组
                    temp_result[0] = index
                    temp_result[1] = value
                    index_and_value_batch[m] = temp_result
            else:
                for m in range(data_1.shape[0]):
                    relation_m = relation[m, :]
                    index = np.argpartition(relation_m, kth=k)[:k]  # argsort将元素从小到大排列，提取其index
                    value = relation_m[index]
                    index = index + ave_num * j

                    temp_result = np.zeros((2, k))  # 把index和value合并成一个2×k的数组
                    temp_result[0] = index
                    temp_result[1] = value
                    index_and_value_batch[m] = temp_result
            # 把这一批的关系保存下来，把这一批每一个点与后续所有批的index和value拼在一起，得到第i批中的点与全数据关系最近的 批数×k 个值
            if j == 0:
                index_and_value = index_and_value_batch
            else:
                index_and_value = np.concatenate((index_and_value, index_and_value_batch), axis=2)
            print('计算一次SAM距离的时间：', time.time() - a)
        # 从第i批的这些点与全数据的关系中提取出关系最大的前k个，保留下来
        for m in range(data_1.shape[0]):
            temp_1 = index_and_value[m, 1, :]
            temp_index = np.argpartition(temp_1, kth=k)[:k]  # 取前k个值
            index = index_and_value[m, 0, temp_index]
            value = index_and_value[m, 1, temp_index]
            all_index_value[ave_num * i + m, 0, :] = index
            all_index_value[ave_num * i + m, 1, :] = value
        print('完成一个批次计算所需的时间:', time.time() - b)
        print('*************************************')

    # 把矩阵转成coo的格式存储
    A_coo = np.zeros((3, (data.shape[0] * k)))
    for i in range(data.shape[0]):
        A_coo[0, k * i: k * (i + 1)] = i
        A_coo[1, k * i: k * (i + 1)] = all_index_value[i, 0, :]
        A_coo[2, k * i: k * (i + 1)] = all_index_value[i, 1, :]

    time_end = time.time()
    print('totally cost', time_end - time_start, 's')
    return A_coo


def diff_loss(feature_com, feature_pri):
    feature_com = feature_com.flatten(1)
    feature_pri = feature_pri.flatten(1)
    # 将数据正则化
    feature_com = F.normalize(feature_com)
    feature_pri = F.normalize(feature_pri)
    # 计算loss
    loss = torch.mean((feature_com.t() @ feature_pri).pow(2))

    return loss


def active_loss(bin, label_index, unlabel_index):
    bin = bin.view(-1)
    # 计算域鉴别器损失
    loss_label = torch.mean(torch.log(bin[label_index]))
    loss_unlabel = torch.mean(torch.log(1 - bin[unlabel_index]))
    loss = -loss_label - loss_unlabel

    return loss


def simse(real, pred):
    diffs = torch.add(real, - pred)
    n = torch.numel(diffs.data)
    simse = torch.sum(diffs).pow(2) / (n ** 2)
    return simse


def re_loss(data, data_recon):
    # 计算重构损失
    lossfun = nn.MSELoss()
    loss = lossfun(data, data_recon) - simse(data, data_recon)

    return loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        with torch.no_grad():
            delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
            loss = delta.dot(delta.T)
        torch.cuda.empty_cache()
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            # with torch.no_grad():
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            del XX, YY, XY, YX
            return loss


def mmd_data_standard(data):
    d_min = data.min()
    if d_min < 0:
        data = data + torch.abs(d_min)
        d_min = data.min()
    d_max = data.max()
    dst = d_max - d_min
    norm_data = (data - d_min).true_divide(dst)
    return norm_data


def mmd_loss(source, target):
    mmd = MMD_loss()
    batch = source.shape[0]
    source = source.view(batch, -1)
    target = target.view(batch, -1)
    source = mmd_data_standard(source)
    target = mmd_data_standard(target)
    loss = mmd(source, target)
    loss.requires_grad_(True)
    return loss


def select(bin, result, train_label):
    device = bin.device
    unlabel_index = torch.where(train_label == 0)[-1]
    class_num = train_label.max().item()

    # 对无标签数据分数排序，取出最接近决策边界的点
    bin = torch.abs(bin - 0.3)
    _, bin_index = torch.topk(bin, k=300, largest=False)
    # 获取选择出的点的伪标签
    class_index = unlabel_index[bin_index]
    m = nn.Softmax(dim=1)
    result = m(result)
    sample_selected = result[class_index]
    [confidence, label] = torch.max(sample_selected, dim=1)
    confidence = confidence.view(-1)
    label = label.view(-1)
    confidence_index = torch.tensor([]).to(device)

    th = torch.mean(confidence)
    print('D_q中的均值为：{}'.format(th))
    confidence_th = torch.where(confidence > th)[-1]
    sample_selected_number = torch.zeros(class_num)
    for i in range(class_num):
        sample_selected_number[i] = torch.where(label[confidence_th] == i)[-1].shape[0]
    class_th_para_stand = sample_selected_number / sample_selected_number.mean()
    class_th_para_stand = class_th_para_stand.to(device)
    for i in range(class_num):
        th_class = th * class_th_para_stand[i]
        if th_class > 1:
            th_class = 0.95
        confidence_index_temp = torch.where((label == i) & (confidence > th_class))[-1]
        confidence_index = torch.cat([confidence_index, confidence_index_temp])

    # 取出对应的伪标签以及所选取样本的索引
    confidence_index = confidence_index.long()
    select_index = class_index[confidence_index]
    label = label[confidence_index] + 1.0
    train_label[select_index] = label.long()

    return train_label


# 将全部数据送入网络
def test_all(net, test_data):
    device = next(net.parameters()).device
    result = torch.tensor([]).to(device)
    data = torch.tensor([]).to(device)
    label = torch.tensor([]).to(device)
    with torch.no_grad():
        for i, (HSI, LiDAR, label_temp) in enumerate(test_data):
            result_temp, data_temp, _ = net(HSI, LiDAR, 'label', 'test')
            result = torch.cat([result, result_temp], dim=0)
            data = torch.cat([data, data_temp], dim=0)
            label = torch.cat([label, label_temp], dim=0)
    return result, data, label


# 计算指标
def analy(label, result):
    label = label.cpu().detach().numpy()
    result = result.cpu().detach().numpy()
    result = result + 1
    # 计算混淆矩阵
    matrix = confusion_matrix(label, result)
    # 计算总体精度
    oa = accuracy_score(label, result)
    # 计算每类精度
    each_aa = recall_score(label, result, average=None, zero_division=0.0)
    each_aa = each_aa * 100
    each_aa = [round(i, 2) for i in each_aa]
    # 计算平均精度
    aa = recall_score(label, result, average='macro', zero_division=0.0)
    # 计算kappa系数
    kappa = cohen_kappa_score(label, result)
    return round(oa * 100, 2), each_aa, round(aa * 100, 2), round(kappa * 100, 2)
