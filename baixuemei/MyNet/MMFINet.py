import torch
import argparse
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os

from torch.nn import CrossEntropyLoss

from functions import normalize, get_data, GET_A2, metrics, show_results, train_and_test_data, output_metric, applyPCA, \
    get_cls_map, mmfinet_train_epoch, common_valid_epoch
from models.MMFINet.MMFINet import get_model

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'PaviaU', 'Salinas', 'NewTest', 'NC12', 'NC13', 'NC16', 'gf5'],
                    default='NewTest', help='dataset to use')
parser.add_argument("--num_run", type=int, default=5)
parser.add_argument('--epoches', type=int, default=20, help='epoch number')
parser.add_argument('--patches', type=int, default=64, help='number of patches')
parser.add_argument('--n_gcn', type=int, default=21, help='number of related pix')
parser.add_argument('--pca_band', type=int, default=30, help='pca_components')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')

parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=20, help='number of batch size')  # 64
parser.add_argument('--test_freq', type=int, default=10, help='number of evaluation')
parser.add_argument('--train_acc', type=float, default=87, help='accuracy of train')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
# -------------------------------------------------------------------------------

# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.cpu.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data

input, num_classes, total_pos_train, total_pos_test, total_pos_true, y_train, y_test, y_true, gt_hsi = get_data(
    args.dataset)
##########得到原始图像 训练测试以及所有点坐标 每一类训练测试的个数############


input = applyPCA(input, numComponents=args.pca_band)
# normalize data by band norm
input_normalize = normalize(input)
height, width, band = input_normalize.shape  # 145*145*200
print("height={0},width={1},band={2}".format(height, width, band))
# -------------------------------------------------------------------------------
# obtain train and test data
x_train_band, corner_train, indexs_train = train_and_test_data(input_normalize, band, total_pos_train,
                                                               patch=args.patches, w=height, h=width, n_gcn=args.n_gcn)
x_test_band, corner_test, indexs_test = train_and_test_data(input_normalize, band, total_pos_test, patch=args.patches,
                                                            w=height, h=width, n_gcn=args.n_gcn)
##########得到训练测试以及所有点的光谱############


input2 = torch.from_numpy(input_normalize).type(torch.FloatTensor)

x_train = torch.from_numpy(x_train_band).type(torch.FloatTensor)  # [695, 200, 7, 7]
y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # [695]
Label_train = Data.TensorDataset(x_train, y_train)

x_test = torch.from_numpy(x_test_band).type(torch.FloatTensor)  # [9671, 200, 7, 7]
y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # [9671]
Label_test = Data.TensorDataset(x_test, y_test)

label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
##########训练集的光谱值及标签##########
label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)
##########测试集的光谱值及标签##########


results = []

best_OA2 = 0.0
best_AA_mean2 = 0.0
best_Kappa2 = 0.0
model = get_model(num_classes)
# criterion
ce_dice_loss_main = CrossEntropyLoss()
ce_dice_loss_1 = CrossEntropyLoss()
ce_dice_loss_2 = CrossEntropyLoss()
ce_dice_loss_3 = CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
# -------------------------------------------------------------------------------

print("start training")
tic = time.time()

result_flag = False
run_results = {}
for epoch in range(args.epoches):
    # train model
    model.train()
    train_acc, train_obj, tar_t, pre_t = mmfinet_train_epoch(model, label_train_loader, ce_dice_loss_main,
                                                             ce_dice_loss_1, ce_dice_loss_2, ce_dice_loss_3, optimizer,
                                                             permute=[0, 3, 1, 2])
    OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
    print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
          .format(epoch + 1, train_obj, train_acc))

    if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1) and epoch >= args.epoches * 0.6:

        model.eval()
        tar_v, pre_v = common_valid_epoch(model, label_test_loader, ce_dice_loss_main, permute=True)
        OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
        if OA2 >= best_OA2 and AA_mean2 >= best_AA_mean2 and Kappa2 >= best_Kappa2:
            best_OA2 = OA2
            best_AA_mean2 = AA_mean2
            best_Kappa2 = Kappa2
            run_results = metrics(best_OA2, best_AA_mean2, best_Kappa2, AA2)
            result_flag = True
if result_flag:
    show_results(run_results, agregated=False)
    results.append(run_results)
toc = time.time()

'''
保存模型参数
'''
torch.save(model.state_dict(), 'save_models/MMFINet/mmfinet.pth')


def valid(_count, *_args):
    _x_start, _x_end, _y_start, _y_end = _args
    _total_pos_true = total_pos_true[_count: _count + (_y_end - _y_start) * (_x_end - _x_start)]
    _y_true = y_true[_count: _count + (_y_end - _y_start) * (_x_end - _x_start)]
    x_true_band, corner_true, indexs_true = train_and_test_data(input_normalize, band, _total_pos_true,
                                                                patch=args.patches, w=height, h=width, n_gcn=args.n_gcn)
    _x_true = torch.from_numpy(x_true_band).type(torch.FloatTensor)
    _y_true = torch.from_numpy(_y_true).type(torch.LongTensor)
    _Label_true = Data.TensorDataset(_x_true, _y_true)
    label_true_loader = Data.DataLoader(_Label_true, batch_size=args.batch_size, shuffle=False)
    ##########所有地物的光谱值及标签##########

    tar_v, pre_v = common_valid_epoch(model, label_true_loader, ce_dice_loss_main, permute=True)
    return pre_v


model.load_state_dict(torch.load('save_models/MMFINet/mmfinet.pth'))

model.eval()
all_prev = np.zeros([len(total_pos_true)])
total_height, total_wight = gt_hsi.shape
step = 50
x_start, x_end, y_start, y_end = 0, step, 0, step
count = 0
img_idx = 0
while x_start < total_height:
    print(f'patch: {img_idx}' + '*' * 20)
    if y_end >= total_wight + step:
        x_start += step
        x_end += step
        y_start = 0
        y_end = step
    if y_end > total_wight:
        y_end = total_wight
    if x_end > total_height:
        x_end = total_height
    all_prev[count:count + (y_end - y_start) * (x_end - x_start)] = valid(count, x_start, x_end, y_start, y_end)
    count += (y_end - y_start) * (x_end - x_start)
    y_start += step
    y_end += step
    img_idx += 1

get_cls_map(total_pos_true, all_prev, 'mmfinet')
