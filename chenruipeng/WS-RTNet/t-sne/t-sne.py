import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets, transforms

from sklearn import preprocessing
from sklearn.manifold import TSNE

from net import Model


class Config:
    image_resize = 256
    image_crop = 224
    batch_size = 32                 # 64 for MIT-Indoor, 128 for others
    backbone = "resnet50"           # vgg19 resnet18 resnet50 resnet101 densenet161
    dataset_name = "dtd_t-SNE"     #  "FMD" "dtd-r1.0.1" "4D_Light" "MIT-Indoor"
    data_dir = Path(f"data/{dataset_name}/splits/split_1")
    output_dir = Path(f"outputs/confusion_matrix")
    checkpoint = "outputs/resnet50_dtd-tsne_epoch42_0.8608.pth.tar"

DataTransforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(Config.image_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    'test' : transforms.Compose([
        transforms.Resize(Config.image_resize),
        transforms.CenterCrop(Config.image_crop),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# function for training model
def inference(model, dataloaders):
    since = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    features_list = []
    predict_list = []
    lable_list = []

    # for phase in ['train', 'test']:
    t1 = time.time()
    for phase in ['test']:
        model.eval()   # Set model to evaluate mode

        # Iterate over data.
        running_corrects = 0
        i = 0
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            features, outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data)
            predict_list.extend(list(preds.cpu().numpy()))
            lable_list.extend(list(labels.data.cpu().numpy()))
            features_list.append(features)

        accuracy = running_corrects.double() / len(dataloaders[phase].dataset)
        print('{} : Acc = {:.4f}'.format(phase, accuracy))

    t2 = time.time()
    print('-' * 35)
    print(f'Start Time : {since}')
    print(f'End Time : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Inference Time : {(t2-t1)/60:.4f} minute  ({(t2-t1):.4f}s)')
    print('=' * 39, '\n')

    return accuracy, features_list, predict_list, lable_list


def t_SNE(X, y, class_names):
    # t-SNE降维处理
    tsne = TSNE(n_components=2,
                verbose=1,
                n_iter=5000,
                random_state=0)
    # tsne = TSNE(
    #     n_components=2,
    #     init="pca",
    #     learning_rate="auto",
    #     n_iter=500,
    #     n_iter_without_progress=150,
    #     n_jobs=2,
    #     random_state=0,
    # )
    result = tsne.fit_transform(X)

    # 归一化处理
    scaler = preprocessing.MinMaxScaler(feature_range=(-100,100))
    scaler = preprocessing.MinMaxScaler()
    result = scaler.fit_transform(result)

    # 颜色设置
    color = ['#DC143C', '#FF0000', '#FFA07A', '#FFB6C1', '#FF69B4',
             '#FF1493', '#FF7F50', '#FFD700', '#FFFF00', '#FFDAB9',
             '#D2691E', '#B8860B', '#800000', '#9400D3', '#836FFF',
             '#FF00FF', '#DDA0DD', '#808000', '#A2CD5A', '#00FF00',
             '#228B22', '#20B2AA', '#00FFFF', '#00BFFF', '#1E90FF',
             '#0000CD', '#000000', '#808080', '#778899', '#DCDCDC']

    # 可视化展示, 加图例
    # plt.figure(figsize=(30, 30))
    plt.title('t-SNE process')
    legend_dict = {}
    for i in range(len(result)):
        sh = plt.scatter(result[i,0], result[i,1], c=color[y[i]], s=50)
        if class_names[y[i]] not in legend_dict.keys():
            legend_dict[class_names[y[i]]] = sh
    legend_dict = dict(sorted(legend_dict.items(), key = lambda x:x[0]))
    plt.legend(legend_dict.values(), legend_dict.keys(), loc=(1.04901,0.1))
    plt.show()



# prepare dataset
image_datasets = {x: datasets.ImageFolder(Config.data_dir / x,
                    transform=DataTransforms[x]) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                  batch_size=Config.batch_size, shuffle=True, num_workers=4)
                  for x in ['train', 'test']}
class_names = image_datasets['train'].classes

# prepare net and load checkpoint
net = Model(Config.backbone, len(class_names))
net.load_state_dict(torch.load(Config.checkpoint)["state_dict"])

# assign device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = net.to(device)

accuracy, features_list, predict_list, lable_list = inference(model_ft, dataloaders)
predicts = np.array(predict_list)
lables = np.array(lable_list)
features = np.concatenate(features_list)
print(type(predicts), predicts.shape)
print(type(lables), lables.shape)
print(type(features), features.shape)

t_SNE(features, lables, class_names)
