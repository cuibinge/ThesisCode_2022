import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
import imageio
import glob
import os
import cv2
# from torchvision import transforms
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, images_path, labels_path, Transform=None):
        """"""
        # 在这里写，获得所有image路径，所有label路径的代码，并将路径放在分别放在images_path_list和labels_path_list中
        """"""
        self.images_path_list = sorted(glob.glob(os.path.join(images_path, '*.png')))
        self.labels_path_list = sorted(glob.glob(os.path.join(labels_path, '*.png')))
        self.transform = ToTensor()

    def __getitem__(self, index):

        image_path = self.images_path_list[index]
        label_path = self.labels_path_list[index]
        image = imageio.imread(image_path)
        label = imageio.imread(label_path)
        # print("111:",label.shape)
#         B1, B2, B3 = cv2.split(image)

#         B1_normalization = ((B1 - np.min(B1)) / (np.max(B1) - np.min(B1)) * 1).astype('float32')
#         B2_normalization = ((B2 - np.min(B2)) / (np.max(B2) - np.min(B2)) * 1).astype('float32')
#         B3_normalization = ((B3 - np.min(B3)) / (np.max(B3) - np.min(B3)) * 1).astype('float32')


        # 原始输入
        # image = cv2.merge([B1_normalization, B2_normalization, B3_normalization])
#         image = image
        # image = np.expand_dims(image, axis=2)
        image = np.array(image)
        label = np.array(label) / 255
        image = image.astype(float)
        label = label.astype(float)

        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.from_numpy(label)
        label = torch.squeeze(label, 0)

        return image.float(), label

    def __len__(self):
        return len(self.images_path_list)


def main():
    imagePath = r"/chenruipeng/RedTideDetection/data/img"  # 图像路径
    labelPath = r"/chenruipeng/RedTideDetection/data/label"  # 真值路径
    mydataset = MyDataset(imagePath, labelPath)

    Data = DataLoader(mydataset, batch_size=1, shuffle=False, pin_memory=True)
    for i, data in enumerate(Data):
        if data is not None:
            img, lab = data
            print(img.shape)
            print(lab.shape)


if __name__ == '__main__':
    main()
