import torch
import numpy as np
from skimage.segmentation import slic
from skimage import img_as_float
from scipy.spatial.distance import cdist

class SpatialSimilarity:
    def __init__(self, num_segments=32, compactness=10, sigma=5.0):
        """
        初始化超像素处理器。

        :param num_segments: 超像素的数量。
        :param compactness: 用于SLIC算法的紧凑度参数。
        :param sigma: 用于计算相似度的sigma参数。
        """
        self.num_segments = num_segments
        self.compactness = compactness
        self.sigma = sigma

    def process(self, batch_images):
        """
        处理图像批次，计算每个图像的超像素特征和空间相似度。

        :param batch_images: 批量图像，形状为 (B, 3, H, W)，类型为torch.Tensor
        :return: Tensor格式的segments, superpixel_features, spatial_similarity
        """
        batch_segments = []
        batch_superpixel_features = []
        batch_spatial_similarity = []

        if not isinstance(batch_images, torch.Tensor):
            raise TypeError("batch_images should be a torch.Tensor.")

        # 遍历批处理中的每个图像
        for image_tensor in batch_images:
            # 在转换之前分离Tensor
            # print("hdahdalkdhaslik:",image_tensor.shape)
            image = image_tensor.detach().permute(1, 2, 0).cpu().numpy()
            image = img_as_float(image)
            
            # 使用SLIC算法进行超像素分割
            segments = slic(image, n_segments=self.num_segments, compactness=self.compactness)

            # 计算每个超像素的平均颜色值
            superpixel_features = np.array([np.mean(image[segments == i], axis=0) for i in range(self.num_segments)])

            # 处理超像素特征中的nan值
            superpixel_features = np.nan_to_num(superpixel_features)

            # 计算超像素间的欧氏距离
            distances = cdist(superpixel_features, superpixel_features, 'euclidean')

            # 转换距离为相似度
            spatial_similarity = np.exp(-distances ** 2 / (2.0 * self.sigma ** 2))

            # 处理相似度矩阵中的nan值
            spatial_similarity = np.nan_to_num(spatial_similarity)

            # 将结果添加到对应的列表中
            batch_segments.append(torch.tensor(segments, dtype=torch.int))
            batch_superpixel_features.append(torch.tensor(superpixel_features, dtype=torch.float))
            batch_spatial_similarity.append(torch.tensor(spatial_similarity, dtype=torch.float))

        # 将列表转换为Tensor
        batch_segments_tensor = torch.stack(batch_segments)
        batch_superpixel_features_tensor = torch.stack(batch_superpixel_features)
        batch_spatial_similarity_tensor = torch.stack(batch_spatial_similarity)

        return batch_segments_tensor, batch_superpixel_features_tensor, batch_spatial_similarity_tensor
