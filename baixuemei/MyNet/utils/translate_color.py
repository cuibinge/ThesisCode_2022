import numpy as np
from PIL import Image

color_map = {
    0: np.array([99, 255, 247]) / 255.,  # 浅蓝色，池塘pond
    1: np.array([170, 255, 0]) / 255.,  # 浅绿，芦苇reed
    2: np.array([36, 125, 20]) / 255.,  # 芦苇柽柳混生reed-tamarix mixed
    3: np.array([230, 152, 0]) / 255.,  # 碱蓬salsa
    4: np.array([0, 133, 250]) / 255.,  # 海洋sea
    5: np.array([212, 0, 0]) / 255.,  # 红色，互花米草spartina alterniflora
    6: np.array([194, 99, 31]) / 255.,  # 深棕色，柽柳tamarix
    7: np.array([245, 202, 122]) / 255.,  # 潮滩tidal flat
    8: np.array([115, 178, 255]) / 255.,  # 浅蓝色，黄河yellow river
    9: np.array([100, 0, 0]) / 255.,
    10: np.array([0, 100, 0]) / 255.,
    11: np.array([0, 0, 100]) / 255.,
    12: np.array([0, 100, 100]) / 255.,
    13: np.array([100, 0, 100]) / 255.,
    14: np.array([100, 100, 0]) / 255.,
    15: np.array([0, 100, 200]) / 255.,
    16: np.array([100, 0, 200]) / 255.,
    17: np.array([100, 200, 0]) / 255.,
}

translate_color_map = {
    (36, 125, 20): np.array([170, 255, 0]),  # 深绿色-浅绿色
    (170, 255, 0): np.array([115, 178, 255]),  # 浅绿色-浅蓝色
    (99, 255, 247): np.array([212, 0, 0]),  # 浅蓝色-红色
    (212, 0, 0): np.array([245, 202, 122]),  # 红色-肉色
    (245, 202, 122): np.array([194, 99, 31]),  # 肉色-深棕色
    (194, 99, 31): np.array([100, 200, 0]),  # 深棕色-中绿色
    (0, 133, 250): np.array([255, 255, 255]),  # 深蓝色-白色
    (230, 152, 0): np.array([230, 152, 0])  # 棕色-棕色
}

old_img = np.array(Image.open(r'D:\code\python\xuemei\GTFN\test_images\666.png'))
print(old_img.shape)
new_img = np.zeros([740, 770, 3])
for i in range(old_img.shape[0]):
    for j in range(old_img.shape[1]):
        p = tuple(old_img[i, j, :3])
        new_img[i, j, :3] = translate_color_map[p]
        # print(new_img[i, j, :3])
Image.fromarray(np.uint8(new_img)).save('tmp.png')
