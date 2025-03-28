# # 划分出一批验证集
# import os
# import random
# import shutil
#
# def split_dataset(dataset_dir, output_dir, validation_ratio):
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 获取文件夹列表
#     folder_list = os.listdir(dataset_dir)
#
#     for folder in folder_list:
#         folder_path = os.path.join(dataset_dir, folder)
#         if not os.path.isdir(folder_path):
#             continue
#
#         # 创建验证集目录
#         validation_folder = os.path.join(output_dir, folder)
#         os.makedirs(validation_folder, exist_ok=True)
#
#         # 获取文件列表
#         file_list = os.listdir(folder_path)
#
#         # 计算验证集的数量
#         num_validation = int(len(file_list) * validation_ratio)
#
#         # 随机选择验证集文件
#         validation_files = random.sample(file_list, num_validation)
#
#         for file in validation_files:
#             file_path = os.path.join(folder_path, file)
#             validation_file_path = os.path.join(validation_folder, file)
#
#             # 移动文件到验证集目录
#             shutil.move(file_path, validation_file_path)
#
# # 示例用法
# dataset_dir = r'D:\Project of lab\Project\变化检测\代码\VPGCDNet-master\LEVIR-CD\train'
# output_dir = r'D:\Project of lab\Project\变化检测\代码\VPGCDNet-master\LEVIR-CD\val\val'
# validation_ratio = 0.2  # 验证集比例为 20%
#
# split_dataset(dataset_dir, output_dir, validation_ratio)
import os
import random

def split_files(folder_path, output_dir, train_ratio, validation_ratio, test_ratio):
    # 获取文件列表
    file_list = os.listdir(folder_path)

    # 计算文件数量
    num_files = len(file_list)

    # 计算划分后的文件数量
    num_train = int(num_files * train_ratio)
    num_validation = int(num_files * validation_ratio)
    num_test = num_files - num_train - num_validation

    # 随机打乱文件列表
    random.shuffle(file_list)

    # 划分文件名列表
    train_files = file_list[:num_train]
    validation_files = file_list[num_train:num_train+num_validation]
    test_files = file_list[num_train+num_validation:]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存文件名列表到文本文件
    save_file_list(train_files, os.path.join(output_dir, 'train.txt'))
    save_file_list(validation_files, os.path.join(output_dir, 'val.txt'))
    save_file_list(test_files, os.path.join(output_dir, 'test.txt'))

def save_file_list(file_list, output_file):
    with open(output_file, 'w') as f:
        for file_name in file_list:
            f.write(file_name + '\n')

# 示例用法
folder_path = r'./LEVIR/A'  # 文件夹路径
output_dir = r'./LEVIR'  # 输出目录路径
train_ratio = 0.7  # 训练集比例
validation_ratio = 0.2  # 验证集比例
test_ratio = 0.1  # 测试集比例

split_files(folder_path, output_dir, train_ratio, validation_ratio, test_ratio)