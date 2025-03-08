import os
from glob import glob

def get_image_names(folder_path):
    # 使用glob列出文件夹中所有的图片文件
    image_files = glob(os.path.join(folder_path, '*.jpg')) + glob(os.path.join(folder_path, '*.png'))  # 你可以根据需要添加其他图片格式的扩展名

    # 提取文件名
    image_names = [os.path.basename(file) for file in image_files]

    return image_names

def save_to_txt(image_names, output_file):
    with open(output_file, 'w') as file:
        for name in image_names:
            file.write(name + '\n')

if __name__ == "__main__":
    folder_path = r"./LEVIR/A"  # 替换成你的图片文件夹的路径
    output_file = "train.txt"  # 输出的txt文件名

    image_names = get_image_names(folder_path)
    save_to_txt(image_names, output_file)

    print(f"Image names have been saved to {output_file}")
