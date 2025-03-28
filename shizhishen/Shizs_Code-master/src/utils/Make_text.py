import os

# 图片所在文件夹路径
image_folder = r"LEVIR\A"
# 要创建的 txt 文件所在文件夹路径
txt_folder = r"LEVIR\txt"

# 获取图片文件夹中的所有文件名
image_files = os.listdir(image_folder)

for image_file in image_files:
    image_file_path = os.path.join(image_folder, image_file)
    if os.path.isfile(image_file_path):
        base_name, extension = os.path.splitext(image_file)
        # 确保是图片文件（可以根据实际需求修改判断条件）
        if extension.lower() in [".jpg", ".jpeg", ".png", ".gif"]:
            txt_file_path = os.path.join(txt_folder, base_name + "_A.txt")
            with open(txt_file_path, "w") as txt_file:
                pass  # 这里可以根据需要添加一些额外的操作或内容到 txt 文件中

print("文件创建完成")