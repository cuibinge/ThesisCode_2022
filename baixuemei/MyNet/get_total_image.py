import cv2
import numpy as np


def get_total_image(_i):
    total_image = np.zeros([1024, 1024, 3])
    x_start, x_end, y_start, y_end = 0, 256, 0, 256
    for i in range(1, 17):
        file_path = f'test_images/{i}.png'
        img = cv2.imread(file_path)
        if y_end > 1024:
            x_start += 256
            x_end += 256
            y_start = 0
            y_end = 256
        total_image[x_start: x_end, y_start: y_end, :] = img
        y_start += 256
        y_end += 256

    cv2.imwrite(f'total_images/{_i}.png', total_image)
