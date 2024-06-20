import cv2
import numpy as np
import os
import argparse


def fsobel_gradient(img):
    # 计算x方向和y方向上的梯度
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # 计算梯度大小和方向
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2) / 4  # 为了在数值上和ffour保持一致
    # gradient_direction = np.arctan2(sobely, sobelx)
    # 计算平均梯度
    average_gradient = np.mean(gradient_magnitude)
    return average_gradient


def ffour_neighbor_gradient(img):
    # 使用像素周围上下左右四个像素的差异来估计梯度
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
    gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
    average_gradient = np.mean(gradient_magnitude)
    return average_gradient


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    opt = parser.parse_args()

    # 图像文件夹路径
    img_folder = opt.path
    # 初始化总平均梯度
    total_sobel_gradient = 0
    total_four_neighbor_gradient = 0
    # 遍历文件夹中所有图片
    for filename in os.listdir(img_folder):
        if filename.endswith('.JPG'):  # 假设只有jpg格式的图片
            img_path = os.path.join(img_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 4:
                alpha_channel = img[:, :, 3]
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                gray_img[alpha_channel == 0] = 255  # 0: black  255: white
            else:
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # 计算平均梯度
            sobel_gradient = fsobel_gradient(gray_img)
            four_neighbor_gradient = ffour_neighbor_gradient(gray_img)
            # 更新总平均梯度
            total_sobel_gradient += sobel_gradient
            total_four_neighbor_gradient += four_neighbor_gradient

    # 计算所有图片的平均梯度
    num_images = len(os.listdir(img_folder))
    average_sobel_gradient = total_sobel_gradient / num_images
    average_four_neighbor_gradient = total_four_neighbor_gradient / num_images

    # 输出结果
    print("Average gradient over all pixels and all images (Sobel operator):", average_sobel_gradient)
    print("Average gradient over all pixels and all images (four neighbor pixels):", average_four_neighbor_gradient)

