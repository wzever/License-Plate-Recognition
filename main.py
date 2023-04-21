import cv2
import os
from detect import *
from recognize import recognize
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

levels = ['easy', 'medium', 'difficult']
colors = {'green': '（绿牌）', 'blue': '（蓝牌）'}
test_dir = 'resources/images/'

def standard_test():
    plt.figure(figsize=(15, 10))
    plt.suptitle('测试车牌识别结果', fontsize=28)
    idx = 1
    for i, level in enumerate(levels):
        for j in range(3):
            img_path = test_dir + level + f'/{i+1}-{j+1}.jpg'
            ori_img = cv2.imread(img_path)
            imgs, color, _ = detectAndSegment(ori_img, i)
            ret = recognize(imgs, color)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            plt.subplot(3, 3, idx)
            plt.title(colors[color] + ret, pad = 3)
            plt.imshow(ori_img)
            idx += 1
    plt.show()

def standard_test_2():
    for i, level in enumerate(levels):
        for j in range(3):
            fig = plt.figure(figsize=(10, 4))
            img_path = test_dir + level + f'/{i+1}-{j+1}.jpg'
            ori_img = cv2.imread(img_path)
            imgs, color, img = detectAndSegment(ori_img, i)
            ret = recognize(imgs, color)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            plt.suptitle('识别结果：' + ret + colors[color], fontsize=18)
            plt.subplot(1, 2, 1)
            plt.title('输入图像')
            plt.imshow(ori_img)
            plt.subplot(1, 2, 2)
            plt.title('车牌区域')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.show()

def extra_test():
    for img_path in os.listdir('test'):
        img_path = 'test/' + img_path
        try:
            fig = plt.figure(figsize=(10, 4))
            ori_img = cv2.imread(img_path)
            imgs, color, img = detectAndSegment(ori_img, 1)
            ret = recognize(imgs, color)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            plt.suptitle('识别结果：' + ret + colors[color], fontsize=18)
            plt.subplot(1, 2, 1)
            plt.title('输入图像')
            plt.imshow(ori_img)
            plt.subplot(1, 2, 2)
            plt.title('车牌区域')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.show()
        except:
            print('Sorry, but there\'s something wrong.')

if __name__ == '__main__':
    standard_test()
    extra_test()