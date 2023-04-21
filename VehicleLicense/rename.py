##################################################################################
# 数据清理
# 作者: Xinyu Ou (http://ouxinyu.cn)
# 数据集名称：车牌识别数据集
# 数据集简介: VehicleLicense车牌识别数据集包含16151张单字符数据，所有的单字符均为严格切割且都转换为黑白二值图像。
# 本程序功能:
# 1. 对样本文件进行改名，屏蔽特殊命名符号对训练的影响
###################################################################################

import os
dataset_root_path = 'VehicleLicense'

data_path = os.path.join(dataset_root_path, 'Data')
character_folders = os.listdir(data_path)

num_image = 0 
for character_folder in character_folders:
    character_imgs = os.listdir(os.path.join(data_path, character_folder))
    
    id = 0
    for character_img in character_imgs:
        newname = character_folder + '_' + str(id).rjust(4,'0') + os.path.splitext(character_img)[1]
        os.rename(os.path.join(data_path, character_folder, character_img), os.path.join(data_path, character_folder, newname))
        id += 1
        num_image += 1

    print('\r 已完成{}副图片的改名'.format(num_image), end='')
        
print('，已完成。')