##################################################################################
# 数据集预处理
# 作者: Xinyu Ou (http://ouxinyu.cn)
# 数据集名称：车牌识别数据集
# 数据集简介: VehicleLicense车牌识别数据集包含16151张单字符数据，所有的单字符均为严格切割且都转换为黑白二值图像。
# 本程序功能:
# 1. 将数据集按照7:1:2的比例划分为训练验证集、训练集、验证集、测试集
# 2. 代码将生成4个文件：训练验证集trainval.txt, 训练集列表train.txt, 验证集列表val.txt, 测试集列表test.txt, 数据集信息dataset_info.json
# 3. 图像列表已生成, 其中训练验证集样本13192，训练集样本11508个, 验证集样本1684个, 测试集样本3352个, 共计16544个。
# 4. 生成数据集标签词典时，需要根据标签-文件夹列表匹配标签列表
###################################################################################

import os
import json
import codecs

num_trainval = 0
num_train = 0
num_val = 0
num_test = 0
class_dim = 0
dataset_info = {
    'dataset_name': '',
    'num_trainval': -1,
    'num_train': -1,
    'num_val': -1,
    'num_test': -1,
    'class_dim': -1,
    'label_dict': {}
}
# 本地运行时，需要修改数据集的名称和绝对路径，注意和文件夹名称一致
dataset_name = 'VehicleLicense'
dataset_path = 'D:\\Workspace\\ExpDatasets\\'
dataset_root_path = os.path.join(dataset_path, dataset_name)
excluded_folder = ['.DS_Store', '.ipynb_checkpoints']      # 被排除的文件夹

# 获取标签和文件夹的对应关系，即省市拼音和中文对照关系
json_label_match = os.path.join(dataset_root_path, 'label_match.json')
label_match = json.loads(open(json_label_match, 'r', encoding='utf-8').read()) 
    
# 定义生成文件的路径
data_path = os.path.join(dataset_root_path, 'Data')
trainval_list = os.path.join(dataset_root_path, 'trainval.txt')
train_list = os.path.join(dataset_root_path, 'train.txt')
val_list = os.path.join(dataset_root_path, 'val.txt')
test_list = os.path.join(dataset_root_path, 'test.txt')
dataset_info_list = os.path.join(dataset_root_path, 'dataset_info.json')

# 检测数据集列表是否存在，如果存在则先删除。其中测试集列表是一次写入，因此可以通过'w'参数进行覆盖写入，而不用进行手动删除。
if os.path.exists(trainval_list):
    os.remove(trainval_list)
if os.path.exists(train_list):
    os.remove(train_list)
if os.path.exists(val_list):
    os.remove(val_list)
if os.path.exists(test_list):
    os.remove(test_list)

# 按照比例进行数据分割
class_name_list = os.listdir(data_path)
with codecs.open(trainval_list, 'a', 'utf-8') as f_trainval:
    with codecs.open(train_list, 'a', 'utf-8') as f_train:
        with codecs.open(val_list, 'a', 'utf-8') as f_val:
            with codecs.open(test_list, 'a', 'utf-8') as f_test:
                for class_name in class_name_list:
                    if class_name not in excluded_folder:
                        dataset_info['label_dict'][class_name] = label_match[class_name]        # 按照文件夹名称和label_match进行标签匹配
                        images = os.listdir(os.path.join(data_path, class_name))
                        count = 0
                        for image in images:
                            if count % 10 == 0:  # 抽取大约10%的样本作为验证数据
                                f_val.write("{0}\t{1}\n".format(os.path.join(data_path, class_name, image), class_dim))
                                f_trainval.write("{0}\t{1}\n".format(os.path.join(data_path, class_name, image), class_dim))
                                num_val += 1
                                num_trainval += 1
                            elif count % 10 == 1 or count % 10 == 2:  # 抽取大约20%的样本作为测试数据
                                f_test.write("{0}\t{1}\n".format(os.path.join(data_path, class_name, image), class_dim))
                                num_test += 1
                            else:
                                f_train.write("{0}\t{1}\n".format(os.path.join(data_path, class_name, image), class_dim))
                                f_trainval.write("{0}\t{1}\n".format(os.path.join(data_path, class_name, image), class_dim))
                                num_train += 1
                                num_trainval += 1
                            count += 1
                    class_dim += 1        

# 将数据集信息保存到json文件中供训练时使用
dataset_info['dataset_name'] = dataset_name
dataset_info['num_trainval'] = num_trainval
dataset_info['num_train'] = num_train
dataset_info['num_val'] = num_val
dataset_info['num_test'] = num_test
dataset_info['class_dim'] = class_dim

with codecs.open(dataset_info_list, 'w', encoding='utf-8') as f_dataset_info:
    json.dump(dataset_info, f_dataset_info, ensure_ascii=False, indent=4, separators=(',', ':')) # 格式化字典格式的参数列表

print("图像列表已生成, 其中训练验证集样本{}，训练集样本{}个, 验证集样本{}个, 测试集样本{}个, 共计{}个。".format(num_trainval, num_train, num_val, num_test, num_train+num_val+num_test))