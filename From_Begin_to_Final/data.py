import numpy as np
import torch
from torch.utils.data import Dataset
import os, cv2


class MyOwndataset(Dataset):  # 重写了一个类，继承Dataset这个类
    def __init__(self, root, is_train=True):  # 初始化，root变量含义为数据集所存放的位置，is_train变量是为了判断当前加载的是训练集还是测试集
        # 导入数据，如果是图片，我们一般保存图片的路径，避免内存加载太大的数据
        self.dataset = []  # 定义一个列表，用来存放一系列数据
        dir = 'train' if is_train else 'test'  # 如果is_train为true，则加载训练集，如果为false，则加载测试集
        sub_dir = os.path.join(root, dir)  # 将数据集所在的路径与train/test文件夹进行拼接，若is_train=False则拼接的为测试集所在文件夹路径
        image_list = os.listdir(sub_dir)  # 获取当前文件夹中所有图像的文件名，并保存为列表
        for i in image_list:  # 遍历该列表，并且对路径进行拼接
            image_dir = os.path.join(sub_dir, i)  # 拼接操作
            self.dataset.append(image_dir)  # 将拼接完成后的文件名保存在列表之中

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):  # 当数据被调用就会触发该函数
        data = self.dataset[index]  # 返回的是图片的地址
        img = cv2.imread(data) / 255  # opencv读到的图片类型为 H高 W宽 C通道数 除以255是为了归一化
        # print(img.shape)
        # 根据模型输入数据类型，转换图像的数据格式
        """new_img=np.transpose(img,(2,0,1))"""  # 使用numpy换轴，可直接操作
        new_img = torch.tensor(img).permute(2, 0, 1)  # 使用torch换轴，需要先将ndarray转化为tensor
        # print(new_img.shape)
        """label=data.split(.)"""  # 标签值或者其他值可以使用从文件名中读取得到，文件名示例：‘1.0.0.0.0.0.0.jpg’
        data_list = data.split('.')  # 以点分隔得到初步的label结果，并存入列表
        # print(data_list)
        label = int(data_list[1])  # 获取label值,是否包含
        position = data_list[2:6]  # 从2到5，虽然是写的6，但是实际不包含6
        position = [int(i) / 300 for i in position]  # 对坐标归一化一下，假设图片大小为300
        sort = int(data_list[6])-1  # 类别，属于哪一类物品

        return np.float32(new_img), np.float32(label), np.float32(position), np.int(sort)
