import datetime
import os.path

import torch

from net import MyNet
from data import MyOwndataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim

DEVICE = 'cuda'


class Train:
    def __init__(self, root,weight_path):  # root为数据集存放的文件夹,weight_path为预训练权重存放的文件夹
        self.summaryWriter = SummaryWriter('log')  # 对可视化进行一个初始化并存放在log里面
        # 创建数据加载器
        self.train_dataset = MyOwndataset(root=root, is_train=True)  # 加载训练数据
        self.test_dataset = MyOwndataset(root=root, is_train=False)  # 加载测试数据
        self.train_dataLoader = DataLoader(self.train_dataset, batch_size=50,
                                           shuffle=True)  # 加载训练dataLoader，batch_size为批次数，shuffle为打乱
        self.test_dataLoader = DataLoader(self.test_dataset, batch_size=50, shuffle=True)
        # 创建模型并导入cuda
        self.net = MyNet().to(DEVICE)
        if os.path.exists(weight_path):
            torch.net.load_state_dict(torch.load(weight_path))
        # 创建优化器
        self.opt = optim.Adam(self.net.parameters())
        # 构建损失函数
        # 由于目标检测问题涉及三个损失函数，所以需要构建三个损失函数
        self.label_loss_fn = nn.BCEWithLogitsLoss()
        self.position_loss_fn = nn.MSELoss()
        self.sot_loss_fn = nn.CrossEntropyLoss()

        self.train = True
        self.test = True

    def __call__(self):
        index1,index2=0,0
        # 训练步骤，外加模型保存和损失显示
        for epoch in range(1000):
            if self.train:
              # 设置一个迭代器
                for i, (img, label, position, sort) in enumerate(self.train_dataLoader):
                    self.net.train()
                # 将数据全部放到cuda上
                    img, label, position, sort = img.to(DEVICE), label.to(DEVICE), position.to(DEVICE), sort.to(DEVICE)
                # 将图片传入网络得到预测值
                    out_label, out_position, out_sort = self.net(img)
                # print(img.shape)
                # print(label.shape)
                # 计算损失
                    label_loss = self.label_loss_fn(out_label, label)
                    position_loss = self.position_loss_fn(out_position, position)
                    sort = sort[torch.where(sort >= 0)]
                    out_sort = out_sort[torch.where(sort >= 0)]
                    sort_loss = self.sot_loss_fn(out_sort, sort)

                    train_loss = label_loss + position_loss + sort_loss
                # 更新参数
                    self.opt.zero_grad()  # 梯度置零
                    train_loss.backward()  # 误差传播
                    self.opt.step()  # 修改参数

                    if i%10==0:
                        print(f'train_loss {i}=====>',train_loss.item())
                        self.summaryWriter.add_scalar('train_loss',train_loss,index1)
                        index1+=1
            data_time=str(datetime.now()).replace(':','-').replace('.','_').replace(':','_')
            torch.save(self.net.state_dict(),f'param/{data_time}-{epoch}.pt')

        #测试过程，大部分代码与训练相同，但是不需要保存模型和更新参数
        if self.test:
            sum_sort_acc,avg_sort_acc=0,0
            for i, (img, label, position, sort) in enumerate(self.train_dataLoader):
                self.net.train()
                # 将数据全部放到cuda上
                img, label, position, sort = img.to(DEVICE), label.to(DEVICE), position.to(DEVICE), sort.to(DEVICE)
                # 将图片传入网络得到预测值
                out_label, out_position, out_sort = self.net(img)
                # print(img.shape)
                # print(label.shape)
                # 计算损失
                label_loss = self.label_loss_fn(out_label, label)
                position_loss = self.position_loss_fn(out_position, position)
                sort = sort[torch.where(sort >= 0)]
                out_sort = out_sort[torch.where(sort >= 0)]
                sort_loss = self.sot_loss_fn(out_sort, sort)

                test_loss = label_loss + position_loss + sort_loss
                #先将label转化为0和1的值，0.5为阈值
                out_label=torch.tensor(torch.sigmoid(out_label))
                out_label=out_label[torch.where(out_label>=0.5)]=1
                out_label = out_label[torch.where(out_label < 0.5)] = 0

                out_sort=torch.argmax(torch.softmax(out_sort,dim=1))
                #求解准确度
                label_acc=torch.mean(torch.eq(out_label.float(),label.float()))
                sort_acc=torch.mean(torch.eq(out_sort,sort).float())
                #如下准确度label，position同理
                sum_sort_acc+=sort_acc
                if i % 10 == 0:
                    print(f'test_loss {i}=====>', test_loss.item())
                    self.summaryWriter.add_scalar('test_loss', test_loss, index2)
                    index2 += 1
            avg_sort_acc=sum_sort_acc/i
            print(f'avg_sort_acc{epoch}==>',avg_sort_acc)
            self.summaryWriter.add_scalar('avg_sort_acc', avg_sort_acc,epoch)

if __name__ == '__main__':
    train=Train('数据集路径','预训练模型路径')
    train()