from torch import nn
import torch


class MyNet(nn.Module):
    def __init__(self):  # 第一个方法，初始化
        super(MyNet, self).__init__()  # 继承父类的方法
        self.layers = nn.Sequential(

            nn.Conv2d(3, 11, 3),  # 输入通道为3，输出通道为11，卷积核为3*3
            nn.LeakyReLU(),  # 激活一下
            nn.MaxPool2d(3),  # 池化一下

            nn.Conv2d(11, 22, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(22, 32, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU()

        )
        self.label_layer = nn.Sequential(  # 对label的卷积
            nn.Conv2d(128, 1, 19),
            nn.ReLU(),
        )
        self.position_layer = nn.Sequential(  # 对position的卷积
            nn.Conv2d(128, 4, 19),
            nn.LeakyReLU(),
        )
        self.sort_layer = nn.Sequential(  # 对sort的卷积
            nn.Conv2d(128, 20, 19),
            nn.LeakyReLU(),
        )

    def forward(self, x):  # 前向计算
        out = self.layers(x)
        label = self.label_layer(out)
        label = torch.squeeze(label, dim=2)  # 对第二个位置进行降维，只输出一个值
        label = torch.squeeze(label, dim=2)
        label = torch.squeeze(label, dim=1)
        position = self.position_layer(out)
        position = torch.squeeze(position, dim=2)
        position = torch.squeeze(position, dim=2)
        sot = self.sort_layer(out)
        sot = torch.squeeze(sot, dim=2)
        sot = torch.squeeze(sot, dim=2)
        return label, position, sot


"""
if __name__=='__main__':
    net=MyNet().cuda()
    x=torch.rand(3,3,300,300).cuda() #参数含义，批次，通道，维度，维度
    print(net(x)[0].shape)
    print(net(x)[1].shape)
    print(net(x)[2].shape)
"""
