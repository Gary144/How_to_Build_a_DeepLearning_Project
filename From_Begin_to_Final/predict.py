import os

import torch

from net import MyNet
import cv2

if __name__ == '__main__':
    model = MyNet()
    model.load_state_dict(torch.load('param/2021-07-23-20_27_07_213811-11.pt'))
    root='data/test/test/'
    for i in os.listdir(root):
        img = cv2.imread(root+i)
        img_data = torch.tensor(img).permute(2, 0, 1)
        img_data = torch.unsqueeze(img_data, dim=0) / 255
        print(img_data.shape)
        rst = model(img_data)
        label = torch.sigmoid(rst[0])
        sort = torch.softmax(rst[2], dim=1)
        print(label)
        print(rst[1] * 300)
        print(torch.argmax(sort))
