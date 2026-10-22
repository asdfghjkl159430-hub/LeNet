import torch
from torch import nn #搭建网络的各个层
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__() #先初始化
        self.sig = nn.Sigmoid() #定义激活函数
#       然后开始定义要使用的网络
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2) #定义卷积层1，输入通道为1，输出通道为6，卷积核大小为5x5，padding为2
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2) #定义平均池化层1，窗口大小为2x2，步长为2
        self.c3 = nn.Conv2d(6,16,5) #定义卷积层3，输入通道为6，输出通道为16，卷积核大小为5x5
        self.s4 = nn.AvgPool2d(2,2) #定义平均池化层2，窗口大小为2x2，步长为2

        self.flatten = nn.Flatten() #定义展平层
        self.f5 = nn.Linear(400,120) #定义全连接层5，输入大小为400，输出大小为120
        self.f6 = nn.Linear(120,84) #定义全连接层6，输入大小为120，输出大小为84
        self.f7 = nn.Linear(84,10) #定义全连接层7，输入大小为84，输出大小为10

    def forward(self,x):
        x = self.sig(self.c1(x))
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.sig(self.f5(x))
        x = self.sig(self.f6(x))
        x = self.f7(x)
        return x

if __name__ =="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    summary(model, (1, 28, 28))
