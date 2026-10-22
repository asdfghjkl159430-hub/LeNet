import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet

#还是先处理测试数据
def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=28),
                                                          transforms.ToTensor()]),
                             download=True)
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0)
    return test_dataloader

#模型测试函数
def test_model_process(model,test_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # 初始化测试函数的记录
    # 用来累计预测正确的图片数量
    test_corrects = 0.0
    # 用来累计已经测试的样本数量
    test_num = 0

    model.eval()
    # torch.no_grad()关闭自动梯度记录，因为测试时不需要反向传播和参数更新，可以节省内存
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_lab == test_data_y)
            test_num += test_data_y.size(0)
    test_acc = test_corrects / test_num
    print("测试的准确率为",test_acc)

if __name__ == "__main__":
    model = LeNet()
    model.load_state_dict(torch.load('best_model.pth'))
    test_dataloader = test_data_process()
    test_model_process(model,test_dataloader)


