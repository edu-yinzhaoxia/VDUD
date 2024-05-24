import random
import datetime
import numpy as np
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn

from Model.Cifar10_model import *
from Model.Mnist_model import *


# 定义一个可以设置随机种子的函数
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(124)

# 设置全局超参数
num_epochs = 10
batch_size = 128
lr = 0.01
num_workers = 4


def get_network(model, model_name,
                train_data, test_data, data_name,
                device, optimizer, criterion):
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers)
    # 训练模型
    model.to(device)
    model.train()
    total_step = len(train_data) // batch_size
    for epoch in range(num_epochs):
        aver_loss = 0.0
        for i, data in enumerate(train_dataloader):
            x, labels = data
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            y = model(x)
            loss = criterion(y, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Average_Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, aver_loss / 10))
                aver_loss = loss.item()
            else:
                aver_loss += loss.item()

        # 模型评估
        model.eval()
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        print('Accuracy of the network on the {} test images: {}%'
              .format(len(test_data), 100 * correct / total))

    # 保存模型参数
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    time = datetime.datetime.now().strftime("%H-%M-%S")
    torch.save(model.state_dict(), data_name + '_' + model_name + '_' + str(date) + '_' + str(time) + '.pth')


def cifar10_cnn(cifar10_model, model_name):
    model = cifar10_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    trans = transforms.Compose([
        transforms.Resize(224),  # 将图像大小调整为224x224
        transforms.RandomHorizontalFlip(),        # 图像一半概率翻转，一半概率不翻转
        transforms.RandomRotation(10),              # 随机旋转图像，最大旋转角度为10度
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移图像，最大平移比例为0.1
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))      # RGB每层归一化
    ])
    train_data = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, transform=trans, download=True)
    test_data = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, transform=trans, download=True)

    get_network(model=model, model_name=model_name,
                train_data=train_data, test_data=test_data, data_name='cifar10',
                device=device, optimizer=optimizer, criterion=criterion)


def mnist_cnn(mnist_model, model_name):
    model = mnist_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    trans = transforms.Compose([
        transforms.Resize(224),  # 将图像大小调整为224x224
        transforms.RandomRotation(10),              # 随机旋转图像，最大旋转角度为10度
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移图像，最大平移比例为0.1
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.MNIST(root='./data/mnist', train=True, transform=trans, download=True)
    test_data = torchvision.datasets.MNIST(root='./data/mnist', train=False, transform=trans, download=True)

    get_network(model=model, model_name=model_name,
                train_data=train_data, test_data=test_data, data_name='mnist',
                device=device, optimizer=optimizer, criterion=criterion)


def mnist_def():
    model = CNN_MNIST()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)

    trans = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转图像，最大旋转角度为10度
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移图像，最大平移比例为0.1
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.MNIST(root='.././data/mnist', train=True, transform=trans, download=True)
    test_data = torchvision.datasets.MNIST(root='.././data/mnist', train=False, transform=trans, download=True)

    get_network(model=model, model_name='cnn_def',
                train_data=train_data, test_data=test_data, data_name='mnist',
                device=device, optimizer=optimizer, criterion=criterion)


def cifar10_def():
    # 训练cifar10数据集
    model = CNN_CIFAR()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 图像一半概率翻转，一半概率不翻转
        transforms.RandomRotation(10),  # 随机旋转图像，最大旋转角度为10度
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移图像，最大平移比例为0.1
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB每层归一化
    ])
    train_data = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, transform=trans, download=True)
    test_data = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, transform=trans, download=True)

    get_network(model=model, model_name='cnn_def',
                train_data=train_data, test_data=test_data, data_name='cifar10',
                device=device, optimizer=optimizer, criterion=criterion)


if __name__ == "__main__":
    # mnist_cnn(VGG16_MNIST, 'vgg16')

    # mnist_cnn(Resnet18_MNIST, 'resnet18')
    # cifar10_def()
    mnist_def()

