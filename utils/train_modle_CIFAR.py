import random
import numpy as np
import torch
import torchvision.datasets
from torch.utils import data
from Model.Mnist_model import *
from Model.classifer import Classifer_cifar
from Model.reconstruction.comdefend_model import Decoder_Minist, Encoder_Minist, Encoder_CIFAR, Decoder_CIFAR
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
# 定义一个可以设置随机种子的函数

from Model.reconstruction.unet import CIFAR_Net, MNIST_Net
from utils.eval import predict


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def loss_fn(y, y_hat, mean, logvar, kl_weight):
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
    loss = recons_loss + kl_loss * kl_weight
    return loss

def train_epoch_den(epoch, unet, device, CIFAR_train_dataloader,optimizer,simple_classifer,criterion):
    unet.train()
    simple_classifer.train()
    loss = 0.0
    for i, images in enumerate(CIFAR_train_dataloader):
        # 前向传播
        images, targets = images
        images = images.to(device)
        targets = targets.to(device)
        # gauss_noise = torch.randn(images.size()) * 0.1
        # inputs = images + gauss_noise.to(device)
        outputs, latent, mean, log_var = unet(images)
        y_hat = simple_classifer(latent)
        loss1 = criterion(y_hat, targets)
        # 计算损失函数
        loss2 = loss_fn(images, outputs, mean, log_var, 0.00025)
        loss = loss1 + loss2
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录训练过程
        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Train_Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    return loss.item()


def test_epoch_den(epoch, unet, device, CIFAR_test_dataloader,classifer,simple_classifer):
    unet.eval()
    classifer.eval()
    simple_classifer.eval()
    classifer = classifer.to(device)
    total_loss = 0.0
    total = 0
    correct = 0
    correct1 = 0
    with torch.no_grad():
        for i, images in enumerate(CIFAR_test_dataloader):
            # 前向传播
            images, targets = images
            images = images.to(device)
            targets = targets.to(device)
            # gauss_noise = torch.randn(images .size())*0.1
            # inputs = images + gauss_noise.to(device)
            outputs, latent, mean, log_var = unet(images)
            y_hat = simple_classifer(latent)
            _, pred1 = torch.max(y_hat.data, 1)
            correct1 += (pred1 == targets).sum().item()
            # 计算损失函数
            pred = predict(classifer, outputs)
            total += targets.size(0)
            success_id = (targets, pred)
            correct += len(success_id)
            # 计算损失函数
            loss = ((outputs - images) ** 2).mean()
            total_loss += loss
            # 记录训练过程
        print('Accuracy of the network on the {} test images: {}%'
                .format(len(CIFAR_test_dataloader), 100 * correct / total))
        print('Accuracy of the network on the {} test images: {}%'
                .format(len(CIFAR_test_dataloader), 100 * correct1 / total))

        print('Test_Loss: {:.4f}'
              .format(total_loss))
    return loss.item()


# 设置随机数种子
setup_seed(1234)

if __name__ == "__main__":
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    Unet = CIFAR_Net()
    simple_classifer = Classifer_cifar()
    trans = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor()  # 将原有数据转化成张量图像，值在(0,1)
        ])

    classifer = nn.Sequential(torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                              model)
    CIFAR_train = torchvision.datasets.CIFAR10(root='.././data/cifar10', train=True, transform=trans, download=True)
    CIFAR_test = torchvision.datasets.CIFAR10(root='.././data/cifar10', train=False, transform=trans, download=True)

    CIFAR_train_dataloader = torch.utils.data.DataLoader(CIFAR_train, batch_size=128, shuffle=True, num_workers=1)
    CIFAR_test_dataloader = torch.utils.data.DataLoader(CIFAR_test, batch_size=128, shuffle=True, num_workers=1)
    # 预处理数据以及训练模型

    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Unet.to(device)
    simple_classifer.to(device)
    criterion = nn.CrossEntropyLoss()
    params = list(Unet.parameters()) + list(simple_classifer.parameters())
    optimizer = optim.Adam(params, lr=0.0001)

    # 训练模型
    total_step = len(CIFAR_train) // 128
    for epoch in range(num_epochs):
        train_loss = train_epoch_den(epoch, Unet, device, CIFAR_train_dataloader, optimizer, simple_classifer, criterion)
        test_loss = test_epoch_den(epoch, Unet, device, CIFAR_test_dataloader, classifer, simple_classifer)
    # 保存模型
    torch.save(Unet.state_dict(), '../Model/cifar_Unet_10.pth')
    torch.save(simple_classifer.state_dict(), '../Model/simple_classifer.pth')


