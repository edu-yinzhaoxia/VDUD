import torch.nn as nn
import torchvision.models


class CNN_MNIST(nn.Module):
    def __init__(self, num_classes: int = 10, init_weights: bool = True):
        super(CNN_MNIST, self).__init__()
        self.Drop_rate = 0.3
        # 添加卷积层
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.Activation1 = nn.ReLU()

        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.Activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.Conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.Activation3 = nn.ReLU()

        self.Conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.Activation4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.Activation5 = nn.ReLU(True)
        self.Drop = nn.Dropout(p=self.Drop_rate, inplace=False)
        self.fc2 = nn.Linear(512, 256)
        self.Activation5 = nn.ReLU()
        self.fc3 = nn.Linear(256, num_classes)
        self.sm = nn.Softmax(dim=-1)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 前向传播
        x = self.Conv1(x)
        x = self.Activation1(x)

        x = self.Conv2(x)
        x = self.Activation2(x)
        x = self.pool2(x)

        x = self.Conv3(x)
        x = self.Activation3(x)

        x = self.Conv4(x)
        x = self.Activation4(x)
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.Activation5(x)
        x = self.Drop(x)
        x = self.fc2(x)
        x = self.Activation5(x)
        x = self.fc3(x)
        x = self.sm(x)
        return x


# vgg16_MNIST
VGG16_MNIST = torchvision.models.vgg16(pretrained=True)
VGG16_MNIST.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
VGG16_MNIST.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)


# resnet18_MNIST
Resnet18_MNIST = torchvision.models.resnet18(pretrained=True)
Resnet18_MNIST.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
Resnet18_MNIST.fc = nn.Linear(in_features=512, out_features=10, bias=True)
