import torch.nn as nn
import torchvision.models


class CNN_CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, init_weights: bool = True):
        super(CNN_CIFAR, self).__init__()
        self.Drop_rate = 0.1
        # 添加卷积层
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.BatchNormalization1 = nn.BatchNorm2d(64)
        self.Activation1 = nn.ReLU()

        self.Conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.BatchNormalization2 = nn.BatchNorm2d(64)
        self.Activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.Drop2 = nn.Dropout(p=self.Drop_rate, inplace=False)

        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.BatchNormalization3 = nn.BatchNorm2d(128)
        self.Activation3 = nn.ReLU()

        self.Conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.BatchNormalization4 = nn.BatchNorm2d(128)
        self.Activation4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.Drop4 = nn.Dropout(p=self.Drop_rate+0.1, inplace=False)

        self.Conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.BatchNormalization5 = nn.BatchNorm2d(256)
        self.Activation5 = nn.ReLU()

        self.Conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.BatchNormalization6 = nn.BatchNorm2d(256)
        self.Activation6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2))
        self.Drop6 = nn.Dropout(p=self.Drop_rate+0.2, inplace=False)

        self.Conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.BatchNormalization7 = nn.BatchNorm2d(512)
        self.Activation7 = nn.ReLU()
        self.pool7 = nn.MaxPool2d(kernel_size=(2, 2))
        self.Drop7 = nn.Dropout(p=self.Drop_rate+0.3, inplace=False)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512*2*2, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.sm = nn.Softmax(dim=-1)
        # if init_weights:
        #     for m in self.modules():
        #         if isinstance(m, nn.Conv2d):
        #             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #             if m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)
        #         elif isinstance(m, nn.BatchNorm2d):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)
        #         elif isinstance(m, nn.Linear):
        #             nn.init.normal_(m.weight, 0, 0.01)
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 前向传播

        x = self.Conv1(x)
        x = self.BatchNormalization1(x)
        self.Activation1 = nn.ReLU(x)

        x = self.Conv2(x)
        x = self.BatchNormalization2(x)
        x = self.Activation2(x)
        x = self.pool2(x)
        x = self.Drop2(x)

        x = self.Conv3(x)
        x = self.BatchNormalization3(x)
        x = self.Activation3(x)

        x = self.Conv4(x)
        x = self.BatchNormalization4(x)
        x = self.Activation4(x)
        x = self.pool4(x)
        x = self.Drop4(x)

        x = self.Conv5(x)
        x = self.BatchNormalization5(x)
        x = self.Activation5(x)

        x = self.Conv6(x)
        x = self.BatchNormalization6(x)
        x = self.Activation6(x)
        x = self.pool6(x)
        x = self.Drop6(x)

        x = self.Conv7(x)
        x = self.BatchNormalization7(x)
        x = self.Activation7(x)
        x = self.pool7(x)
        x = self.Drop7(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sm(x)
        return x


# vgg16_cifar10
VGG16_CIFAR = torchvision.models.vgg16(pretrained=True)
VGG16_CIFAR.classifier[6] = nn.Linear(4096, 10)

# resnet18_cifar10
Resnet18_ft = torchvision.models.resnet18(pretrained=True)
nums_feature = Resnet18_ft.fc.in_features
Resnet18_ft.fc = nn.Linear(nums_feature, 10)
Resnet18_CIFAR = Resnet18_ft

