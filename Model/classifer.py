import torch
from torch import nn

def full_connected(dim_in, dim_out):
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.BatchNorm1d(dim_out),
        nn.ReLU(),
        nn.Dropout(p=0.5)
    )
class Classifer_cifar(nn.Module):
    def __init__(self, num_classes: int = 10, init_weights: bool = True):
        super(Classifer_cifar, self).__init__()
        # 添加卷积层

        self.linear1 = full_connected(2048, 1024)
        self.linear2 = full_connected(1024, 512)
        self.linear3 = full_connected(512, num_classes)
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
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.sm(x)
        return x


class Classifer_MNIST(nn.Module):
    def __init__(self, num_classes: int = 10, init_weights: bool = True):
        super(Classifer_MNIST, self).__init__()
        # 添加卷积层
        self.linear1 = full_connected(512, 1024)
        self.linear2 = full_connected(1024, 512)
        self.linear3 = full_connected(512, num_classes)
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
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.sm(x)
        return x