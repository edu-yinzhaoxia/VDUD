import joblib
import matplotlib
import numpy as np
import torch
import torchvision
from torch import nn
import torch.utils.data as data

from Model.classifer import Classifer_cifar


from Model.reconstruction.unet import MNIST_Net, CIFAR_Net



from attacks.pre_attack import BIM, FGSM, f_PGD_l2, f_PGD_linf, CW, JSMA, SA, PA, VMI, DIM, TIM, MIM, autoattack
from utils.eval import predict, common, detect_rate, attack_suc

if __name__ == '__main__':
    path = './Model/cifar_Unet_10.pth'  # 去噪网络
    simple_path = './Model/simple_classifer.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Unet = CIFAR_Net()
    simple_classifer = Classifer_cifar()
    Unet.load_state_dict(torch.load(path))
    simple_classifer.load_state_dict(torch.load(simple_path))
    Unet.eval()
    simple_classifer.eval()
    resnet20 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    resnet56 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    vgg16 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    resnet20.eval()
    resnet56.eval()
    vgg16.eval()
    trans = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor()  # 将原有数据转化成张量图像，值在(0,1)
        ])

    Resnet20 = nn.Sequential(
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        resnet20
    ).to(device)

    Resnet56 = nn.Sequential(
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        resnet56
    ).to(device)

    VGG16 = nn.Sequential(
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        vgg16
    ).to(device)

    Unet.to(device)
    simple_classifer.to(device)
    test_cifar = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, transform=trans, download=True)
    test_dataloader = torch.utils.data.DataLoader(test_cifar, batch_size=32, shuffle=True, num_workers=1)

    total = 0
    total0 = 0
    total1 = 0
    total2 = 0
    total3 = 0

    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0

    for i, imgs in enumerate(test_dataloader):
        images, labels = imgs
        images = images.to(device)
        labels = labels.to(device)

        outputs, latent, _, _ = Unet(images)
        common_id0 = common(labels, predict(simple_classifer, latent))
        images0 = images[common_id0]
        labels0 = labels[common_id0]
        total0 += len(common_id0)

        common_id1 = common(labels0, predict(Resnet20, images0))
        inputs = images0[common_id1]
        targets = labels0[common_id1]
        total += len(common_id1)

        or_pre = predict(Resnet20, inputs)
        advs_images = JSMA(Resnet20, inputs, targets)
        adv_pre = predict(Resnet20, advs_images)
        outputs1, latent1, _, _ = Unet(inputs)

        simple_pre = predict(simple_classifer, latent1)

        outputs2, latent2, _, _ = Unet(advs_images)
        simple_adv_pre = predict(simple_classifer, latent2)

        common_id2 = common(targets, or_pre)
        correct1 += len(common_id2)  # 原始样本识别精度

        attack_id2 = attack_suc(targets, adv_pre)
        total2 += len(attack_id2)  # 对抗样本攻击成功率
        full_simple_pre = simple_adv_pre[attack_id2]
        full_adv_pre = adv_pre[attack_id2]

        common_id3 = detect_rate(or_pre, simple_pre)
        correct3 += len(common_id3)  # 重构干净识别精度和目标模型识别精度

        common_id4 = detect_rate(full_simple_pre, full_adv_pre)
        correct4 += len(common_id4)  # 重构对抗识别精度和目标模型识别精度

    print('原始样本模型识别精度: {}%'.
          format(100 * correct1 / total))
    print('对抗样本模型识别精度: {}%'.
          format(100 * total2 / total))
    print('干净样本误检率: {}%'.
          format(100 * correct3 / total))
    print('对抗样本检测率: {}%'.
          format(100 * correct4 / total2))
