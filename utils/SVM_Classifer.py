import joblib
import matplotlib
import numpy as np
import torch
import torchvision
from torch import nn
import torch.utils.data as data
from torchmetrics.image import RelativeAverageSpectralError, UniversalImageQualityIndex, VisualInformationFidelity

from Model.Mnist_model import CNN_MNIST
from Model.reconstruction.comdefend_model import Decoder_Minist, Encoder_Minist, Encoder_CIFAR, Decoder_CIFAR
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


from attacks.pre_attack import f_FGSM, BIM
from utils.eval import predict


def Reconstuct_image(inputs, Encoder, Decoder):
    Encoder.eval()
    Decoder.eval()
    tmp_out = Encoder(inputs)
    noisy_code = tmp_out - torch.randn(tmp_out.size()).to(device) * 20
    binary_code = torch.round(torch.sigmoid(noisy_code))
    outputs = Decoder(binary_code)
    return outputs

def normalized_fature(data_2d):
    np.random.seed(42)
    # 计算每一列的最小值和最大值
    min_vals = np.min(data_2d, axis=0)
    max_vals = np.max(data_2d, axis=0)

    # 对每一列进行归一化
    normalized_data = (data_2d - min_vals) / (max_vals - min_vals)
    return normalized_data

def Calculate_difference(inputs, re_inputs, batch_size):
    Rase = RelativeAverageSpectralError()
    Uqi = UniversalImageQualityIndex()
    Vif = VisualInformationFidelity()
    new_size = (41, 41)

    index_diff = torch.zeros((batch_size, 3))
    for i in range(batch_size):
        input = torch.unsqueeze(inputs[i], 0)
        re_input = torch.unsqueeze(re_inputs[i], 0)
        rase = Rase(input.cpu(), re_input.cpu())
        uqi = Uqi(input.cpu(), re_input.cpu())
        resized_input = F.interpolate(input, size=new_size, mode='bilinear', align_corners=False)
        resized_re_input = F.interpolate(re_input, size=new_size, mode='bilinear', align_corners=False)
        vif = Vif(resized_input.cpu(), resized_re_input.cpu())
        index_diff[i, 0] = rase/10000
        index_diff[i, 1] = uqi
        index_diff[i, 2] = vif/10
    return index_diff


def Construct_dataset(x_true, x_false):
    y_true = np.ones(len(x_true))
    y_false = np.zeros(len(x_false))
    true_value = np.array(x_true)
    false_value = np.array(x_false)
    # 数据归一化
    true_value = normalized_fature(true_value)
    false_value = normalized_fature(false_value)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(true_value[:, 0], true_value[:, 1], true_value[:, 2], c='b', label='clean example')
    ax.scatter(false_value[:, 0], false_value[:, 1], false_value[:, 2], c='y', label='adversarial example')
    ax.set_xlabel('VIF')
    ax.set_ylabel('UQI')
    ax.set_zlabel('RASE')
    ax.legend()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文字体，如宋体
    plt.title("重构特征差异的三维分布")
    plt.show()
    # 将数据和标签合并成一个数据集
    x = np.concatenate([true_value, false_value])
    y = np.concatenate([y_true, y_false])

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def SVM_classifer(X_train, X_test, y_train, y_test):
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    # 预测测试集
    y_pred = classifier.predict(X_test)

    # 评估分类器性能
    accuracy = accuracy_score(y_test, y_pred)
    # 保存二分类模型
    model_filename = '../checkpoint/svm_cifar_encoder.joblib'
    joblib.dump(classifier, model_filename)
    print(f'Model parameters saved to {model_filename}')
    print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    encoder_path = '.././checkpoint/Cifar_Comrec/cifar_encoder.pth'
    decoder_path = '.././checkpoint/Cifar_Comrec/cifar_decoder.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder_CIFAR()
    decoder = Decoder_CIFAR()
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    batch = 50
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    trans = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor()  # 将原有数据转化成张量图像，值在(0,1)
        ])

    classifer = nn.Sequential(
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        model
    ).to(device)


    encoder.to(device)
    decoder.to(device)
    train_cifar = torchvision.datasets.CIFAR10(root='.././data/cifar10', train=True, transform=trans, download=True)
    train_size = int(0.1* len(train_cifar))
    val_size = len(train_cifar) - train_size
    train_dataset, val_dataset = data.random_split(train_cifar, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=1)
    or_differences = []
    adv_differences = []
    total = 0
    correct = 0
    correct1 = 0
    for i, imgs in enumerate(train_dataloader):
        images, labels = imgs
        images = images.to(device)
        labels = labels.to(device)
        adv_images = BIM(classifer, images, labels)
        re_or_images = Reconstuct_image(images, encoder, decoder)
        re_adv_images = Reconstuct_image(adv_images, encoder, decoder)
    #     pred = predict(classifer, adv_images)
    #     total += labels.size(0)
    #     success_id = attack_success(labels, pred)
    #     correct += len(success_id)
    #     pred1 = predict(classifer, re_adv_images)
    #     success_id1 = attack_success(labels, pred1)
    #     correct1 += len(success_id1)
    # print('Accuracy of the network on the test images: {}%'.
    #       format(100 * correct / total))
    # print('Accuracy of the network on the test images: {}%'.
    #       format(100 * correct1 / total))

        or_fea_difference = Calculate_difference(images, re_or_images, batch)
        adv_fea_difference = Calculate_difference(adv_images, re_adv_images, batch)
        or_fea_tmp = or_fea_difference.detach().cpu()
        adv_fea_tmp = adv_fea_difference.detach().cpu()
        for j in range(batch):
            or_differences.append(or_fea_tmp[j].numpy())
            adv_differences.append(adv_fea_tmp[j].numpy())

    X_train, X_test, y_train, y_test = Construct_dataset(or_differences, adv_differences)
    SVM_classifer(X_train, X_test, y_train, y_test)

