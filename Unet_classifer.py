import torch
import torchvision
from torch import nn, optim
from Model.classifer import Classifer_cifar
from Model.reconstruction.unet import CIFAR_Net
from attacks.pre_attack import BIM, FGSM
from utils.eval import predict

# def normalized_fature(data_2d):
#     np.random.seed(42)
#     # 计算每一列的最小值和最大值
#     min_vals = np.min(data_2d, axis=0)
#     max_vals = np.max(data_2d, axis=0)
#
#     # 对每一列进行归一化
#     normalized_data = (data_2d - min_vals) / (max_vals - min_vals)
#     return normalized_data
#
# def Calculate_difference(inputs, re_inputs, batch_size):
#     Rase = RelativeAverageSpectralError()
#     Uqi = UniversalImageQualityIndex()
#     Vif = VisualInformationFidelity()
#     new_size = (41, 41)
#
#     index_diff = torch.zeros((batch_size, 3))
#     for i in range(batch_size):
#         input = torch.unsqueeze(inputs[i], 0)
#         re_input = torch.unsqueeze(re_inputs[i], 0)
#         rase = Rase(input.cpu(), re_input.cpu())
#         uqi = Uqi(input.cpu(), re_input.cpu())
#         resized_input = F.interpolate(input, size=new_size, mode='bilinear', align_corners=False)
#         resized_re_input = F.interpolate(re_input, size=new_size, mode='bilinear', align_corners=False)
#         vif = Vif(resized_input.cpu(), resized_re_input.cpu())
#         index_diff[i, 0] = rase
#         index_diff[i, 1] = uqi
#         index_diff[i, 2] = vif
#     return index_diff
#
# # test
# def Construct_dataset_test(x_test):
#     value = np.array(x_test)
#     value_test = normalized_fature(value)
#     return value_test
#
# def SVM_classifer_test(value_test):
#     model_filename = './checkpoint/svm_cifar_unet.joblib'
#
#     # 加载模型
#     loaded_classifier = joblib.load(model_filename)
#
#
#     # 进行预测
#     y_pred_new = loaded_classifier.predict(value_test)
#     print(y_pred_new)
#  # train
# # def Construct_dataset(x_true, x_false):
# #     y_true = np.ones(len(x_true))
# #     y_false = np.zeros(len(x_false))
# #     true_value = np.array(x_true)
# #     false_value = np.array(x_false)
# #     # 数据归一化
# #     true_value = normalized_fature(true_value)
# #     false_value = normalized_fature(false_value)
# #
# #     fig = plt.figure(figsize=(8, 6))
# #     ax = fig.add_subplot(111, projection='3d')
# #     ax.scatter(true_value[:, 0], true_value[:, 1], true_value[:, 2], c='b', label='clean example')
# #     ax.scatter(false_value[:, 0], false_value[:, 1], false_value[:, 2], c='y', label='adversarial example')
# #     ax.set_xlabel('VIF')
# #     ax.set_ylabel('UQI')
# #     ax.set_zlabel('RASE')
# #     ax.legend()
# #     plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文字体，如宋体
# #     plt.title("重构特征差异的三维分布")
# #     plt.show()
# #     # 将数据和标签合并成一个数据集
# #     x = np.concatenate([true_value, false_value])
# #     y = np.concatenate([y_true, y_false])
# #
# #     # 划分数据集为训练集和测试集
# #     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# #     return X_train, X_test, y_train, y_test
#
# # def SVM_classifer(X_train, X_test, y_train, y_test):
# #     classifier = SVC(kernel='linear')
# #     classifier.fit(X_train, y_train)
# #
# #     # 预测测试集
# #     y_pred = classifier.predict(X_test)
# #
# #     # 评估分类器性能
# #     accuracy = accuracy_score(y_test, y_pred)
# #     # 保存二分类模型
# #     model_filename = './checkpoint/svm_cifar_unet.joblib'
# #     joblib.dump(classifier, model_filename)
# #     print(f'Model parameters saved to {model_filename}')
# #     print(f'Accuracy: {accuracy}')
# train_size = int(1 * len(train_cifar))
    # val_size = len(train_cifar) - train_size
    # train_dataset, val_dataset = data.random_split(train_cifar, [train_size, val_size])
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=1)

    # train_size = int(0.1 * len(test_cifar))
    # val_size = len(test_cifar) - train_size
    # test_dataset, val_dataset = data.random_split(test_cifar, [train_size, val_size])
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=True, num_workers=1)
    # or_differences = []
    # adv_differences = []
    # total = 0
    # correct = 0
    # correct1 = 0
    # correct2 = 0
    # for i, imgs in enumerate(train_dataloader):
    #     images, labels = imgs
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     re_or_images = Reconstuct_image(images, Unet2)
    #     adv_images = BIM(classifer, images, labels)
    #     re_adv_images = Reconstuct_image(adv_images, Unet2)



if __name__ == '__main__':
    path = './Model/cifar_Unet_10.pth'  # 去噪网络
    # cnn_path = '.././checkpoint/MNIST_CNN/mnist_cnn.pth'
    # model = CNN_MNIST()
    # model.load_state_dict(torch.load(cnn_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Unet = CIFAR_Net()
    Unet.load_state_dict(torch.load(path))

    Unet.eval()
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    model.eval()
    trans = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor()  # 将原有数据转化成张量图像，值在(0,1)
        ])

    classifer = nn.Sequential(
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        model
    ).to(device)
    Unet.to(device)
    test_cifar = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, transform=trans, download=True)
    CIFAR_test_dataloader = torch.utils.data.DataLoader(test_cifar, batch_size=128, shuffle=True, num_workers=1)

    total = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    for i, imgs in enumerate(CIFAR_test_dataloader):
        images, labels = imgs
        images = images.to(device)
        labels = labels.to(device)
        total += labels.size(0)
        adv_images = FGSM(classifer, images, labels)
        outputs, latent, _, _ = Unet(adv_images)

