import os
from imageio import imsave
import os
from skimage import io
import torchvision.datasets.mnist as mnist

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
def convert_to_img(train=True):
    if (train):
        data_path = root + '/train/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            path = data_path + str(int(label))
            if (not os.path.exists(path)):
                os.makedirs(path)
            img_name = str(i) +'.jpg'
            img_path = os.path.join(path, img_name)
            imsave(img_path, img)
    else:
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            path = data_path + str(label)
            if (not os.path.exists(path)):
                os.makedirs(path)
            img_path = path + str(i) + '.jpg'
            imsave(img_path, img)







if __name__ == "__main__":
    # filename = 'C:/Users/123/Desktop/RC_decetc/data/cifar10/cifar-10-batches-py'  # 图片的路径
    # meta = unpickle(filename + '/batches.meta')
    # label_name = meta[b'label_names']
    #
    # for i in range(len(label_name)):
    #     file = label_name[i].decode()
    #     path = 'C:/Users/123/Desktop/RC_decetc/data/cifar10/cifar-10-batches-py/test/' + file
    #     isExist = os.path.exists(path)
    #     if not isExist:
    #         os.makedirs(path)
    #
    # # for i in range(1, 6):
    # #     content = unpickle(filename + '/data_batch_' + str(i))  # 解压后的每个data_batch_
    # content = unpickle(filename + '/test_batch' )  # 解压后的每个data_batch_
    # print('load data...')
    # print(content.keys())
    # print('tranfering test_batch')
    # for j in range(10000):
    #     img = content[b'data'][j]
    #     img = img.reshape(3, 32, 32)
    #     img = img.transpose(1, 2, 0)
    #     img_name = 'C:/Users/123/Desktop/RC_decetc/data/cifar10/cifar-10-batches-py/test/' + label_name[
    #             content[b'labels'][j]].decode() + '/test_batch'+ '_num_' + str(
    #         j) + '.jpg'
    #     imsave(img_name, img)

    root = "C:/Users/123/Desktop/RC_decetc/data/mnist/MNIST/raw"
    train_set = (
        mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
            )
    test_set = (
        mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
            )
    print("training set :", train_set[0].size())
    print("test set :", test_set[0].size())




    # convert_to_img(True)#转换训练集
    convert_to_img(False)#转换测试集