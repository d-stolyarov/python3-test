import numpy as np
from data_utils import *
from models.FCN import *
from models.DeepLabV2 import *
from models.UNet import *
import matplotlib.pyplot as plt

# main function
def main():
    # параметры по умолчанию
    model_type = 'unet'
    epoch = 1
    batch_size = 8
    learning_rate = 0.001

    # data load
    IMG_HEIGHT = 384
    IMG_WIDTH = 256
    num_of_class = 4

    x_train = np.load('dataset/x_train.npy').astype(np.float32)
    x_test = np.load('dataset/x_test.npy').astype(np.float32)
    y_train = np.load('dataset/y_train_onehot.npy').astype(np.float32)
    y_test = np.load('dataset/y_test_onehot.npy').astype(np.float32)

    if model_type == 'unet':
        model = UNet(img_shape = x_train[0].shape, num_of_class = num_of_class, learning_rate = learning_rate)
    elif model_type == 'fcn':
        model = FCN8s(img_shape = x_train[0].shape, num_of_class = num_of_class, learning_rate = learning_rate)
    elif model_type == 'deeplabv2':
        model = DeepLabV2(img_shape = x_train[0].shape, num_of_class = num_of_class, learning_rate = learning_rate)

    # model train
    history = model.train_generator(x_train, y_train,
                                x_test, y_test,
                                model_type,
                                epoch = epoch,
                                batch_size = batch_size)

    # model history plot - loss and accuracy
    plot_acc(history)
    plot_loss(history)
    plt.show()

if __name__ == '__main__':
    main()