import numpy as np
import sys
sys.path.append('/content/python3-test')
from data_utils import *
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

    # создаем модель UNet
    model = UNet(img_shape=x_train[0].shape, num_of_class=num_of_class, learning_rate=learning_rate)

    # Задайте train_gen и val_gen или используйте x_train и y_train напрямую, если не используете генераторы
    # Например, если train_gen и val_gen не определены:
    train_gen = (x_train, y_train)
    val_gen = (x_test, y_test)

    # Также убедитесь, что cb_checkpoint и reduce_lr определены:
    # (Пример создания обратных вызовов, если они не созданы)
    cb_checkpoint = ModelCheckpoint('model_checkpoint.keras', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # model train
    history = model.train_generator(x_train, y_train, x_test, y_test, name_model='unet', epoch=epoch, batch_size=batch_size)

    # model history plot - loss and accuracy
    plot_acc(history)
    plot_loss(history)
    plt.show()

if __name__ == '__main__':
    main()