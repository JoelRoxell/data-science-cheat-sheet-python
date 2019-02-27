# %%
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop


# %%
(mnist_train_images, mnist_train_labels), (minst_test_images,
                                           mnist_test_labels) = mnist.load_data()


# %%
train_images = mnist_train_images.reshape(60000, 784)
test_images = minst_test_images.reshape(10000, 784)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255


# %%
train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)


# %%
def display_sample(num):
    print(train_labels[num])
    label = train_labels[num].argmax(axis=0)
    image = train_images[num].reshape(28, 28)

    plt.title("sample {} {}".format(num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


# %%
display_sample(1234)


# %%
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.summary()


# %%
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(), metrics=['accuracy'])


# %%
history = model.fit(train_images, train_labels, batch_size=100,
                    epochs=10, verbose=2, validation_data=(test_images, test_labels))


# %%
score = model.evaluate(test_images, test_labels, verbose=0)
print('test loss {}'.format(score[0]))
print('test accuracy {}'.format(score[1]))

# %%
