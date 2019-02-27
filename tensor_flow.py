# %%
# Highly efficient C++ imp. with easy python APIs.
# Train is costly, however, we push or finished models to the potentially offline devices.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

a = tf.Variable(1, name="a")
b = tf.Variable(2, name="b")

f = a + b

init = tf.global_variables_initializer()

with tf.Session() as s:
    init.run()
    print(f.eval())

# %%
sess = tf.InteractiveSession()

mnist = input_data.read_data_sets('MNIST_DATA/', one_hot=True)

# %%


def display_sample(num):
    print(mnist.train.labels[num])

    label = mnist.train.labels[num].argmax(axis=0)

    image = mnist.train.images[num].reshape([28, 28])

    plt.title('sample: {} Label: {}'.format(num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


display_sample(14)

# %%
images = mnist.train.images[0].reshape([1, 784])

for i in range(1, 500):
    images = np.concatenate((images, mnist.train.images[i].reshape(1, 784)))

plt.imshow(images, cmap=plt.get_cmap('gray_r'))
plt.show()

# %%

input_images = tf.placeholder(tf.float32, shape=[None, 784])
target_labels = tf.placeholder(tf.float32, shape=[None, 10])

# %%
hidden_nodes = 512

input_weights = tf.Variable(tf.truncated_normal([784, hidden_nodes]))
input_biases = tf.Variable(tf.zeros([hidden_nodes]))

hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes, 10]))
hidden_biases = tf.Variable(tf.zeros([10]))


# %%
input_layer = tf.matmul(input_images, input_weights)
hidden_layer = tf.nn.relu(input_layer + input_biases)
digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases


# %%
# Optimizing
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=digit_weights, labels=target_labels))

optimizer = tf.train.GradientDescentOptimizer(0.8).minimize(loss_function)


# %%
correct_prediction = tf.equal(
    tf.argmax(digit_weights, 1), tf.argmax(target_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# %%
tf.global_variables_initializer().run()

for x in range(2000):
    batch = mnist.train.next_batch(100)
    optimizer.run(feed_dict={input_images: batch[0], target_labels: batch[1]})

    if ((x+1) % 100):
        print("training epoch {}".format(str(x + 1)))
        print("accuracy {}".format(accuracy.eval(feed_dict={
              input_images: mnist.test.images, target_labels: mnist.test.labels})))
