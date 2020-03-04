import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from utils import getBatch


def train(model, x_train, y_train, x_validation, y_validation, learning_rate=0.0005, batch_size = 32, n_epochs = 10):

    """ defining loss function and the optimizer to use in the training phase """
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    iterations = x_train.shape[0] / batch_size
    if x_train.shape[0] % batch_size != 0:
        iterations += 1

    for e in range(n_epochs):
        loss_iteration = 0
        x_train, y_train = shuffle(x_train, y_train, random_state=0)
        total_loss = 0.0
        for ibatch in range(int(iterations)):
            batch_x = getBatch(x_train, ibatch, batch_size)
            batch_y = getBatch(y_train, ibatch, batch_size)

            with tf.GradientTape() as tape:
                predictions = model(batch_x, training=True)
                loss = loss_object(batch_y, predictions)
                loss_iteration += loss.numpy()
                gradients = tape.gradient(loss, model.trainable_variables)
                #print("gradients ",gradients)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_loss = loss_iteration / int(iterations)

        pred = model(x_train, training=False)
        y_argmax = np.argmax(y_train, axis=1)
        pred_argmax = np.argmax(pred, axis=1)
        ac = accuracy_score(y_argmax, pred_argmax)
        print("epoch %d loss %f Train Accuracy %f" % (
            e, total_loss, np.round(ac, 4)))
        pred = model(x_validation, training=False)
        print("vs. Validation Accuracy %f" % accuracy_score(np.argmax(y_validation, axis=1), np.argmax(pred, axis=1)))
        print("===============")