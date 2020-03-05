import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from  utils import getBatch


def train(model, x_train, y_train, x_validation, y_validation, loss_object, optimizer, ckpt, manager, batch_size = 32, n_epochs = 10):
    # trying to restore a previous checkpoint. If it fails, starts from scratch
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    iterations = x_train.shape[0] / batch_size
    if x_train.shape[0] % batch_size != 0:
        iterations += 1

    best_step = -1
    best_epoch_val_loss = 100.0
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
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_loss = loss_iteration / int(iterations)

        pred = model(x_train, training=False)
        y_argmax = np.argmax(y_train, axis=1)
        pred_argmax = np.argmax(pred, axis=1)
        ac = accuracy_score(y_argmax, pred_argmax)

        # increment of checkpoint step
        ckpt.step.assign_add(1)
        if total_loss <= best_epoch_val_loss:
            # new best model found, so save the checkpoint into a file
            best_epoch_val_loss = total_loss
            best_step = int(ckpt.step)
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(best_step, save_path))
            print("loss {:1.2f}".format(best_epoch_val_loss))

        print("epoch %d loss %f Train Accuracy %f" % (
            e, total_loss, np.round(ac, 4)))
        pred = model(x_validation, training=False)
        print("vs. Validation Accuracy %f" % accuracy_score(np.argmax(y_validation, axis=1), np.argmax(pred, axis=1)))
        print("===============")