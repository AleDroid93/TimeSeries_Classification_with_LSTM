import numpy as np
import tensorflow as tf
import os
from constants import N_CLASSES, EPOCHS, N_TIMESTAMPS, N_FOLDS, BATCH_SIZE, EXTENSION, BASE_DIR_FOLDS, BASE_FILE_NAMES
from model import MyLSTM
from utils import reshapeToTensor
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from train import train


for i in range(N_FOLDS):
    current_fold = str(i + 1)
    train_fn = '[YOUR TRAIN DATA FILE PATH HERE]'
    validation_fn = '[YOUR VALIDATION DATA FILE PATH HERE]'
    test_fn = '[YOUR TEST DATA FILE PATH HERE]'
    target_train_fn = '[YOUR TRAIN LABELS FILE PATH HERE]'
    target_validation_fn = '[YOUR VALIDATION LABELS FILE PATH HERE]'
    target_test_fn = '[YOUR TEST LABELS FILE PATH HERE]'
    
    # loading the data already splitted
    x_train = np.load(train_fn)
    x_validation = np.load(validation_fn)
    x_test = np.load(test_fn)

    x_train = reshapeToTensor(x_train)
    x_validation = reshapeToTensor(x_validation)
    x_test = reshapeToTensor(x_test)

    print(x_train.shape)

    y_train = np.load(target_train_fn)
    y_validation = np.load(target_validation_fn)
    y_test = np.load(target_test_fn)
    model = MyLSTM(512, N_CLASSES, dropout_rate=0.5)
    print(x_train.shape)
    print("Fold %s metrics:\n" % current_fold)

    # TRAINING
    outputFolder = './MLP_output'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    """ defining loss function and the optimizer to use in the training phase """
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, outputFolder + '/MLP_ckpts_fold ' + current_fold, max_to_keep=1)
    train(model, x_train, y_train, x_validation, y_validation, loss_object, optimizer, ckpt, manager, n_epochs=EPOCHS)

    # TESTING
    pred = model.predict(x_test)
    print("Accuracy score on test set: ", accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))
    print("F-score on test set: ", f1_score(y_test, pred, average='macro'))
    print("K-score on test set: ", cohen_kappa_score(y_test, pred))