import numpy as np
import constants
from model import MyLSTM
from utils import reshapeToTensor
from sklearn.metrics import accuracy_score
from train import train


for i in range(constants.N_FOLDS):
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
    n_classes = 12
    model = MyLSTM(512, n_classes, dropout_rate=0.5)
    print(x_train.shape)
    print("Fold %s metrics:\n" % current_fold)
    # TRAINING
    train(model, x_train, y_train, x_validation, y_validation, n_epochs=constants.EPOCHS)

    # TESTING
    pred = model.predict(x_test)
    print(accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))