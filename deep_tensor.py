import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
import numpy as np
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
import xgboost as xgb


# load or create your dataset
def tensor_deep():
    print('Load data...')
    data = pd.read_csv('training.csv')
    data = data.values
    print(type(list(data[:, 0])))
    label_ = LabelEncoder()
    for i in range(len(data[0]) - 1):
        if (i % 2 == 0):
            label_.fit(['C', 'D', 'H', 'S'])
            data[:, i] = label_.transform(list(data[:, i]))
        else:
            label_.fit(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'])
            data[:, i] = label_.transform(list(data[:, i]))
    print(data)
    train_samples = int(len(data) * 0.8)
    data_x = data[:, 0:10]
    data_y = pd.get_dummies(data[:, 10]).values
    print(data_y)

    x = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="input")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="output")

    w = tf.get_variable("weight", shape=[10, 10], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
    bais = tf.get_variable("bais", shape=[10], dtype=tf.float32, initializer=tf.constant_initializer(0))
    y_1 = tf.nn.bias_add(tf.matmul(x, w), bais)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_1))
    # x0min, x0max=data_x[:, 0].min(), data_x[:, 0].max()
    # x1min, x1max=data_x[:, 1].min(), data_x[:, 1].max()
    # x2min, x2max=data_x[:, 2].min(), data_x[:, 2].max()
    # x3min, x3max=data_x[:, 3].min(), data_x[:, 3].max()
    # x4min, x4max=data_x[:, 4].min(), data_x[:, 4].max()
    # x5min, x5max=data_x[:, 5].min(), data_x[:, 5].max()
    # x6min, x6max=data_x[:, 6].min(), data_x[:, 6].max()
    # x7min, x7max=data_x[:, 7].min(), data_x[:, 7].max()
    # x8min, x8max=data_x[:, 8].min(), data_x[:, 8].max()
    # x9min, x9max=data_x[:, 9].min(), data_x[:, 9].max()

    with tf.Session() as sess:
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y, 1), tf.arg_max(y_1, 1)), tf.float32))
        train_step = tf.train.AdamOptimizer().minimize(loss)
        my = tf.arg_max(y_1, 1)
        sess.run(tf.global_variables_initializer())
        for i in range(len(data_x)):
            sess.run(train_step, feed_dict={x: data_x, y: data_y})
            if i % 500 == 0:
                accuracy_print = sess.run(accuracy, feed_dict={x: data_x, y: data_y})
                print(accuracy_print)

        data = pd.read_csv('preliminary-testing.csv')
        data = data.values
        label_ = LabelEncoder()
        for i in range(len(data[0])):
            if (i % 2 == 0):
                label_.fit(['C', 'D', 'H', 'S'])
                data[:, i] = label_.transform(list(data[:, i]))
            else:
                label_.fit(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'])
                data[:, i] = label_.transform(list(data[:, i]))
        test_x = data[:, 0:10]

        # h=0.05
        # xx, yy = np.meshgrid(np.arange(x0min-1,x0max+1, h), np.arange(x1min-1,x1max+1, h))
        # x_ = xx.reshape([xx.shape[0]*xx.shape[1],1])
        # y_ = yy.reshape([yy.shape[0]*yy.shape[1],1])
        # test_x = np.c_[x_,y_]
        my_p = sess.run(my, feed_dict={x: test_x})
        coef = w.eval()
        intercept = bais.eval()
    print(my_p)
    a = {'label': list(my_p)}
    print(len(my_p))
    for i in range(10):
        print(i)
        print(list(my_p).count(i))
    # create dataset for lightgbm
    b = pd.DataFrame(a)
    b.to_csv('label.csv')


# Generate dummy data
def keras_mul():
    # data = pd.read_csv('training.csv')
    data = pd.read_csv('Data/data.csv')
    data = data.values
    # print(type(list(data[:, 0])))
    # label_ = LabelEncoder()
    # for i in range(len(data[0]) - 1):
    #     if (i % 2 == 0):
    #         label_.fit(['C', 'D', 'H', 'S'])
    #         data[:, i] = label_.transform(list(data[:, i]))
    #     else:
    #         label_.fit(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'])
    #         data[:, i] = label_.transform(list(data[:, i]))

    data_x = data[:18000, 0:5]
    x_train = data_x
    y_train = keras.utils.to_categorical(data[:18000, 5], num_classes=10)
    test_x = data[18000:, 0:5]
    x_test = test_x
    y_test = keras.utils.to_categorical(data[18000:, 5], num_classes=10)
    # data = pd.read_csv('preliminary-testing.csv')
    data = pd.read_csv('Data/test.csv')
    data = data.values
    # label_ = LabelEncoder()
    # for i in range(len(data[0])):
    #     if (i % 2 == 0):
    #         label_.fit(['C', 'D', 'H', 'S'])
    #         data[:, i] = label_.transform(list(data[:, i]))
    #     else:
    #         label_.fit(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'])
    #         data[:, i] = label_.transform(list(data[:, i]))
    test = data[:, 0:5]
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=5))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    op = model.predict(test, verbose=1)
    g = []
    for i in range(len(op)):
        g.append(list(op[i]).index(max(list(op[i]))))
    a = {'label': g}
    # for i in range(10):
    #     print(i)
    #     print(list(score).count(i))
    # create dataset for lightgbm
    b = pd.DataFrame(a)
    b.to_csv('label0626.csv', index=False)


def xgboost_mul():
    # label need to be 0 to num_class -1
    # if col 33 is '?' let it be 1 else 0, col 34 substract 1
    data = pd.read_csv('training.csv')
    data = data.values
    label_ = LabelEncoder()
    for i in range(len(data[0]) - 1):
        if (i % 2 == 0):
            label_.fit(['C', 'D', 'H', 'S'])
            data[:, i] = label_.transform(list(data[:, i]))
        else:
            label_.fit(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'])
            data[:, i] = label_.transform(list(data[:, i]))
    sz = data.shape

    train = data[:16000, :]
    test = data[16000:, :]  # take row 257-366 as testing set

    train_X = train[:, 0:10]
    train_Y = train[:, 10]

    test_X = test[:, 0:10]
    test_Y = test[:, 10]

    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 10
    param['num_class'] = 10

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 5
    bst = xgb.train(param, xg_train, num_round, watchlist)
    # get prediction
    pred = bst.predict(xg_test)

    print('predicting, classification error=%f' % (
            sum(int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))))

    # do the same thing again, but output probabilities
    param['objective'] = 'multi:softprob'
    bst = xgb.train(param, xg_train, num_round, watchlist)
    # Note: this convention has been changed since xgboost-unity
    # get prediction, this is in 1D array, need reshape to (ndata, nclass)
    yprob = bst.predict(xg_test).reshape(test_Y.shape[0], 10)
    print(yprob)
    ylabel = np.argmax(yprob, axis=1)  # return the index of the biggest pro

    print('predicting, classification error=%f' % (
            sum(int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))))


keras_mul()
