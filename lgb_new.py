import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import numpy as np


# 读取文件中的数据
def read_data(filename):
    data = pd.read_csv(filename, header=None)
    data = data.values
    data = encode_data(data)
    return data


# 将带有字母的字段进行数字编码
def encode_data(data):
    label_ = LabelEncoder()
    for i in range(10):
        if (i % 2 == 0):
            label_.fit(['C', 'D', 'H', 'S'])
            data[:, i] = label_.transform(list(data[:, i]))
        else:
            label_.fit(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'])
            data[:, i] = label_.transform(list(data[:, i]))
    return data


# 数据标准化
def normal_data(data):
    da = preprocessing.MinMaxScaler()
    X = da.fit_transform(data)
    return X


# 将测试的结果存为一个文件
def save_predict(pred, filename):
    df = pd.DataFrame(pred)
    df.to_csv(filename, index=False, header=None)


# 模型训练，预测
def train_predict(X_train, Y_train, X_test):
    gbm = lgb.LGBMClassifier(
        max_depth=6,  # 树的最大深度
        min_data_in_leaf=20,  # 叶子可能具有的最小记录数
        num_leaves=126,  # 叶子数量
        silent=True,
        learning_rate=0.698,  # 学习速率
        n_estimators=700,  # 树的数量
    )
    gbm.fit(X_train,
            Y_train.astype('int')  # 模型训练
            )
    pred_y = gbm.predict(X_test)  # 模型预测
    return pred_y


data_train = read_data('training 1.csv')  # 读取训练集数据
data_test = read_data('preliminary-testing 1.csv')  # 读取测试集数据
Data = np.concatenate((data_train[:, 0:10], data_test), axis=0)  # 将训练集的前十列特征与测试集按行拼接
leg = len(data_train)
X_train = normal_data(Data)[:leg, ]  # 训练集输入值
y_train = data_train[:leg, 10]  # 训练集的目标值
X_test = normal_data(Data)[len(data_train):, ]  # 测试集的输入值

# i = 0
# h = 0
# for g in pred_Y:
#     if (g == y_test[i]):
#         h = h + 1
#     i = i + 1
# print(h)
# print(h / len(pred_Y))
pred_Y = train_predict(X_train, y_train, X_test)  # 调用函数进行训练预测，得到预测结果
save_predict(pred_Y, 'label0627.txt')  # 调用函数，将预测结果存为一个文件
