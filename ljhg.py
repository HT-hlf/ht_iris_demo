#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: logistic_regression.py

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from time import time

__author__ = 'yasaka'

iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris['DESCR'])
# print(iris['feature_names'])

X = iris['data'][:, 3:]
# print(X)

# print(iris['target'])
y = iris['target']
# y = (iris['target'] == 2).astype(np.int)
# print(y)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#
start = time()
param_grid = {"tol": [1e-4, 1e-3, 1e-2],
              "C": [0.4, 0.6, 0.8]}
log_reg = LogisticRegression(multi_class='ovr', solver='sag')
grid_search = GridSearchCV(log_reg, param_grid=param_grid, cv=3)
# log_reg.fit(X, y)
grid_search.fit(X, y)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
# print(grid_search.cv_results_)
report(grid_search.cv_results_)


X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# print(np.linspace(0, 3, 1000))

# print(X_new)

# y_proba = log_reg.predict_proba(X_new)
# y_hat = log_reg.predict(X_new)
# print(y_proba)
# print(y_hat)

# plt.plot(X_new, y_proba[:, 2], 'g-', label='Iris-Virginica')
# plt.plot(X_new, y_proba[:, 1], 'r-', label='Iris-Versicolour')
# plt.plot(X_new, y_proba[:, 0], 'b--', label='Iris-Setosa')
# plt.show()

# print(log_reg.predict([[1.7], [1.5]]))
print(grid_search.predict([[1.7], [1.5]]))


# sklearn.linear_model.LogisticRegression(penalty=l2, # 惩罚项，可选l1,l2，对参数约束，减少过拟合风险
#                                         dual=False, # 对偶方法（原始问题和对偶问题），用于求解线性多核（liblinear)的L2的惩罚项上。样本数大于特征数时设置False
#                                         tol=0.0001, # 迭代停止的条件，小于等于这个值停止迭代，损失迭代到的最小值。
#                                         C=1.0, # 正则化系数λ的倒数，越小表示越强的正则化。
#                                         fit_intercept=True, # 是否存在截距值，即b
#                                         intercept_scaling=1, #
#                                         class_weight=None, # 类别的权重，样本类别不平衡时使用，设置balanced会自动调整权重。为了平横样本类别比例，类别样本多的，权重低，类别样本少的，权重高。
#                                         random_state=None, # 随机种子
#                                         solver=’liblinear’, # 优化算法的参数，包括newton-cg,lbfgs,liblinear,sag,saga,对损失的优化的方法
#                                         max_iter=100,# 最大迭代次数，
#                                         multi_class=’ovr’,# 多分类方式，有‘ovr','mvm'
#                                         verbose=0, # 输出日志，设置为1，会输出训练过程的一些结果
#                                         warm_start=False, # 热启动参数，如果设置为True,则下一次训练是以追加树的形式进行（重新使用上一次的调用作为初始化）
#                                         n_jobs=1 # 并行数，设置为1，用1个cpu运行，设置-1，用你电脑的所有cpu运行程序
#                                             )