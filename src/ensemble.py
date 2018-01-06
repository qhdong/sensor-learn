import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report

from utils.log import setup_logging
from sensor import load_dataset
from utils.supervise import plot_learning_curve, plot_loss

# 配置日志
################################################################################
setup_logging()
logger = logging.getLogger(__name__)

# 载入数据集
################################################################################
sensor_data = load_dataset()
data = sensor_data['data']
target = sensor_data['target']
n_samples = len(data)
n_feature = len(data[0])

# 预处理
################################################################################
X, y = sensor_data['data'], sensor_data['target']

# 处理NaN
imp = Imputer()
imp = imp.fit(data)
X = imp.transform(X)

# 归一化
X = StandardScaler().fit_transform(X)

# 分配训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 配置分类器
################################################################################
Classifer = namedtuple('Classifer', ['name', 'clf'])
classifiers = [
    # Classifer("Linear SVM", SVC(C=0.025)),
    Classifer("RBF SVM 1", SVC()),
    Classifer("RBF SVM 2", SVC()),
    Classifer("RBF SVM 3", SVC()),
    # Classifer("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
    # Classifer("Decision Tree", DecisionTreeClassifier(max_depth=5)),
    # Classifer("Random Forest", RandomForestClassifier(n_jobs=-1, verbose=True)),
    # Classifer("AdaBoost", AdaBoostClassifier()),
    # Classifer("Neural Net", MLPClassifier(activation='relu', max_iter=400, solver='adam', verbose=True)),
    # classifer("Naive Bayes", GaussianNB()),
    # classifer("QDA", QuadraticDiscriminantAnalysis()),
]


def ensemble_learn():
    logger.info("集成学习")
    kfold = KFold(n_splits=10, random_state=42)
    estimators = [(name, clf) for name, clf in classifiers]
    ensemble = VotingClassifier(estimators)
    results = cross_val_score(ensemble, X, y, cv=kfold)
    print(results.mean())

# 训练
################################################################################
def train_with_supervise_chart():
    for name, clf in classifiers:
        logger.info("正在训练分类器[%s]", name)

        title = "Learning Curves (%s)" % name
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

        t = time.time()
        plot_learning_curve(clf, title, X, y, cv=cv, n_jobs=4)
        elapsed_time = time.time() - t
        logger.info("分类器[%s] 训练完成，花费时间 %.2s", name, elapsed_time)

        plt.show()


def train():

    scores = []
    steps = range(100, 500, 10)
    for iter in steps:

        clf = MLPClassifier(activation='relu', max_iter=iter, solver='adam', verbose=True)

        t = time.time()
        clf.fit(X_train, y_train)
        elapsed_time = time.time() - t

        # logger.info("分类器[%s] 训练完成，花费时间 %.2s", name, elapsed_time)
        score = clf.score(X_test, y_test)
        scores.append(score)

        # logger.info("分类器 [%s] 得分: %s", name, score)
        # summary.append([name, score, elapsed_time])
    # logger.info("训练结果如下：\r%s", pd.DataFrame(data=summary, columns=['name', 'score', 'time']))
    plt.figure()
    plt.plot(steps, scores)
    plt.show()


def train_and_show_table():
    summary = []
    for name, clf in classifiers:
        logger.info("正在训练分类器[%s]", name)
        t = time.time()
        clf.fit(X_train, y_train)
        elapsed_time = time.time() - t

        # 绘制loss
        if isinstance(clf, MLPClassifier):
            plot_loss(name, clf)
            plt.show()

        logger.info("分类器[%s] 训练完成，花费时间 %.2s", name, elapsed_time)
        score = clf.score(X_test, y_test)
        predictions = clf.predict(X_test)
        logger.info("分类器 [%s] 得分: %s", name, score)
        logger.info("测试类结果如下：\r%s", classification_report(y_test, predictions))
        summary.append([name, score, elapsed_time])
    logger.info("训练结果如下：\r%s", pd.DataFrame(data=summary, columns=['name', 'score', 'time']))


if __name__ == '__main__':
    # train_and_show_table()
    ensemble_learn()
    # train_with_supervise_chart()
    # train()
