import logging
import time
import pandas as pd
import matplotlib.pyplot as plt

from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit

from utils.log import setup_logging
from feature import load_dataset
from utils.supervise import plot_learning_curve


# 配置日志
setup_logging()
logger = logging.getLogger(__name__)

# 载入数据集
sensor_data = load_dataset()
data = sensor_data['data']
# data = sensor_data['data'][:, 24:]
target = sensor_data['target']
n_samples = len(data)

# 配置分类器
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

Classifer = namedtuple('Classifer', ['name', 'clf'])
classifiers = [
    # Classifer("Nearest Neighbors", KNeighborsClassifier()),
    Classifer("Linear SVM", SVC(C=0.025)),
    # classifer("RBF SVM", SVC(gamma=2, C=1)),
    # Classifer("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
    Classifer("Decision Tree", DecisionTreeClassifier(max_depth=5)),
    Classifer("Random Forest", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    Classifer("Neural Net", MLPClassifier(solver='lbfgs', alpha=1)),
    # classifer("AdaBoost", AdaBoostClassifier()),
    # classifer("Naive Bayes", GaussianNB()),
    # classifer("QDA", QuadraticDiscriminantAnalysis()),
    ]

# 预处理数据，归一化
X, y = sensor_data['data'], sensor_data['target']
# X = StandardScaler().fit_transform(X)

# 分配训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练
columns = ['name', 'score', 'time']
summary = []
for name, clf in classifiers:
    title = "Learning Curves (%s)" % name
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    plot_learning_curve(clf, title, X, y, cv=cv, n_jobs=4)

    plt.show()

#     logger.info("正在训练分类器[%s]", name)
#     t = time.perf_counter()
#     clf.fit(X_train, y_train)
#     elapsed_time = time.perf_counter() - t
#
#     logger.info("分类器[%s] 训练完成，花费时间 %.2s", name, elapsed_time)
    score = clf.score(X_test, y_test)
#     logger.info("分类器 [%s] 得分: %s", name, score)
#     summary.append([name, score, elapsed_time])
#
# logger.info(pd.DataFrame(data=summary, columns=columns))
