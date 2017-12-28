import pymongo
import logging
import pandas as pd
import numpy as np
import os.path
import time
from sklearn import preprocessing
from pymongo import MongoClient
from datetime import datetime


DATASET_FILENAME = "sensor_dataset.npz"

logger = logging.getLogger(__name__)

# connect to MongoDB
client = MongoClient()

# select the database
db = client.sensor

# get the collection
coll = db.sensor


def get_sampleid_by_pin(pin):
    """根据PIN获取无重复的sampleID"""
    cursor = db.sensor.find(
        {'pin': str(pin)}
    )
    return cursor.distinct('sampleID')


def normalize(data):
    """归一化预处理"""
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    return np_scaled


def get_samples_by_sampleid(sampleid):
    """根据sampleID组装4种传感器数据并返回Pandas DataFrame"""

    logger.info('find samples with simpleid %s', sampleid)

    cursor_acc = db.sensor.find(
        {'sampleID': sampleid, 'data.acc-x': {'$exists': True}},
        {'data': 1, 'time': 1}
    ).sort([
        ('time', pymongo.ASCENDING)
    ])

    cursor_ori = db.sensor.find(
        {'sampleID': sampleid, 'data.ox-gamma': {'$exists': True}},
        {'data': 1, 'time': 1}
    ).sort([
        ('time', pymongo.ASCENDING)
    ])

    acc_data = (sample for sample in cursor_acc)
    ori_data = (sample for sample in cursor_ori)

    # rfc3339 时间格式
    fmt = '%Y-%m-%dT%H:%M:%S.%fZ'

    dates = []
    samples = []
    columns = ['acc-x', 'acc-y', 'acc-z', 'gacc-x', 'gacc-y', 'gacc-z', 'rot-alpha', 'rot-beta', 'rot-gamma',
               'ox-gamma', 'oy-beta', 'oz-alpha']
    for acc, ori in zip(acc_data, ori_data):
        raw_data = dict(acc['data'], **ori['data'])
        samples.append([raw_data[field] for field in columns])
        dates.append(datetime.strptime(acc['time'], fmt))

    # data = normalize(np.array(samples))
    data = np.array(samples)
    df = pd.DataFrame(data, index=dates, columns=columns)
    return df


def extract_feature_vector(samples):
    return pd.concat([
        extract_column_feature_vector(samples),
        extract_matrix_feature_vector(samples),
        extract_corr_feature_vector(samples)
    ])


def extract_column_feature_vector(samples):
    """从n行12列数据中提取出特征，返回128维特征向量"""

    feature_max = samples.max()

    feature_min = samples.min()

    feature_mean = samples.mean()

    feature_var = samples.var()

    feature_rms = np.sqrt(np.square(samples).mean())

    feature_energy = np.square(samples).sum()

    # 曲线的锐度
    feature_kurtosis = samples.kurt()

    # 统计分布的不对称性
    feature_skewness = samples.skew()

    return pd.concat([feature_max, feature_min, feature_mean, feature_var, feature_rms,
                      feature_energy, feature_kurtosis, feature_skewness])


def extract_matrix_feature_vector(samples):
    """提取样本中的矩阵特征"""

    acc = samples.loc[:, ['acc-x', 'acc-y', 'acc-z']]
    gacc = samples.loc[:, ['gacc-x', 'gacc-y', 'gacc-z']]
    rot = samples.loc[:, ['rot-alpha', 'rot-beta', 'rot-gamma']]
    ori = samples.loc[:, ['ox-gamma', 'oy-beta', 'oz-alpha']]

    # 矩阵中每列元素的绝对值求和后的最大值
    feature_1_norm = pd.Series([
        acc.abs().sum().max(),
        gacc.abs().sum().max(),
        rot.abs().sum().max(),
        ori.abs().sum().max()
    ])

    # Infinity范数（矩阵每行元素绝对值之和的最大值）
    feature_inifinity_norm = pd.Series([
        acc.abs().sum(axis=1).max(),
        gacc.abs().sum(axis=1).max(),
        rot.abs().sum(axis=1).max(),
        ori.abs().sum(axis=1).max()
    ])

    # Frobenius范数（矩阵中所有元素的平方和的平方根）
    feature_frobenius_norm = pd.Series([
        np.sqrt(np.square(acc).values.sum()),
        np.sqrt(np.square(gacc).values.sum()),
        np.sqrt(np.square(rot).values.sum()),
        np.sqrt(np.square(ori).values.sum())
    ])

    return pd.concat([feature_1_norm, feature_inifinity_norm, feature_frobenius_norm])


def extract_corr_feature_vector(samples):
    """提取样本中的相关性特征"""
    x = samples.loc[:, ['acc-x', 'gacc-x', 'rot-alpha', 'ox-gamma']]
    y = samples.loc[:, ['acc-y', 'gacc-y', 'rot-beta', 'oy-beta']]
    z = samples.loc[:, ['acc-z', 'gacc-z', 'rot-gamma', 'oz-alpha']]

    def make_cor_feature(cor):
        return pd.concat([cor.iloc[0, 1:], cor.iloc[1, 2:], cor.iloc[2, 3:]])

    return pd.concat([
        make_cor_feature(x.corr()),
        make_cor_feature(y.corr()),
        make_cor_feature(z.corr()),
    ])


def generate_dataset_from_mongo():
    """从mongodb数据库获取label:feature格式的所有样本的特征向量"""
    pins = list(range(10))
    data = []
    target = []

    t = time.process_time()
    for pin in pins:
        logger.info("processing pin: %s", pin)
        count = 0
        for sampleid in get_sampleid_by_pin(pin):
            samples = get_samples_by_sampleid(sampleid)
            feature_vector = extract_feature_vector(samples)
            data.append(feature_vector.values)
            target.append(pin)
            count += 1
        logger.info("Get %s samples for pin: %s", count, pin)

    elapsed_time = time.process_time() - t
    logger.info("Samples: %s, Feature length: %s, Time elapsed %s", len(data), len(data[0]), elapsed_time)
    return {
        'data': np.array(data),
        'target': np.array(target)
    }


def load_dataset():
    """获取数据集，默认从文件中读取处理好的数据，如果没有就从数据库读取，并保存到文件"""
    if os.path.exists(DATASET_FILENAME):
        logger.info("load dataset from file: %s", DATASET_FILENAME)
        npzfile = np.load(DATASET_FILENAME)
        return {
            'data': npzfile['data'],
            'target': npzfile['target']
        }
    else:
        logger.info("dataset dosen't exist in disk, generate from mongodb instead, it may take some time.")
        dataset = generate_dataset_from_mongo()
        logger.info("save dataset to disk file %s", DATASET_FILENAME)
        np.savez(DATASET_FILENAME, data=dataset['data'], target=dataset['target'])
        return dataset

