import random
import pymongo
import logging
import pandas as pd
import numpy as np
import os.path
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.interpolate import interp1d
from tqdm import tqdm
from pymongo import MongoClient
from datetime import datetime


# rfc3339 时间格式
TIME_FMT = '%Y-%m-%dT%H:%M:%S.%fZ'

DATASET_FILENAME = "sensor_dataset.npz"
LSTM_DATASET_FILENAME = "raw_dataset.npz"

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


def normalize(samples):
    """归一化预处理"""
    # return (samples - samples.mean()) / samples.std()
    return (samples - samples.mean()) / (samples.max() - samples.min())
    # standard_scaler = preprocessing.StandardScaler()
    # return pd.DataFrame(standard_scaler.fit_transform(samples.T).T,
    #                     columns=samples.columns,
    #                     index=samples.index)


def get_key_time_by_sampleid(sampleid):
    """通过sampleID获取按键的时间戳信息"""
    logger.info('find key timestamp with sampleID %s', sampleid)
    cursor = db.key.find(
        {'sampleID': sampleid},
        {'time': 1, 'keyIndex': 1}
    ).sort([
        ('time', pymongo.ASCENDING)
    ])

    ts = []
    for doc in cursor:
        ts.append(datetime.strptime(doc['time'], TIME_FMT))
    return ts


def get_max_sample_length():
    """获取 0-9 PIN码的最长采样点数"""
    return max([get_sample_length_by_pin(pin)['len'].max() for pin in range(10)])


def get_samples_by_sampleid(sampleid):
    """根据sampleID组装3种传感器数据并返回Pandas DataFrame"""

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

    dates = []
    samples = []
    columns = ['acc-x', 'acc-y', 'acc-z', 'gacc-x', 'gacc-y', 'gacc-z', 'rot-alpha', 'rot-beta', 'rot-gamma',
               'ox-gamma', 'oy-beta', 'oz-alpha']
    for acc, ori in zip(acc_data, ori_data):
        raw_data = dict(acc['data'], **ori['data'])
        samples.append([raw_data[field] for field in columns])
        dates.append(datetime.strptime(acc['time'], TIME_FMT))

    # data = normalize(np.array(samples))
    data = np.array(samples)
    df = pd.DataFrame(data, index=dates, columns=columns)
    # df.drop(columns=['gacc-x', 'gacc-y', 'gacc-z'])
    return df


def get_sample_length_by_pin(pin):
    """获取PIN码的所有采样的长度"""
    cur = db.sensor.aggregate([
        {
            "$match": {"pin": str(pin)}
        },
        {
            "$group": {
                "_id": "$sampleID",
                "total": {"$sum": 1}
            }
        }
    ])
    data = []
    for doc in cur:
        sample = [str(doc['_id']), doc['total']]
        data.append(sample)
    return pd.DataFrame(data, columns=['sampleID', 'len'])


def get_a_sample_by_pin(pin):
    """通过pin随机获取一组原始数据"""
    sample_ids = get_sampleid_by_pin(pin)
    return get_samples_by_sampleid(random.choice(sample_ids))


def extract_feature_vector(samples):
    """获取所有的特征向量"""

    samples = normalize(samples)
    return pd.concat([
        extract_column_feature_vector(samples),
        extract_polynomial_feature(samples),
        extract_matrix_feature_vector(samples),
        extract_corr_feature_vector(samples)
    ])


def extract_polynomial_feature(samples, degree=5):
    """获取多项式拟合系数作为特征参数"""

    t = time.time()
    n_sensor = samples.shape[1]
    X = np.array(range(samples.shape[0]))

    features = []
    for col in range(n_sensor):
        try:
            p = pd.Series(np.polyfit(X, samples.iloc[:, col], degree))
        except ValueError:
            logger.exception("多项式拟合错误", exc_info=True)
            p = pd.Series([None for x in range(degree+1)])
        features.append(p)

    logger.info("提取多项式特征, 花费时间：%.2fs", time.time() - t)
    return pd.concat(features)


def extract_column_feature_vector(samples):
    """从n行9列数据中提取出特征，返回128维特征向量"""

    t = time.time()

    feature_max = samples.max()

    feature_min = samples.min()

    feature_mean = samples.mean()

    feature_var = samples.var()

    feature_std = samples.std()

    feature_rms = np.sqrt(np.square(samples).mean())

    feature_energy = np.square(samples).sum()

    # 曲线的锐度
    feature_kurtosis = samples.kurt()

    # 统计分布的不对称性
    feature_skewness = samples.skew()

    logger.info("提取统计特征, 花费时间：%.2fs", time.time() - t)

    return pd.concat([feature_max, feature_min, feature_mean, feature_var,
                      feature_rms, feature_std, feature_energy,
                      feature_kurtosis, feature_skewness])


def get_samples_total_number_by_pin(pins=None):
    """根据PIN获取MongoDB中的样本总数"""
    if pins is None:
        pins = range(10)
    return np.array([get_sample_length_by_pin(pin)['len'].size for pin in pins])


def extract_matrix_feature_vector(samples):
    """提取样本中的矩阵特征"""

    t = time.time()

    acc = samples.loc[:, ['acc-x', 'acc-y', 'acc-z']]
    rot = samples.loc[:, ['rot-alpha', 'rot-beta', 'rot-gamma']]
    ori = samples.loc[:, ['ox-gamma', 'oy-beta', 'oz-alpha']]

    # 矩阵中每列元素的绝对值求和后的最大值
    feature_1_norm = pd.Series([
        acc.abs().sum().max(),
        rot.abs().sum().max(),
        ori.abs().sum().max()
    ])

    # Infinity范数（矩阵每行元素绝对值之和的最大值）
    feature_inifinity_norm = pd.Series([
        acc.abs().sum(axis=1).max(),
        rot.abs().sum(axis=1).max(),
        ori.abs().sum(axis=1).max()
    ])

    # Frobenius范数（矩阵中所有元素的平方和的平方根）
    feature_frobenius_norm = pd.Series([
        np.sqrt(np.square(acc).values.sum()),
        np.sqrt(np.square(rot).values.sum()),
        np.sqrt(np.square(ori).values.sum())
    ])

    logger.info("提取相关性特征, 花费时间：%.2fs", time.time() - t)

    return pd.concat([feature_1_norm, feature_inifinity_norm, feature_frobenius_norm])


def extract_corr_feature_vector(samples):
    """提取样本中的相关性特征"""
    x = samples.loc[:, ['acc-x', 'rot-alpha', 'ox-gamma']]
    y = samples.loc[:, ['acc-y', 'rot-beta', 'oy-beta']]
    z = samples.loc[:, ['acc-z', 'rot-gamma', 'oz-alpha']]

    acc = samples.loc[:, ['acc-x', 'acc-y', 'acc-z']]
    rot = samples.loc[:, ['rot-alpha', 'rot-beta', 'rot-gamma']]
    ori = samples.loc[:, ['ox-gamma', 'oy-beta', 'oz-alpha']]

    def make_cor_feature(cor):
        return pd.concat([cor.iloc[0, 1:], cor.iloc[1, 2:]])

    features = pd.concat([
        make_cor_feature(x.corr()),
        make_cor_feature(y.corr()),
        make_cor_feature(z.corr()),
        make_cor_feature(acc.corr()),
        make_cor_feature(rot.corr()),
        make_cor_feature(ori.corr()),
    ])
    logger.debug("获取 %s 维相关性特征", features.size)
    return features


def generate_dataset_from_mongo():
    """从mongodb数据库获取label:feature格式的所有样本的特征向量"""
    pins = list(range(10))
    data = []
    target = []

    begin_time = time.time()
    for pin in pins:
        start = time.time()
        logger.info("processing pin: %s", pin)
        n_samples = len(get_sampleid_by_pin(pin))
        for count, sampleid in enumerate(get_sampleid_by_pin(pin)):
            samples = get_samples_by_sampleid(sampleid)
            # 扔掉含有NaN的样本
            if samples.isnull().values.any():
                continue
            t = time.time()
            feature_vector = extract_feature_vector(samples)
            logger.info("提取特征，PIN: %s, 花费时间：%.2f，进度[%4d/%4d]", pin, time.time() - t, count, n_samples)
            data.append(feature_vector.values)
            target.append(pin)
        elapsed_time = time.time() - start
        logger.info("Get %s samples for pin: %s, Time elapsed: %.2s", n_samples, pin, elapsed_time)

    total_time = time.time() - begin_time
    logger.info("Samples: %s, Feature length: %s, Time elapsed %s", len(data), len(data[0]), total_time)
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


def load_lstm_data(path=LSTM_DATASET_FILENAME):
    """载入lstm格式要求的数据，默认从硬盘读取，如果没有则生成"""
    if os.path.exists(path):
        logger.info("从文件读取LSTM格式的数据集: %s", path)
        npzfile = np.load(path)
        return npzfile['X'], npzfile['y']
    else:
        logger.info("没有找到文件 %s, 从MongoDB中生成数据", path)
        X, y = generate_lstm_data()
        logger.info("将数据保存到文件 %s", path)
        np.savez(path, X=X, y=y)
        return X, y


def generate_lstm_data():
    """以LSTM要求的格式整合数据"""
    X = []
    y = []

    LOW, HIGH = 30, 128
    logger.info("选取在[%d, %d]范围内的样本，并补齐到：%d", LOW, HIGH, HIGH)

    total_samples = get_samples_total_number_by_pin().sum()
    pbar = tqdm(total=total_samples)
    for pin in range(10):
        start = time.time()
        logger.info("正在处理PIN：%s 的样本", pin)

        # 组装 PAD_LEN * 12 的矩阵
        sample_ids = get_sampleid_by_pin(pin)
        total_ids = len(sample_ids)
        for count, sample_id in enumerate(sample_ids):
            # logger.info("正在处理sampleID：%s, 进度[%3d/%3d]", sample_id, count, total_ids)
            pbar.update(1)
            samples = get_samples_by_sampleid(sample_id)
            row = samples.shape[0]
            col = samples.shape[1]
            if row < LOW or row > HIGH:
                logger.info("该样本采样数量异常 %d 不在范围[%d,%d]内，抛弃", row, LOW, HIGH)
                continue

            # 对每一列数据进行归一化
            acc_norm = normalize(samples.loc[:, ['acc-x', 'acc-y', 'acc-z', 'gacc-x', 'gacc-y', 'gacc-z', 'rot-alpha', 'rot-beta', 'rot-gamma']])
            ori_norm = samples.loc[:, ['ox-gamma', 'oy-beta', 'oz-alpha']] / 360.0
            samples_norm = np.concatenate((acc_norm, ori_norm.values), axis=1)

            # 补齐长度
            pad = np.zeros((HIGH - row, col))
            data = np.concatenate((samples_norm, pad))
            X.append(data)
            y.append(pin)
        logger.info("PIN: %s 处理完成，共有%3d个样本，花费时间：%.2fs", pin, total_ids, time.time() - start)

    pbar.close()
    return np.array(X), np.array(y)

