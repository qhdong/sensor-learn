import logging

import random
import os.path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pymongo import MongoClient
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.interpolate import interpolate
from sklearn.preprocessing import normalize

import sensor
from utils.log import setup_logging

DATASET_FILENAME = "sensor_dataset.npz"

logger = logging.getLogger(__name__)

# 连接数据库
################################################################################
client = MongoClient()

# select the database
db = client.sensor

# get the collection
coll = db.sensor

# 从文件中读取特征向量
################################################################################
npzfile = np.load(DATASET_FILENAME)
data = npzfile['data']
target = npzfile['target']


def get_features_by_pin(pin):
    """获取pin的特征向量"""
    return data[target == pin]


def print_feature(pin=0, row=0):
    """打印某个Key的某一行特征向量"""


def plot_corr(x, y):
    """绘制两个pin的特征向量的相关系数矩阵"""
    feature_x = get_features_by_pin(x)
    feature_y = get_features_by_pin(y)

    plt.figure()
    plt.suptitle("corr matrix for %s and %s" % (x, y))
    plt.matshow(np.corrcoef(feature_x, feature_y))
    plt.show()


def plot_raw_data_split(pin, sampleid=None, sensor_list=None, sensor_axis=None):
    """绘制原始采集数据的图像"""
    if sampleid is None:
        sampleid = random.choice(sensor.get_sampleid_by_pin(pin))
        logger.info("没有提供sampleID，随机选取 %s", sampleid)
    samples = sensor.get_samples_by_sampleid(sampleid)
    key_times = sensor.get_key_time_by_sampleid(sampleid)
    logger.info("PIN:%s 按键的时间戳为：%s", pin, key_times)

    # 绘制图像
    if sensor_list is None:
        sensor_list = ['acc']
    if sensor_axis is None:
        sensor_axis = ['x']

    def make_label(sensor_name, sensor_axis):
        rot_map = {
            'x': 'rot-alpha',
            'y': 'rot-beta',
            'z': 'rot-gamma'
        }
        ori_map = {
            'x': 'ox-gamma',
            'y': 'oy-beta',
            'z': 'oz-alpha',
        }

        if sensor_name == 'acc':
            return ['acc-' + axis for axis in sensor_axis]
        elif sensor_name == 'gacc':
            return ['gacc-' + axis for axis in sensor_axis]
        elif sensor_name == 'rot':
            return [rot_map[axis] for axis in sensor_axis]
        elif sensor_name == 'ori':
            return [ori_map[axis] for axis in sensor_axis]
        else:
            raise ValueError("传感器名称不正确，目前只支持 acc, gacc, rot, ori")

    x_time = samples.index
    logger.info("传感器数据时间轴信息：%s", x_time)

    # 行是传感器数量，列是轴的数量
    row, col = len(sensor_list), len(sensor_axis)
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Sensor Data for PIN %s" % pin)
    for index, sensor_name in enumerate(sensor_list):
        for i, label in enumerate(make_label(sensor_name, sensor_axis)):
            plt.subplot(row, col, col * index + i + 1)
            plt.plot(x_time, samples[label])
            plt.title("%s sensor data" % label)
            plt.xticks(key_times, ['down', 'up'] * 4)

    plt.tight_layout()
    logger.info("绘制传感器 %s 三个轴的图像", sensor_list)
    plt.show()


def plot_raw_data(pin, sampleid=None, sensor_list=None, sensor_axis=None):
    """绘制原始采集数据的图像,三个轴在一个图内"""
    if sampleid is None:
        sampleid = random.choice(sensor.get_sampleid_by_pin(pin))
        logger.info("没有提供sampleID，随机选取 %s", sampleid)
    samples = sensor.get_samples_by_sampleid(sampleid)
    key_times = sensor.get_key_time_by_sampleid(sampleid)
    logger.info("PIN:%s 按键的时间戳为：%s", pin, key_times)

    # 绘制图像
    if sensor_list is None:
        sensor_list = ['acc']
    if sensor_axis is None:
        sensor_axis = ['x']

    def make_label(sensor_name, sensor_axis):
        rot_map = {
            'x': 'rot-alpha',
            'y': 'rot-beta',
            'z': 'rot-gamma'
        }
        ori_map = {
            'x': 'ox-gamma',
            'y': 'oy-beta',
            'z': 'oz-alpha',
        }

        if sensor_name == 'acc':
            return ['acc-' + axis for axis in sensor_axis]
        elif sensor_name == 'gacc':
            return ['gacc-' + axis for axis in sensor_axis]
        elif sensor_name == 'rot':
            return [rot_map[axis] for axis in sensor_axis]
        elif sensor_name == 'ori':
            return [ori_map[axis] for axis in sensor_axis]
        else:
            raise ValueError("传感器名称不正确，目前只支持 acc, gacc, rot, ori")

    x_time = samples.index
    logger.info("传感器数据时间轴信息：%s", x_time)

    # 行是传感器数量，列是轴的数量
    row, col = 1, len(sensor_list)
    fig = plt.figure(figsize=(12, 6))
    # fig.suptitle("Sensor Data for PIN %s" % pin)
    for index, sensor_name in enumerate(sensor_list):
        plt.subplot(row, col, col * index + 1)
        for i, label in enumerate(make_label(sensor_name, sensor_axis)):
            plt.plot(x_time, samples[label])
            # plt.title("%s sensor data" % label)
            plt.xticks(key_times, ['down', 'up'] * 4)

    plt.tight_layout()
    logger.info("绘制传感器 %s 三个轴的图像", sensor_list)
    plt.show()


def plot_polynomial_fit(pin, label='oz-alpha'):
    """对原始数据进行不同维度的多项式拟合，并绘制图形"""
    samples = sensor.get_a_sample_by_pin(pin)
    y = samples[label]
    X = np.array(range(len(y)))
    X = X[:, np.newaxis]
    plt.scatter(X, y, color='navy', s=30, marker='o', label='training points')

    colors = ['teal', 'yellowgreen', 'gold']
    for count, degree in enumerate([3, 4, 5]):
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X, y)
        y_plot = model.predict(X)
        plt.plot(X, y_plot, color=colors[count], linewidth=2, label="degree %d" % degree)

    plt.legend(loc='lower left')
    plt.show()


def describe_sample_length():
    """画出PIN码的每次采样的点数"""
    for pin in range(3):
        sample_len_df = sensor.get_sample_length_by_pin(pin)
        fig, ax = plt.subplots()
        ax.hist(sample_len_df['len'], 50)
        ax.set_title("sample length")
        plt.show()
        logger.info("PIN: %s 平均长度：%s", pin, sample_len_df['len'].mean)


def get_max_pin_length():
    lens = pd.concat(sensor.get_sample_length_by_pin(pin)['len']

                     for pin in range(10))
    plt.hist(lens, 50)
    plt.show()


def main():
    # plot_raw_data("6316", sensor_list=['gacc'], sensor_axis=['x', 'y', 'z'])
    # plot_raw_data("6316", sensor_list=['rot'], sensor_axis=['x', 'y', 'z'])
    # plot_raw_data("6316", sensor_list=['ori'], sensor_axis=['x', 'y', 'z'])
    # plot_raw_data("1021", sensor_list=['acc'], sensor_axis=['x', 'y', 'z'])
    # plot_raw_data("1", sensor_list=['acc'], sensor_axis=['x', 'y', 'z'])
    # plot_raw_data("1", sensor_list=['acc'], sensor_axis=['x', 'y', 'z'])
    # plot_raw_data("1", sensor_list=['ori'], sensor_axis=['x', 'y', 'z'])
    # plot_raw_data("1", sensor_list=['rot'], sensor_axis=['x', 'y', 'z'])
    # plot_raw_data("2", sensor_list=['ori'], sensor_axis=['x', 'y', 'z'])
    # plot_raw_data("1021", sensor_list=['acc', 'rot', 'ori'])
    # plot_raw_data(1, sensor_list=['acc', 'gacc', 'rot', 'ori'])
    # plot_raw_data(2, sensor_list=['acc', 'gacc', 'rot', 'ori'])
    # plot_corr(0, 1)
    # sample = sensor.get_a_sample_by_pin("1")
    plot_polynomial_fit(6316, 'acc-x')
    # describe_sample_length()
    # print(sensor.get_sample_length_by_pin(0))
    # X, y = sensor.load_lstm_data()
    # get_max_pin_length()



if __name__ == '__main__':
    setup_logging(default_path="../logging.json")
    main()
    # X = sensor.get_a_sample_by_pin(0)
    # X_norm = normalize(X, axis=0)
    # X, y = sensor.load_lstm_data()

