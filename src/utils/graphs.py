"""绘制论文中的所有图表"""
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sensor
import feature
from utils.log import setup_logging


def plot_pin_samples_length_hist():
    """绘制采集到的样本数据点数直方图"""
    lens = pd.concat(sensor.get_sample_length_by_pin(pin)['len']
                     for pin in range(10))
    plt.hist(lens, 100)
    plt.xlabel('sample length')
    plt.show()


def plot_raw_acc_data():
    samples = sensor.get_a_sample_by_pin(6316)
    y = samples['acc-x']
    x = range(y)
    plt.scatter(x, y, color='navy', s=30, marker='o', label='training points')
    plt.show()


def main():
    plot_pin_samples_length_hist()
    plot_raw_acc_data()


if __name__ == '__main__':
    setup_logging(default_path="../logging.json")
    main()
