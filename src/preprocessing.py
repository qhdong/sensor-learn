import pymongo
import logging
from pymongo import MongoClient
from .utils.log import setup_logging

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# connect to MongoDB
client = MongoClient()

# select the database
db = client.sensor

# get the collection
coll = db.sensor


def find_sampleid_by_pin(pin):
    """根据PIN获取无重复的sampleID"""
    cursor = db.sensor.find(
        {'pin': str(pin)}
    )
    return cursor.distinct('sampleID')


def get_samples_by_sampleid(sampleid):
    """根据sampleID组装4种传感器数据并返回"""

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

    for acc, ori in zip(acc_data, ori_data):
        data = {
            'time': acc['time'],
            'data': dict(acc['data'], **ori['data'])
        }
        print(data)


if __name__ == '__main__':
    setup_logging()
    # for pin in range(0, 10):
    #     print(len(find_sampleid_by_pin(pin)))
    sampleid = find_sampleid_by_pin(0)[0]
    # for sampleid in find_sampleid_by_pin(0):
    #     if len(list(get_samples_by_sampleid(sampleid))) % 2 != 0:
    #         print('error, is odd, sampleid = %s' % sampleid)
    samples = get_samples_by_sampleid(sampleid)
    # print(list(samples))
