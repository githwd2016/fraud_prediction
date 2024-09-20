# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: preprocess.py
@Time: 2018/5/17 下午9:24
@Description:
"""
import json
import pickle
import collections

import pandas as pd
import numpy as np
# import pymongo
# from sklearn import preprocessing
from tqdm import tqdm

import setting


def one_hot(index, depth):
    temp = [0] * depth
    temp[index] = 1
    return temp


def get_logistic_data(path,
                      field,
                      train_start_year,
                      train_end_year,
                      test_start_year,
                      test_end_year):
    data = pd.read_csv(path, sep=',', header=0)
    data['year_copy'] = data['year']
    data = pd.get_dummies(data[field + ['DV_ICW', 'year_copy']], columns=['year', 'Industry'])
    train_data = data[(data.year_copy >= train_start_year) & (data.year_copy <= train_end_year)]
    test_data = data[(data.year_copy >= test_start_year) & (data.year_copy <= test_end_year)]
    x_field = list(train_data.columns)
    x_field.remove('DV_ICW')
    x_field.remove('year_copy')
    train_x = train_data[x_field]
    train_y = train_data['DV_ICW']
    test_x = test_data[x_field]
    test_y = test_data['DV_ICW']
    return train_x, train_y, test_x, test_y


def get_crf_data(path, field, end_year, test=False):
    data = pd.read_csv(path, sep=',', header=0)
    data = pd.get_dummies(data[field + ['DV_ICW', 'stkcd']], columns=['Industry'])
    data_group = data.groupby('stkcd')
    x_field = list(data.columns)
    x_field.remove('DV_ICW')
    x_field.remove('year')
    x_field.remove('stkcd')
    print(len(x_field))
    if test:
        max_len = end_year - 2006 + 1
    else:
        max_len = end_year - 2006 + 2
    x = []
    y = []
    seq_len = []
    for name, value in data_group:
        if test and value[value.year == end_year].empty:
            continue
        temp = value[value.year <= end_year]
        if not temp.empty:
            temp_x = np.array(temp[x_field])
            temp_y = np.array(temp['DV_ICW'])
            len_ = temp_x.shape[0]
            zeros = np.zeros((max_len - len_, temp_x.shape[1]))
            temp_x = np.concatenate((temp_x, zeros), axis=0)
            zero = np.zeros(max_len - len_)
            temp_y = np.concatenate((temp_y, zero), axis=0)
            x.append(temp_x)
            y.append(temp_y)
            seq_len.append(len_)
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int32), np.array(seq_len, dtype=np.int32)


def get_svm(path,
            field,
            train_start_year,
            train_end_year,
            test_start_year,
            test_end_year):
    data = pd.read_csv(path, sep=',', header=0)
    data['year_copy'] = data['year']
    data = pd.get_dummies(data[field + ['DV_ICW', 'year_copy', 'stkcd']], columns=['year', 'Industry'])
    train_data = data[(data.year_copy >= train_start_year) & (data.year_copy <= train_end_year)]
    test_data = data[(data.year_copy >= test_start_year) & (data.year_copy <= test_end_year)]
    x_field = list(train_data.columns)
    x_field.remove('DV_ICW')
    x_field.remove('year_copy')
    x_field.remove('stkcd')
    train_f = train_data[x_field]
    train_l = train_data['DV_ICW']
    train_g = train_data['stkcd']
    test_f = test_data[x_field]
    test_l = test_data['DV_ICW']
    test_g = test_data['stkcd']
    return np.array(train_f), train_g, train_l, np.array(test_f), test_g, test_l


def get_paiwise_year():
    ids = {2006: {1: [], 0: []},
           2007: {1: [], 0: []},
           2008: {1: [], 0: []},
           2009: {1: [], 0: []},
           2010: {1: [], 0: []},
           2011: {1: [], 0: []},
           2012: {1: [], 0: []},
           2013: {1: [], 0: []},
           2014: {1: [], 0: []},
           2015: {1: [], 0: []},
           }
    with pymongo.MongoClient(setting.HOST) as conn:
        conn['admin'].authenticate(setting.USER, setting.PASSWORD)
        db = conn['quexian']
        coll = db['feature1']
        coll2 = db['feature1_pair_year']
        for content in coll.find():
            ids[content['year']][content['label']].append(content['_id'])
        for year in ids:
            for pos in ids[year][1]:
                for neg in ids[year][0]:
                    coll2.insert({
                        'year': year,
                        'pos_id': pos,
                        'neg_id': neg
                    })


def get_paiwise_stkcd(year):
    ids = collections.defaultdict(dict)
    with pymongo.MongoClient(setting.HOST) as conn:
        conn['admin'].authenticate(setting.USER, setting.PASSWORD)
        db = conn['quexian']
        coll = db['feature1']
        coll2 = db['feature1_pair_stkcd_{}'.format(year)]
        for content in coll.find({'year': {'$lt': year}}):
            if content['label'] not in ids[content['stkcd']]:
                ids[content['stkcd']][content['label']] = [content['_id']]
            else:
                ids[content['stkcd']][content['label']].append(content['_id'])
        for stkcd in ids:
            if 1 in ids[stkcd] and 0 in ids[stkcd]:
                for pos in ids[stkcd][1]:
                    for neg in ids[stkcd][0]:
                        coll2.insert({
                            'stkcd': stkcd,
                            'pos_id': pos,
                            'neg_id': neg
                        })


def get_padding_data(path):
    data = pd.read_csv(path, sep=',', header=0)
    # del data['Unnamed: 0']
    data_group = data.groupby('stkcd')
    x_field = list(data.columns)
    x_field.remove('DV_ICW')
    x_field.remove('year')
    x_field.remove('stkcd')
    print(len(x_field))
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(data[x_field])
    with pymongo.MongoClient(setting.HOST) as conn:
        conn['admin'].authenticate(setting.USER, setting.PASSWORD)
        db = conn['quexian']
        coll = db['feature1_padding_5_norm']
        coll.remove()
        for name, value in tqdm(data_group):
            temp_data = []
            for year in range(2006, 2016):
                temp = value[value.year <= year]
                if len(temp) > 1:
                    temp = temp.sort_values(['year'], ascending=1)
                temp_x = min_max_scaler.transform(temp[x_field])
                temp_y = int(temp[temp.year == year]['DV_ICW'])
                temp_y_2 = np.array(temp[temp.year <= year]['DV_ICW'])
                seq_len = temp_x.shape[0]
                temp_data.append({
                    'stkcd': str(int(name)),
                    'year': year,
                    'seq_len': seq_len,
                    'feature': pickle.dumps(temp_x),
                    'label': temp_y,
                    'label2': pickle.dumps(temp_y_2)
                })
            coll.insert_many(temp_data)


def get_rnn_data(path, field):
    data = pd.read_csv(path, sep=',', header=0)
    data = pd.get_dummies(data[field + ['DV_ICW', 'stkcd']], columns=['Industry'])
    data_group = data.groupby('stkcd')
    x_field = list(data.columns)
    x_field.remove('DV_ICW')
    x_field.remove('year')
    x_field.remove('stkcd')
    print(len(x_field))
    # min_max_scaler = preprocessing.MinMaxScaler()
    # min_max_scaler.fit(data[x_field])
    with pymongo.MongoClient(setting.HOST) as conn:
        db = conn['quexian']
        coll = db['feature2_new']
        coll.remove()
        for name, value in tqdm(data_group):
            temp_data = []
            for year in range(2006, 2017):
                temp = value[value.year <= year]
                test = value[value.year == year]
                if not test.empty:
                    # temp_x = min_max_scaler.transform(temp[x_field])
                    temp_x = np.array(temp[x_field])
                    temp_y = int(temp[temp.year == year]['DV_ICW'])
                    temp_y_2 = np.array(temp[temp.year <= year]['DV_ICW'])
                    len_ = year - 2005 - temp_x.shape[0]
                    # padding
                    if len_ > 0:
                        temp_x = np.concatenate((np.zeros((len_, temp_x.shape[1])), temp_x))
                        temp_y_2 = np.concatenate((np.zeros(len_), temp_y_2))
                    seq_len = temp_x.shape[0]
                    temp_data.append({
                        'stkcd': str(name),
                        'year': year,
                        'seq_len': seq_len,
                        'feature': pickle.dumps(temp_x),
                        'label': temp_y,
                        'label2': pickle.dumps(temp_y_2)
                    })
            if temp_data:
                coll.insert_many(temp_data)


def get_rnn_data2(path, field):
    data = pd.read_csv(path, sep=',', header=0)
    data = pd.get_dummies(data[field + ['DV_ICW', 'stkcd']], columns=['Industry'])
    data_group = data.groupby('stkcd')
    print(list(data.columns))
    x_field = list(data.columns)
    x_field.remove('DV_ICW')
    x_field.remove('year')
    x_field.remove('stkcd')
    with pymongo.MongoClient(setting.HOST) as conn:
        db = conn['quexian']
        coll = db['feature2']
        coll.remove()
        for name, data_frame in tqdm(data_group):
            temp_data = {'stkcd': str(name), 'data': {}}
            for index, item in data_frame.iterrows():
                year = item['year']
                label = item['DV_ICW']
                feature = item[x_field]
                temp_data['data'].update({
                    str(int(year)): {
                        'label': label,
                        'feature': list(feature)
                    }
                })
            coll.insert_one(temp_data)


def get_rnn_json_data2(path, field, out_path):
    data = pd.read_csv(path, sep=',', header=0)
    data = pd.get_dummies(data[field + ['DV_ICW', 'stkcd']], columns=['Industry'])
    data_group = data.groupby('stkcd')
    print(list(data.columns))
    x_field = list(data.columns)
    x_field.remove('DV_ICW')
    x_field.remove('year')
    x_field.remove('stkcd')
    res = []
    for name, data_frame in tqdm(data_group):
        temp_data = {'stkcd': str(name), 'data': {}}
        for index, item in data_frame.iterrows():
            year = item['year']
            label = item['DV_ICW']
            feature = item[x_field]
            temp_data['data'].update({
                str(int(year)): {
                    'label': label,
                    'feature': list(feature)
                }
            })
        res.append(temp_data)
    with open(out_path, 'w') as f:
        json.dump(res, f)


if __name__ == '__main__':
    # get_rnn_data('data/ICD_dataset_180716.csv', field=setting.FEATURE2_NEW)
    # get_rnn_data2('data/ICD_dataset_180716.csv', field=setting.FEATURE1_NEW)
    # for end_year in [2011, 2012, 2013, 2014]:
    #     a, b, c, d, e, f = get_svm('data/ICD_dataset_180515.csv',
    #                                setting.FEATURE1,
    #                                2006,
    #                             pre   end_year,
    #                                end_year + 1,
    #                                end_year + 1)
    #     with open('data/{}.train.svm'.format(end_year + 1), 'w') as f1:
    #         for i, j, k in zip(a, b, c):
    #             f1.write('{} qid:{} '.format(int(k), int(j)))
    #             for index, value in enumerate(i):
    #                 f1.write('{}:{} '.format(index + 1, value))
    #             f1.write('#\n')
    #     with open('data/{}.test.svm'.format(end_year + 1), 'w') as f2:
    #         for i, j, k in zip(d, e, f):
    #             f2.write('{} qid:{} '.format(int(k), int(j)))
    #             for index, value in enumerate(i):
    #                 f2.write('{}:{} '.format(index + 1, value))
    #             f2.write('#\n')
    # get_paiwise_year()
    # get_paiwise_stkcd(2012)
    # get_padding_data('data/feature1_padding_5.csv')
    # get_rnn_json_data2('data/ICD_dataset_180716.csv', field=setting.FEATURE1_NEW, out_path='data/rnn_feature1_new.json')
    # get_rnn_json_data2('data/ICD_dataset_180716.csv', field=setting.FEATURE2_NEW, out_path='data/rnn_feature2_new.json')
    get_rnn_json_data2('data/ICD_dataset_240918_1700.csv',
                       field=setting.FEATURE1_NEW,
                       out_path='data/rnn_feature1_240918_1700.json')
