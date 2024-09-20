# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: rnn_with_json_data.py
@Time: 2024/2/26 15:14
@Description:
"""
import json
import os

import tensorflow as tf
import numpy as np
import photinia as ph

from evaluation import evaluation

# hyper parameter
tf.flags.DEFINE_string('gpu', '0', 'Choose which GPU to use.')
tf.flags.DEFINE_string('input_file', 'data/rnn_feature1_240918_1700.json', 'Input json file.')
tf.flags.DEFINE_string('unit', 'gru', 'Type of RNN unit.')
tf.flags.DEFINE_integer('emb', 200, 'Embedding size.')
tf.flags.DEFINE_integer('hidden', 300, 'Hidden layer size.')
tf.flags.DEFINE_integer('layers', 2, 'RNN layers.')
tf.flags.DEFINE_integer('batch', 256, 'Batch size.')
tf.flags.DEFINE_integer('loops', 1000, 'Num loops.')
tf.flags.DEFINE_integer('start', 2006, 'Train start year.')
tf.flags.DEFINE_integer('end', 2011, 'Train end year.')
tf.flags.DEFINE_float('threshold', 0.3, 'Threshold.')
tf.flags.DEFINE_float('keep', 0.9, 'Keep prob.')
tf.flags.DEFINE_bool('bi', True, 'If use bidirectional RNN.')

FLAGS = tf.flags.FLAGS
DATA_TYPE = tf.float32


class JSONData(object):
    def __init__(self,
                 file,
                 start_year,
                 end_year,
                 feature=None
                 ):
        self._train_stkcd = []
        self._train_year = []
        self._train_x = []
        self._train_y = []
        self._train_len = []
        self._test_x = []
        self._test_y = []
        self._test_len = []
        self._test_stkcd = []
        self._feature = feature
        train_max_len = end_year - start_year + 1
        test_max_len = end_year - start_year + 2
        end_year = end_year
        test_year = end_year + 1
        with open(file, 'r') as f:
            data = json.load(f)
        for doc in data:
            temp = list(sorted(doc['data'].items(), key=lambda d: int(d[0])))  # sort by year
            temp_x, temp_y, temp_len, temp_year = self._get_data(temp, start_year, end_year, train_max_len)
            if temp_len > 0:
                self._train_stkcd.append(doc['stkcd'])
                self._train_year.append(temp_year)
                self._train_x.append(temp_x)
                self._train_y.append(temp_y)
                self._train_len.append(temp_len)
            if str(test_year) in doc['data']:
                self._test_stkcd.append(doc['stkcd'])
                temp_x, temp_y, temp_len, _ = self._get_data(temp, start_year, test_year, test_max_len)
                self._test_x.append(temp_x)
                self._test_y.append(temp_y)
                self._test_len.append(temp_len)
        self._data_set = ph.Dataset(self._train_x, self._train_y, self._train_len)

    def next_batch(self, size=0):
        return self._data_set.next_batch(size)

    def _get_data(self, data, start, end, max_len):
        x = []
        y = []
        years = []
        for key, value in data:
            if start <= int(key) <= end:
                years.append(int(key))
                if self._feature is not None:
                    temp = np.array(value['feature'])
                    x.append(temp[self._feature])
                else:
                    x.append(value['feature'])
                y.append([value['label']])
        len_ = len(y)
        if len_ > 0:
            feature_size = len(x[0])
            for i in range(max_len - len_):
                x.append([0] * feature_size)
                y.append([0])
        return x, y, len_, years

    @property
    def test_data(self):
        return self._test_x, self._test_y, self._test_len

    @property
    def test_stkcd(self):
        return self._test_stkcd

    @property
    def train_size(self):
        return self._data_set.size

    @property
    def train_data(self):
        return self._train_x, self._train_y, self._train_len

    @property
    def train_stkcd(self):
        return self._train_stkcd

    @property
    def train_year(self):
        return self._train_year


class Model(object):
    def __init__(self, is_training, input_size, output_size):
        self._input = tf.placeholder(name='input', dtype=DATA_TYPE, shape=(None, None, input_size))
        self._seq_len = tf.placeholder(name='seq_len', dtype=tf.int32, shape=(None,))
        self._true_output = tf.placeholder(name='true_output', dtype=DATA_TYPE, shape=(None, None, output_size))
        embedding = self.linear_layer('emb', self._input, input_size, FLAGS.emb, axes=[[2], [0]])
        # embedding = tf.nn.leaky_relu(embedding, alpha=0.2)
        # dropout for input
        if is_training and FLAGS.keep < 1:
            embedding = tf.nn.dropout(embedding, FLAGS.keep)
        # cell
        if FLAGS.unit == 'lstm':
            # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
            rnn_fw_cells = [tf.nn.rnn_cell.LSTMCell(name='fw_cell_{}'.format(i), num_units=FLAGS.hidden) for i in
                            range(FLAGS.layers)]
            rnn_bw_cells = [tf.nn.rnn_cell.LSTMCell(name='bw_cell_{}'.format(i), num_units=FLAGS.hidden) for i in
                            range(FLAGS.layers)]
        elif FLAGS.unit == 'gru':
            rnn_fw_cells = [tf.nn.rnn_cell.GRUCell(name='fw_cell_{}'.format(i), num_units=FLAGS.hidden) for i in
                            range(FLAGS.layers)]
            rnn_bw_cells = [tf.nn.rnn_cell.GRUCell(name='bw_cell_{}'.format(i), num_units=FLAGS.hidden) for i in
                            range(FLAGS.layers)]
        else:
            raise ValueError('Unit must be lstm or gru!')
        if FLAGS.bi:
            # multi-layers bidirectional RNN
            outputs, output_states_fw, output_states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                rnn_fw_cells,
                rnn_bw_cells,
                embedding,
                dtype=DATA_TYPE,
                sequence_length=self._seq_len
            )
            self._inter_output = self.linear_layer('out', outputs, 2 * FLAGS.hidden, output_size, axes=[[2], [0]])
        else:
            # multi-layers RNN
            multi_rnn = tf.contrib.rnn.MultiRNNCell(cells=rnn_fw_cells, state_is_tuple=True)
            outputs, state = tf.nn.dynamic_rnn(
                multi_rnn,
                embedding,
                dtype=DATA_TYPE,
                sequence_length=self._seq_len
            )
            self._inter_output = self.linear_layer('out', outputs, FLAGS.hidden, output_size, axes=[[2], [0]])
        self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._true_output,
                                                                            logits=self._inter_output,
                                                                            name='loss'))
        self._pred_output = tf.nn.sigmoid(self._inter_output)
        if not is_training:
            return

        self._global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(1e-3, self._global_step, 1000, 0.5, staircase=True)
        self._train_step = tf.train.AdamOptimizer(lr).minimize(self._loss)
        # self._train_step = tf.train.AdamOptimizer(1e-3).minimize(self._loss)

    def linear_layer(self, name, x, input_size, output_size, axes=None):
        with tf.variable_scope(name):
            weight = tf.get_variable(
                name='w',
                shape=(input_size, output_size),
                dtype=DATA_TYPE,
                initializer=tf.keras.initializers.glorot_uniform()
            )
            bias = tf.get_variable(
                name='b',
                shape=(output_size,),
                dtype=DATA_TYPE,
                initializer=tf.zeros_initializer()
            )
            y = tf.matmul(x, weight) if axes is None else tf.tensordot(x, weight, axes=axes)
            y = tf.add(y, bias)
            return y

    @property
    def input(self):
        return self._input

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def true_output(self):
        return self._true_output

    @property
    def pred_output(self):
        return self._pred_output

    @property
    def global_step(self):
        return self._global_step

    @property
    def loss(self):
        return self._loss

    @property
    def train_step(self):
        return self._train_step

    @property
    def inter_output(self):
        return self._inter_output


def main(_):
    feature = list(range(0, 21)) + list(range(22, 67))
    # print(feature)
    # feature = None
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    ds = JSONData(FLAGS.input_file, FLAGS.start, FLAGS.end, feature=feature)
    print('Data loaded, size={}.'.format(ds.train_size))
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope('model', initializer=tf.keras.initializers.glorot_uniform()):
            model = Model(True, 66, 1)
        with tf.variable_scope('model', reuse=True, initializer=tf.keras.initializers.glorot_uniform()):
            model_test = Model(False, 66, 1)
        session.run(tf.global_variables_initializer())
        temp = '{:<5}' + 10 * ' {:<15}'
        print(temp.format('loop', 'train_loss', 'test_loss', 'Accuracy', 'Precision', 'Recall', 'F1',
                          'NDCG', 'Log Loss', 'MSPE', 'AUC'))
        acc = 0
        best_test_loss = 100
        for iter in range(1, FLAGS.loops + 1):
            # train
            train_x, train_y, train_seq = ds.next_batch(FLAGS.batch)
            train_fetches = [
                model.train_step,
                model.loss,
                model.pred_output
            ]
            train_feed = {
                model.input: train_x,
                model.true_output: train_y,
                model.seq_len: train_seq,
                model.global_step: iter
            }
            _, train_loss, output = session.run(train_fetches, train_feed)
            # in sample
            # test_x, test_y, test_seq = ds.train_data
            # out sample
            test_x, test_y, test_seq = ds.test_data
            test_fetches = [
                model_test.loss,
                model_test.pred_output
            ]
            test_feed = {
                model_test.input: test_x,
                model_test.true_output: test_y,
                model_test.seq_len: test_seq
            }
            test_loss, output = session.run(test_fetches, test_feed)
            # evaluation
            true_y = []
            pred_y = []
            # in sample
            # train_stkcds = []
            # train_years = []
            # for i, j, k, p, q in zip(output, test_y, test_seq, ds.train_stkcd, ds.train_year):
            #     for m in range(k):
            # train_stkcds.append(p)
            # train_years.append(q[m])
            # pred_y.append(i[m][0])
            # true_y.append(j[m][0])
            # out sample
            test_stkcd = ds.test_stkcd
            for i, j, k in zip(output, test_y, test_seq):
                pred_y.append(i[k - 1][0])
                true_y.append(j[k - 1][0])
            acc_temp, precision_temp, recall_temp, f1_temp, auc_temp, ndcg_temp, log_temp, mspe_temp = evaluation(
                true_y,
                pred_y,
                threshold=FLAGS.threshold)
            temp = '{:<5d}' + 10 * ' {:<15.8f}'
            print(temp.format(iter, train_loss, test_loss, acc_temp, precision_temp, recall_temp, f1_temp,
                              ndcg_temp, log_temp, mspe_temp, auc_temp))
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_loss_loop = iter
            if acc_temp > acc:
                acc = acc_temp
                best_acc_loop = iter
                # result = [acc_temp, precision_temp, recall_temp, f1_temp, ndcg_temp, log_temp, mspe_temp, auc_temp]
                result = [precision_temp, recall_temp, ndcg_temp, auc_temp]
                best_pred_y = pred_y
        print('Result({}): {}'.format(best_acc_loop, result))
        print('Test loss({}): {}'.format(best_loss_loop, best_test_loss))
        # save
        with open('result/rnn/ICD_dataset_240918_1700/{}_{}_{}_{}_{}.txt'.format(FLAGS.end + 1, FLAGS.layers, FLAGS.hidden,
                                                                            FLAGS.keep, FLAGS.emb), 'w') as f:
            for i, j, k in zip(test_stkcd, true_y, best_pred_y):
                f.write('{} {} {}\n'.format(i, j, k))
        # with open('result/rnn/insample.txt', 'w') as f:
        #     for i, j, k, l in zip(train_stkcds, train_years, true_y, best_pred_y):
        #         f.write('{} {} {} {}\n'.format(i, j, k, l))


if __name__ == '__main__':
    tf.app.run()
