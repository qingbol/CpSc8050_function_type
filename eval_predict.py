import tensorflow as tf
import numpy as np
import dataset
# import dataset_caller
import os
import sys

import argparse
import functools
import pickle
import inspect


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


# def placeholder_inputs(class_num, max_length= 500, embedding_dim= 256):
def placeholder_inputs(class_num, max_length, embedding_dim=256):
    data_placeholder = tf.placeholder(
        tf.float32, [None, max_length, embedding_dim])
    label_placeholder = tf.placeholder(tf.float32, [None, class_num])
    length_placeholder = tf.placeholder(tf.int32, [None, ])
    keep_prob_placeholder = tf.placeholder(
        tf.float32)  # dropout (keep probability)
    return data_placeholder, label_placeholder, length_placeholder, keep_prob_placeholder


class Model(object):
    # def __init__(self, session, my_data, config_info, data_pl, label_pl, length_pl, keep_prob_pl):
    def __init__(self, session,  feed_data_dict, config_info, data_pl, label_pl, length_pl, keep_prob_pl):
        self.session = session
        # self.datasets = my_data
        self.feed_data_dict = feed_data_dict
        self.emb_dim = int(config_info['embed_dim'])
        self.dropout = float(config_info['dropout'])
        self.num_layers = int(config_info['num_layers'])
        self.num_classes = int(config_info['num_classes'])
        self.batch_size = int(config_info['batch_size'])

        self._data = data_pl
        self._label = label_pl
        self._length = length_pl
        # self._keep_prob = 1.0
        self._keep_prob = keep_prob_pl
        self.run_count = 0
        self.build_graph()

    @lazy_property
    def probability(self):
        def lstm_cell():
            if 'reuse' in inspect.getargspec(tf.contrib.rnn.GRUCell.__init__).args:
                return tf.contrib.rnn.GRUCell(self.emb_dim, reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.GRUCell(self.emb_dim)

        attn_cell = lstm_cell
        if self.dropout < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=self._keep_prob)
        single_cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(self.num_layers)], state_is_tuple=True)

        output, state = tf.nn.dynamic_rnn(single_cell, self._data, dtype=tf.float32,
                                          sequence_length=self._length)
        weight = tf.Variable(tf.truncated_normal(
            [self.emb_dim, self.num_classes], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))

        self.output = output
        probability = tf.matmul(self.last_relevant(
            output, self._length), weight) + bias
        return probability

    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_len = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant

    @lazy_property
    def pred_label(self):
        true_probability = tf.nn.softmax(self.probability)
        print "type of true_probability", type(true_probability)
        print "data of true_probability", true_probability
        pred_output = tf.argmax(true_probability, 1)
        # label_output = tf.argmax(self._label, 1)
        output_result = {
            'pred': pred_output,
            # 'label': label_output
        }
        return pred_output
        # return output_result

    def build_graph(self):
        # self.optimize
        # self.calc_accuracy
        self.pred_label
        self.saver = tf.train.Saver(tf.trainable_variables())
        tf.global_variables_initializer().run()

    def test(self):
        predict_result = {
            'pred': []
        }
        feed_dict = {
            self._data: self.feed_data_dict['data'],
            self._label: self.feed_data_dict['label'],
            self._length: self.feed_data_dict['length'],
            self._keep_prob: self.feed_data_dict['keep_prob_pl']
        }

        pred_result = self.session.run(self.pred_label, feed_dict=feed_dict)
        print "type in pred_result", type(pred_result)
        print "len in pred_result", len(pred_result)
        # print "data in pred_result", pred_result
        predict_result['pred'].append(pred_result)
        # print "type in predict_result['pred']", type(predict_result['pred'][0])
        # print "len in predict_result['pred']", len(predict_result['pred'][0])
        # print "data in predict_result['pred']", predict_result['pred'][0]
        return pred_result


def get_model_id_list(folder_path):
    file_list = os.listdir(folder_path)
    model_id_set = set()
    for file_name in file_list:
        if file_name[:6] == 'model-':
            model_id_set.add(int(file_name.split('.')[0].split('-')[-1]))
        else:
            pass
    model_id_list = sorted(list(model_id_set))
    return model_id_list


def testing(feed_data_dict, config_info, func_name):
    embed_dim = int(config_info['embed_dim'])
    max_length = int(config_info['max_length'])
    num_classes = int(config_info['num_classes'])
    model_dir = config_info['model_dir']
    output_dir = config_info['output_dir']

    '''create model & log folder'''
    if os.path.exists(output_dir):
        pass
    else:
        os.mkdir(output_dir)
    print('Created all folders!')

    '''load dataset'''

    '''get model id list'''
    # model_id_list = sorted(get_model_id_list(model_dir), reverse=True)
    print "befor get model_list"
    model_id_list = sorted(get_model_id_list(model_dir))
    # print "model_id_list", model_id_list
    print "after get model_list"

    with tf.Graph().as_default(), tf.Session() as session:
        # generate placeholder
        data_pl, label_pl, length_pl, keep_prob_pl = placeholder_inputs(
            num_classes, max_length, embed_dim)
        # generate model
        model = Model(session, feed_data_dict, config_info,
                      data_pl, label_pl, length_pl, keep_prob_pl)
        # model = Model(session, my_data, config_info, data_pl, label_pl, length_pl, keep_prob_pl)
        print('Created the model!')

        for model_id in model_id_list:
            print "entering for model_id"
            # result_path = os.path.join(
            #     output_dir, 'predict_result_%d_.label' % model_id)
            predicted_file = 'model' + \
                str(model_id) + '_' + func_name + '.predict'
            result_path = os.path.join(output_dir, predicted_file)
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    total_result = pickle.load(f)
                continue
            else:
                pass
            model_path = os.path.join(model_dir, 'model-%d' % model_id)
            model.saver.restore(session, model_path)

            print "before model test"
            total_result = model.test()
            print "after model test"
            # my_data._index_in_test = 0
            # my_data.test_tag = True
            with open(result_path, 'w') as f:
                pickle.dump(total_result, f)
            print('Save the test result !!! ... %s' % result_path)
    return total_result


def predict_main(feed_data_dict, config_info, func_name):
    total_result = testing(feed_data_dict, config_info, func_name)
    return total_result


def main(feed_data_dict):
    config_info = get_config()
    total_result = testing(config_info, feed_data_dict)
    return total_result


if __name__ == '__main__':
    feed_data_dict = {}
    main(feed_data_dict)
