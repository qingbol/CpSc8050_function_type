'''
script for explain argument nums of function type
'''
import time
from rpy2 import robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

from fidelity_eval import Fidelity_test
import dataset
import converter
import eval_predict
from configure import get_config

import os
import sys
import cPickle as pickle
import numpy as np
np.random.seed(1234)
# np.set_printoptions(threshold = 1e6)
np.set_printoptions(threshold=sys.maxsize)

r = robjects.r
rpy2.robjects.numpy2ri.activate()
importr('genlasso')
# importr('gsubfn')


class XaiFunction(object):
    def __init__(self, config_info, file_all_path, file_acc_path):
        print "entering tst main"
        self.config_info = config_info
        self.data_folder = config_info['data_folder']
        self.func_path = config_info['func_path']
        self.embed_path = config_info['embed_path']
        self.tag = config_info['tag']
        self.process_num = int(config_info['process_num'])
        self.embed_dim = int(config_info['embed_dim'])
        # self.max_length = int(config_info['max_length'])
        self.num_classes = int(config_info['num_classes'])
        self.output_dir = config_info['output_dir']
        self.int2insn_path = config_info['int2insn_path']
        self.batch_size = config_info['batch_size']
        self.sample_num = int(config_info['sample_num'])
        self.func_index = int(config_info['func_index'])
        self.feature_num = int(config_info['feature_num'])

        self.file_all_path = file_all_path
        self.file_acc_path = file_acc_path

    def __enter__(self):
        if os.path.exists(self.file_all_path):
            os.remove(self.file_all_path)
        self.log_all_file = open(self.file_all_path, 'a+')
        self.log_acc_file = open(self.file_acc_path, 'a+')

    def __exit__(self, exception_type, exception_value, traceback):
        self.log_all_file.close()
        self.log_acc_file.close()

    def workfolow(self):
        print "func_path: ", self.func_path
        with open(self.func_path) as f:
            func_info = pickle.load(f)
        # print "type of func_info:", type(func_info)
        # print "shape of func_info:", len(func_info)
        # print "data of func_info", func_info
        # self.func_lst = func_info['test']
        self.func_lst = func_info['train']
        # print "type of self.func_lst:", type(self.func_lst)
        print "shape of self.func_lst: ", len(self.func_lst)
        # print "data of self.func_lst:", self.func_lst

        # cal the match(prediction/label) number
        self.match_num_true = 0
        self.match_num_false = 0
        self.n_pos_lemna = 0
        self.n_pos_rand = 0
        self.n_new_lemna = 0
        self.n_new_rand = 0
        self.n_neg_lemna = 0
        self.n_neg_rand = 0
        for index, func_name in enumerate(self.func_lst):
            self.index = index
            self.func_name = func_name
            print "---------start of new function: %d------------------------------" % self.index
            # print "index in self.func_lst:", index
            print "func_name in self.func_lst:", self.func_name
            if self.func_index != -1 and index != self.func_index:
                continue
            # --------------start(read data )-----------------------------------
            func_lst_in_loop = []
            func_lst_in_loop.append(func_name)
            # print "type of  in func_lst_in_loop:", type(func_lst_in_loop)
            # print "data of  in func_lst_in_loop:", func_lst_in_loop
            data_batch = self.read_func_data(func_lst_in_loop)
            # -------------- end (read data )-----------------------------------

            # --------------start(convert data )--------------------------------
            self.convert_insn2int(data_batch)
            # -------------- end (convert data )--------------------------------

            # --------------start(check the correctness of prediction)----------
            data_embed = self.embed_data_array.reshape(
                1, self.instrunction_length, self.embed_dim)
            predicted_result = self.predict(data_embed, 1)
            # print "type of predicted_result", type(predicted_result)
            # print "shape of predicted_result", predicted_result.shape
            # print "data of predicted_result", predicted_result[0]
            self.predicted_arg_num = predicted_result[0]
            if self.predicted_arg_num != self.real_arg_num:
                print "data of real_arg_num of 1: ", self.real_arg_num
                print "data of predicted_arg_num of 1: ", self.predicted_arg_num
                print "Error: predicted_arg_num don't match real_arg_num"
                self.match_num_false += 1
                continue
            else:
                self.match_num_true += 1
                pass
            # -------------- end (check the correctness of prediction)----------

            # --------------start(explain the prediction)-----------------------
            self.xai_function_type()
            # -------------- end (explain the prediction)-----------------------

            # --------------start(fidelity evaluation)--------------------------
            fidelity_test = Fidelity_test(self)
            pos_lemna, pos_rand = fidelity_test.pos_exp(self.feature_num)
            self.n_pos_lemna += pos_lemna
            self.n_pos_rand += pos_rand
            neg_lemna, neg_rand = fidelity_test.neg_exp(self.feature_num)
            self.n_neg_lemna += neg_lemna
            self.n_neg_rand += neg_rand
            new_lemna, new_rand = fidelity_test.new_exp(self.feature_num)
            self.n_new_lemna += new_lemna
            self.n_new_rand += new_rand
            # -------------- end (fidelity evaluation)--------------------------
            self.log_all()
            print "--------- end of new function: %d------------------------------" % self.index
        print "-----------------match(predict/label)----------"
        print "match_num_false: ", self.match_num_false
        print "match_num_true: ", self.match_num_true
        print "-----------------Acc(pos_exp)------------------"
        print "Acc pos of LEMNA: {0:.2f}% ".format(
            float(self.n_pos_lemna)/self.match_num_true*100)
        print "Acc pos of Random: {0:.2f}%".format(
            float(self.n_pos_rand)/self.match_num_true*100)
        print "-----------------Acc(neg_exp)------------------"
        print "Acc neg of LEMNA: {0:.2f}% ".format(
            float(self.n_neg_lemna)/self.match_num_true*100)
        print "Acc neg of Random: {0:.2f}%".format(
            float(self.n_neg_rand)/self.match_num_true*100)
        print "-----------------Acc(new_exp)------------------"
        print "Acc new of LEMNA: {0:.2f}% ".format(
            float(self.n_new_lemna)/self.match_num_true*100)
        print "Acc new of Random: {0:.2f}%".format(
            float(self.n_new_rand)/self.match_num_true*100)

        self.log_acc()
        sys.exit(0)

    def read_func_data(self, func_lst_in_loop):
        # ------------start(retriev the target function data)------------------------
        function_data_file = func_lst_in_loop[0] + ".dat"
        function_data_path = os.path.join(self.output_dir, function_data_file)
        # result_path = os.path.join(self.output_dir, 'data_batch_result.pkl')
        if os.path.exists(function_data_path):
            with open(function_data_path, 'r') as f:
                data_batch = pickle.load(f)
            print('read the function data !!! ... %s' % function_data_path)
        else:
            my_data = dataset.Dataset(self.data_folder, func_lst_in_loop,
                                      self.embed_path, self.process_num, self.embed_dim,
                                      self.num_classes, self.tag, self.int2insn_path)
            data_batch = my_data.get_batch(batch_size=self.batch_size)
            with open(function_data_path, 'w') as f:
                pickle.dump(data_batch, f)
            print('Save the function_data_path !!! ... %s' %
                  function_data_path)

        # *******start(used to predict the label of this data_batch)********
        # keep_prob = 1.0
        # feed_batch_dict1 = {
        #     'data': data_batch['data'],
        #     'label': data_batch['label'],
        #     'length': data_batch['length'],
        #     'keep_prob_pl': keep_prob
        # }
        # print "type of feed_batch_dict1['data']", type(feed_batch_dict1['data'])
        # print "len of feed_batch_dict1['data']", len(feed_batch_dict1['data'])
        # print "data of feed_batch_dict1['data']", feed_batch_dict1['data']
        # eval_predict.main(feed_batch_dict1)
        # ******* end (used to predict the label of this data_batch)********
        # ------------ end (retriev the target function data)------------------------
        return data_batch

    def convert_insn2int(self, data_batch):
        # ------------start(convert insn2int )---------------------------------------
        # **********start(get mat_length/label )*********************
        self.instrunction_length = int(data_batch['length'])
        # print "data of instrunction_length:", self.instrunction_length

        # print "************label of {}**********".format(self.func_name)
        # print "type of  in data_batch['label']:", type(data_batch['label'])
        print "data of  in data_batch['label']:", data_batch['label']
        # print "************label of {}**********".format(self.func_name)
        self.real_arg_num = np.argmax(data_batch['label'])
        print "data of real_arg_num:", self.real_arg_num
        # ********** end (get mat_length/label )*********************

        # original instruction string data
        inst_asm_list = data_batch['inst_strings']
        self.inst_asm_array = np.asarray(
            inst_asm_list).reshape(self.instrunction_length, 1)
        print "type of inst_strings", type(self.inst_asm_array)
        print "shape of inst_strings", self.inst_asm_array.shape
        print "data of inst_strings", self.inst_asm_array

        # original embedding data
        # print "type of data_batch['data']", type(data_batch['data'][0])
        # print "shape of data_batch['data']", len(data_batch['data'][0])
        # print "data of data_batch['data']", data_batch['data'][0]
        self.embed_data_array = data_batch['data'][0]
        # print "type of self.embed_data_array", type(self.embed_data_array)
        # print "data of self.embed_data_array", self.embed_data_array
        # self.embed_data_array[0].fill(0)
        # print "data of self.embed_data_array", self.embed_data_array

        # original hex data
        # print "type of data_batch['inst_types']", type(data_batch['inst_bytes'][0])
        print "shape of data_batch['inst_types']", len(
            data_batch['inst_bytes'][0])
        print "data of data_batch['inst_types']", data_batch['inst_bytes'][0]
        hex_data_list = data_batch['inst_bytes'][0]
        # print "type of hex_data_list", type(hex_data_list)
        # print "data of hex_data_list", hex_data_list
        self.hex_data_array = np.asarray(hex_data_list)
        # self.hex_data_array = np.array(hex_data_list)
        # self.hex_data_array = np.array([np.array(x) for x in hex_data_list])
        # print "data of self.hex_data_array", self.hex_data_array

        # int of hex data
        int2insn_map, int_data_list = converter.main(hex_data_list)
        print "type of int_data_list:", type(int_data_list)
        print "int data of int_data_list:", int_data_list
        # print "type of int2insn_map:", type(int2insn_map)
        # print "data of int2insn_map:", int2insn_map
        self.int_data_array = np.asarray(int_data_list)
        # print "type of self.int_data_array:", type(self.int_data_array)
        # print "shape of self.int_data_array:", self.int_data_array.shape
        # print "data of self.int_data_array", self.int_data_array

        # bin_data_list = [int2insn_map[k]
        #                  for k in int_data_list if k in int2insn_map]
        # bin_data_list = [int2insn_map[int(k)] for k in int_data_list if int(k) in int2insn_map]
        # print "type of bin_data_list", type(bin_data_list)
        # print "len of bin_data_list", len(bin_data_list)
        # print "data of bin_data_list", bin_data_list
        # ------------ end (convert insn2int )---------------------------------------

    def xai_function_type(self):
        # sample_num = 500
        # print "self.max_length", self.max_length
        self.embed_row = self.embed_data_array.shape[0]
        print "embed_row of self.embed_data_array.shape[0]", self.embed_data_array.shape[0]
        self.embed_col = self.embed_data_array.shape[1]
        print "embed_col of self.embed_data_array.shape[1]", self.embed_data_array.shape[1]
        # half_tl = self.embed_row/2
        sample = np.random.randint(
            1, self.instrunction_length+1, self.sample_num)
        # print "sample len", len(sample)
        # print "sample shape: ", sample.shape
        # print "type of smaple", type(sample)

        features_range = range(self.instrunction_length)
        # features_range = range(tl+1)
        # print "feature_range type: ", type(features_range)
        # print "feature_range len", len(features_range)
        # print "feature_range data: ", features_range

        data_embed = np.copy(self.embed_data_array).reshape(
            1, self.instrunction_length, self.embed_dim)
        data_int = np.copy(self.int_data_array).reshape(
            1, self.instrunction_length)
        # print "data of self.int_data_array", self.int_data_array
        # print "data of data_int", data_int
        for i, size in enumerate(sample, start=1):
            inactive = np.random.choice(features_range, size, replace=False)
            # print "type of inactive", type(inactive)
            # print 'inactive --->', inactive
            tmp_embed = np.copy(self.embed_data_array)
            tmp_embed[inactive] = 0
            tmp_embed = tmp_embed.reshape(
                1, self.instrunction_length, self.embed_dim)
            data_embed = np.concatenate((data_embed, tmp_embed), axis=0)

            tmp_int = np.copy(self.int_data_array)
            tmp_int[inactive] = 0
            tmp_int = tmp_int.reshape(1, self.instrunction_length)
            data_int = np.concatenate((data_int, tmp_int), axis=0)
        # print "type of data_embed", type(data_embed)
        # print "shape of data_embed", data_embed.shape
        # print "type of tmp_int", type(data_int)
        # print "shape of tmp_int", data_int.shape

        print "self.sample_num: ", self.sample_num
        total_result = self.predict(data_embed, self.sample_num + 1)
        print "data of real_arg_num of 1: ", self.real_arg_num
        print "data of predicted_arg_num of 1: ", self.predicted_arg_num
        print "data of predicted_arg_num of 501: ", total_result
        label_sampled = total_result.reshape(self.sample_num + 1, 1)
        # print "type in label_sampled: ", type(label_sampled)
        # print "shape in label_sampled: ", label_sampled.shape
        # print "data  in label_sampled:", label_sampled

        # **********convert the value in label to 1 or 0 **************
        # label_sampled[label_sampled != 4] = 0
        # print "data  in total_result['pred']", label_sampled
        # label_sampled[label_sampled == 4] = 1
        # print "data  in total_result['pred']", label_sampled

        # ---------start(prepare the input data for regression model)---------------
        X = r.matrix(data_embed, nrow=data_embed.shape[0],
                     ncol=data_embed.shape[1])
        # X = r.matrix(data_int, nrow = data_int.shape[0], ncol = data_int.shape[1])
        # print "type of X", type(X)
        # print "X data: ", X
        Y = r.matrix(label_sampled, nrow=label_sampled.shape[0],
                     ncol=label_sampled.shape[1])
        # print "type of Y", type(Y)
        # print "Y data: ", Y

        n = r.nrow(X)
        p = r.ncol(X)
        results = r.fusedlasso1d(y=Y, X=X)
        # print "type of results: {}|row: {}|col: {}".format(
        #         type(results),r.nrow(results),r.ncol(results))
        result_original = np.array(r.coef(results, np.sqrt(n*np.log(p)))[0])
        # print "type of result_original: ", type(result_original)
        print "shape of result_original: ", result_original.shape
        self.coef = np.array(r.coef(results, np.sqrt(n*np.log(p)))[0])[:, -1]
        # print "type of result: ", type(result)
        print "shape of predicted result: ", self.coef.shape
        # print "data of real_arg_num:", self.real_arg_num
        # result_round=np.around(result, decimals=1)
        # print "data of predicted self.coef:{res:.2e} ".format(res=self.coef)
        print "data of predicted result: ", np.array_str(
            self.coef, precision=4)
        significant_index = np.argsort(self.coef)[::-1]
        self.sig_idx = significant_index
        print "data of self.sig_idx: ", self.sig_idx

        fea_hex = np.zeros_like(self.hex_data_array)
        # print "shape of fea", fea.shape
        # print "data of self.hex_data_array", self.hex_data_array
        print "type of self.hex_data_array", type(self.hex_data_array)
        print "shape of self.hex_data_array", self.hex_data_array.shape
        fea_hex[self.sig_idx[0:self.feature_num]
                ] = self.hex_data_array[self.sig_idx[0:self.feature_num]]
        print "hex value of feature: ", fea_hex.tolist()

        fea_asm = np.zeros_like(self.inst_asm_array)
        fea_asm[self.sig_idx[0:self.feature_num]
                ] = self.inst_asm_array[self.sig_idx[0:self.feature_num]]
        print "assembly of feature: ", fea_asm.tolist()

        # --------- end (prepare the input data for regression model)---------------

    def predict(self, data_embed, sample_num):
        # ---------start(prepare the dict which feed to eval)------------------------
        data_length = np.empty(sample_num)
        data_length.fill(self.instrunction_length)
        # print "type of data_length", type(data_length)
        # print "len of data_length", len(data_length)
        # print "shape of data_length", data_length.shape
        # print "data of data_length", data_length
        data_label = np.empty([sample_num, 16])
        data_label.fill(0)
        # print "type of data_label", type(data_label)
        # print "len of data_label", len(data_label)
        # print "shape of data_label", data_label.shape
        # print "data of data_label", data_label
        keep_prob = 1.0
        feed_batch_dict2 = {
            'data': data_embed,
            'label': data_label,
            'length': data_length,
            'keep_prob_pl': np.asarray(keep_prob, dtype=np.float32)
        }
        # print "type of feed_dict2[data_pl]", type(feed_batch_dict2['data'][0])
        # print "len of feed_dict2[data_pl]", len(feed_batch_dict2['data'][0])
        # print "data of feed_dict2[data_pl]", feed_batch_dict2['data'][0]
        # --------- end (prepare the dict which feed to eval)------------------------

        # ---------start(predict the label of 500 data)-----------------------------
        # print "func_name in func_lst:", self.func_name

        predicted_result = eval_predict.predict_main(feed_batch_dict2,
                                                     self.config_info,
                                                     self.func_name,
                                                     self.instrunction_length,
                                                     sample_num)

        # print "type in predicted_result['pred']", type(predicted_result)
        # print "shape in predicted_result['pred']", predicted_result.shape
        # print "label in predicted_result['pred']", predicted_result
        # --------- end (predict the label of 500 data)-----------------------------
        return predicted_result

    def write_all(self, msg):
        return self.log_all_file.write(msg)

    def write_acc(self, msg):
        return self.log_acc_file.write(msg)

    def log_acc(self):
        self.write_acc(
            "**********************feature num: {}******************************\n"
            .format(self.feature_num))
        self.write_acc("-----------------match(predict/label)----------\n")
        self.write_acc("match_num_false: {} \n".format(self.match_num_false))
        self.write_acc("match_num_true: {} \n".format(self.match_num_true))
        self.write_acc("-----------------Acc(pos_exp)------------------\n")
        self.write_acc("Acc pos of LEMNA: {0:.2f}% \n".format(
            float(self.n_pos_lemna)/self.match_num_true*100))
        self.write_acc("Acc pos of Random: {0:.2f}%\n".format(
            float(self.n_pos_rand)/self.match_num_true*100))
        self.write_acc("-----------------Acc(neg_exp)------------------\n")
        self.write_acc("Acc neg of LEMNA: {0:.2f}% \n".format(
            float(self.n_neg_lemna)/self.match_num_true*100))
        self.write_acc("Acc neg of Random: {0:.2f}%\n".format(
            float(self.n_neg_rand)/self.match_num_true*100))
        self.write_acc("-----------------Acc(new_exp)------------------\n")
        self.write_acc("Acc new of LEMNA: {0:.2f}% \n".format(
            float(self.n_new_lemna)/self.match_num_true*100))
        self.write_acc("Acc new of Random: {0:.2f}%\n".format(
            float(self.n_new_rand)/self.match_num_true*100))
        self.write_acc("\n")
        self.write_acc("\n")

    def log_all(self):
        # self.write_all(
            # "*********function num = {} ************\n".format(self.match_num_true))
        self.write_all("---------start of new function. no|index : {}|{} -----------------------\n"
                       .format(self.match_num_true, self.index))
        self.write_all(
            "func_name in self.func_lst: " + self.func_name + "\n")
        self.write_all("\n")
        self.write_all(
            "data of real_arg_num : %s\n" % self.real_arg_num)
        self.write_all(
            "data of predicted_arg_num: {} \n".format(self.predicted_arg_num))
        self.write_all("shape of assembly code: {}\n".format(
            self.inst_asm_array.shape))
        self.write_all("Assembly code:\n{}\n".format(self.inst_asm_array))
        self.write_all("\n")
        self.write_all(
            "shape of predicted result: {} \n".format(self.coef.shape))
        self.write_all("coefficients of each feature:\n{}\n".format(
            np.array_str(self.coef, precision=4)))
        self.write_all(
            "ranked index of most important feature:\n{}\n".format(self.sig_idx))
        self.write_all("\n")
        self.write_all("\n")


def main(options):
    config_info = get_config(options)
    # config_info = configure.get_config()
    time_str = time.strftime("%Y%m%d-%H%M")
    file_all = "./log/log_all_" + \
        str(config_info['feature_num']) + "_" + time_str + ".txt"
    file_acc = "./log/log_acc" + \
        str(config_info['feature_num']) + "_" + time_str + ".txt"
    # file_acc = "./log/log_acc.txt"
    xai_func = XaiFunction(config_info, file_all, file_acc)
    with xai_func:
        xai_func.workfolow()


if __name__ == '__main__':
    print "sys.argv[1:]", sys.argv
    main(sys.argv[1:])
