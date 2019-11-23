'''
script for explain argument nums of function type
'''
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from rpy2 import robjects
from configure import get_config
# import configure
import eval_predict
import converter
import dataset
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
    def __init__(self, config_info):
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

    def workfolow(self):
        print "func_path: ", self.func_path
        with open(self.func_path) as f:
            func_info = pickle.load(f)
        # print "type of func_info:", type(func_info)
        # print "shape of func_info:", len(func_info)
        # print "data of func_info", func_info
        func_lst = func_info['train']
        # print "type of func_lst:", type(func_lst)
        print "shape of func_lst: ", len(func_lst)
        # print "data of func_lst:", func_lst

        # cal the match(prediction/label) number
        match_num_true = 0
        match_num_false = 0
        for index, func_name in enumerate(func_lst):
            self.func_name = func_name
            print "---------start of new function: %d------------------------------" % index
            # print "index in func_lst:", index
            print "func_name in func_lst:", self.func_name
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
            embed_data_array, int_data_array, hex_data_array = \
                self.convert_insn2int(data_batch)
            # -------------- end (convert data )--------------------------------

            # --------------start(check the correctness of prediction)----------
            data_embed = embed_data_array.reshape(
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
                match_num_false += 1
                continue
            else:
                match_num_true += 1
                pass
            # -------------- end (check the correctness of prediction)----------

            # --------------start(explain the prediction)-----------------------
            self.xai_function_type(embed_data_array, int_data_array,
                                   hex_data_array)
            # -------------- end (explain the prediction)-----------------------

            print "--------- end of new function: %d------------------------------" % index
        print "match_num_false: ", match_num_false
        print "match_num_true: ", match_num_true
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

    def xai_function_type(self, embed_data_array, int_data_array,
                          hex_data_array):
        # sample_num = 500
        # print "self.max_length", self.max_length
        self.embed_row = embed_data_array.shape[0]
        print "embed_row of embed_data_array.shape[0]", embed_data_array.shape[0]
        self.embed_col = embed_data_array.shape[1]
        print "embed_col of embed_data_array.shape[1]", embed_data_array.shape[1]
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

        data_embed = np.copy(embed_data_array).reshape(
            1, self.instrunction_length, self.embed_dim)
        data_int = np.copy(int_data_array).reshape(1, self.instrunction_length)
        # print "data of int_data_array", int_data_array
        # print "data of data_int", data_int
        for i, size in enumerate(sample, start=1):
            inactive = np.random.choice(features_range, size, replace=False)
            # print "type of inactive", type(inactive)
            # print 'inactive --->', inactive
            tmp_embed = np.copy(embed_data_array)
            tmp_embed[inactive] = 0
            tmp_embed = tmp_embed.reshape(
                1, self.instrunction_length, self.embed_dim)
            data_embed = np.concatenate((data_embed, tmp_embed), axis=0)

            tmp_int = np.copy(int_data_array)
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
        result = np.array(r.coef(results, np.sqrt(n*np.log(p)))[0])[:, -1]
        # print "type of result: ", type(result)
        print "shape of result: ", result.shape
        # print "data of real_arg_num:", self.real_arg_num
        # result_round=np.around(result, decimals=1)
        # print "data of result:{res:.2e} ".format(res=result)
        print "data of result: ", np.array_str(result, precision=2)
        significant_index = np.argsort(result)[::-1]
        print "data of significant_index: ", significant_index
        fea = np.zeros_like(hex_data_array)
        # print "shape of fea", fea.shape
        # print "data of hex_data_array", hex_data_array
        fea[significant_index[0:7]] = hex_data_array[significant_index[0:7]]
        print "data of feature", fea.tolist()

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

    def convert_insn2int(self, data_batch):
        # ------------start(convert insn2int )---------------------------------------
        # original embedding data
        # print "type of data_batch['data']", type(data_batch['data'][0])
        # print "shape of data_batch['data']", len(data_batch['data'][0])
        # print "data of data_batch['data']", data_batch['data'][0]
        embed_data_array = data_batch['data'][0]
        # print "type of embed_data_array", type(embed_data_array)
        # print "data of embed_data_array", embed_data_array
        # embed_data_array[0].fill(0)
        # print "data of embed_data_array", embed_data_array

        # original hex data
        # print "type of data_batch['inst_types']", type(data_batch['inst_bytes'][0])
        print "shape of data_batch['inst_types']", len(
            data_batch['inst_bytes'][0])
        print "data of data_batch['inst_types']", data_batch['inst_bytes'][0]
        hex_data_list = data_batch['inst_bytes'][0]
        # print "type of hex_data_list", type(hex_data_list)
        # print "data of hex_data_list", hex_data_list
        hex_data_array = np.asarray(hex_data_list)
        # hex_data_array = np.array(hex_data_list)
        # hex_data_array = np.array([np.array(x) for x in hex_data_list])
        # print "data of hex_data_array", hex_data_array

        # int of hex data
        int2insn_map, int_data_list = converter.main(hex_data_list)
        print "type of int_data_list:", type(int_data_list)
        print "int data of int_data_list:", int_data_list
        # print "type of int2insn_map:", type(int2insn_map)
        # print "data of int2insn_map:", int2insn_map
        int_data_array = np.asarray(int_data_list)
        # print "type of int_data_array:", type(int_data_array)
        # print "shape of int_data_array:", int_data_array.shape
        # print "data of int_data_array", int_data_array

        # bin_data_list = [int2insn_map[k]
        #                  for k in int_data_list if k in int2insn_map]
        # bin_data_list = [int2insn_map[int(k)] for k in int_data_list if int(k) in int2insn_map]
        # print "type of bin_data_list", type(bin_data_list)
        # print "len of bin_data_list", len(bin_data_list)
        # print "data of bin_data_list", bin_data_list

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

        # ------------ end (convert insn2int )---------------------------------------
        return embed_data_array, int_data_array, hex_data_array


def main(options):
    config_info = get_config(options)
    # config_info = configure.get_config()
    xai_func = XaiFunction(config_info)
    xai_func.workfolow()


if __name__ == '__main__':
    print "sys.argv[1:]", sys.argv
    main(sys.argv[1:])
