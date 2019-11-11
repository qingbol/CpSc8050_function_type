import os
import sys
#os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32"
import numpy as np
import cPickle as pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from scipy import io
from rpy2 import robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
# from keras.models import load_model
np.random.seed(1234)

r = robjects.r
rpy2.robjects.numpy2ri.activate()

#np.set_printoptions(threshold = 1e6)
importr('genlasso')
importr('gsubfn')

def perf_measure(y_true, y_pred):
    TP_FN = np.count_nonzero(y_true)
    FP_TN = y_true.shape[0] * y_true.shape[1] - TP_FN
    FN = np.where((y_true - y_pred) == 1)[0].shape[0]
    TP = TP_FN - FN
    FP = np.count_nonzero(y_pred) - TP
    TN = FP_TN - FP
    Precision = float(float(TP) / float(TP + FP + 1e-9))
    Recall = float(float(TP) / float((TP + FN + 1e-9)))
    accuracy = float(float(TP + TN) / float((TP_FN + FP_TN + 1e-9)))
    F1 =  2*((Precision * Recall) / (Precision + Recall))
    return Precision, Recall, accuracy

class xai_rnn(object):
    """class for explaining the rnn prediction"""
    def __init__(self, model, data, start_binary, real_start_sp):
        """
        Args:
            model: target rnn model.
            data: data sample needed to be explained.
            label: label of the data sample.
            start: value of function start.
        """
        self.model = model
        self.data = data
        self.seq_len = data.shape[1]
        self.seq_len = data[0].shape[0]
        self.start = start_binary
        self.sp = np.where((self.data == self.start))
        self.real_sp = real_start_sp
        self.pred = self.model.predict(self.data, verbose = 0)[self.sp]

        #print 'Seq_len ... ', self.seq_len
        #print 'Start ...', self.start
        #print 'Sp ...',self.sp
        #print 'Real_start_sp ...',self.real_sp
        #print 'Pred ...\n',self.pred

    def truncate_seq(self, trunc_len):
        """ Generate truncated data sample
        function: take the data which are in tunc_len/2 far from function start
        Args:
            trun_len: the lenght of the truncated data sample.

        return:
            trunc_data: the truncated data samples.
        """
        self.trunc_data_test = np.zeros((1, self.seq_len), dtype=int)
        #self.trunc_data_test = np.ones((1, self.seq_len),dtype = int)
        self.tl = trunc_len
        cen = self.seq_len/2
        half_tl = trunc_len/2

        if self.real_sp < half_tl:
            #print 'self.real_sp < half_tl'
            self.trunc_data_test[0, (cen - self.real_sp):cen] = self.data[0, 0:self.real_sp]
            self.trunc_data_test[0, cen:(cen+half_tl+1)] = self.data[0, self.real_sp:(self.real_sp+half_tl+1)]

        elif self.real_sp >= self.seq_len - half_tl:
            #print 'self.real_sp >= self.seq_len - half_tl:'
            self.trunc_data_test[0, (cen - half_tl):cen] = self.data[0, (self.real_sp-half_tl):self.real_sp]
            self.trunc_data_test[0, cen:(cen + (self.seq_len-self.real_sp))] = self.data[0, self.real_sp:self.seq_len]

        else:
            #print 'else'
            self.trunc_data_test[0, (cen - half_tl):(cen + half_tl + 1)] = self.data[0, (self.real_sp - half_tl):(self.real_sp + half_tl + 1)]

        self.trunc_data = self.trunc_data_test[0, (cen - half_tl):(cen + half_tl + 1)]
        return self.trunc_data


    def pos_bootstrap_trun(self):
        """ Generate positive bootstrap sample and test it
        return:
            test_data: generated positive boostrap sample
            P_pos: prediction probability
        """
        cen = self.seq_len/2
        half_tl = self.tl/2
        test_data = np.copy(self.data)

        if self.real_sp < half_tl:
            test_data[0, 0:self.real_sp] = 0
            test_data[0, (self.real_sp+1):(self.real_sp+half_tl+1)] = 0

        elif self.real_sp >= self.seq_len - half_tl:
            test_data[0, (self.real_sp-half_tl):self.real_sp] = 0
            test_data[0, (self.real_sp+1):self.seq_len] = 0

        else:
            test_data[0, (self.real_sp - half_tl):self.real_sp] = 0
            test_data[0, (self.real_sp+1):(self.real_sp+half_tl+1)] = 0
        P_pos = self.model.predict(test_data, verbose=0)[0, self.real_sp, 1]

        return test_data, P_pos

    def neg_bootstrap_trun(self, test_seed, neg_pos):
        """ Generate negative bootstrap sample and test it
        Arg:
            test_seed: seed for negative bootstrap sample
        return:
            neg_data: generated negative bootstrap sample
            P_neg: prediction probability
        """
        test_data = np.copy(test_seed)
        pos = neg_pos
        cen = self.seq_len/2
        half_tl = self.tl/2
        if pos < half_tl:
            test_data[0, 0:pos] = self.trunc_data[(half_tl - pos):half_tl]
            test_data[0, (pos+1):(pos+half_tl+1)] = self.trunc_data[(half_tl+1):]

        elif pos >= self.seq_len - half_tl:
            test_data[0, (pos-half_tl):pos] = self.trunc_data[0:half_tl]
            test_data[0, (pos+1):self.seq_len] = self.trunc_data[(half_tl+1):(half_tl+self.seq_len - pos)]
        else:
            test_data[0,(pos - half_tl):pos] = self.trunc_data[ 0:half_tl]
            test_data[0,(pos+1):(pos+half_tl+1)] = self.trunc_data[(half_tl+1):]
        P_neg = self.model.predict(test_data, verbose=0)[0, pos, 1]
        return test_data, P_neg

    def new_testing_trun(self):
        """ Generate new testing sample and test it
        return:
            new_data: generated negative bootstrap sample
            P: prediction probability
        """
        P = self.model.predict(self.trunc_data_test, verbose=0)[0, 100, 1]
        return self.trunc_data_test, P


    def xai_feature(self, samp_num, option= 'None'):
        """extract the important features from the input data
        Arg:
            fea_num: number of features that needed by the user
            samp_num: number of data used for explanation
        return:
            fea: extracted features
        """
        cen = self.seq_len/2
        half_tl = self.tl/2
        sample = np.random.randint(1, self.tl+1, samp_num)
        print "sample len", len(sample)
        print "sample shape: ", sample.shape
        print "type of smaple", type(sample)
        # print "sample data: ", sample
        features_range = range(self.tl+1)
        print "feature_range type: ", type(features_range)
        print "feature_range len", len(features_range)
        print "feature_range data: ", features_range
        data_explain = np.copy(self.trunc_data).reshape(1, self.trunc_data.shape[0])
        data_sampled = np.copy(self.trunc_data_test)
        for i, size in enumerate(sample, start=1):
            inactive = np.random.choice(features_range, size, replace=False)
            #print '\ninactive --->',inactive
            tmp_sampled = np.copy(self.trunc_data)
            tmp_sampled[inactive] = 0
            #tmp_sampled[inactive] = np.random.choice(range(257), size, replace = False)
            # print "shape of tmp_sampled befor reshape", tmp_sampled.shape
            tmp_sampled = tmp_sampled.reshape(1, self.trunc_data.shape[0])
            # print "shape of tmp_sampled after reshape", tmp_sampled.shape
            data_explain = np.concatenate((data_explain, tmp_sampled), axis=0)
            # print "type of data_explain", type(data_explain)
            # print "shape of data_explain", data_explain.shape
            data_sampled_mutate = np.copy(self.data)
            if self.real_sp < half_tl:
                data_sampled_mutate[0, 0:tmp_sampled.shape[1]] = tmp_sampled
            elif self.real_sp >= self.seq_len - half_tl:
                data_sampled_mutate[0, (self.seq_len - tmp_sampled.shape[1]): self.seq_len] = tmp_sampled
            else:
                data_sampled_mutate[0, (self.real_sp - half_tl):(self.real_sp + half_tl + 1)] = tmp_sampled
            data_sampled = np.concatenate((data_sampled, data_sampled_mutate),axis=0)
        print "type of data_sampled", type(data_sampled)
        print "shape of data_sampled", data_sampled.shape
        print "type of data_explain", type(data_explain)
        print "shape of data_explain", data_explain.shape

        if option == "Fixed":
            print "Fix start points"
            data_sampled[:, self.real_sp] = self.start
        label_sampled_original = self.model.predict(data_sampled, verbose = 0)
        print "type of label_sampled_original", type(label_sampled_original)
        print "shape of label_sampled_original", label_sampled_original.shape
        label_sampled = self.model.predict(data_sampled, verbose = 0)[:, self.real_sp, 1]
        print "type of label_sampled", type(label_sampled)
        print "shape of label_sampled", label_sampled.shape
        label_sampled = label_sampled.reshape(label_sampled.shape[0], 1)
        print 'num of label_sampled is...',len(label_sampled), 'row...', label_sampled.shape[0], 'col...', label_sampled.shape[1]
        print 'num of data_explain is ...',len(data_explain), 'row...',data_explain.shape[0], 'col...', data_explain.shape[1]
        X = r.matrix(data_explain, nrow = data_explain.shape[0], ncol = data_explain.shape[1])
        print "type of X", type(X)
        # print "X data: ", X
        Y = r.matrix(label_sampled, nrow = label_sampled.shape[0], ncol = label_sampled.shape[1])
        print "type of Y", type(Y)
        # print "Y data: ", Y

        n = r.nrow(X)
        p = r.ncol(X)
        results = r.fusedlasso1d(y=Y,X=X)
        print "type of results: {}|row: {}|col: {}".format(type(results),r.nrow(results),r.ncol(results))
        result_original = np.array(r.coef(results, np.sqrt(n*np.log(p)))[0])
        print "type of result_original: ", type(result_original)
        print "shape of result_original: ", result_original.shape
        result = np.array(r.coef(results, np.sqrt(n*np.log(p)))[0])[:,-1]
        print "type of result: ", type(result)
        print "shape of result: ", result.shape
        print "data of result: ", result

        importance_score_original = np.argsort(result)
        print "type of importance_score_original: ", type(importance_score_original)
        print "shape of importance_score_original: ", importance_score_original.shape
        print "data of importance_score_original: ", importance_score_original

        importance_score = np.argsort(result)[::-1]
        print "type of importance_score: ", type(importance_score)
        print "shape of importance_score: ", importance_score.shape
        print "data of importance_score: ", importance_score
        #print 'importance_score ...',importance_score 
        self.fea = (importance_score-self.tl/2)+self.real_sp
        print "type of self.fea 1: ", type(self.fea)
        print "data of self.fea 1: ", self.fea
        self.fea = self.fea[np.where(self.fea<200)]
        print "data of self.fea 2: ", self.fea
        self.fea = self.fea[np.where(self.fea>=0)]
        print "data of self.fea 3: ", self.fea
        #print 'self.fea ...',self.fea
        return self.fea

class fid_test(object):
    def __init__(self, xai_rnn):
        self.xai_rnn = xai_rnn

    def pos_boostrap_exp(self, num_fea):
        test_data = np.copy(self.xai_rnn.data)
        selected_fea = self.xai_rnn.fea[0:num_fea]
        test_data[0, selected_fea] = 0
        P1 = self.xai_rnn.model.predict(test_data, verbose=0)[0, self.xai_rnn.real_sp,1]
        #test_data[0, self.xai_rnn.sp] = self.xai_rnn.start
        #P2 = self.xai_rnn.model.predict(test_data, verbose=0)[0, self.xai_rnn.real_sp,1]

        random_fea = np.random.randint(0, 200, num_fea)
        test_data_1 = np.copy(self.xai_rnn.data)
        test_data_1[0, random_fea] = 0
        P2 = self.xai_rnn.model.predict(test_data_1, verbose=0)[0, self.xai_rnn.real_sp, 1]
        return test_data, P1, P2


    def neg_boostrap_exp(self, test_seed, num_fea):
        test_seed = test_seed.reshape(1, 200)
        test_data = np.copy(test_seed)
        selected_fea = self.xai_rnn.fea[0:num_fea]
        test_data[0, selected_fea] = self.xai_rnn.data[0,selected_fea]
        P_neg_1 = self.xai_rnn.model.predict(test_data, verbose=0)[0, self.xai_rnn.real_sp,1]

        random_fea = np.random.randint(0, 200, num_fea)
        test_data_1 = np.copy(test_seed)
        test_data_1[0, random_fea] = self.xai_rnn.data[0,random_fea]
        P_neg_2 = self.xai_rnn.model.predict(test_data_1, verbose=0)[0, self.xai_rnn.real_sp, 1]

        return test_data, P_neg_1, P_neg_2


    def new_test_exp(self, num_fea):
        test_data = np.zeros_like(self.xai_rnn.data)
        selected_fea = self.xai_rnn.fea[0:num_fea]
        test_data[0, selected_fea] = self.xai_rnn.data[0, selected_fea]
        P_test_1 = self.xai_rnn.model.predict(test_data, verbose=0)[0, self.xai_rnn.real_sp,1]

        random_fea = np.random.randint(0, 200, num_fea)
        test_data_1 = np.zeros_like(self.xai_rnn.data)
        test_data_1[0, random_fea] = self.xai_rnn.data[0, random_fea]
        P_test_2 = self.xai_rnn.model.predict(test_data_1, verbose=0)[0, self.xai_rnn.real_sp, 1]

        return test_data, P_test_1, P_test_2


if __name__ == "__main__":
    print '[Load model...]'
    model = load_model('target_model/O0_Bi_Rnn.h5')
    PATH_TEST_DATA = 'data/elf_x86_32_gcc_O0_test.pkl'
    n_fea_select = 25

    #PATH_TEST_DATA = 'elf_x86_32_gcc_O0_test.pkl'
    print '[Load data...]'
    data = pickle.load(file(PATH_TEST_DATA))
    print "type of data %s"%(type(data))
    print "type of data[0] %s"%(type(data[0]))
    data_num = len(data[0])
    print 'Data_num:',data_num
    print "type of data[1] %s"%(type(data[1]))
    seq_len = len(data[0][0])
    print 'Sequence length:', seq_len
    # print 'binary snippet[0:26]: ', data[0][17][0:26]
    # print 'label snippet[0-26]: ', data[1][17][0:26]
    print "type of data[0][17] %s"%(type(data[0][17]))
    print "len of data[0][17]: %d"%len(data[0][17])
    print 'data[0][17] is ', data[0][17]
    print 'data[1][17] is: ', data[1][17]
    """
    dat0[0] is the dataset with 7337rows*200cols. 
              1   2  3 ... 200
          1  88 97 46 ... 91
          2  94 34 21 ... 92
               . . .
        7337 89 34 64 ... 86

    data[1] is the ground truth label with 7337rows*200cols. 
    """

    print "--------------------------------------------"
    ### Padding sequence ....
    x_test = pad_sequences(data[0], maxlen=seq_len, dtype='int32', padding='post', truncating='post', value=0)
    print "type of x_test %s"%type(x_test)
    print "shape of x_test: {}".format(x_test.shape)
    x_test = x_test + 1
    y = pad_sequences(data[1], maxlen=seq_len, dtype='int32', padding='post', truncating='post', value=0)
    print "type of y %s"%type(y)
    print "shape of y: {}".format(y.shape)
    y_test = np.zeros((data_num, seq_len, 2), dtype=y.dtype)
    print "type of y_test %s"%type(y_test)
    print "shape of y_test: {}".format(y_test.shape)
    # print "y_test[1]: {}".format(y_test[1])
    flag = 0
    for test_id in xrange(data_num):
        flag += 1
        y_test[test_id, np.arange(seq_len), y[test_id]] = 1
        # if test_id == 7:
        # if flag > 16 and flag < 25:
            # print "y[{}]: {}".format(test_id, y[test_id])
            # print "y_test[1]: {}".format(y_test[test_id])

    # sys.exit(0)

    #idx is the row of nonzero datapoint
    idx = np.nonzero(y)[0]
    n1 = idx.shape[0]
    print 'num of nonzero datapoints by row is(have duplicate row) ', n1
    print 'idx[0:20] is ', idx[0:20]

    #start_points is the col of nonzero datapoint  
    start_points = np.nonzero(y)[1]
    n2 = start_points.shape[0]
    print "num of nozero datapoints by col is: ", n2
    print 'start_points[0:20] is', start_points[0:20]

    n_pos = 0
    n_new = 0
    n_neg = 0

    n_pos_rand = 0
    n_new_rand = 0
    n_neg_rand = 0
    # n is used to cal the number of function start
    n = 0
    #print y[np.nonzero(y)]
    #print x_test[np.nonzero(y)]
    #print x_test[idx,start_points]
    # sys.exit(0)

    print "-----------------start(loop)------------------------------------"
    # for i in xrange(len(x_test)):
    for i in range(7,8):
    # for i in range(44,45):
    # for i in range(1,20):
        if i in idx:
            print "===========start(seq_id = row of function start: {} ): ======".format(i)
            #print '\n------>', i
            idx_Col = np.where(idx == i)
            #idx_Row is the col list of a function start
            idx_Row = start_points[idx_Col]
            # print 'idx_Col= the col index list of function start ...',idx_Col
            print 'idx_Row= the col list of a function start ...', idx_Row
            
            # binary_func_start is the binary value list of these function start
            binary_func_start = x_test[i][idx_Row]
            print 'binary_func_start ...', x_test[i][idx_Row]
            x_test_d = x_test[i:i + 1]
            print "x_test_d = the binary val of this line : ", x_test_d 
            # sys.exit(0)

            for j in xrange(len(idx_Row)):
                print '===========start(row=i | col=j of function start: {}|{})===='.format(i, idx_Row[j])
                # print 'seq_id', i
                # print 'function_start', binary_func_start[j]
                # print 'start_position', idx_Row[j]
                n = n + 1
                xai_test = xai_rnn(model, x_test_d, binary_func_start[j], idx_Row[j])
                prob_start = xai_test.pred[0, 1]
                print "the prob of {}|{} is the function start: {}".format(i, idx_Row[j], prob_start)
                if prob_start > 0.5:
                # if xai_test.pred[0, 1] > 0.5:
                    truncate_seq_data = xai_test.truncate_seq(40)
                    print "self.tunc_data :", truncate_seq_data
                    xai_fea = xai_test.xai_feature(500)
                    fea = np.zeros_like(xai_test.data)
                    print "type of fea: ", type(fea)
                    print "shape of fea: ", fea.shape
                    fea[0, xai_fea[0:25]] = xai_test.data[0, xai_fea[0:25]]
                    print 'fea...',fea
                    print 'xai fea-idx row[j]...',xai_fea - idx_Row[j]
                    # print '==================================================='
                    fid_tt = fid_test(xai_test)

                    test_data, P1, P2 = fid_tt.pos_boostrap_exp(n_fea_select)
                    # print 'Pos fide test probability >>>', P1, P2
                    # print 'Expect a low probability'
                    if P1 > 0.5:
                       n_pos = n_pos + 1
                    if P2 > 0.5:
                       n_pos_rand = n_pos_rand + 1

                    test_data, P_test_1, P_test_2 = fid_tt.new_test_exp(n_fea_select)
                    # print 'New fide test probability >>>', P_test_1
                    # print 'Expect a high probability'
                    if P_test_1> 0.5:
                       n_new = n_new + 1
                    if P_test_2 > 0.5:
                       n_new_rand = n_new_rand + 1

                    test_seed = x_test[0, ]
                    neg_test_data, P_neg_1, P_neg_2 = fid_tt.neg_boostrap_exp(test_seed, n_fea_select)
                    # print 'Neg fide test probability >>>', P_neg_1
                    # print 'Expect a high probability'
                    if P_neg_1 > 0.5:
                       n_neg = n_neg + 1
                    if P_neg_2 > 0.5:
                       n_neg_rand = n_neg_rand + 1

    print n
    print 'Our method'
    print 'Acc pos:', float(n_pos)/n
    print 'Acc new:', float(n_new)/n
    print 'Acc neg:', float(n_neg)/n

    print 'Random'
    print 'Acc pos:', float(n_pos_rand) / n
    print 'Acc new:', float(n_new_rand) / n
    print 'Acc neg:', float(n_neg_rand) / n

