import random
import numpy as np
np.random.seed(1234)


class Fidelity_test(object):
    def __init__(self, xai_obj):
        self.xai_obj = xai_obj

    def pos_exp(self, num_fea):
        test_data_pos_lemna = np.copy(self.xai_obj.embed_data_array)
        # print "type of test_data_pos_lemna before removing feature: ", type(test_data_pos_lemna)
        print "shape of test_data before removing feature: ", test_data_pos_lemna.shape
        # print "test_data_pos_lemna before removing feature: ", test_data_pos_lemna[0:1]
        selected_fea = self.xai_obj.sig_idx[0:num_fea]
        print "selected_fea:", selected_fea
        test_data_pos_lemna[selected_fea] = 0
        # print "test_data_pos_lemna after removing feature: ", test_data_pos_lemna[0:1]
        test_data_pos_lemna = test_data_pos_lemna.reshape(
            1, self.xai_obj.instrunction_length, self.xai_obj.embed_dim)
        pred_pos_lemna = self.xai_obj.predict(test_data_pos_lemna, 1)
        print "pred_pos_lemna: ", pred_pos_lemna
        n_pos_lemna = 1 if pred_pos_lemna[0] == self.xai_obj.real_arg_num else 0
        print "n_pos_lemna: ", n_pos_lemna

        random_fea = random.sample(range(0, 40), num_fea)
        # random_fea = np.random.randint(0, 40, num_fea)
        test_data_pos_rand = np.copy(self.xai_obj.embed_data_array)
        test_data_pos_rand[random_fea] = 0
        test_data_pos_rand = test_data_pos_rand.reshape(
            1, self.xai_obj.instrunction_length, self.xai_obj.embed_dim)
        pred_pos_rand = self.xai_obj.predict(test_data_pos_rand, 1)
        print "pred_pos_rand: ", pred_pos_rand
        n_pos_rand = 1 if pred_pos_rand[0] == self.xai_obj.real_arg_num else 0
        print "n_pos_ran: ", n_pos_rand

        return n_pos_lemna, n_pos_rand

    def neg_exp(self, num_fea):
        # --------------start(generate test_data_seed)---------------------------
        func_list_len = len(self.xai_obj.func_lst)
        print "func_list_len is: ", func_list_len
        while (True):
            index_rand_array = np.random.randint(0, func_list_len, 1)
            index_rand = index_rand_array[0]
            print "rand index in neg_exp is: ", index_rand
            print "xai_obj.index in neg_exp is: ", self.xai_obj.index
            if index_rand != self.xai_obj.index:
                break
        func_name_rand = self.xai_obj.func_lst[index_rand]
        func_list_rand = []
        func_list_rand.append(func_name_rand)
        data_batch_rand = self.xai_obj.read_func_data(func_list_rand)
        test_data_seed = data_batch_rand['data'][0]
        # -------------- end (generate test_data_seed)---------------------------

        test_data_neg_lemna = np.copy(test_data_seed)
        selected_fea = self.xai_obj.sig_idx[0:num_fea]
        # print "selected_fea:", selected_fea
        test_data_neg_lemna[selected_fea] = self.xai_obj.embed_data_array[selected_fea]
        # print "test_data_neg_lemna after removing feature: ", test_data_neg_lemna[0:1]
        test_data_neg_lemna = test_data_neg_lemna.reshape(
            1, self.xai_obj.instrunction_length, self.xai_obj.embed_dim)
        pred_neg_lemna = self.xai_obj.predict(test_data_neg_lemna, 1)
        print "pred_neg_lemna: ", pred_neg_lemna
        n_neg_lemna = 1 if pred_neg_lemna[0] == self.xai_obj.real_arg_num else 0
        print "n_neg_lemna: ", n_neg_lemna

        test_data_neg_rand = np.copy(test_data_seed)
        random_fea = random.sample(range(0, 40), num_fea)
        # random_fea = np.random.randint(0, 40, num_fea)
        test_data_neg_rand[random_fea] = self.xai_obj.embed_data_array[random_fea]
        test_data_neg_rand = test_data_neg_rand.reshape(
            1, self.xai_obj.instrunction_length, self.xai_obj.embed_dim)
        pred_neg_rand = self.xai_obj.predict(test_data_neg_rand, 1)
        print "pred_neg_rand: ", pred_neg_rand
        n_neg_rand = 1 if pred_neg_rand[0] == self.xai_obj.real_arg_num else 0
        print "n_neg_rand: ", n_neg_rand

        return n_neg_lemna, n_neg_rand

    def new_exp(self, num_fea):
        test_data_new_lemna = np.zeros_like(self.xai_obj.embed_data_array)
        selected_fea = self.xai_obj.sig_idx[0:num_fea]
        # print "selected_fea:", selected_fea
        test_data_new_lemna[selected_fea] = self.xai_obj.embed_data_array[selected_fea]
        test_data_new_lemna = test_data_new_lemna.reshape(
            1, self.xai_obj.instrunction_length, self.xai_obj.embed_dim)
        pred_new_lemna = self.xai_obj.predict(test_data_new_lemna, 1)
        print "pred_new_lemna: ", pred_new_lemna
        n_new_lemna = 1 if pred_new_lemna[0] == self.xai_obj.real_arg_num else 0
        print "n_new_lemna: ", n_new_lemna

        test_data_new_rand = np.zeros_like(self.xai_obj.embed_data_array)
        # random_fea = np.random.randint(0, 40, num_fea)
        random_fea = random.sample(range(0, 40), num_fea)
        test_data_new_rand[random_fea] = self.xai_obj.embed_data_array[random_fea]
        test_data_new_rand = test_data_new_rand.reshape(
            1, self.xai_obj.instrunction_length, self.xai_obj.embed_dim)
        pred_new_rand = self.xai_obj.predict(test_data_new_rand, 1)
        print "pred_new_rand: ", pred_new_rand
        n_new_rand = 1 if pred_new_rand[0] == self.xai_obj.real_arg_num else 0
        print "n_new_rand: ", n_new_rand

        return n_new_lemna, n_new_rand
