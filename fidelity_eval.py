import numpy as np


class Fidelity_test(object):
    def __init__(self, xai_obj):
        self.xai_obj = xai_obj

    def pos_exp(self, num_fea):
        test_data_lemna = np.copy(self.xai_obj.embed_data_array)
        # print "type of test_data before removing feature: ", type(test_data)
        print "shape of test_data before removing feature: ", test_data_lemna.shape
        # print "test_data before removing feature: ", test_data[0:1]
        selected_fea = self.xai_obj.sig_idx[0:num_fea]
        print "selected_fea:", selected_fea
        test_data_lemna[selected_fea] = 0
        # print "test_data after removing feature: ", test_data[0:1]
        test_data_lemna = test_data_lemna.reshape(
            1, self.xai_obj.instrunction_length, self.xai_obj.embed_dim)
        pred_pos_lemna = self.xai_obj.predict(test_data_lemna, 1)
        print "pred_pos_lemna: ", pred_pos_lemna
        n_pos_lemna = 0 if pred_pos_lemna[0] == self.xai_obj.real_arg_num else 1
        print "n_pos_lemna: ", n_pos_lemna

        random_fea = np.random.randint(0, 40, num_fea)
        test_data_rand = np.copy(self.xai_obj.embed_data_array)
        test_data_rand[random_fea] = 0
        test_data_rand = test_data_rand.reshape(
            1, self.xai_obj.instrunction_length, self.xai_obj.embed_dim)
        pred_pos_rand = self.xai_obj.predict(test_data_rand, 1)
        print "pred_pos_rand: ", pred_pos_rand
        n_pos_rand = 0 if pred_pos_rand[0] == self.xai_obj.real_arg_num else 1
        print "n_pos_ran: ", n_pos_rand

        return n_pos_lemna, n_pos_rand
