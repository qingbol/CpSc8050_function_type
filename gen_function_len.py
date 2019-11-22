'''
Input: the whole dataset (in order to get the whole vocabulary)
Output:
output_path: the input for embedding model
error_path: save all the error information in (especially when two distinct instructions map to same integer)
int2insn_map_path: the map information(int -> insn (int list))
'''
### split func_path
# python gen_function_len.py -i /Users/tarus/OnlyInMac/dataset/eklavya/clean_pickles
# python gen_function_len.py -i /scratch2/qingbol/CpSc8580EKLAVYA/dataset/clean_pickles

import pickle
import argparse
# import sys
import os
# import insn_int
import random
import collections



def get_file_tree(folder_path):
    file_tree = open ("file_tree.txt","w")
    i = 0
    split_func_dict_x64 = {}
    split_func_dict_x64["train"] = []
    split_func_dict_x64["test"] = []
    # file_list_x64 = []
    # func_list_x64 = []
    # list_x64_train = []
    # list_x64_test = []

    split_func_dict_x86 = {}
    split_func_dict_x86["train"] = []
    split_func_dict_x86["test"] = []
    # list_x86 = []
    # list_x86_train = []
    # list_x86_test = []
    
    func_len_dict_x64 = {}
    func_len_dict_x86 = {}

    for root, dirs, files in os.walk(folder_path):
        i += 1
        file_tree.write("---------{} layer----------\n".format(i))
        file_tree.write("root is {}:\n".format(root))
        file_tree.write("dirs is :{} \n".format(dirs))
        file_tree.write("files is : {} \n".format(files))

        root_basename = os.path.basename(root)
        if root_basename == "x64":
            # file_list_x64 = files
            # split_func_dict_x64["train"], split_func_dict_x64["test"] \
                # = get_func_list(root, files)
            func_len_dict_x64 = get_func_list(root, files)
            pickle.dump(func_len_dict_x64, open("./func_list/func_len_dict_x64.pkl","wb"))
            # pickle.dump(split_func_dict_x64, open("./func_list/func_dict_x64.lst","wb"))
            # print func_list_x64
            # split_func_dict_x64["train"] = list_x64_train
            # split_func_dict_x64["test"] = list_x64_test
        # elif root_basename == "x86":
        #     split_func_dict_x86["train"], split_func_dict_x86["test"] \
        #         = get_func_list(root, files)
        #     pickle.dump(split_func_dict_x86, open("./func_list/func_dict_x86.lst","wb"))
            # list_x86 = files
            # random.shuffle(list_x86)
            # index = len(list_x86) * 0.8
            # list_x86_train = list_x86[:index]
            # list_x86_test = list_x86[index:]
            # split_func_dict_x86["train"] = list_x86_train
            # split_func_dict_x86["test"] = list_x86_test
    # return split_func_dict_x64, split_func_dict_x86

def get_func_list(root, file_list):
    func_list = []
    pkl_file_num = 0 
    func_len_dict = {}
    pkl_file_total = len(file_list)
    max_len = 0
    func_num = 0
    for pkl_file in file_list:
        pkl_file_num += 1
        # print pkl_file_num
        # print pkl_file_total
        progress = float(pkl_file_num)/pkl_file_total
        # print progress
        print "root:{0}, len(func_len_dict):{1}, progress : {2:.1f}%". \
            format(os.path.basename(root), len(func_len_dict), progress*100)
        pkl_file_path = os.path.join(root, pkl_file)
        f = open(pkl_file_path)
        temp = pickle.load(f)
        f.close()
        for func_name in temp["functions"]:
            func_num += 1
            # func_list.append(pkl_file + "#" + func_name)
            func_len = len(temp['functions'][func_name]['inst_bytes'])
            max_len = max(max_len, func_len)
            func_len_dict[pkl_file + "#" + func_name] = func_len
            # if func_num < 10:
            #     print func_len
            # else:
            #     break
        # print func_list
    # random.shuffle(func_list)
    # index = int(len(func_list) * 0.8)
    # print index
    # func_list_train = func_list[:index]
    # func_list_test = func_list[index:]
    # return func_list_test, func_list_test
    print "saved func_num : ", len(func_len_dict)
    print "total func_num : ", func_num
    print "max_len : ", max_len
    return func_len_dict

        # print root
        # print dirs
        # print files

def load_func_tree(func_list):
    f1 = open(func_list) 
    func_len_dict = pickle.load(f1)
    f1.close()
    func_len_dict_lst=sorted(func_len_dict.items(),key=lambda item:item[1])
    print "type of func_len_dict", type(func_len_dict)
    print "len of func_len_dict", len(func_len_dict)
    # split_func_dict = pickle.load(open(func_list))
    print func_len_dict_lst[0:20]
    print '----------------------'
    print func_len_dict_lst[-20:-1]
    print '----------------------'
    print func_len_dict_lst[len(func_len_dict)/2:len(func_len_dict)/2+20]
    print '----------------------'

    val_list = func_len_dict.values()
    len_freq = collections.Counter(val_list)
    len_freq_dict = dict(len_freq)
    len_freq_dict_sorted = sorted(len_freq_dict.items(), key=lambda item:item[1])
    print "type of len_freq", type(len_freq)
    print "type of len_freq_dict", type(len_freq_dict)
    print "len of len_freq_dict", len(len_freq_dict)
    print "data of len_freq", len_freq_dict_sorted
    # print "data of len_freq", len_freq_dict_sorted


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_folder_path', dest='input_folder_path', help='The data folder saving binaries information.', type=str, required=True)
    # parser.add_argument('-o', '--output_path', dest='output_path' ,help='The file saving the input for embedding model.', type=str, required=False, default='embed_input')
    # parser.add_argument('-e', '--error_path', dest='error_path' ,help='The file saving all error information. ', type=str, required=False, default='error_log')
    # parser.add_argument('-m', '--int2insn_map_path', dest='int2insn_map_path', help='The file saving the map information (int -> instruction (int list)).', type=str, required=False, default='int2insn.map')

    args = parser.parse_args()

    config_info = {
        'input_folder_path': args.input_folder_path,
        # 'output_path': args.output_path,
        # 'error_path': args.error_path,
        # 'int2insn_map_path': args.int2insn_map_path
    }
    return config_info

def main():
    config_info = get_config()
    # get_file_tree(config_info['input_folder_path'])
    load_func_tree("./func_list/func_len_dict_x64.pkl")
    # load_bin_file(config_info['input_folder_path'])
    # load_func_tree("./func_list/func_dict_x86.pkl")
    # load_func_tree("./func_list/func_dict_x64.pkl")
    # my_vocab=GetVocab(config_info)

if __name__ == '__main__':
    main()