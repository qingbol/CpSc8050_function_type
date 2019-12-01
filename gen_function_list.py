'''
Input: the whole dataset (in order to get the whole vocabulary)
Output:
output_path: the input for embedding model
error_path: save all the error information in (especially when two distinct instructions map to same integer)
int2insn_map_path: the map information(int -> insn (int list))
'''
# split func_path
# python gen_function_list.py -i /Users/tarus/OnlyInMac/dataset/eklavya/clean_pickles
# python gen_function_list.py -i /scratch2/qingbol/CpSc8580EKLAVYA/dataset/clean_pickles

import pickle
import argparse
# import sys
import os
# import insn_int
import random


def get_file_tree(folder_path):
    file_tree = open("./func_list/file_tree.txt", "w")
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

    # func_len_dict_x64 = {}
    # func_len_dict_x86 = {}

    for root, dirs, files in os.walk(folder_path):
        i += 1
        file_tree.write("---------{} layer----------\n".format(i))
        file_tree.write("root is {}:\n".format(root))
        file_tree.write("dirs is :{} \n".format(dirs))
        file_tree.write("files is : {} \n".format(files))

        root_basename = os.path.basename(root)
        if root_basename == "x64":
            # file_list_x64 = files
            split_func_dict_x64["train"], split_func_dict_x64["test"] \
                = get_func_list(root, files)
            pickle.dump(split_func_dict_x64, open(
                "./func_list/func_dict_x64.lst", "wb"))
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
    func_num = 0
    func_num_sub = 0
    # func_len_dict = {}
    pkl_file_total = len(file_list)
    for pkl_file in file_list:
        pkl_file_num += 1
        # print pkl_file_num
        # print pkl_file_total
        progress = float(pkl_file_num)/pkl_file_total
        # print progress
        print "root:{0}, len(func_list):{1},progress : {2:.1f}%\r" . \
            format(os.path.basename(root), len(func_list), progress*100),
        if 'gcc' in pkl_file:
            pkl_file_path = os.path.join(root, pkl_file)
            f = open(pkl_file_path)
            temp = pickle.load(f)
            f.close()
            for func_name in temp["functions"]:
                func_num += 1
                func_len = len(temp['functions'][func_name]['inst_bytes'])
                num_args = temp['functions'][func_name]['num_args']
                # if func_len == 40:
                # if func_len == 40 and num_args > 3:
                if func_len == 40 and num_args ==3 :
                # if func_len == 40:
                    func_num_sub += 1
                    func_list.append(pkl_file + "#" + func_name)
                # func_list.append(pkl_file + "#" + func_name)
                # if func_num < 10:
                #     print func_len
                # else:
                #     break
            # print func_list
    random.shuffle(func_list)
    index = int(len(func_list) * 0.8)
    print "sub_num{}|total_num{}".format(func_num_sub, func_num)
    print "len of func_list", index
    func_list_train = func_list[:index]
    func_list_test = func_list[index:]
    return func_list_train, func_list_test

    # print root
    # print dirs
    # print files


def load_func_tree(func_list):
    split_func_dict = pickle.load(open(func_list))
    print split_func_dict["train"][0:20]


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_folder_path', dest='input_folder_path',
                        help='The data folder saving binaries information.',
                        type=str, required=False,
                        default='/Users/tarus/OnlyInMac/dataset/eklavya/clean_pickles')
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
    get_file_tree(config_info['input_folder_path'])
    # load_func_tree("./func_list/fun_dict_x86.pkl")
    # load_func_tree("./func_list/fun_dict_x64.pkl")
    # my_vocab=GetVocab(config_info)


if __name__ == '__main__':
    main()
