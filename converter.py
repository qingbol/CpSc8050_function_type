'''
Input: the instruction set of one function
Output:
output_path: the input for embedding model
error_path: save all the error information in (especially when two distinct instructions map to same integer)
int2insn_map_path: the map information(int -> insn (int list))
'''

import pickle
import argparse
import sys
import os
import insn_int

class ConvertInsn2int(object):
    def __init__(self, list2d):
        self.list2d = list2d
        self.int2insn_map = {}
        self.insn2int_list = []

    def convert_insn2int(self):
        print_flag = 0
        for insn in self.list2d:
            print_flag += 1
            # if print_flag < 10:
            #     print type(insn)
            #     print insn
            int_value = self.insn2int_inverse(insn)
            if int_value in self.int2insn_map:
                if self.int2insn_map[int_value] != insn:
                    print "error, dont match"
            else:
                self.int2insn_map[int_value] = insn
            self.insn2int_list.append(int_value)
            # self.insn2int_list.append(str(int_value))
        return self.int2insn_map, self.insn2int_list

    '''
    transfer the instruction to integer with inverse order
    example:
    [72,137,229] ==>15042888 (72+137*256+229*256*256)
    [243, 15, 16, 13, 205, 0, 0, 0] ==> 880687452147 (243+15*256+16*256*256+13*256*256*256+13*256*256*256*256+205*256*256*256*256*256)
    :param insn_list:
    :return insn_int:
    '''
    def insn2int_inverse(self, insn_list):
        insn_int=0
        for idx, value in enumerate(insn_list):
            insn_int = insn_int + value*(256**idx)
        return insn_int

def get_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input_folder_path', dest='input_folder_path', help='The data folder saving binaries information.', type=str, required=True)
    parser.add_argument('-o', '--output_path', dest='output_path' ,help='The file saving the input for embedding model.', type=str, required=False, default='embed_input')
    parser.add_argument('-e', '--error_path', dest='error_path' ,help='The file saving all error information. ', type=str, required=False, default='error_log')
    parser.add_argument('-m', '--int2insn_map_path', dest='int2insn_map_path', help='The file saving the map information (int -> instruction (int list)).', type=str, required=False, default='int2insn.map')
    args = parser.parse_args()
    config_info = {
        # 'input_folder_path': args.input_folder_path,
        'output_path': args.output_path,
        'error_path': args.error_path,
        'int2insn_map_path': args.int2insn_map_path
    }
    return config_info

# def main():
def main(list2d):
    config_info = get_config()
    my_vocab=ConvertInsn2int(list2d)
    int2insn_map,insn2int_list = my_vocab.convert_insn2int()
    return int2insn_map,insn2int_list

if __name__ == '__main__':
    list2d = []
    main(list2d)