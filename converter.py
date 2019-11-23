'''
Input: the instruction set of one function
Output:
output_path: the input for embedding model
error_path: save all the error information in (especially when two distinct instructions map to same integer)
int2insn_map_path: the map information(int -> insn (int list))
'''

# import pickle
import cPickle as pickle
# import argparse
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
        insn_int = 0
        for idx, value in enumerate(insn_list):
            insn_int = insn_int + value*(256**idx)
        return insn_int


def main(list2d):
    # config_info = get_config()
    my_vocab = ConvertInsn2int(list2d)
    int2insn_map, insn2int_list = my_vocab.convert_insn2int()
    return int2insn_map, insn2int_list


if __name__ == '__main__':
    list2d = []
    main(list2d)
