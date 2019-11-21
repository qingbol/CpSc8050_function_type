'''
for test the converter
'''
import numpy as np
import pickle
import argparse
import eval
import eval_predict
import converter
import dataset
import sys
import os

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--int2insn_map_path', dest='int2insn_path', help='The pickle file saving int -> instruction mapping.', type=str, required=False, default='/Users/tarus/TarusHome/10SrcFldr/CpSc8580EKLAVYA_py27/code/embedding/int2insn.map')
    # parser.add_argument('-i', '--int2insn_map_path', dest='int2insn_path', help='The pickle file saving int -> instruction mapping.', type=str, required=True)
    parser.add_argument('-d', '--data_folder', dest='data_folder', help='The data folder of testing dataset.', type=str, required=False,default='/Users/tarus/OnlyInMac/dataset/eklavya/clean_pickles/x64')
    # parser.add_argument('-d', '--data_folder', dest='data_folder', help='The data folder of testing dataset.', type=str, required=True)
    parser.add_argument('-f', '--split_func_path', dest='func_path', help='The path of file saving the training & testing function names.', type=str, required=False, default='/Users/tarus/TarusHome/10SrcFldr/CpSc8580EKLAVYA_py27/code/embedding/func_list/func_dict_x64_len10.lst')
    # parser.add_argument('-f', '--split_func_path', dest='func_path', help='The path of file saving the training & testing function names.', type=str, required=True)
    parser.add_argument('-e', '--embed_path', dest='embed_path', help='The path of file saving embedding vectors.', type=str, required=False,default='/Users/tarus/OnlyInMac/dataset/eklavya/embed.pkl')
    # parser.add_argument('-e', '--embed_path', dest='embed_path', help='The path of file saving embedding vectors.', type=str, required=True)
    parser.add_argument('-o', '--output_dir', dest='output_dir', help='The directory to saved the evaluation result.', type=str, required=False, default='eval_output')
    # parser.add_argument('-o', '--output_dir', dest='output_dir', help='The directory to saved the evaluation result.', type=str, required=True)
    parser.add_argument('-m', '--model_dir', dest='model_dir', help='The directory saved the models.', type=str, required=False,default='/Users/tarus/OnlyInMac/dataset/eklavya/rnn_output/model')
    # parser.add_argument('-m', '--model_dir', dest='model_dir', help='The directory saved the models.', type=str, required=True)
    parser.add_argument('-t', '--label_tag', dest='tag', help='The type of labels. Possible value: num_args, type#0, type#1, ...', type=str, required=False, default='num_args')
    parser.add_argument('-dt', '--data_tag', dest='data_tag', help='The type of input data.', type=str, required=False, choices=['caller', 'callee'], default='callee')
    parser.add_argument('-pn', '--process_num', dest='process_num', help='Number of processes.', type=int, required=False, default=4)
    parser.add_argument('-ed', '--embedding_dim', dest='embed_dim', help='The dimension of embedding vector.', type=int, required=False, default=256)
    # parser.add_argument('-ml', '--max_length', dest='max_length', help='The maximun length of input sequences.', type=int, required=False, default=10)
    parser.add_argument('-ml', '--max_length', dest='max_length', help='The maximun length of input sequences.', type=int, required=False, default=10)
    # parser.add_argument('-ml', '--max_length', dest='max_length', help='The maximun length of input sequences.', type=int, required=False, default=500)
    parser.add_argument('-nc', '--num_classes', dest='num_classes', help='The number of classes', type=int, required=False, default=16)
    parser.add_argument('-do', '--dropout', dest='dropout', help='The dropout value.', type=float, required=False, default=1.0)
    parser.add_argument('-nl', '--num_layers', dest='num_layers', help='Number of layers in RNN.', type=int, required=False, default=3)
    parser.add_argument('-b', '--batch_size', dest='batch_size', help='The size of batch.', type=int, required=False, default=1)

    args = parser.parse_args()
    config_info = {
        'data_folder': args.data_folder,
        'func_path': args.func_path,
        'embed_path': args.embed_path,
        'tag': args.tag,
        'data_tag': args.data_tag,
        'process_num': args.process_num,
        'embed_dim': args.embed_dim,
        'max_length': args.max_length,
        'num_classes': args.num_classes,
        'output_dir': args.output_dir,
        'model_dir': args.model_dir,
        'dropout': args.dropout,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'int2insn_path': args.int2insn_path
    }
    return config_info

def main():
    print "entering tst main"
    config_info = get_config()
    data_folder = config_info['data_folder']
    func_path = config_info['func_path']
    embed_path = config_info['embed_path']
    tag = config_info['tag']
    data_tag = config_info['data_tag']
    process_num = int(config_info['process_num'])
    embed_dim = int(config_info['embed_dim'])
    max_length = int(config_info['max_length'])
    num_classes = int(config_info['num_classes'])
    model_dir = config_info['model_dir']
    output_dir = config_info['output_dir']
    int2insn_path = config_info['int2insn_path']
    batch_size = config_info['batch_size']

    #------------start(retriev the target function data)------------------------
    result_path = os.path.join(output_dir, 'data_batch_result_.pkl') 
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            data_batch = pickle.load(f)
    else:
        my_data = dataset.Dataset(data_folder, func_path, embed_path, process_num, embed_dim, max_length, num_classes, tag, int2insn_path)
        data_batch = my_data.get_batch(batch_size=batch_size)
        with open(result_path, 'w') as f:
            pickle.dump(data_batch, f)
        print('Save the test result !!! ... %s' % result_path)

    keep_prob = 1.0
    feed_batch_dict1 = {
        'data': data_batch['data'],
        'label': data_batch['label'],
        'length': data_batch['length'],
        'keep_prob_pl': keep_prob 
    }
    print "type of feed_batch_dict1['data']", type(feed_batch_dict1['data'])
    print "len of feed_batch_dict1['data']", len(feed_batch_dict1['data'])
    print "data of feed_batch_dict1['data']", feed_batch_dict1['data']
    eval_predict.main(feed_batch_dict1)
    #------------ end (retriev the target function data)------------------------


    #------------start(convert insn2int )---------------------------------------
    #original embedding data
    print "type of data_batch['inst_types']",type(data_batch['inst_bytes'][0])
    print "shape of data_batch['inst_types']",len(data_batch['inst_bytes'][0])
    print "data of data_batch['inst_types']",data_batch['inst_bytes'][0]
    hex_data_list = data_batch['inst_bytes'][0]
    print "type of embed_data_list", type(hex_data_list)
    print "embedding data of embed_data_list", hex_data_list

    #int of embedding data
    int2insn_map, int_data_list= converter.main(hex_data_list)
    print "type of int_data_list", type(int_data_list)
    print "int data of int_data_list", int_data_list
    print "type of int2insn_map", type(int2insn_map)
    print "data of int2insn_map", int2insn_map
    bin_data_list = [int2insn_map[k] for k in int_data_list if k in int2insn_map]
    # bin_data_list = [int2insn_map[int(k)] for k in int_data_list if int(k) in int2insn_map]
    print "type of bin_data_list", type(bin_data_list)
    print "len of bin_data_list", len(bin_data_list)
    print "data of bin_data_list", bin_data_list
    #------------ end (convert insn2int )---------------------------------------

    sys.exit(0)

    #--------------start(read data from eval result)----------------------------
    # eval.main()

    # total_result['cost'].append(cost_result)
    # total_result['pred'].append(pred_result)
    # total_result['func_name'].append(func_name_list)
    # total_result['data'].append(data_list)
    print "config_info['input_data']",input_data
    result_file = open(config_info['input_data'])
    total_result = pickle.load(result_file)
    result_file.close()

    #hex of binary data
    print "type of total_result['data']",type(total_result['data'][0][0][:])
    print "shape of total_result['data']",total_result['data'][0][0][:].shape
    embed_data_list = total_result['data'][0][0][:]
    print "len of total_result['data']",len(embed_data_list)
    print "hex_data_list",embed_data_list[:]
    
    #original embedding data
    print "type of total_result['inst_types']",type(total_result['inst_bytes'][0][0][:])
    print "shape of total_result['inst_types']",len(total_result['inst_bytes'][0][0][:])
    print "data of total_result['inst_types']",total_result['inst_bytes'][0][0][:]
    hex_data_list = total_result['inst_bytes'][0][0]
    print "type of embed_data_list", type(hex_data_list)
    print "embedding data of embed_data_list", hex_data_list

    #int of embedding data
    int2insn_map, int_data_list= converter.main(hex_data_list)
    print "type of int_data_list", type(int_data_list)
    print "int data of int_data_list", int_data_list
    print "type of int2insn_map", type(int2insn_map)
    print "data of int2insn_map", int2insn_map
    bin_data_list = [int2insn_map[k] for k in int_data_list if k in int2insn_map]
    # bin_data_list = [int2insn_map[int(k)] for k in int_data_list if int(k) in int2insn_map]
    print "type of bin_data_list", type(bin_data_list)
    print "len of bin_data_list", len(bin_data_list)
    print "data of bin_data_list", bin_data_list

    new_batch_data = {
        'data': [],
        'label': [],
        'length': [],
        # 'func_name':[],
        # 'inst_bytes':[]
    }
    new_batch_data['data'].append(embed_data_list)
    new_batch_data['length'].append(len(embed_data_list))
    # prepare_data()
    # data_pl, label_pl, length_pl, keep_prob_pl = eval_predict.placeholder_inputs(num_classes, max_length, embed_dim)
    keep_prob = 1.0
    feed_batch_dict = {
        'data': np.asarray(new_batch_data['data'],dtype=np.float32),
        'label': np.asarray(new_batch_data['label'],dtype=np.float32), 
        'length': np.asarray(new_batch_data['length'],dtype=np.uint16),
        'keep_prob_pl': np.asarray(keep_prob,dtype=np.float32)
    }
    print "type of feed_dict[data_pl]", type(feed_batch_dict['data'][0])
    print "len of feed_dict[data_pl]", len(feed_batch_dict['data'][0])
    print "data of feed_dict[data_pl]", feed_batch_dict['data'][0]
    #-------------- end (read data from eval result)----------------------------

    # eval_predict.predict()
    eval_predict.main(feed_batch_dict)



if __name__ == '__main__':
    main()