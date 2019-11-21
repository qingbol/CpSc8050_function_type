'''
for test the converter
'''
import numpy as np
import pickle
import argparse
import eval
import eval_predict
import converter

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-id','--input_data', dest='input_data', 
        help='import the result of eval.py', type=str, required=False, 
        default='/Users/tarus/TarusHome/10SrcFldr/CpSc8580EKLAVYA_py27_lemna/code/RNN/test/eval_output/test_result_4600.pkl')
    parser.add_argument('-nc', '--num_classes', dest='num_classes', help='The number of classes', type=int, required=False, default=16)
    parser.add_argument('-ed', '--embedding_dim', dest='embed_dim', help='The dimension of embedding vector.', type=int, required=False, default=256)
    parser.add_argument('-ml', '--max_length', dest='max_length', help='The maximun length of input sequences.', type=int, required=False, default=10)
    
    args = parser.parse_args()
    config_info = {
        'input_data':args.input_data,
        'max_length': args.max_length,
        'num_classes': args.num_classes,
        'embed_dim': args.embed_dim
    }

    return config_info

def main():
    config_info = get_config()
    input_data = config_info['input_data']
    embed_dim = int(config_info['embed_dim'])
    max_length = int(config_info['max_length'])
    num_classes = int(config_info['num_classes'])
    print "entering tst main"

    #retriev the target function data
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
        # 'label': [],
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
        # label_pl: data_batch['label'],
        'length': np.asarray(new_batch_data['length'],dtype=np.uint16),
        'keep_prob_pl': np.asarray(keep_prob,dtype=np.float32)
    }
    print "type of feed_dict[data_pl]", type(feed_batch_dict['data'][0])
    print "len of feed_dict[data_pl]", len(feed_batch_dict['data'][0])
    print "data of feed_dict[data_pl]", feed_batch_dict['data'][0]

    # eval_predict.predict()
    eval_predict.main(feed_batch_dict)



if __name__ == '__main__':
    main()