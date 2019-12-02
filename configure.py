
import sys
import argparse


def get_config(options):
    print "options: ", options
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', '--feature_num', dest='feature_num',
                        help='the feature_num for evaluation', type=int, required=False, default=15)
    parser.add_argument('-idx', '--index_of_function', dest='func_index',
                        help='the index of function to be test', type=int, required=False, default=1)
    parser.add_argument('-sn', '--sample_num', dest='sample_num',
                        help='the number of artificial data', type=int, required=False, default=500)
    # parser.add_argument('-ml', '--max_length', dest='max_length',
    #                     help='The maximun length of input sequences.', type=int, required=False, default=40)
    # parser.add_argument('-ml', '--max_length', dest='max_length', help='The maximun length of input sequences.', type=int, required=False, default=500)
    parser.add_argument('-i', '--int2insn_map_path', dest='int2insn_path', help='The pickle file saving int -> instruction mapping.',
                        type=str, required=False, default='/Users/tarus/TarusHome/10SrcFldr/CpSc8580EKLAVYA_py27/code/embedding/int2insn.map')
    # parser.add_argument('-i', '--int2insn_map_path', dest='int2insn_path', help='The pickle file saving int -> instruction mapping.', type=str, required=True)
    parser.add_argument('-d', '--data_folder', dest='data_folder', help='The data folder of testing dataset.',
                        type=str, required=False, default='/Users/tarus/OnlyInMac/dataset/eklavya/clean_pickles/x64')
    # parser.add_argument('-d', '--data_folder', dest='data_folder', help='The data folder of testing dataset.', type=str, required=True)
    parser.add_argument('-f', '--split_func_path', dest='func_path', help='The path of file saving the training & testing function names.',
                        type=str, required=False, default='/Users/tarus/TarusHome/11git_repo/CpSc8580Lemna_fuction_type/func_list/func_dict_x64_len40_gcc_arg4-n.lst')
    # parser.add_argument('-f', '--split_func_path', dest='func_path', help='The path of file saving the training & testing function names.', type=str, required=True)
    parser.add_argument('-e', '--embed_path', dest='embed_path', help='The path of file saving embedding vectors.',
                        type=str, required=False, default='/Users/tarus/OnlyInMac/dataset/eklavya/embed.pkl')
    # parser.add_argument('-e', '--embed_path', dest='embed_path', help='The path of file saving embedding vectors.', type=str, required=True)
    parser.add_argument('-m', '--model_dir', dest='model_dir', help='The directory saved the models.',
                        type=str, required=False, default='/Users/tarus/OnlyInMac/dataset/eklavya/rnn_output/model')
    # parser.add_argument('-m', '--model_dir', dest='model_dir', help='The directory saved the models.', type=str, required=True)

    parser.add_argument('-o', '--output_dir', dest='output_dir',
                        help='The directory to saved the evaluation result.', type=str, required=False, default='eval_output')
    # parser.add_argument('-o', '--output_dir', dest='output_dir', help='The directory to saved the evaluation result.', type=str, required=True)
    parser.add_argument('-t', '--label_tag', dest='tag',
                        help='The type of labels. Possible value: num_args, type#0, type#1, ...', type=str, required=False, default='num_args')
    parser.add_argument('-dt', '--data_tag', dest='data_tag', help='The type of input data.',
                        type=str, required=False, choices=['caller', 'callee'], default='callee')
    parser.add_argument('-pn', '--process_num', dest='process_num',
                        help='Number of processes.', type=int, required=False, default=40)
    parser.add_argument('-ed', '--embedding_dim', dest='embed_dim',
                        help='The dimension of embedding vector.', type=int, required=False, default=256)
    # parser.add_argument('-ml', '--max_length', dest='max_length', help='The maximun length of input sequences.', type=int, required=False, default=10)
    parser.add_argument('-nc', '--num_classes', dest='num_classes',
                        help='The number of classes', type=int, required=False, default=16)
    parser.add_argument('-do', '--dropout', dest='dropout',
                        help='The dropout value.', type=float, required=False, default=1.0)
    parser.add_argument('-nl', '--num_layers', dest='num_layers',
                        help='Number of layers in RNN.', type=int, required=False, default=3)
    parser.add_argument('-b', '--batch_size', dest='batch_size',
                        help='The size of batch.', type=int, required=False, default=1)

    args = parser.parse_args(options)
    config_info = {
        'data_folder': args.data_folder,
        'func_path': args.func_path,
        'embed_path': args.embed_path,
        'tag': args.tag,
        'data_tag': args.data_tag,
        'process_num': args.process_num,
        'embed_dim': args.embed_dim,
        # 'max_length': args.max_length,
        'num_classes': args.num_classes,
        'output_dir': args.output_dir,
        'model_dir': args.model_dir,
        'dropout': args.dropout,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'int2insn_path': args.int2insn_path,
        'sample_num': args.sample_num,
        'func_index': args.func_index,
        'feature_num': args.feature_num
    }
    return config_info
