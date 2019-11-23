def prepare_data_deprecated():
    # --------------start(read data from eval result)----------------------------
    # eval.main()

    # total_result['cost'].append(cost_result)
    # total_result['pred'].append(pred_result)
    # total_result['func_name'].append(func_name_list)
    # total_result['data'].append(data_list)
    print "config_info['input_data']", input_data
    result_file = open(config_info['input_data'])
    total_result = pickle.load(result_file)
    result_file.close()

    # hex of binary data
    print "type of total_result['data']", type(total_result['data'][0][0][:])
    print "shape of total_result['data']", total_result['data'][0][0][:].shape
    embed_data_list = total_result['data'][0][0][:]
    print "len of total_result['data']", len(embed_data_list)
    print "hex_data_list", embed_data_list[:]

    # original embedding data
    print "type of total_result['inst_types']", type(
        total_result['inst_bytes'][0][0][:])
    print "shape of total_result['inst_types']", len(
        total_result['inst_bytes'][0][0][:])
    print "data of total_result['inst_types']", total_result['inst_bytes'][0][0][:]
    hex_data_list = total_result['inst_bytes'][0][0]
    print "type of embed_data_list", type(hex_data_list)
    print "embedding data of embed_data_list", hex_data_list

    # int of embedding data
    int2insn_map, int_data_list = converter.main(hex_data_list)
    print "type of int_data_list", type(int_data_list)
    print "int data of int_data_list", int_data_list
    print "type of int2insn_map", type(int2insn_map)
    print "data of int2insn_map", int2insn_map
    bin_data_list = [int2insn_map[k]
                     for k in int_data_list if k in int2insn_map]
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
        'data': np.asarray(new_batch_data['data'], dtype=np.float32),
        'label': np.asarray(new_batch_data['label'], dtype=np.float32),
        'length': np.asarray(new_batch_data['length'], dtype=np.uint16),
        'keep_prob_pl': np.asarray(keep_prob, dtype=np.float32)
    }
    print "type of feed_dict[data_pl]", type(feed_batch_dict['data'][0])
    print "len of feed_dict[data_pl]", len(feed_batch_dict['data'][0])
    print "data of feed_dict[data_pl]", feed_batch_dict['data'][0]
    # -------------- end (read data from eval result)----------------------------

    # eval_predict.predict()
    eval_predict.main(feed_batch_dict)
