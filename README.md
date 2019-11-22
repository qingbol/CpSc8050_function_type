# CpSc8580_function_type

## Get the function list

- gen_function_len.py is used to get the statics of all the functions. After
  run it, we can know how many instructions each function has. So we can choose
  the function which has the intended length.<br />
  > python gen_function_len.py -i /Users/tarus/OnlyInMac/dataset/eklavya/clean_pickles
- gen_function_list.py is used to get the list of function name combind with
  the binary file name. Only after we get this list, we can train the rnn model
  and use LEMNA to explain the result.
  Also you can set the rate of dataset used for training and testing.<br />
  > python gen_function_list.py -i /Users/tarus/OnlyInMac/dataset/eklavya/clean_pickles
