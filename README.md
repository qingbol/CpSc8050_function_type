# CpSc8580_function_type

## Introduction:

This project aimed to explian the result of RNN, which is used to predict the
function type signature(argument numbers/types), by the method proposed in LEMNA.
I follow paper of "Neural Nets Can Learn Function Type Signatures From Binaries"
to build the RNN based on the dataset that contains 5168 binaries. The author
didn't give some detailed information to run the project and didn't release some
scripts, such as the scripts used to get the function list.
So I developed some scripts by myself and managed to run the project. <br />
Because the structure of the dataset is totally different from that of LEMNA, so
I almost rewriter all the codes in LEMNA to make the result of RNN explainable.

## Dataset:

You can get all the needed data from below link:
https://drive.google.com/drive/folders/1oiSp5Ak1Lg6Dyg6WIJylIkW9dgmpJLBE?usp=sharing

- [clean_pickles.tar.gz]. The compressed file saves the binary code and the ground truth of the function arguments for sanitized functions. For this dataset, we removed the functions which are duplicates of other functions in the dataset. Given that the same piece of code compiled with different binaries will result in different offsets generated,we chose to remove all direct address used by instructions found in the function. For example, the instruction _'je 0x98'_ are represented as _'je '_. After the substituion, we hash the function and remove functions with the same hashes. Other than duplicates, we also removed functions with less than four instructions as these small functions typically do not have any operation on arguments.

- [embed.pkl] The data after word embedding which contains the embedding value
  of each instruction.

- [int2insn.map] The dictionary which contains (int value: bianary value) pairs
  of each instruction.

- [model] The folder contains the trained model of RNN.

- [func_list] The folder contains the list of each function name. We need this
  to retrieve the needed data of certain function.

## How to run this project

### Get the function list:

- gen_function_len.py is used to get the statics of all the functions. After
  run it, we can know how many instructions each function has. So we can choose
  the function which has the intended length.<br />

  > python gen_function_len.py -i /folder_of_clean_pickles

  <!-- > python gen_function_len.py -i /Users/tarus/OnlyInMac/dataset/eklavya/clean_pickles -->

- gen_function_list.py is used to get the list of function name combind with
  the binary file name. Only after we get this list, we can train the rnn model
  and use LEMNA to explain the result.
  Also you can set the rate of dataset used for training and testing.<br />
  > python gen_function_list.py -i /folder_of_clean_pickles
  <!-- > python gen_function_list.py -i /Users/tarus/OnlyInMac/dataset/eklavya/clean_pickles -->

### Set the function list and length of this function.

In xai_arg_num.py, modify the parameter in these two below statements.

<pre><code>
parser.add_argument('-f', '--split_func_path', dest='func_path', help='The path of file saving the training & testing function names.',type=str, required=False, default='/path_to_your_function_list/func_dict_xxx.lst')

parser.add_argument('-ml', '--max_length', dest='max_length', help='The maximun length of input sequences.', type=int, required=False, default=40)
</code></pre>

### Set the specific function you want to explain

In xai_arg_num.py, modify the parameter in this statement.

<pre><code>
for index, func_name in enumerate(func_lst):
    self.func_name = func_name
    if index != 1 :
        continue
</code></pre>

### Run the entry function.

<pre><code>
 python xai_arg_num.py
</code></pre>

## The workflow in entry function

Take function humidity_str in gcc-64-O1-sg3utils-sg_logs.pkl as example, which
contains 40 instructions.
The workflow in this entry function can be summarized as folllow:

### Read embedding value and binary value of the target function.<br />

The binary format of this function:

> [[65, 85], [65, 84], [73, 137, 245], [85], [83], [72, 99, 239], [72, 141, 117, 5], [73, 137, 204], [72, 99, 218], [72, 131, 236, 8], [73, 139, 125, 0], [72, 131, 195, 14], [72, 193, 227, 5], [232, 53, 194, 11, 0], [73, 137, 69, 0], [199, 4, 40, 122, 58, 80, 58], [72, 137, 222], [198, 68, 40, 4, 0], [73, 139, 60, 36], [232, 25, 194, 11, 0], [73, 137, 4, 36], [72, 141, 140, 24, 64, 254, 255, 255], [72, 139, 5, 230, 177, 19, 0], [190, 128, 130, 86, 0], [72, 141, 121, 8], [72, 137, 1], [72, 139, 5, 139, 179, 19, 0], [72, 131, 231, 248], [72, 137, 129, 184, 1, 0, 0], [72, 41, 249], [72, 41, 206], [129, 193, 192, 1, 0, 0], [193, 233, 3], [243, 72, 165], [72, 131, 196, 8], [91], [93], [65, 92], [65, 93], [195]]

The embedding value of this function:

> [[ 4.9242502e-01 -6.3213439e+00 -2.1821052e-01 ... 5.3236794e-01
> -4.7061494e-01 7.2240531e-02][-1.4824426e-01 7.6710269e-02 2.1449502e-03 ... -9.8106312e-03 -1.5178236e-01 -4.9693279e-02] > [-1.4415282e-01 6.8571813e-02 -8.1709184e-04 ... 2.6113811e-04
> > -1.5341897e-01 -4.9748953e-02]
> ...
> [ 1.7497748e+00 -1.2347011e-01 -1.6310322e+00 ... -1.3755658e+00
> -1.6481979e-01 -1.5915378e+00][ 7.6243961e-01 -2.6772411e+00 1.3527184e+00 ... 9.2628604e-01 -1.7080919e-01 1.4312527e+00]
> [ 5.9697060e-03 -4.4744587e-01 5.3603011e-01 ... -1.8529317e-01

    1.2032903e+00  2.1198967e+00]]]

The label of this function:

> [[ 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

### Generate artificial data.

generate 500(or the number you think is better which is a superparameter) artificial function by randomly set some instructions to 0.

### Predict the argument number of each above functions.

The result are like below:

> [3 5 2 4 5 4 2 5 2 1 2 2 3 3 4 5 0 2 0 5 0 5 3 4 2 3 3 5 4 5 4 3 5 3 2 2 2
> 2 2 2 5 2 0 1 2 3 1 4 3 6 4 5 3 4 0 0 0 3 2 4 3 5 2 5 2 3 4 0 1 1 2 5 3 5
> 2 2 3 5 4 0 3 4 4 3 5 4 1 3 2 2 5 5 3 2 3 5 3 4 3 2 3 3 3 2 2 3 3 2 0 1 1
> 4 2 2 0 3 1 3 2 5 2 3 2 2 2 4 0 3 5 5 2 0 0 1 2 3 2 3 3 0 0 5 0 5 3 2 0 2
> 2 1 2 1 5 2 0 0 1 0 5 2 2 4 0 4 3 3 2 2 2 3 0 4 5 2 4 2 3 5 3 4 2 1 3 0 4
> 2 3 3 3 3 1 2 4 4 3 4 2 5 3 0 2 3 3 2 3 2 3 3 3 4 2 2 2 2 3 3 1 3 2 2 5 3
> 3 2 4 5 2 2 2 2 2 2 4 2 3 3 2 4 2 2 0 5 2 4 3 3 0 5 4 5 3 2 3 5 5 5 2 2 2
> 2 2 5 3 3 2 5 3 4 0 3 2 2 4 0 2 0 2 5 3 4 4 3 3 3 5 3 0 0 5 2 2 3 0 2 2 5
> 2 0 2 5 5 3 2 5 2 3 4 3 5 3 0 0 3 5 5 3 3 1 2 2 2 2 3 3 3 5 3 3 2 3 3 0 0
> 4 3 3 2 1 1 2 0 3 1 5 4 2 5 3 3 4 2 3 1 3 3 2 4 0 5 1 4 2 3 3 3 3 3 3 3 4
> 0 3 3 0 4 0 4 2 2 5 2 4 3 3 5 1 2 1 2 3 3 3 2 0 5 4 2 4 5 2 2 2 1 5 3 5 3
> 2 4 2 2 2 4 3 2 0 4 3 5 5 5 5 3 0 3 5 5 3 3 4 5 3 3 3 4 2 2 3 2 2 5 0 3 2
> 3 2 4 5 4 3 2 2 2 3 3 3 2 3 0 4 3 3 4 4 2 2 2 0 2 5 4 2 3 2 0 4 2 2 5 2 4
> 3 2 5 2 5 4 3 3 5 2 2 5 3 2 5 2 2 2 4 3]

Where the argument numbers of original function is 4, but after set some
instructions to 0, the model gives different prediction. The further away from
number 4, the more importance those instructions are.

### Use regression model to explain the result.

Feed the above 501 functions and it's corresponding predicted arguments number
to a regression model which can give the most significant instructions to
determin the argument number in this function. The coefficients array is as below:

> [-0.33 -0.33 -0.33 1.62 1.62 1.62 1.62 1.62 1.62 1.62 1.27 0.32
> 0.32 0.32 0.32 0.32 0.32 0.32 0.32 0.32 0.32 0.32 0.32 0.32
> 0.32 0.32 0.32 0.32 0.32 0.32 0.32 0.12 0.12 0.12 -1.05 -1.05
> -1.05 -1.05 -1.05 -1.05]
