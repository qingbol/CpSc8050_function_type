import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import random
np.set_printoptions(precision=3)
plt.style.use('seaborn-whitegrid')

fea_num = np.arange(0, 41, 5)
print ("fea_num: {}".format(fea_num))

# acc_lemna_pos = np.random.uniform(0.1, 0.9, 9)
acc_lemna_pos = np.array(
    [1, 0.728, 0.506, 0.462, 0.317, 0.304, 0.223, 0.184, 0.076])
acc_rand_pos = np.array(
    [1, 0.798, 0.728, 0.696, 0.519, 0.462, 0.354, 0.291, 0.076])

print ("type of acc_lemna_pos: {}".format(type(acc_lemna_pos)))
print ("acc_lemna_pos: {}".format(acc_lemna_pos))
print ("type of acc_rand_pos: {}".format(type(acc_rand_pos)))
print ("acc_rand_pos: {}".format(acc_rand_pos))

fig1 = plt.figure(figsize=(6, 4))
ax1 = fig1.add_subplot(111)
ax1.plot(fea_num, acc_lemna_pos, label="lemna", color='#FF8E00', marker='o')
ax1.plot(fea_num, acc_rand_pos, label="random", color='#004D7B', marker='o')
ax1.fill_between(fea_num, acc_lemna_pos, acc_rand_pos,
                 facecolor="0.75", alpha=0.7)
plt.title("Feature Deduction Test By my method")
plt.xticks(fea_num)
plt.xlabel("feature number")
plt.ylabel("PCR")
handles, labels = ax1.get_legend_handles_labels()
lgd = ax1.legend(handles, labels, loc='upper center',
                 bbox_to_anchor=(0.9, 0.9))
# lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.15,1))
ax1.grid('on')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.savefig('positive_without.png')
# plt.show()
# -------------------------------------------------------------------------
acc_lemna_neg = np.array(
    [0.260, 0.399, 0.582, 0.677, 0.804, 0.842, 0.854, 0.886, 1])
acc_rand_neg = np.array(
    [0.260, 0.329, 0.348, 0.430, 0.506, 0.601, 0.703, 0.873, 1])

fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(111)
ax2.plot(fea_num, acc_lemna_neg, label="lemna", color='#FF8E00', marker='o')
ax2.plot(fea_num, acc_rand_neg, label="random", color='#004D7B', marker='o')
ax2.fill_between(fea_num, acc_lemna_neg, acc_rand_neg,
                 facecolor="0.75", alpha=0.7)

handles, labels = ax2.get_legend_handles_labels()
lgd = ax2.legend(handles, labels, loc='upper center',
                 bbox_to_anchor=(0.1, 0.9))

plt.title("Feature Augmentation Test By my method")
plt.xticks(fea_num)
plt.xlabel("feature number")
plt.ylabel("PCR")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.savefig('negtive_without.png')
# plt.show()
# -----------------------------------------------------------------------------

acc_lemna_new = np.array(
    [0.076, 0.386, 0.608, 0.753, 0.848, 0.873, 0.886, 0.930, 1])
acc_rand_new = np.array(
    [0.076, 0.253, 0.348, 0.424, 0.557, 0.608, 0.798, 0.861, 1])
fig3 = plt.figure(figsize=(6, 4))
ax3 = fig3.add_subplot(111)
ax3.plot(fea_num, acc_lemna_new, label="lemna", color='#FF8E00', marker='o')
ax3.plot(fea_num, acc_rand_new, label="random", color='#004D7B', marker='o')
ax3.fill_between(fea_num, acc_lemna_new, acc_rand_new,
                 facecolor="0.75", alpha=0.7)

handles, labels = ax3.get_legend_handles_labels()
lgd = ax3.legend(handles, labels, loc='upper center',
                 bbox_to_anchor=(0.1, 0.9))

plt.title("Synthetic Test By my method")
plt.xticks(fea_num)
plt.xlabel("feature number")
plt.ylabel("PCR")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.savefig('new_without.png')

plt.show()
# rand1 = random.sample(range(0, 4), 4)
# print ("rand1 :{}".format(rand1))
# acc_lemna_pos = sorted(acc_lemna_pos)
# acc_lemna_pos = np.array(acc_lemna_pos)


# fake up some data
# spread = np.random.rand(50) * 100
# center = np.ones(25) * 50
# flier_high = np.random.rand(10) * 100 + 100
# flier_low = np.random.rand(10) * -100
# data = np.concatenate((spread, center, flier_high, flier_low), 0)

# # basic plot
# plt.boxplot(data)

# # notched plot
# plt.figure()
# plt.boxplot(data, 1)

# # change outlier point symbols
# plt.figure()
# plt.boxplot(data, 0, 'gD')

# # don't show outlier points
# plt.figure()
# plt.boxplot(data, 0, '')

# # horizontal boxes
# plt.figure()
# plt.boxplot(data, 0, 'rs', 0)

# # change whisker length
# plt.figure()
# plt.boxplot(data, 0, 'rs', 0, 0.75)

# # fake up some more data
# spread = np.random.rand(50) * 100
# center = np.ones(25) * 40
# flier_high = np.random.rand(10) * 100 + 100
# flier_low = np.random.rand(10) * -100
# d2 = np.concatenate((spread, center, flier_high, flier_low), 0)
# data.shape = (-1, 1)
# d2.shape = (-1, 1)
# # data = concatenate( (data, d2), 1 )
# # Making a 2-D array only works if all the columns are the
# # same length.  If they are not, then use a list instead.
# # This is actually more efficient because boxplot converts
# # a 2-D array into a list of vectors internally anyway.
# data = [data, d2, d2[::2, 0]]
# # multiple box plots on one figure
# plt.figure()
# plt.boxplot(data)

# plt.show()
