from numpy import genfromtxt
import numpy as np
import sys
import os
import shutil

if (os.path.exists('data')):
    shutil.rmtree('data')
    os.mkdir('data')
else:
    os.mkdir('data')

output_number_files = int(sys.argv[1])


input_array = genfromtxt('path/to/data', delimiter=',')

"""

It is remaking values , if you do not need just comment

"""
for i in range(input_array.shape[0]):
    input_array[i][7] -= 1

np.random.shuffle(input_array)

step = int(input_array.shape[0] / output_number_files)
print('Step of split is: ', step)

for i in range(output_number_files):
    file_name = "data/scs_part{}.data".format(i)
    print('Step %d is starting ...' % (i))
    print('Split from %d to %d ' % (step*i, step*(i+1)))
    output_array = input_array[step*i:step*(i+1)]
    output_array.tofile(file_name, sep=' ')

