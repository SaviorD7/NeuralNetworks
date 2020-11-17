import numpy as np
import sys

feature_num = 15
filename = 'car.data'

output_number_files = int(sys.argv[1])


data = np.fromfile(filename, sep=' ')
data = data.reshape(data.shape[0] // feature_num, feature_num)

np.random.shuffle(data)


step = int(data.shape[0] / output_number_files)
print('Step of split is: ', step)

for i in range(output_number_files):
    file_name = "data/car_part{}.data".format(i)
    print('Step %d is starting ...' % (i))
    print('Split from %d to %d ' % (step*i, step*(i+1)))
    output_array = data[step*i:step*(i+1)]
    output_array.tofile(file_name, sep=' ')
