from paddle_fl.paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
import paddle_fl.paddle_fl.dataset.femnist as femnist
import numpy as np
import sys
import paddle
import paddle.fluid as fluid
import logging
import math
import time
import xray_dataset



trainer_id = int(sys.argv[1]) 
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
job._scheduler_ep = "127.0.0.1:9091"  # Inform the scheduler IP to trainer
print(job._target_names)
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer._current_ep = "127.0.0.1:{}".format(9000 + trainer_id)
place = fluid.CPUPlace()
trainer.start(place)
print('Trainer step: %d' % (trainer._step))
test_program = trainer._main_program.clone(for_test=True)

img = fluid.layers.data(name='img', shape=[1, 150, 150], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
#feeder = fluid.DataFeeder(feed_list=[img, label], place=fluid.CPUPlace())

data_path_test = '../data/data52182/chest_xray/test'
data_path_train = '../data/data52182/chest_xray/train'

uploaded_train_data = xray_dataset.upload_data(data_path_train)
uploaded_test_data = xray_dataset.upload_data(data_path_test)

def train_test(train_test_program):
    acc_set = []
    for data in uploaded_test_data:
        acc_np = trainer.run( 
            feed={'img': np.array([data['data']]).astype('float32'), 'label': np.array([data['label']]).astype('int64')}, fetch=["accuracy_0.tmp_0"])
        #print(acc_np)
        acc_set.append(float(acc_np[0]))
    acc_val_mean = np.array(acc_set).mean()
    return acc_val_mean

data_path_test = 'data/data52182/chest_xray/test'
data_path_train = 'data/data52182/chest_xray/train'

epoch_id = 0
step = 0
epoch = 20
count_by_step = False
if count_by_step:
    output_folder = "model_node%d" % trainer_id
else:
    output_folder = "model_node%d" % trainer_id

file_name = "logs/trainer%d.log" % trainer_id
print("Log file with %d id is ready to be created." % (trainer_id))


while not trainer.stop():
    count = 0
    epoch_id += 1
    if epoch_id > epoch:
        break

    f1 = open(file_name, 'a')
    f1.write("{} Epoch {} start train\n".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), epoch_id))
    f1.close()
    #train_data,test_data= data_generater(trainer_id,inner_step=trainer._step,batch_size=64,count_by_step=count_by_step)
      
    for data in uploaded_train_data:
        acc =  trainer.run(
                feed={'img': np.array([data['data']]).astype('float32'), 'label': np.array([data['label']]).astype('int64')}, fetch=["accuracy_0.tmp_0"])

    acc_val = train_test(train_test_program=test_program)
    f1 = open(file_name, 'a')
    f1.write("\nTest with epoch %d, accuracy: %s\n" % (epoch_id, acc_val))
    f1.close()
    save_dir = (output_folder + "/epoch_%d") % epoch_id
    trainer.save_inference_program(output_folder)