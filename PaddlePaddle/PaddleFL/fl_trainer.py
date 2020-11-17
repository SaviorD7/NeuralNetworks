from paddle_fl.paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
import numpy as np
import six
import sys
import paddle
import paddle.fluid as fluid
import logging
import math
import time
import scs_dataset
import car_dataset

logging.basicConfig(
    filename="test.log",
    filemode="w",
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.DEBUG)

# file_name = str(sys.argv[2])
trainer_id = int(sys.argv[1])
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
job._scheduler_ep = "127.0.0.1:9091"  # Inform scheduler IP address to trainer
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer._current_ep = "127.0.0.1:{}".format(9000 + trainer_id)
place = fluid.CPUPlace()
trainer.start(place)

test_program = trainer._main_program.clone(for_test=True)

file_name = 'data/car_part{}.data'.format(trainer_id)

train_reader = paddle.batch(
    paddle.reader.shuffle(
        car_dataset.train(file_name), buf_size=5000),
    batch_size=32)
test_reader = paddle.batch(car_dataset.test(file_name), batch_size=32)

x = fluid.layers.data(name='x', shape=[1, 14], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
feeder = fluid.DataFeeder(feed_list=[x, label], place=fluid.CPUPlace())


def train_test(train_test_program, train_test_feed, train_test_reader):
    acc_set = []
    loss_set = []
    for test_data in train_test_reader():
        acc_np, loss = trainer.exe.run(program=train_test_program,
                                       feed=train_test_feed.feed(test_data),
                                       fetch_list=["accuracy_0.tmp_0", "reduce_mean_0.tmp_0"])
        acc_set.append(float(acc_np[0]))
        loss_set.append(float(loss[0]))
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_mean = np.array(loss_set).mean()
    return acc_val_mean, avg_loss_mean


output_folder = "models/model_node%d" % trainer_id
epoch_id = 0
step = 0
while not trainer.stop():
    epoch_id += 1
    if epoch_id > 10:
        break
    print("{} Epoch {} start train".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), epoch_id))
    for step_id, data in enumerate(train_reader()):
        acc, loss = trainer.run(feeder.feed(data), fetch=["accuracy_0.tmp_0", "reduce_mean_0.tmp_0"])
        step += 1

    acc_val, avg_loss = train_test(
        train_test_program=test_program,
        train_test_reader=test_reader,
        train_test_feed=feeder)

    print("Test with epoch %d, accuracy: %s , loss: %s" % (epoch_id, acc_val, avg_loss))

    save_dir = (output_folder + "/epoch_%d") % epoch_id
    trainer.save_inference_program(output_folder)

print("{} Train is ended".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
