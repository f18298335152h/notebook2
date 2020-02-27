#!/usr/bin/env python
# coding=utf-8
import mxnet as mx
import numpy as np
import tensorflow as tf
from mxboard import SummaryWriter
import tensorboard

#logwriter = SummaryWriter(logdir='./logs', flush_secs=30, verbose=False)
#batch_size=1000
#cnt_step=0
num_examples = 455576
batch_size = 1000
num_batches = num_examples / batch_size
train_iter = mx.io.ImageRecordIter(
                path_imgrec="prefix_train.rec", 
                data_shape=(3,32,32),
                batch_size=1000,
                 )
test_iter = mx.io.ImageRecordIter(            
                path_imgrec="prefix_test.rec",
                data_shape=(3,32,32),          
                batch_size=1000,               
                 )
 
data = mx.symbol.Variable('data')
#first cov
conv1 = mx.symbol.Convolution(data = data ,kernel=(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data = conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data = tanh1,pool_type="max",kernel=(2,2), stride=(2,2))

#sencond conv
conv2 = mx.symbol.Convolution(data = pool1,kernel=(4,4), num_filter=50)
tanh2 = mx.symbol.Activation(data = conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data = tanh2,pool_type="max",kernel=(2,2), stride=(2,2))

#first fc
#Flatten [batch,channel,length,width] ---->[batch,list]
flatten = mx.symbol.Flatten(data = pool2)
fc1 =  mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.symbol.Activation(data = fc1, act_type="tanh")
#second fc
fc2 =  mx.symbol.FullyConnected(data=tanh3, num_hidden=2)
#loss
lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
mx.viz.plot_network(lenet)

#module
model = mx.mod.Module(symbol=lenet,context=mx.gpu())
optimizer_params={
        'learning_rate':0.01,
        'momentum':0.9,
        'wd':0.0001,
        'lr_scheduler':mx.lr_scheduler.FactorScheduler(num_batches,factor=0.9)
        
        }
batch_end_callbacks=[mx.callback.Speedometer(batch_size,100)]
epoch_end_callbacks = []
eval_batch_end_callbacks = []
eval_end_callbacks = []
summary_writer = SummaryWriter('./logs',flush_secs=30, verbose=False)
# monitor train accuracy
def monitor_train_acc(param):
    if param.nbatch + 1 == num_batches:
        metric = dict(param.eval_metric.get_name_value())
        summary_writer.add_scalar(tag='train-accuracy', metric['accuracy'])
batch_end_callbacks.append(monitor_train_acc)
#monitor validation accuracy
def monitor_eval_acc(param):
    metric = dict(param.eval_metric.get_name_value())
    summary_writer.add_scalar(tag='eval-accuracy', metric['accuracy'])
eval_end_callbacks.append(monitor_eval_acc)

def monitor_fc1_gradient(g):
    summary_writer.add_scalar(tag='fc1-backward-weight', g.asnumpy().flatten())
    stat = mx.nd.norm(g) / np.sqrt(g.size)
    return stat
monitor = mx.mon.Monitor(100,monitor_fc1_gradient,pattern='fc1_backward_weight')

def monitor_fc1_weight(param):
    if param.nbatch % 100 ==0:
        arg_params,aux_params = param.locals['self'].get_params()
        summary_writer.add_scalar(tag='fc1-weight', value=arg_params['fc1_weight'].asnumpy().flatten())
batch_end_callbacks.append(monitor_fc1_weight)

model.fit(
            train_data = train_iter,
            begin_epoch = 0,
            num_epoch = 20,
            eval_data = test_iter,
            eval_metric = 'accuracy',
            optimizer = 'sgd',
            optimizer_params=optimizer_params,
            initializer = mx.init.Uniform(),
            batch_end_callback = batch_end_callbacks,
            eval_end_callback = eval_end_callbacks,
            monitor = monitor
            )
summary_writer.close()




