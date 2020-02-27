#!/usr/bin/env python
# coding=utf-8
import mxnet as mx
import numpy as np
import logging
logging.getLogger().setLevel(logging.DEBUG)
batch_size=1000
def get_lenet():
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
    return lenet
def train():
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
    model = mx.model.FeedForward(
            ctx = mx.gpu(0),
            symbol = get_lenet(),
            num_epoch = 50,
            learning_rate = 0.01,
            )
    model.fit(
            X = train_iter,
            eval_data = test_iter,
            batch_end_callback = mx.callback.Speedometer(batch_size,100)
            )
train()
