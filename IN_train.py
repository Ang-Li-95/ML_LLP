#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import uproot
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt

import numpy as np
import awkward as ak

from utils.parameterSet import *
from utils.utilities import *

year="2017"

if not os.path.exists(dir_model):
      os.makedirs(dir_model)

def distance_corr(var_1, var_2, normedweight, power=2):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    
    va1_1, var_2 and normedweight should all be 1D tf tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """

    xx = tf.reshape(var_1, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_1)])
    xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])
 
    yy = tf.transpose(xx)
    amat = tf.math.abs(xx-yy)
    
    xx = tf.reshape(var_2, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_2)])
    xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])
    
    yy = tf.transpose(xx)
    bmat = tf.math.abs(xx-yy)
   
    amatavg = tf.reduce_mean(amat*normedweight, axis=1)
    bmatavg = tf.reduce_mean(bmat*normedweight, axis=1)
 
    minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)
    Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg*normedweight)

    minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)
    Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg*normedweight)

    ABavg = tf.reduce_mean(Amat*Bmat*normedweight,axis=1)
    AAavg = tf.reduce_mean(Amat*Amat*normedweight,axis=1)
    BBavg = tf.reduce_mean(Bmat*Bmat*normedweight,axis=1)
   
    epsilon = 1e-08
    if power==1:
        dCorr = tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight) + epsilon)
    elif power==2:
        dCorr = (tf.reduce_mean(ABavg*normedweight))**2/(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight) + epsilon)
    else:
        dCorr = (tf.reduce_mean(ABavg*normedweight)/tf.math.sqrt(tf.reduce_mean(AAavg*normedweight)*tf.reduce_mean(BBavg*normedweight) + epsilon))**power
  
    return dCorr

def m(O,Rr,Rs,Ra):
    '''
    The marshalling function that rearranges the object and relations into interacting terms
    In the code, ORr-ORs is used instead of ORr, ORs seperately
    '''
    return tf.concat([(tf.matmul(O,Rr)-tf.matmul(O,Rs)), Ra],1)

def phi_R(B):
    '''
    The phi_R function that predict the effect of each interaction by applying f_R to each column of B
    '''
    h_size = 50
    B_trans = tf.transpose(B,[0,2,1])
    B_trans = tf.reshape(B_trans, [-1,(Ds+Dr)]) 
    w1 = tf.Variable(tf.random.truncated_normal([(Ds+Dr),h_size], stddev=0.1), name="r_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h_size]), name="r_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(B_trans, w1)+b1)
    w2 = tf.Variable(tf.random.truncated_normal([h_size,h_size], stddev=0.1), name="r_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h_size]), name="r_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2)+b2)
    w3 = tf.Variable(tf.random.truncated_normal([h_size,h_size], stddev=0.1), name="r_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([h_size]), name="r_b3", dtype=tf.float32)
    h3 = tf.nn.relu(tf.matmul(h2, w3)+b3)
    w4 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w4", dtype=tf.float32)
    b4 = tf.Variable(tf.zeros([h_size]), name="r_b4", dtype=tf.float32)
    h4 = tf.nn.relu(tf.matmul(h3, w4) + b4)
    w5 = tf.Variable(tf.truncated_normal([h_size, De], stddev=0.1), name="r_w5", dtype=tf.float32)
    b5 = tf.Variable(tf.zeros([De]), name="r_b5", dtype=tf.float32)
    h5 = tf.matmul(h4, w5) + b5
    h5_trans=tf.reshape(h5,[-1,Nr,De])
    h5_trans=tf.transpose(h5_trans,[0,2,1])
    return h5_trans

def a(O,Rr,E):
    '''
    sum all effect applied on given receiver and then combine it with all other components
    '''
    E_bar = tf.matmul(E,tf.transpose(Rr,[0,2,1]))
    return tf.concat([O,E_bar],1)

def phi_O(C):
    '''
    the phi_O function that predict the final result by applying f_O on each object
    '''
    h_size = 50
    C_trans = tf.transpose(C,[0,2,1])
    C_trans = tf.reshape(C_trans,[-1,Ds+De])
    w1 = tf.Variable(tf.random.truncated_normal([Ds+De, h_size], stddev=0.1), name="o_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h_size]), name="o_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(C_trans,w1)+b1)
    w2 = tf.Variable(tf.random.truncated_normal([h_size,Dp], stddev=0.1), name="o_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([Dp]), name="o_b2", dtype=tf.float32)
    h2 = tf.matmul(h1, w2)+b2
    h2_trans = tf.reshape(h2, [-1, No, Dp])
    h2_trans = tf.transpose(h2_trans, [0,2,1])
    return h2_trans

def sumrows_O(P):
    '''
    sums rows of input P, take input with shape (None, Dp, No)
    output shape (None, Dp, 1)
    '''
    return tf.reduce_sum(P, axis=2, keepdims=True)

def phi_output_sum(P):
    '''
    phi_output: NN that output the score of classifier
    '''
    h_size=100
    w1 = tf.Variable(tf.random.truncated_normal([h_size, Dp], stddev=0.1), name="out_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h_size,1]), name="out_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(w1, P)+b1)
    w2 = tf.Variable(tf.random.truncated_normal([h_size, h_size], stddev=0.1), name="out_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h_size,1]), name="out_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(w2, h1)+b2)
    w3 = tf.Variable(tf.random.truncated_normal([1, h_size], stddev=0.1), name="out_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name="out_b3", dtype=tf.float32)
    h3 = tf.matmul(w3, h2)+b3
    #h3 = tf.nn.relu(tf.matmul(w2, h1)+b2)
    #h1 = tf.math.sigmoid(tf.matmul(w1, P)+b1)
    h3 = tf.reshape(h3, [-1,1])
    return h3

def phi_output(P):
    '''
    phi_output: NN that output the score of classifier
    '''
    h_size=100
    w1 = tf.Variable(tf.random.truncated_normal([No, h_size], stddev=0.1), name="out_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([Dp, h_size]), name="out_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(P, w1)+b1)
    w2 = tf.Variable(tf.random.truncated_normal([h_size, 1], stddev=0.1), name="out_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([Dp,1]), name="out_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2)+b2)
    w3 = tf.Variable(tf.random.truncated_normal([1, Dp], stddev=0.1), name="out_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name="out_b3", dtype=tf.float32)
    h3 = tf.matmul(w3, h2)+b3
    #h1 = tf.math.sigmoid(tf.matmul(w1, P)+b1)
    h3 = tf.reshape(h3, [-1,1])
    return h3

def phi_output_nd(P):
    '''
    phi_output: NN that output the score of classifier
    '''
    h_size=100
    w1 = tf.Variable(tf.random.truncated_normal([No, h_size], stddev=0.1), name="out_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([Dp, h_size]), name="out_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(P, w1)+b1)
    w2 = tf.Variable(tf.random.truncated_normal([h_size, 1], stddev=0.1), name="out_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([Dp,1]), name="out_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2)+b2)
    h2 = tf.reshape(h2, [-1,Dp])
    return h2

def phi_tk(T):
    '''
    phi_tk: NN that calculate final results using tracking information
    '''
    T_tk = tf.reshape(T, [-1,1,Dp])
    h_size=100
    w1 = tf.Variable(tf.random.truncated_normal([Dp, 1], stddev=0.1), name="tk_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([1, 1]), name="tk_b1", dtype=tf.float32)
    h1 = tf.matmul(T_tk, w1)+b1
    h1 = tf.reshape(h1, [-1,1])
    return h1

def phi_vtx(T, vtx):
    '''
    phi_vtx: NN that add a 1d vtx information to previous output and output the final result
    '''
    tkvtx = tf.concat([T,vtx], 1)
    tkvtx = tf.reshape(tkvtx, [-1,1,Dp+Dv])
    m,v = tf.nn.moments(tkvtx,[0,1])
    #tkvtx = tf.nn.batch_normalization(tkvtx, m, v, 0, 1, 1e-12)
    h_size=20
    w1 = tf.Variable(tf.random.truncated_normal([Dp+Dv, h_size], stddev=0.1), name="vtx_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([1,h_size]), name="vtx_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(tkvtx, w1)+b1)
    w2 = tf.Variable(tf.random.truncated_normal([h_size,h_size], stddev=0.1), name="vtx_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([1,h_size]), name="vtx_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2)+b2)
    w3 = tf.Variable(tf.random.truncated_normal([h_size,1], stddev=0.1), name="vtx_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1,1]), name="vtx_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3)+b3
    h3 = tf.reshape(h3, [-1,1])
    return h3

train, val, test, pos_weight = importData([0.7,0.15,0.15], year, True, True, True)

#def train():

O = tf.placeholder(tf.float32, [None, Ds, No], name='O')
Rr = tf.placeholder(tf.float32, [None, No, Nr], name="Rr")
Rs = tf.placeholder(tf.float32, [None, No, Nr], name="Rs")
Ra = tf.placeholder(tf.float32, [None, Dr, Nr], name="Ra")

label = tf.placeholder(tf.float32, [None, 1], name="label")
ntk_max = tf.placeholder(tf.float32, [None, 1], name="ntk_max")
#met = tf.placeholder(tf.float32, [None, 1], name="met")
evtweight = tf.placeholder(tf.float32, [None, 1], name="evtweight")
lambda_dcorr = tf.placeholder(tf.float32, [], name="lambda_dcorr")

B = m(O,Rr,Rs,Ra)

E = phi_R(B)

C = a(O,Rr,E)

P = phi_O(C)

out = phi_output(P)
#P = tf.reduce_sum(P, axis=2, keepdims=True)
#out_tk = phi_output_nd(P)
#out = phi_tk(out_tk)
#out_tkonly_sigmoid = tf.math.sigmoid(out_tkonly, name="INscore_tk")
#out = phi_vtx(out_tk,vtx)
out_sigmoid = tf.math.sigmoid(out, name="INscore")

params_list = tf.global_variables()
#for i in range(len(params_list)):
    #variable_summaries(params_list[i],i)
    
loss_bce = tf.nn.weighted_cross_entropy_with_logits(labels=label,logits=out,pos_weight=pos_weight)
loss_bce = tf.reduce_mean(loss_bce)
#loss_bce_tk = tf.nn.weighted_cross_entropy_with_logits(labels=label,logits=out_tkonly,pos_weight=pos_weight)
#loss_bce_tk = tf.reduce_mean(loss_bce_tk)
loss_param = tf.nn.l2_loss(E)
#loss = 0
for i in params_list:
    loss_param+=tf.nn.l2_loss(i)
dcorr = distance_corr(ntk_max, out_sigmoid, evtweight)
#dcorr_met = distance_corr(met, out_sigmoid, evtweight)
loss = loss_bce+lambda_param*loss_param+lambda_dcorr*dcorr
#loss = loss_bce+lambda_param*loss_param
optimizer = tf.train.AdamOptimizer(lr)
trainer=optimizer.minimize(loss)

#dcorr_tk = distance_corr(ntk_max, out_tkonly_sigmoid, evtweight)
#loss_tk = loss_bce_tk+lambda_param*loss_param+lambda_dcorr*dcorr_tk
#trainer_tk=optimizer.minimize(loss_tk)

# tensorboard
tf.summary.scalar('loss_bce',loss_bce)
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter(dir_model)

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
batch_num = 128
#batch_num = 1024
if use_dR:
    Rr_train, Rs_train, Ra_train = getRmatrix_dR2(train[4][:,1:3,:])
    Rr_val, Rs_val, Ra_val = getRmatrix_dR2(val[4][:,1:3,:])
else:
    Rr_train, Rs_train, Ra_train = getRmatrix(batch_num)
    Rr_val, Rs_val, Ra_val = getRmatrix(batch_num)

# training
num_epochs_tk=0
history = []
history_val = []
h_bce = []
h_bce_val = []
h_dcorr = []
h_dcorr_val = []
h_dcorr_met = []
h_dcorr_met_val = []
min_loss = 100
for i in range(num_epochs):
    lambda_dcorr_epoch = 0
    if i>=10:
      lambda_dcorr_epoch = 0.5
    if i==num_epochs_tk:
      print("training on Vertices info")
    loss_train = 0
    l_bce_train = 0
    l_dcorr_train = 0
    l_dcorr_met_train = 0
    for j in range(int(len(train[0])/batch_num)):
        batch_tk = train[0][j*batch_num:(j+1)*batch_num]
        batch_ntk = train[1][j*batch_num:(j+1)*batch_num][:,0]
        batch_ntk = np.reshape(batch_ntk, (-1,1))
        batch_label = train[2][j*batch_num:(j+1)*batch_num]
        if use_dR:
          batch_Rr = Rr_train[j*batch_num:(j+1)*batch_num]
          batch_Rs = Rs_train[j*batch_num:(j+1)*batch_num]
          batch_Ra = Ra_train[j*batch_num:(j+1)*batch_num]
        else:
          batch_Rr = Rr_train
          batch_Rs = Rs_train
          batch_Ra = Ra_train

        batch_weight = (batch_label-1)*(-1)
        batch_weight[batch_weight==0] = 1e-08
        #batch_weight = np.ones(batch_label.shape)

        #if i<num_epochs_tk:
        #  l_train,_,bce_train,dcorr_ntk_train=sess.run([loss_tk,trainer_tk,loss_bce_tk,dcorr_tk],feed_dict={O:batch_tk,Rr:batch_Rr,Rs:batch_Rs,Ra:batch_Ra,vtx:batch_vtx,label:batch_label,ntk_max:batch_ntk,evtweight:batch_weight})
        #  loss_train+=l_train
        #  l_bce_train+=bce_train
        #  l_dcorr_train+=dcorr_ntk_train
        #else:
        l_train,_,bce_train,dcorr_ntk_train=sess.run([loss,trainer,loss_bce,dcorr],feed_dict={O:batch_tk,Rr:batch_Rr,Rs:batch_Rs,Ra:batch_Ra,label:batch_label,ntk_max:batch_ntk,evtweight:batch_weight, lambda_dcorr:lambda_dcorr_epoch})
        loss_train+=l_train
        l_bce_train+=bce_train
        l_dcorr_train+=dcorr_ntk_train

    history.append(loss_train)
    h_bce.append(l_bce_train)

    #shuffle data after each epoch
    train_idx = np.array(range(len(train[0])))
    np.random.shuffle(train_idx)
    for ite in range(len(train)):
      train[ite] = train[ite][train_idx]
    if use_dR:
      Rr_train = Rr_train[train_idx]
      Rs_train = Rs_train[train_idx]
      Ra_train = Ra_train[train_idx]
    
    # validation after each epoch
    loss_val = 0
    l_bce_val = 0
    l_dcorr_val = 0
    l_dcorr_met_val = 0
    for j in range(int(len(val[0])/batch_num)):
        batch_tk = val[0][j*batch_num:(j+1)*batch_num]
        batch_ntk = val[1][j*batch_num:(j+1)*batch_num][:,0]
        batch_ntk = np.reshape(batch_ntk, (-1,1))
        batch_label = val[2][j*batch_num:(j+1)*batch_num]
        if use_dR:
          batch_Rr = Rr_val[j*batch_num:(j+1)*batch_num]
          batch_Rs = Rs_val[j*batch_num:(j+1)*batch_num]
          batch_Ra = Ra_val[j*batch_num:(j+1)*batch_num]
        else:
          batch_Rr = Rr_val
          batch_Rs = Rs_val
          batch_Ra = Ra_val
        batch_weight = (batch_label-1)*(-1)
        batch_weight[batch_weight==0] = 1e-08
        #batch_weight = np.ones(batch_label.shape)

        #if i<num_epochs_tk:
        #  l_val,_,bce_val,dcorr_ntk_val=sess.run([loss_tk,out_tkonly_sigmoid,loss_bce_tk,dcorr_tk],feed_dict={O:batch_tk,Rr:batch_Rr,Rs:batch_Rs,Ra:batch_Ra,vtx:batch_vtx,label:batch_label,ntk_max:batch_ntk,evtweight:batch_weight})
        #  loss_val+=l_val
        #  l_bce_val+=bce_val
        #  l_dcorr_val+=dcorr_ntk_val
        #else:
        l_val,_,bce_val,dcorr_ntk_val=sess.run([loss,out_sigmoid,loss_bce,dcorr],feed_dict={O:batch_tk,Rr:batch_Rr,Rs:batch_Rs,Ra:batch_Ra,label:batch_label,ntk_max:batch_ntk,evtweight:batch_weight,lambda_dcorr:lambda_dcorr_epoch})
        loss_val+=l_val
        l_bce_val+=bce_val
        l_dcorr_val+=dcorr_ntk_val
        
    if i>=10 and loss_val < min_loss:
        min_loss = loss_val
        saver = tf.train.Saver()
        saver.save(sess,dir_model+"test_model")
    history_val.append(loss_val)
    h_bce_val.append(l_bce_val)
    val_idx = np.array(range(len(val[0])))
    np.random.shuffle(val_idx)

    for ite in range(len(val)):
      val[ite] = val[ite][val_idx]

    if use_dR:
      Rr_val = Rr_val[val_idx]
      Rs_val = Rs_val[val_idx]
      Ra_val = Ra_val[val_idx]
    
    print("Epoch {}:".format(i))
    print("Training loss: {0}, BCE: {1}, dcorr: {2}"
          .format(loss_train/float(int(len(train[0])/batch_num)), l_bce_train/float(int(len(train[0])/batch_num)), l_dcorr_train/float(int(len(train[0])/batch_num)) ))
    print("Validation loss: {0}, BCE: {1}, dcorr: {2} "
          .format(loss_val/float(int(len(val[0])/batch_num)), l_bce_val/float(int(len(val[0])/batch_num)), l_dcorr_val/float(int(len(val[0])/batch_num)) ))

outputs = ["INscore"]

saver = tf.train.Saver(max_to_keep=20)
saver.save(sess,dir_model+"test_model")
pred = []
truth = []
ntk = []
if isUL:
  normalize_factors_vtx = normalize_factors_vtx_UL
else:
  normalize_factors_vtx = normalize_factors_vtx_EOY

mean = normalize_factors_vtx[0][0]
stddev = normalize_factors_vtx[0][1]
print("vtx ntk mean {}, std.dev. {}".format(mean, stddev))
with tf.Session() as newsess:
    newsaver = tf.train.import_meta_graph(dir_model+"test_model.meta")
    newsaver.restore(newsess, tf.train.latest_checkpoint(dir_model))

    if use_dR:
      Rr_test, Rs_test, Ra_test = getRmatrix_dR2(test[4][:,1:3,:])
    else:
      Rr_test, Rs_test, Ra_test = getRmatrix(batch_num)
    for j in range(int(len(test[0])/batch_num)+1):
        if j==int(len(test[0])/batch_num):
          next_idx = len(test[0])
          if not use_dR:
            Rr_test, Rs_test, Ra_test = getRmatrix(next_idx-j*batch_num)
        else:
          next_idx = (j+1)*batch_num
        batch_tk = test[0][j*batch_num:next_idx]
        batch_ntk = test[1][j*batch_num:next_idx][:,0]
        batch_label = test[2][j*batch_num:next_idx]
        if use_dR:
          batch_Rr = Rr_test[j*batch_num:next_idx]
          batch_Rs = Rs_test[j*batch_num:next_idx]
          batch_Ra = Ra_test[j*batch_num:next_idx]
        else:
          batch_Rr = Rr_test
          batch_Rs = Rs_test
          batch_Ra = Ra_test

        batch_weight = (batch_label-1)*(-1)
        batch_weight[batch_weight==0] = 1e-08
        #batch_weight = np.ones(batch_label.shape)

        b = newsess.run(['INscore:0'],feed_dict={'O:0':batch_tk,'Rr:0':batch_Rr,'Rs:0':batch_Rs,'Ra:0':batch_Ra})
        pred.append(b[0])
        truth.append(batch_label)
        ntk.append(batch_ntk*((stddev)/1.0)+mean)

pred = np.concatenate(pred,axis=None)
truth = np.concatenate(truth,axis=None)
ntk = np.concatenate(ntk,axis=None)
ntk = np.rint(ntk)

#b = b[0]
#plt.hist(b[test[2]==1], bins=50, alpha=0.5, density=True, stacked=True, label="signal")
#plt.hist(b[test[2]==0], bins=50, alpha=0.5, density=True, stacked=True, label="background")
plt.hist(pred[truth==1], bins=50, alpha=0.5, density=True, stacked=True, label="signal")
plt.hist(pred[truth==0], bins=50, alpha=0.5, density=True, stacked=True, label="background")
plt.legend(loc="best")
plt.title("IN score")
plt.xlabel('score')
plt.ylabel('A.U.')
plt.savefig(dir_model+"INscore.png")
plt.close()


#t_A = test[2][(b>0.4) & (test[1]>=5)]
#t_B = test[2][(b<0.4) & (test[1]>=5)]
#t_C = test[2][(b>0.4) & (test[1]<5) & (test[1]>2)]
#t_D = test[2][(b<0.4) & (test[1]<5) & (test[1]>2)]
t_5tkh = truth[(pred>0.4) & (ntk>=5)]
t_5tkl = truth[(pred<0.4) & (ntk>=5)]
t_4tkh = truth[(pred>0.4) & (ntk==4)]
t_4tkl = truth[(pred<0.4) & (ntk==4)]
t_3tkh = truth[(pred>0.4) & (ntk==3)]
t_3tkl = truth[(pred<0.4) & (ntk==3)]

print("5-tk high: signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_5tkh), np.sqrt(np.count_nonzero(t_5tkh)), len(t_5tkh)-np.count_nonzero(t_5tkh), np.sqrt(len(t_5tkh)-np.count_nonzero(t_5tkh))))
print("5-tk low : signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_5tkl), np.sqrt(np.count_nonzero(t_5tkl)), len(t_5tkl)-np.count_nonzero(t_5tkl), np.sqrt(len(t_5tkl)-np.count_nonzero(t_5tkl))))
print("4-tk high: signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_4tkh), np.sqrt(np.count_nonzero(t_4tkh)), len(t_4tkh)-np.count_nonzero(t_4tkh), np.sqrt(len(t_4tkh)-np.count_nonzero(t_4tkh))))
print("4-tk low : signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_4tkl), np.sqrt(np.count_nonzero(t_4tkl)), len(t_4tkl)-np.count_nonzero(t_4tkl), np.sqrt(len(t_4tkl)-np.count_nonzero(t_4tkl))))
print("3-tk high: signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_3tkh), np.sqrt(np.count_nonzero(t_3tkh)), len(t_3tkh)-np.count_nonzero(t_3tkh), np.sqrt(len(t_3tkh)-np.count_nonzero(t_3tkh))))
print("3-tk low : signals: {0} +- {1} backgrounds: {2} +- {3}".format(np.count_nonzero(t_3tkl), np.sqrt(np.count_nonzero(t_3tkl)), len(t_3tkl)-np.count_nonzero(t_3tkl), np.sqrt(len(t_3tkl)-np.count_nonzero(t_3tkl))))

