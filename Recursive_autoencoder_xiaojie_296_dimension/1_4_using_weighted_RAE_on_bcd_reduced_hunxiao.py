#!/usr/bin/python
yl=object
yt=list
yv=range
yM=len
yY=open
yz=id
yC=map
yh=max
yN=True
ym=dict
yK=None
yi=int
yD=enumerate
yp=min
yR=False
yX=float
yg=print
yx=str
yV=input
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import shutil
import tensorflow as tf
from xiaojie_log import xiaojie_log_class
from processWord2vecOutData import preprocess_withAST,preprocess_withAST_experimentID_1,preprocess_withAST_experimentID_10
from scipy.spatial.distance import pdist,squareform
from random import shuffle
import random
import pickle
global dy
dy=xiaojie_log_class()
class du(yl):
 def __init__(dt,dv):
  dy.log("(1)>>>>  before training RvNN,配置超参数")
  dt.label_size=dv['label_size']
  dt.early_stopping=dv['early_stopping']
  dt.max_epochs=dv['max_epochs']
  dt.anneal_threshold=dv['anneal_threshold']
  dt.anneal_by=dv['anneal_by']
  dt.lr=dv['lr']
  dt.l2=dv['l2']
  dt.embed_size=dv['embed_size']
  dt.model_name=dv['model_name']
  dy.log('模型名称为%s'%dt.model_name)
  dt.IDIR=dv['IDIR']
  dt.ODIR=dv['ODIR']
  dt.corpus_fixed_tree_constructionorder_file=dv['corpus_fixed_tree_constructionorder_file']
  dt.corpus_fixed_tree_construction_parentType_weight_file=dv['corpus_fixed_tree_construction_parentType_weight_file']
  dt.MAX_SENTENCE_LENGTH=10000
  dt.batch_size=dv['batch_size']
  dt.batch_size_using_model_notTrain=dv['batch_size_using_model_notTrain']
  dt.MAX_SENTENCE_LENGTH_for_Bigclonebench=dv['MAX_SENTENCE_LENGTH_for_Bigclonebench']
  dy.log("(1)<<<<  before training RvNN,配置超参数完毕")
class dq(yl):
 def __init__(dt,sentence_length=0):
  dt.sl=dM
  dt.nodeScores=np.zeros((2*dt.sl-1,1),dtype=np.double)
  dt.collapsed_sentence=(yt)(yv(0,dt.sl))
  dt.pp=np.zeros((2*dt.sl-1,1),dtype=np.yi)
class dl():
 def load_data(dt):
  dy.log("(2)>>>>  加载词向量数据和语料库")
  (dt.trainCorpus,dt.fullCorpus,dt.trainCorpus_sentence_length,dt.fullCorpus_sentence_length,dt.vocabulary,dt.We,dt.config.max_sentence_length_train_Corpus,dt.config.max_sentence_length_full_Corpus,dt.train_corpus_fixed_tree_constructionorder,dt.full_corpus_fixed_tree_constructionorder)=preprocess_withAST(dt.config.IDIR,dt.config.ODIR,dt.config.corpus_fixed_tree_constructionorder_file,dt.config.MAX_SENTENCE_LENGTH)
  if yM(dt.trainCorpus)>4000:
   dz=yM(dt.trainCorpus)
   dC=yt(yv(dz))
   shuffle(dC)
   dh=dC[0:4000]
   dt.evalutionCorpus=[dt.trainCorpus[dN]for dN in dh]
   dt.config.max_sentence_length_evalution_Corpus=dt.config.max_sentence_length_train_Corpus 
   dt.evalution_corpus_fixed_tree_constructionorder=[dt.train_corpus_fixed_tree_constructionorder[dN]for dN in dh]
  else:
   dt.evalutionCorpus=dt.trainCorpus
   dt.config.max_sentence_length_evalution_Corpus=dt.config.max_sentence_length_train_Corpus
   dt.evalution_corpus_fixed_tree_constructionorder=dt.train_corpus_fixed_tree_constructionorder
  dy.log("(2)>>>>  加载词向量数据和语料库完毕")
 def load_data_experimentID_1(dt):
  dy.log("(2)>>>>  加载词向量数据和语料库")
  (dt.trainCorpus,dt.trainCorpus_sentence_length,dt.vocabulary,dt.We,dt.config.max_sentence_length_train_Corpus,dt.train_corpus_fixed_tree_constructionorder,dt.train_corpus_fixed_tree_parentType_weight)=preprocess_withAST_experimentID_10(dt.config.IDIR,dt.config.ODIR,dt.config.corpus_fixed_tree_constructionorder_file,dt.config.MAX_SENTENCE_LENGTH,dt.config.corpus_fixed_tree_construction_parentType_weight_file)
  dy.log("------------------------------\n对照BigCloneBench中标注的函数，找出在我们的语料库中的编号位置")
  dm='./SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
  dK=[]
  with yY(dm,'rb')as f:
   di=pickle.load(f)
   for yz in di.keys():
    dD=di[yz]
    if(dD==-1):
     continue
    dK.append(dD)
  dt.lines_for_bigcloneBench=dK
  dp=yM(dK)
  dy.log('BigCloneBench中有效函数ID多少个，对应的取出我们语料库中的语料多少个.{}个'.format(dp))
  dt.trainCorpus=[dt.trainCorpus[dN]for dN in dK]
  dt.train_corpus_fixed_tree_constructionorder=[dt.train_corpus_fixed_tree_constructionorder[dN]for dN in dK]
  dt.trainCorpus_sentence_length=[dt.trainCorpus_sentence_length[dN]for dN in dK]
  dt.train_corpus_fixed_tree_parentType_weight=[dt.train_corpus_fixed_tree_parentType_weight[dN]for dN in dK]
  dR=yt(yC(yM,dt.trainCorpus))
  dt.config.max_sentence_length_train_Corpus=yh(dR)
  dt.bigCloneBench_Corpus=dt.trainCorpus 
  dt.bigCloneBench_Corpus_fixed_tree_constructionorder=dt.train_corpus_fixed_tree_constructionorder
  dt.bigCloneBench_Corpus_sentence_length=dt.trainCorpus_sentence_length
  dt.bigCloneBench_Corpus_max_sentence_length=dt.config.max_sentence_length_train_Corpus
  dt.bigCloneBench_Corpus_fixed_tree_parentType_weight=dt.train_corpus_fixed_tree_parentType_weight
  dy.log('(2)>>>>  对照BigCloneBench中标注的函数,从我们的语料库中抽取语料{}个'.format(yM(dK))) 
 def load_data_experimentID_2(dt):
  dX='./vectorTree/valid_dataset_lst.pkl'
  dg=dt.read_from_pkl(dX)
  dy.log("(2)>>>>  加载词向量数据和语料库")
  (dt.trainCorpus,dt.trainCorpus_sentence_length,dt.vocabulary,dt.We,dt.config.max_sentence_length_train_Corpus,dt.train_corpus_fixed_tree_constructionorder)=preprocess_withAST_experimentID_1(dt.config.IDIR,dt.config.ODIR,dt.config.corpus_fixed_tree_constructionorder_file,dt.config.MAX_SENTENCE_LENGTH)
  dy.log("------------------------------\n对照BigCloneBench中标注的函数，找出在我们的语料库中的编号位置")
  dm='./SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
  dx=[]
  dV=[]
  with yY(dm,'rb')as f:
   dt.id_line_dict=pickle.load(f)
   dN=0
   for yz in dg:
    dD=dt.id_line_dict[yz]
    if(dD==-1):
     continue
    dV.append(yz)
    dx.append(dD)
    dN+=1
  dt.need_vectorTree_lines_for_trainCorpus=dx
  dt.need_vectorTree_ids_for_trainCorpus=dV
  dp=yM(dV)
  dy.log('路哥需要取出{}个ID对应的向量树'.format(dp))
  dt.trainCorpus=[dt.trainCorpus[dN]for dN in dx]
  dt.train_corpus_fixed_tree_constructionorder=[dt.train_corpus_fixed_tree_constructionorder[dN]for dN in dx]
  dt.trainCorpus_sentence_length=[dt.trainCorpus_sentence_length[dN]for dN in dx]
  dR=yt(yC(yM,dt.trainCorpus))
  dt.config.max_sentence_length_train_Corpus=yh(dR)
  dt.need_vectorTree_Corpus=dt.trainCorpus 
  dt.need_vectorTree_Corpus_fixed_tree_constructionorder=dt.train_corpus_fixed_tree_constructionorder
  dt.need_vectorTree_Corpus_sentence_length=dt.trainCorpus_sentence_length
  dt.need_vectorTree_Corpus_max_sentence_length=dt.config.max_sentence_length_train_Corpus
 def load_data_experimentID_3(dt):
  dy.log("(2)>>>>  加载词向量数据和语料库")
  (dt.fullCorpus,dt.fullCorpus_sentence_length,dt.vocabulary,dt.We,dt.config.max_sentence_length_full_Corpus,dt.full_corpus_fixed_tree_constructionorder,dt.full_corpus_fixed_tree_parentType_weight)=preprocess_withAST_experimentID_10(dt.config.IDIR,dt.config.ODIR,dt.config.corpus_fixed_tree_constructionorder_file,dt.config.MAX_SENTENCE_LENGTH,dt.config.corpus_fixed_tree_construction_parentType_weight_file)
 def xiaojie_RvNN_fixed_tree(dt):
  dt.add_placeholders_fixed_tree()
  dt.add_model_vars()
  dt.add_loss_fixed_tree()
  dt.train_op=dt.training(dt.tensorLoss_fixed_tree)
 def xiaojie_RvNN_fixed_tree_for_usingmodel(dt):
  dt.add_placeholders_fixed_tree()
  dt.add_model_vars()
  dt.add_loss_and_batchSentenceNodesVector_fixed_tree()
 def buqi_2DmatrixTensor(dt,dj,uo,us,uT,uE):
  dj=tf.pad(dj,[[0,uT-uo],[0,uE-us]])
  return dj
 def modify_one_profile(dt,tensor,dj,uQ,uG,ur,uH):
  dj=tf.expand_dims(dj,axis=0)
  dF=tf.slice(tensor,[0,0,0],[uQ,ur,uH])
  dw=tf.slice(tensor,[uQ+1,0,0],[uG-uQ-1,ur,uH])
  dk=tf.concat([dF,dj,dw],0)
  return dF,dw,dk
 def delete_one_column(dt,tensor,dN,uc,numcolunms):
  dF=tf.slice(tensor,[0,0],[uc,dN])
  dw=tf.slice(tensor,[0,dN+1],[uc,numcolunms-(dN+1)])
  dk=tf.concat([dF,dw],1)
  return dk
 def modify_one_column(dt,tensor,columnTensor,dN,uc,numcolunms):
  dF=tf.slice(tensor,[0,0],[uc,dN])
  dw=tf.slice(tensor,[0,dN+1],[uc,numcolunms-(dN+1)])
  dk=tf.concat([dF,columnTensor,dw],1)
  return dk
 def computeloss_withAST(dt,sentence,lP):
  with tf.variable_scope('Composition',reuse=yN):
   W1=tf.get_variable("W1",dtype=tf.float64)
   b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=yN):
   U=tf.get_variable("U",dtype=tf.float64)
   bs=tf.get_variable("bs",dtype=tf.float64)
  dA=np.ones_like(sentence)
  do=sentence-dA
  ds=np.array(dt.We)
  L=ds[:,do]
  sl=L.shape[1]
  dT=ym()
  for i in yv(0,sl):
   dT[i]=np.expand_dims(L[:,i],1)
  dE=ym()
  if(sl>1):
   for j in yv(0,sl-1):
    dQ=W1.eval()
    dG=b1.eval()
    dr=U.eval()
    dH=bs.eval()
    dI=lP[:,j]
    db=dI[0]-1 
    dO=dT[db]
    dc=dI[1]-1
    dn=dT[dc] 
    dB=dI[2]-1
    dJ=np.concatenate((dO,dn),axis=0)
    p=np.tanh(np.dot(dQ,dJ)+dG)
    dU=p/(np.linalg.norm(p,axis=0)*np.ones(p.shape))
    dT[dB]=dU
    y=np.tanh(np.dot(dr,dU)+dH)
    [y1,y2]=np.split(y,2,axis=0)
    de=y1/(np.linalg.norm(y1,axis=0)*np.ones(y1.shape))
    da=y2/(np.linalg.norm(y2,axis=0)*np.ones(y2.shape))
    dW=1 
    df=1
    dS,dP=dW*(de-dO),df*(da-dn)
    constructionError=np.sum((dS*(de-dO)+dP*(da-dn)),axis=0)*0.5 
    dE[j]=constructionError
    pass
   pass
  ud=0
  for(key,value)in dE.items():
   ud=ud+value
  ud=ud/(sl-1)
  return ud 
 def add_loss_fixed_tree(dt):
  with tf.variable_scope('Composition',reuse=yN):
   dt.W1=tf.get_variable("W1",dtype=tf.float64)
   dt.b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=yN):
   dt.U=tf.get_variable("U",dtype=tf.float64)
   dt.bs=tf.get_variable("bs",dtype=tf.float64)
  uq=tf.zeros(dt.batch_len,tf.float64)
  uq=tf.expand_dims(uq,0)
  dt.numlines_tensor3=tf.constant(1,dtype=tf.int32)
  dt.numcolunms_tensor3=dt.batch_len[0]
  i=tf.constant(0,dtype=tf.int32)
  ul=dt.batch_len[0]
  uy=lambda a,b,c:tf.less(a,ul)
  ut=tf.constant(0)
  uv=[i,uq,ut]
  def _recurrence(i,uq,ut):
   dt.sentence_embeddings=tf.gather(dt.yV,i,axis=0)
   dt.sentence_length=dt.batch_real_sentence_length[i]
   dt.treeConstructionOrders=dt.batch_treeConstructionOrders[i]
   dt.sentence_parentTypes_weight=dt.batch_sentence_parentTypes_weight[i]
   uY=2*dt.sentence_length-1
   dT=tf.zeros(uY,tf.float64)
   dT=tf.expand_dims(dT,0)
   dT=tf.tile(dT,(dt.config.embed_size,1))
   dt.numlines_tensor=tf.constant(dt.config.embed_size,dtype=tf.int32)
   dt.numcolunms_tensor=uY
   ii=tf.constant(0,dtype=tf.int32)
   uz=lambda a,b:tf.less(a,dt.sentence_length)
   uC=[ii,dT]
   def __recurrence(ii,dT):
    uh=tf.expand_dims(dt.sentence_embeddings[:,ii],1)
    dT=dt.modify_one_column(dT,uh,ii,dt.numlines_tensor,dt.numcolunms_tensor)
    ii=tf.add(ii,1)
    return ii,dT
   ii,dT=tf.while_loop(uz,__recurrence,uC,parallel_iterations=1)
   dE=tf.zeros(dt.sentence_length-1,tf.float64)
   dE=tf.expand_dims(dE,0)
   dt.numlines_tensor2=tf.constant(1,dtype=tf.int32)
   dt.numcolunms_tensor2=dt.sentence_length-1
   uN=tf.constant(0,dtype=tf.int32)
   um=lambda a,b,c,d:tf.less(a,dt.sentence_length-1)
   uK=[uN,dE,dT,ut]
   def ____recurrence(uN,dE,dT,ut):
    dI=dt.treeConstructionOrders[:,uN]
    db=dI[0]-1 
    ui=dT[:,db]
    dc=dI[1]-1
    uD=dT[:,dc] 
    dB=dI[2]-1
    up=tf.concat([ui,uD],axis=0)
    up=tf.expand_dims(up,1)
    uR=tf.tanh(tf.add(tf.matmul(dt.W1,up),dt.b1))
    uX=dt.normalization(uR)
    dT=dt.modify_one_column(dT,uX,dB,dt.numlines_tensor,dt.numcolunms_tensor)
    y=tf.tanh(tf.matmul(dt.U,uX)+dt.bs)
    ug=y.shape[1].value
    (y1,y2)=dt.split_by_row(y,ug)
    de=dt.normalization(y1)
    da=dt.normalization(y2)
    dW=1 
    df=1
    ui=tf.expand_dims(ui,1)
    uD=tf.expand_dims(uD,1)
    dS=tf.subtract(de,ui)
    dP=tf.subtract(da,uD) 
    constructionError=dt.constructionError(dS,dP,dW,df)
    ux=dt.sentence_parentTypes_weight[uN]
    constructionError=tf.multiply(constructionError,ux)
    constructionError=tf.expand_dims(constructionError,1)
    dE=dt.modify_one_column(dE,constructionError,uN,dt.numlines_tensor2,dt.numcolunms_tensor2)
    uV=tf.Print(uN,[uN],"\niiii:")
    ut=uV+ut
    ut=uV+ut
    uV=tf.Print(db,[db],"\nleftChild_index:",summarize=100)
    ut=tf.to_int32(uV)+ut
    uV=tf.Print(dc,[dc],"\nrightChild_index:",summarize=100)
    ut=tf.to_int32(uV)+ut
    uV=tf.Print(dB,[dB],"\nparent_index:",summarize=100)
    ut=tf.to_int32(uV)+ut
    uN=tf.add(uN,1)
    return uN,dE,dT,ut
   uN,dE,dT,ut=tf.while_loop(um,____recurrence,uK,parallel_iterations=1)
   pass
   dt.node_tensors_cost_tensor=tf.identity(dE)
   dt.nodes_tensor=tf.identity(dT)
   uj=tf.reduce_sum(dt.node_tensors_cost_tensor)
   uj=tf.expand_dims(tf.expand_dims(uj,0),1)
   uq=dt.modify_one_column(uq,uj,i,dt.numlines_tensor3,dt.numcolunms_tensor3)
   i=tf.add(i,1)
   return i,uq,ut
  i,uq,ut=tf.while_loop(uy,_recurrence,uv,parallel_iterations=10)
  dt.tfPrint=ut
  with tf.name_scope('loss'):
   uF=tf.nn.l2_loss(dt.W1)+tf.nn.l2_loss(dt.U)
   dt.batch_constructionError=tf.reduce_mean(uq)
   dt.tensorLoss_fixed_tree=dt.batch_constructionError+uF*dt.config.l2
  return dt.tensorLoss_fixed_tree
 def add_loss_and_batchSentenceNodesVector_fixed_tree(dt):
  with tf.variable_scope('Composition',reuse=yN):
   dt.W1=tf.get_variable("W1",dtype=tf.float64)
   dt.b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=yN):
   dt.U=tf.get_variable("U",dtype=tf.float64)
   dt.bs=tf.get_variable("bs",dtype=tf.float64)
  uq=tf.zeros(dt.batch_len,tf.float64)
  uq=tf.expand_dims(uq,0)
  uw=dt.batch_real_sentence_length[tf.argmax(dt.batch_real_sentence_length)]
  uk=2*uw-1
  uL=dt.batch_len[0]*dt.config.embed_size*uk
  uA=tf.zeros(uL,tf.float64)
  uA=tf.reshape(uA,[dt.batch_len[0],dt.config.embed_size,uk])
  dt.size_firstDimension=dt.batch_len[0]
  dt.size_secondDimension=tf.constant(dt.config.embed_size,dtype=tf.int32)
  dt.size_thirdDimension=uk
  dt.numlines_tensor3=tf.constant(1,dtype=tf.int32)
  dt.numcolunms_tensor3=dt.batch_len[0]
  dt.numlines_tensor4=tf.constant(dt.config.embed_size,dtype=tf.int32)
  dt.numcolunms_tensor4=dt.batch_len[0]
  i=tf.constant(0,dtype=tf.int32)
  ul=dt.batch_len[0]
  uy=lambda a,b,c,d:tf.less(a,ul)
  ut=tf.constant(0)
  uv=[i,uq,ut,uA]
  def _recurrence(i,uq,ut,uA):
   dt.sentence_embeddings=tf.gather(dt.yV,i,axis=0)
   dt.sentence_length=dt.batch_real_sentence_length[i]
   dt.treeConstructionOrders=dt.batch_treeConstructionOrders[i]
   uY=2*dt.sentence_length-1
   dT=tf.zeros(uY,tf.float64)
   dT=tf.expand_dims(dT,0)
   dT=tf.tile(dT,(dt.config.embed_size,1))
   dt.numlines_tensor=tf.constant(dt.config.embed_size,dtype=tf.int32)
   dt.numcolunms_tensor=uY
   ii=tf.constant(0,dtype=tf.int32)
   uz=lambda a,b:tf.less(a,dt.sentence_length)
   uC=[ii,dT]
   def __recurrence(ii,dT):
    uh=tf.expand_dims(dt.sentence_embeddings[:,ii],1)
    dT=dt.modify_one_column(dT,uh,ii,dt.numlines_tensor,dt.numcolunms_tensor)
    ii=tf.add(ii,1)
    return ii,dT
   ii,dT=tf.while_loop(uz,__recurrence,uC,parallel_iterations=1)
   dE=tf.zeros(dt.sentence_length-1,tf.float64)
   dE=tf.expand_dims(dE,0)
   dt.numlines_tensor2=tf.constant(1,dtype=tf.int32)
   dt.numcolunms_tensor2=dt.sentence_length-1
   uN=tf.constant(0,dtype=tf.int32)
   um=lambda a,b,c,d:tf.less(a,dt.sentence_length-1)
   uK=[uN,dE,dT,ut]
   def ____recurrence(uN,dE,dT,ut):
    dI=dt.treeConstructionOrders[:,uN]
    db=dI[0]-1 
    ui=dT[:,db]
    dc=dI[1]-1
    uD=dT[:,dc] 
    dB=dI[2]-1
    up=tf.concat([ui,uD],axis=0)
    up=tf.expand_dims(up,1)
    uR=tf.tanh(tf.add(tf.matmul(dt.W1,up),dt.b1))
    uX=dt.normalization(uR)
    dT=dt.modify_one_column(dT,uX,dB,dt.numlines_tensor,dt.numcolunms_tensor)
    y=tf.tanh(tf.matmul(dt.U,uX)+dt.bs)
    ug=y.shape[1].value
    (y1,y2)=dt.split_by_row(y,ug)
    de=dt.normalization(y1)
    da=dt.normalization(y2)
    dW=1 
    df=1
    ui=tf.expand_dims(ui,1)
    uD=tf.expand_dims(uD,1)
    dS=tf.subtract(de,ui)
    dP=tf.subtract(da,uD) 
    constructionError=dt.constructionError(dS,dP,dW,df)
    constructionError=tf.expand_dims(constructionError,1)
    dE=dt.modify_one_column(dE,constructionError,uN,dt.numlines_tensor2,dt.numcolunms_tensor2)
    uV=tf.Print(uN,[uN],"\niiii:")
    ut=uV+ut
    ut=uV+ut
    uV=tf.Print(db,[db],"\nleftChild_index:",summarize=100)
    ut=tf.to_int32(uV)+ut
    uV=tf.Print(dc,[dc],"\nrightChild_index:",summarize=100)
    ut=tf.to_int32(uV)+ut
    uV=tf.Print(dB,[dB],"\nparent_index:",summarize=100)
    ut=tf.to_int32(uV)+ut
    uN=tf.add(uN,1)
    return uN,dE,dT,ut
   uN,dE,dT,ut=tf.while_loop(um,____recurrence,uK,parallel_iterations=1)
   pass
   dt.node_tensors_cost_tensor=tf.identity(dE)
   dt.nodes_tensor=tf.identity(dT)
   uj=tf.reduce_mean(dt.node_tensors_cost_tensor)
   uj=tf.expand_dims(tf.expand_dims(uj,0),1)
   uq=dt.modify_one_column(uq,uj,i,dt.numlines_tensor3,dt.numcolunms_tensor3)
   uo=dt.numlines_tensor 
   us=dt.numcolunms_tensor 
   uT=dt.size_secondDimension 
   uE=dt.size_thirdDimension
   dT=dt.buqi_2DmatrixTensor(dT,uo,us,uT,uE)
   dT=tf.reshape(dT,[dt.config.embed_size,uE])
   uQ=i 
   uG=dt.size_firstDimension
   ur=dt.size_secondDimension
   uH=dt.size_thirdDimension
   _,_,uA=dt.modify_one_profile(uA,dT,uQ,uG,ur,uH)
   i=tf.add(i,1)
   return i,uq,ut,uA
  i,uq,ut,uA=tf.while_loop(uy,_recurrence,uv,parallel_iterations=10)
  dt.tfPrint=ut
  dt.batch_sentence_vectors=tf.identity(uA)
  with tf.name_scope('loss'):
   uF=tf.nn.l2_loss(dt.W1)+tf.nn.l2_loss(dt.U)
   dt.batch_constructionError=tf.reduce_mean(uq)
   dt.tensorLoss_fixed_tree=dt.batch_constructionError+uF*dt.config.l2
  return dt.tensorLoss_fixed_tree,dt.batch_sentence_vectors
 def add_placeholders_fixed_tree(dt):
  uI=dt.config.embed_size
  dt.yV=tf.placeholder(tf.float64,[yK,uI,yK],name='input')
  dt.batch_treeConstructionOrders=tf.placeholder(tf.int32,[yK,3,yK],name='treeConstructionOrders')
  dt.batch_real_sentence_length=tf.placeholder(tf.int32,[yK],name='batch_real_sentence_length')
  dt.batch_len=tf.placeholder(tf.int32,shape=(1,),name='batch_len')
  dt.batch_sentence_parentTypes_weight=tf.placeholder(tf.float64,[yK,yK],name='batch_sentence_parentType_weights')
 def add_model_vars(dt):
  with tf.variable_scope('Composition'): 
   tf.get_variable("W1",dtype=tf.float64,shape=[dt.config.embed_size,2*dt.config.embed_size])
   tf.get_variable("b1",dtype=tf.float64,shape=[dt.config.embed_size,1])
  with tf.variable_scope('Projection'):
   tf.get_variable("U",dtype=tf.float64,shape=[2*dt.config.embed_size,dt.config.embed_size])
   tf.get_variable("bs",dtype=tf.float64,shape=[2*dt.config.embed_size,1])
 def normalization(dt,tensor):
  uc=tensor.shape[0].value
  un=tf.pow(tensor,2)
  uB=tf.reduce_sum(un,0)
  uJ=tf.expand_dims(uB,0)
  uU=tf.tile(tf.sqrt(uJ),(uc,1))
  ue=tf.divide(tensor,uU)
  return ue
 def split_by_row(dt,tensor,numcolunms):
  uc=tensor.shape[0].value
  ua=tf.slice(tensor,[0,0],[(yi)(uc/2),numcolunms])
  uW=tf.slice(tensor,[(yi)(uc/2),0],[(yi)(uc/2),numcolunms])
  pass
  return(ua,uW)
 def constructionError(dt,tensor1,tensor2,dW,df):
  uf=tf.multiply(tf.reduce_sum(tf.pow(tensor1,2),0),dW)
  uS=tf.multiply(tf.reduce_sum(tf.pow(tensor2,2),0),df)
  uP=tf.multiply(tf.add(uf,uS),0.5)
  return uP
 def training(dt,qn):
  qd=yK
  qu=tf.train.GradientDescentOptimizer(dt.config.lr)
  qd=qu.minimize(qn)
  return qd
 def __init__(dt,ql,experimentID=yK):
  if(experimentID==yK):
   dt.config=ql
   dt.load_data()
  elif(experimentID==1):
   dt.config=ql
   dt.load_data_experimentID_1()
  elif(experimentID==2):
   dt.config=ql
   dt.load_data_experimentID_2()
  elif(experimentID==3):
   dt.config=ql
   dt.load_data_experimentID_3()
 def computelossAndVector_no_tensor_withAST(dt,sentence,lP):
  with tf.variable_scope('Composition',reuse=yN):
   W1=tf.get_variable("W1",dtype=tf.float64)
   b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=yN):
   U=tf.get_variable("U",dtype=tf.float64)
   bs=tf.get_variable("bs",dtype=tf.float64)
  dA=np.ones_like(sentence)
  do=sentence-dA
  ds=np.array(dt.We)
  L=ds[:,do]
  sl=L.shape[1]
  qy=ym()
  for i in yv(0,sl):
   qy[i]=np.expand_dims(L[:,i],1)
  dE=ym()
  qt=yK
  if(sl>1):
   for j in yv(0,sl-1):
    dQ=W1.eval()
    dG=b1.eval()
    dr=U.eval()
    dH=bs.eval()
    dI=lP[:,j]
    db=dI[0]-1 
    dO=qy[db]
    dc=dI[1]-1
    dn=qy[dc] 
    dB=dI[2]-1
    dJ=np.concatenate((dO,dn),axis=0)
    p=np.tanh(np.dot(dQ,dJ)+dG)
    dU=p/(np.linalg.norm(p,axis=0)*np.ones(p.shape))
    qy[dB]=dU
    qt=dU
    y=np.tanh(np.dot(dr,dU)+dH)
    [y1,y2]=np.split(y,2,axis=0)
    de=y1/(np.linalg.norm(y1,axis=0)*np.ones(y1.shape))
    da=y2/(np.linalg.norm(y2,axis=0)*np.ones(y2.shape))
    dW=1 
    df=1
    dS,dP=dW*(de-dO),df*(da-dn)
    constructionError=np.sum((dS*(de-dO)+dP*(da-dn)),axis=0)*0.5 
    dE[j]=constructionError
    pass
   qv=qy[2*sl-2]
   pass
  ud=0
  for(key,value)in dE.items():
   ud=ud+value
  return(ud,qy,qv,qt)
 def run_epoch_train(dt,lq,qB):
  qM=[]
  qY=dt.trainCorpus
  qz=dt.trainCorpus_sentence_length
  qC=dt.train_corpus_fixed_tree_constructionorder
  qh=dt.config.max_sentence_length_train_Corpus
  qN=yM(dt.trainCorpus)
  qm=dt.config.MAX_SENTENCE_LENGTH_for_Bigclonebench
  dy.log('训练过程设置长短的衡量标准是{}，长的单独成batch，短的集合成batch'.format(qm))
  qK=[]
  qi=[]
  for dN,length in yD(qz):
   if length<qm:
    qK.append(dN)
   else:
    qi.append(dN)
  dy.log("训练集的句子{}个".format(qN))
  dy.log("较长的句子{}个".format(yM(qi)))
  dy.log("较短的句子{}个".format(yM(qK)))
  qD=[qY[dN]for dN in qK]
  qp=[qz[dN]for dN in qK]
  qR=[qC[dN]for dN in qK]
  qX=[qY[dN]for dN in qi]
  qg=[qz[dN]for dN in qi]
  qx=[qC[dN]for dN in qi]
  dy.log("较短的句子，我们走批处理训练网络。长句子，我们单独计算训练网络")
  dy.log("先处理较短的句子的语料，批处理开始")
  qV=yM(qK)
  ds=np.array(dt.We)
  qj=ds.shape[0]
  dC=yt(yv(qV))
  qF=dt.config.batch_size
  qw=0
  for qk in yv(0,qV,qF):
   qL=yp(qk+qF,qV)-qk
   qA=dC[qk:qk+qL]
   qo=[qD[dN]for dN in qA]
   qs=[qp[dN]for dN in qA]
   qT=yh(qs)
   x=[]
   for i,sentence in yD(qo):
    qE=qs[i]
    dA=np.ones_like(sentence)
    do=sentence-dA
    L1=ds[:,do]
    qQ=qT-qE
    L2=np.zeros([qj,qQ],np.float64)
    L=np.concatenate((L1,L2),axis=1)
    x.append(L)
   qG=np.array(x)
   qr=np.array(qG,np.float64)
   x=[]
   qH=[qR[dN]for dN in qA]
   for i,sentence_fixed_tree_constructionorder in yD(qH):
    qI=qs[i]-1
    qQ=(qT-1)-qI
    L2=np.zeros([3,qQ],np.int32)
    L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
    x.append(L)
   qb=np.array(x)
   qO={dt.yV:qr,dt.batch_real_sentence_length:qs,dt.batch_len:[qL],dt.batch_treeConstructionOrders:qb}
   qc=tf.RunOptions(report_tensor_allocations_upon_oom=yN)
   qn,_=lq.run([dt.tensorLoss_fixed_tree,dt.train_op],feed_dict=qO,options=qc)
   qM.append(qn)
   dy.log('\repoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(qB,qw,qL,qn)) 
   dy.log('\repoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(qB,yM(qM),np.mean(qM))) 
   qw=qw+1
   pass 
  dy.log("保存模型到temp")
  qJ=tf.train.Saver()
  if not os.path.exists("./weights"):
   os.makedirs("./weights")
  qJ.save(lq,'./weights/%s.temp'%dt.config.model_name)
  return qM 
 def run_epoch_evaluation(dt,lq,qB):
  qM=[]
  qY=dt.evalutionCorpus
  qz=dt.evalution_corpus_sentence_length
  qC=dt.evalution_corpus_fixed_tree_constructionorder
  qh=dt.config.max_sentence_length_evalution_Corpus
  qU=yM(dt.evalutionCorpus)
  qm=1000; 
  dy.log('训练过程设置长短的衡量标准是{}，长的单独成batch，短的集合成batch'.format(qm))
  qK=[]
  qi=[]
  for dN,length in yD(qz):
   if length<qm:
    qK.append(dN)
   else:
    qi.append(dN)
  dy.log("训练集的句子{}个".format(qN))
  dy.log("较长的句子{}个".format(yM(qi)))
  dy.log("较短的句子{}个".format(yM(qK)))
  qD=[qY[dN]for dN in qK]
  qp=[qz[dN]for dN in qK]
  qR=[qC[dN]for dN in qK]
  qX=[qY[dN]for dN in qi]
  qg=[qz[dN]for dN in qi]
  qx=[qC[dN]for dN in qi]
  dy.log("较短的句子，我们走批处理训练网络。长句子，我们单独计算训练网络")
  dy.log("先处理较短的句子的语料，批处理开始")
  qV=yM(qK)
  ds=np.array(dt.We)
  qj=ds.shape[0]
  dC=yt(yv(qV))
  qF=dt.config.batch_size_using_model_notTrain 
  qw=0
  for qk in yv(0,qV,qF):
   qL=yp(qk+qF,qV)-qk
   qA=dC[qk:qk+qL]
   qo=[qD[dN]for dN in qA]
   qs=[qp[dN]for dN in qA]
   qT=yh(qs)
   x=[]
   for i,sentence in yD(qo):
    qE=qs[i]
    dA=np.ones_like(sentence)
    do=sentence-dA
    L1=ds[:,do]
    qQ=qT-qE
    L2=np.zeros([qj,qQ],np.float64)
    L=np.concatenate((L1,L2),axis=1)
    x.append(L)
   qG=np.array(x)
   qr=np.array(qG,np.float64)
   x=[]
   qH=[qR[dN]for dN in qA]
   for i,sentence_fixed_tree_constructionorder in yD(qH):
    qI=qs[i]-1
    qQ=(qT-1)-qI
    L2=np.zeros([3,qQ],np.int32)
    L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
    x.append(L)
   qb=np.array(x)
   qO={dt.yV:qr,dt.batch_real_sentence_length:qs,dt.batch_len:[qL],dt.batch_treeConstructionOrders:qb}
   qn=lq.run([dt.tensorLoss_fixed_tree],feed_dict=qO)
   qM.append(qn)
   dy.log('\r验证过程epoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(qB,qw,qL,qn)) 
   dy.log('\r验证过程epoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(qB,yM(qM),np.mean(qM))) 
   qw=qw+1
   pass 
  dy.log("再处理较长的句子的语料，每个句子单独处理，开始") 
  qe=yM(qi)
  dC=yt(yv(qe))
  for qk,sentence in yD(qX):
   qL=1
   qA=dC[qk:qk+qL]
   qo=[qX[dN]for dN in qA]
   qs=[qg[dN]for dN in qA]
   qT=yh(qs)
   x=[]
   for i,sentence in yD(qo):
    qE=qs[i]
    dA=np.ones_like(sentence)
    do=sentence-dA
    L1=ds[:,do]
    qQ=qT-qE
    L2=np.zeros([qj,qQ],np.float64)
    L=np.concatenate((L1,L2),axis=1)
    x.append(L)
   qG=np.array(x)
   qr=np.array(qG,np.float64)
   x=[]
   qH=[qx[dN]for dN in qA]
   for i,sentence_fixed_tree_constructionorder in yD(qH):
    qI=qs[i]-1
    qQ=(qT-1)-qI
    L2=np.zeros([3,qQ],np.int32)
    L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
    x.append(L)
   qb=np.array(x)
   qO={dt.yV:qr,dt.batch_real_sentence_length:qs,dt.batch_len:[qL],dt.batch_treeConstructionOrders:qb}
   qn=lq.run([dt.tensorLoss_fixed_tree],feed_dict=qO)
   qM.append(qn)
   dy.log('\r验证过程epoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(qB,qw,qL,qn)) 
   dy.log('\r验证过程epoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(qB,yM(qM),np.mean(qM))) 
   qw=qw+1
   pass
  return qM 
 def train(dt,restore=yR):
  with tf.Graph().as_default():
   dt.xiaojie_RvNN_fixed_tree()
   qa=tf.initialize_all_variables()
   qJ=tf.train.Saver()
   qW=[]
   qf=[]
   qS=yX('inf')
   qP=yX('inf')
   ld=0 
   lu=-1
   qB=0
   ql=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=yN),allow_soft_placement=yN)
   ql.gpu_options.per_process_gpu_memory_fraction=0.95
   with tf.Session(config=ql)as lq:
    lq.run(qa)
    ly=time.time()
    if restore:qJ.restore(lq,'./weights/%s'%dt.config.model_name)
    while qB<dt.config.max_epochs:
     dy.log('epoch %d'%qB)
     qM=dt.run_epoch_train(lq,qB)
     qW.extend(qM)
     lt=dt.run_epoch_evaluation(lq,qB)
     lv=np.mean(lt)
     qf.append(lv)
     dy.log("time per epoch is {} s".format(time.time()-ly))
     lM=lv
     if lM>qS*dt.config.anneal_threshold:
      dt.config.lr/=dt.config.anneal_by
      dy.log('annealed lr to %f'%dt.config.lr)
     qS=lM 
     if lv<qP:
      shutil.copyfile('./weights/%s.temp.data-00000-of-00001'%dt.config.model_name,'./weights/%s.data-00000-of-00001'%dt.config.model_name)
      shutil.copyfile('./weights/%s.temp.index'%dt.config.model_name,'./weights/%s.index'%dt.config.model_name)
      shutil.copyfile('./weights/%s.temp.meta'%dt.config.model_name,'./weights/%s.meta'%dt.config.model_name)
      qP=lv
      ld=qB
     elif qB-ld>=dt.config.early_stopping:
      lu=qB
      break
     qB+=1
     ly=time.time()
     pass
    if(qB<(dt.config.max_epochs-1)):
     dy.log('预定训练{}个epoch,一共训练{}个epoch，在评估集上最优的是第{}个epoch(从0开始计数),最优评估loss是{}'.format(dt.config.max_epochs,lu+1,ld,qP))
    elif(qB==(dt.config.max_epochs-1)):
     dy.log('预定训练{}个epoch,全部轮数达成，在评估集上最优的是第{}个epoch,最优评估loss是{}'.format(dt.config.max_epochs,ld,qP))
    else:
     dy.log('预定训练{}个epoch,全部轮数达成，在评估集上最优的是第{}个epoch,最优评估loss是{}'.format(dt.config.max_epochs,ld,qP))
   return{'complete_loss_history':qW,'evalution_loss_history':qf,}
 def using_model_for_BigCloneBench_experimentID_1(dt):
  dy.log("------------------------------\n读取BigCloneBench的所有ID编号")
  dm='./SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
  lY=[]
  with yY(dm,'rb')as f:
   di=pickle.load(f)
   for lz in di.keys():
    dD=di[lz]
    if(dD==-1):
     continue 
    lY.append(lz)
  lC=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
  lh='./vector/'+lC+'_using_Weigthed_RAE_BigCloneBenchFunctionID_Map_Vector_root.xiaojiepkl'
  if os.path.exists(lh): 
   os.remove(lh) 
  else:
   yg('no such file:%s'%lh)
  lC=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
  lN='./vector/'+lC+'_using_Weigthed_RAE_BigCloneBenchFunctionID_Map_Vector_weighted.xiaojiepkl'
  if os.path.exists(lN): 
   os.remove(lN) 
  else:
   yg('no such file:%s'%lN)
  qY=dt.bigCloneBench_Corpus
  qC=dt.bigCloneBench_Corpus_fixed_tree_constructionorder
  qz=dt.bigCloneBench_Corpus_sentence_length
  lm=dt.bigCloneBench_Corpus_fixed_tree_parentType_weight
  lK={}
  li={}
  del(dt.trainCorpus)
  del(dt.trainCorpus_sentence_length)
  del(dt.train_corpus_fixed_tree_constructionorder)
  del(dt.vocabulary)
  with tf.Graph().as_default():
   dt.xiaojie_RvNN_fixed_tree_for_usingmodel() 
   qJ=tf.train.Saver()
   ql=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=yN))
   ql.gpu_options.allocator_type='BFC'
   with tf.Session(config=ql)as lq:
    lD='./weights/%s'%dt.config.model_name
    qJ.restore(lq,lD)
    qm=300; 
    dy.log('设置长短的衡量标准是{}'.format(qm))
    qK=[]
    qi=[]
    for dN,length in yD(qz):
     if length<qm:
      qK.append(dN)
     else:
      qi.append(dN)
    dy.log("较长的句子{}个".format(yM(qi)))
    dy.log("较短的句子{}个".format(yM(qK)))
    qD=[qY[dN]for dN in qK]
    qp=[qz[dN]for dN in qK]
    qR=[qC[dN]for dN in qK]
    lp=[lm[dN]for dN in qK]
    qX=[qY[dN]for dN in qi]
    qg=[qz[dN]for dN in qi]
    qx=[qC[dN]for dN in qi]
    lR=[lm[dN]for dN in qi]
    dy.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
    dy.log("先处理较短的句子的语料，批处理开始")
    dz=yM(qK)
    ds=np.array(dt.We)
    del(dt.We)
    qj=ds.shape[0]
    dC=yt(yv(dz))
    qF=dt.config.batch_size_using_model_notTrain
    lX=(dz-1)/qF 
    qw=0
    for qk in yv(0,dz,qF):
     dy.log("batch_index:{}/{}".format(qw,lX))
     qL=yp(qk+qF,dz)-qk
     qA=dC[qk:qk+qL]
     qo=[qD[dN]for dN in qA]
     qs=[qp[dN]for dN in qA]
     qT=yh(qs)
     x=[]
     for i,sentence in yD(qo):
      qE=qs[i]
      dA=np.ones_like(sentence)
      do=sentence-dA
      L1=ds[:,do]
      qQ=qT-qE
      L2=np.zeros([qj,qQ],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     qG=np.array(x)
     qr=np.array(qG,np.float64)
     del(qG)
     x=[]
     qH=[qR[dN]for dN in qA]
     for i,sentence_fixed_tree_constructionorder in yD(qH):
      qI=qs[i]-1
      qQ=(qT-1)-qI
      L2=np.zeros([3,qQ],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     qb=np.array(x)
     qO={dt.yV:qr,dt.batch_real_sentence_length:qs,dt.batch_len:[qL],dt.batch_treeConstructionOrders:qb}
     qn,uA=lq.run([dt.tensorLoss_fixed_tree,dt.batch_sentence_vectors],feed_dict=qO)
     x=[]
     lg=[lp[dN]for dN in qA]
     for i,lV in yD(lg):
      lx=np.sum(lV)
      lV=lV/lx
      lj=qs[i]-1
      qQ=(qT-1)-lj
      L2=np.zeros([qQ],np.int32)
      L=np.concatenate((lV,L2))
      x.append(L)
     lF=np.array(x)
     lw=0
     for dN in qA:
      lk=uA[lw,:,:]
      lV=lF[lw]
      dM=qs[lw]
      lL=2*dM-1
      lA=lk[0:dt.config.embed_size,0:lL]
      lA=lA.astype(np.float32)
      lw=lw+1
      lA=np.transpose(lA)
      lo=yt(lA)
      ls=qK[dN]
      lK[ls]=lo[lL-1]
      ls=qK[dN]
      lT=np.zeros_like(lo[0],np.float32)
      for i in yv(dM,lL):
       lE=lo[i]
       j=i-dM
       ux=lV[j]
       lE=np.multiply(lE,ux)
       lT=np.add(lT,lE)
      li[ls]=lT
     qw=qw+1
    dy.log("再处理较长的句子的语料，每个句子单独处理，开始") 
    qe=yM(qi)
    dC=yt(yv(qe))
    for qk,sentence in yD(qX):
     dy.log("long_setence_index:{}/{}".format(qk,qe))
     qL=1
     qA=dC[qk:qk+qL]
     qo=[qX[dN]for dN in qA]
     qs=[qg[dN]for dN in qA]
     qT=yh(qs)
     x=[]
     for i,sentence in yD(qo):
      qE=qs[i]
      dA=np.ones_like(sentence)
      do=sentence-dA
      L1=ds[:,do]
      qQ=qT-qE
      L2=np.zeros([qj,qQ],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     qG=np.array(x)
     qr=np.array(qG,np.float64)
     x=[]
     qH=[qx[dN]for dN in qA]
     for i,sentence_fixed_tree_constructionorder in yD(qH):
      qI=qs[i]-1
      qQ=(qT-1)-qI
      L2=np.zeros([3,qQ],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     qb=np.array(x)
     qO={dt.yV:qr,dt.batch_real_sentence_length:qs,dt.batch_len:[qL],dt.batch_treeConstructionOrders:qb}
     qn,uA=lq.run([dt.tensorLoss_fixed_tree,dt.batch_sentence_vectors],feed_dict=qO)
     x=[]
     lg=[lR[dN]for dN in qA]
     for i,lV in yD(lg):
      lx=np.sum(lV)
      lV=lV/lx
      lj=qs[i]-1
      qQ=(qT-1)-lj
      L2=np.zeros([qQ],np.int32)
      L=np.concatenate((lV,L2))
      x.append(L)
     lF=np.array(x)
     lA=uA[0,:,:]
     lV=lF[0]
     dM=qs[0]
     lL=2*dM-1
     lA=lA.astype(np.float32)
     lA=np.transpose(lA)
     lo=yt(lA)
     ls=qi[qk]
     lK[ls]=lo[lL-1]
     ls=qi[qk]
     lT=np.zeros_like(lo[0],np.float32)
     for i in yv(dM,lL):
      lE=lo[i]
      j=i-dM
      ux=lV[j]
      lE=np.multiply(lE,ux)
      lT=np.add(lT,lE)
     li[ls]=lT
     pass
  lQ={}
  lG={}
  lr={}
  for i,dD in yD(dt.lines_for_bigcloneBench):
   lr[dD]=i
  for lH in lY:
   dD=di[lH]
   lI=lr[dD]
   lb=lK[lI]
   lO=li[lI]
   lQ[lH]=lb
   lG[lH]=lO
  dt.save_to_pkl(lQ,lh)
  dt.save_to_pkl(lG,lN)
  yg(lh)
  yg(lN)
  pass
  return 
 def using_model_for_BigCloneBench_experimentID_2(dt):
  lC=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
  lc='./vectorTree/'+lC+'_vectorTree.xiaojiepkl'
  if os.path.exists(lc): 
   os.remove(lh) 
  else:
   yg('no such file:%s'%lc)
  qY=dt.need_vectorTree_Corpus
  qC=dt.need_vectorTree_Corpus_fixed_tree_constructionorder
  qz=dt.need_vectorTree_Corpus_sentence_length
  ln={}
  with tf.Graph().as_default():
   dt.xiaojie_RvNN_fixed_tree_for_usingmodel() 
   qJ=tf.train.Saver()
   ql=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=yN))
   ql.gpu_options.allocator_type='BFC'
   with tf.Session(config=ql)as lq:
    lD='./weights/%s'%dt.config.model_name
    qJ.restore(lq,lD)
    qm=500; 
    dy.log('设置长短的衡量标准是{}'.format(qm))
    qK=[]
    qi=[]
    for dN,length in yD(qz):
     if length<qm:
      qK.append(dN)
     else:
      qi.append(dN)
    dy.log("较长的句子{}个".format(yM(qi)))
    dy.log("较短的句子{}个".format(yM(qK)))
    qD=[qY[dN]for dN in qK]
    qp=[qz[dN]for dN in qK]
    qR=[qC[dN]for dN in qK]
    qX=[qY[dN]for dN in qi]
    qg=[qz[dN]for dN in qi]
    qx=[qC[dN]for dN in qi]
    dy.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
    dy.log("先处理较短的句子的语料，批处理开始")
    dz=yM(qK)
    ds=np.array(dt.We)
    qj=ds.shape[0]
    dC=yt(yv(dz))
    qF=dt.config.batch_size_using_model_notTrain
    lX=(dz-1)/qF 
    qw=0
    for qk in yv(0,dz,qF):
     dy.log("batch_index:{}/{}".format(qw,lX))
     qL=yp(qk+qF,dz)-qk
     qA=dC[qk:qk+qL]
     qo=[qD[dN]for dN in qA]
     qs=[qp[dN]for dN in qA]
     qT=yh(qs)
     x=[]
     for i,sentence in yD(qo):
      qE=qs[i]
      dA=np.ones_like(sentence)
      do=sentence-dA
      L1=ds[:,do]
      qQ=qT-qE
      L2=np.zeros([qj,qQ],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     qG=np.array(x)
     qr=np.array(qG,np.float64)
     x=[]
     qH=[qR[dN]for dN in qA]
     for i,sentence_fixed_tree_constructionorder in yD(qH):
      qI=qs[i]-1
      qQ=(qT-1)-qI
      L2=np.zeros([3,qQ],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     qb=np.array(x)
     qO={dt.yV:qr,dt.batch_real_sentence_length:qs,dt.batch_len:[qL],dt.batch_treeConstructionOrders:qb}
     qn,uA=lq.run([dt.tensorLoss_fixed_tree,dt.batch_sentence_vectors],feed_dict=qO)
     lw=0
     for dN in qA:
      lk=uA[lw,:,:]
      dM=qs[lw]
      lL=2*dM-1
      lA=lk[0:dt.config.embed_size,0:lL]
      lA=lA.astype(np.float32)
      lw=lw+1
      lA=np.transpose(lA)
      lo=yt(lA)
      ls=qK[dN]
      ln[ls]=lo
     qw=qw+1
    dy.log("再处理较长的句子的语料，每个句子单独处理，开始") 
    qe=yM(qi)
    dC=yt(yv(qe))
    for qk,sentence in yD(qX):
     dy.log("long_setence_index:{}/{}".format(qk,qe))
     qL=1
     qA=dC[qk:qk+qL]
     qo=[qX[dN]for dN in qA]
     qs=[qg[dN]for dN in qA]
     qT=yh(qs)
     x=[]
     for i,sentence in yD(qo):
      qE=qs[i]
      dA=np.ones_like(sentence)
      do=sentence-dA
      L1=ds[:,do]
      qQ=qT-qE
      L2=np.zeros([qj,qQ],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     qG=np.array(x)
     qr=np.array(qG,np.float64)
     x=[]
     qH=[qx[dN]for dN in qA]
     for i,sentence_fixed_tree_constructionorder in yD(qH):
      qI=qs[i]-1
      qQ=(qT-1)-qI
      L2=np.zeros([3,qQ],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     qb=np.array(x)
     qO={dt.yV:qr,dt.batch_real_sentence_length:qs,dt.batch_len:[qL],dt.batch_treeConstructionOrders:qb}
     qn,uA=lq.run([dt.tensorLoss_fixed_tree,dt.batch_sentence_vectors],feed_dict=qO)
     lA=uA[0,:,:]
     dM=qs[0]
     lL=2*dM-1
     lA=lA.astype(np.float32)
     lA=np.transpose(lA)
     lo=yt(lA)
     ls=qi[qk]
     ln[ls]=lo
  lB={}
  lJ={}
  for i,dD in yD(dt.need_vectorTree_lines_for_trainCorpus):
   lJ[dD]=i
  for lH in dt.need_vectorTree_ids_for_trainCorpus:
   dD=dt.id_line_dict[lH]
   lU=lJ[dD]
   le=ln[lU]
   lB[lH]=le
  dt.save_to_pkl(lB,lc)
  yg(lc)
  pass
  return 
 def using_model_for_BigCloneBench_experimentID_3(dt):
  la=0 
  lh='./vector/'+yx(la)+'_fullCorpusLine_Map_Vector_root.xiaojiepkl'
  if os.path.exists(lh): 
   os.remove(lh) 
  else:
   yg('no such file:%s'%lh)
  lN='./vector/'+yx(la)+'_fullCorpusLine_Map_Vector_mean.xiaojiepkl'
  if os.path.exists(lN): 
   os.remove(lN) 
  else:
   yg('no such file:%s'%lN)
  lW=0
  qY=dt.fullCorpus
  qC=dt.full_corpus_fixed_tree_constructionorder
  qz=dt.fullCorpus_sentence_length
  lm=dt.full_corpus_fixed_tree_parentType_weight
  lK={}
  li={}
  del(dt.fullCorpus)
  del(dt.fullCorpus_sentence_length)
  del(dt.full_corpus_fixed_tree_constructionorder)
  with tf.Graph().as_default():
   dt.xiaojie_RvNN_fixed_tree_for_usingmodel() 
   qJ=tf.train.Saver()
   ql=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=yN))
   ql.gpu_options.allocator_type='BFC'
   with tf.Session(config=ql)as lq:
    lD='./weights/%s'%dt.config.model_name
    qJ.restore(lq,lD)
    qm=300; 
    dy.log('设置长短的衡量标准是{}'.format(qm))
    qK=[]
    qi=[]
    for dN,length in yD(qz):
     if length<qm:
      qK.append(dN)
     else:
      qi.append(dN)
    dy.log("较长的句子{}个".format(yM(qi)))
    dy.log("较短的句子{}个".format(yM(qK)))
    qD=[qY[dN]for dN in qK]
    qp=[qz[dN]for dN in qK]
    qR=[qC[dN]for dN in qK]
    lp=[lm[dN]for dN in qK]
    qX=[qY[dN]for dN in qi]
    qg=[qz[dN]for dN in qi]
    qx=[qC[dN]for dN in qi]
    lR=[lm[dN]for dN in qi]
    dy.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
    dy.log("先处理较短的句子的语料，批处理开始")
    dz=yM(qK)
    ds=np.array(dt.We)
    del(dt.We)
    qj=ds.shape[0]
    dC=yt(yv(dz))
    qF=dt.config.batch_size_using_model_notTrain
    lX=(dz-1)/qF 
    qw=0
    for qk in yv(0,dz,qF):
     dy.log("batch_index:{}/{}".format(qw,lX))
     qL=yp(qk+qF,dz)-qk
     qA=dC[qk:qk+qL]
     qo=[qD[dN]for dN in qA]
     qs=[qp[dN]for dN in qA]
     qT=yh(qs)
     x=[]
     for i,sentence in yD(qo):
      qE=qs[i]
      dA=np.ones_like(sentence)
      do=sentence-dA
      L1=ds[:,do]
      qQ=qT-qE
      L2=np.zeros([qj,qQ],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     qG=np.array(x)
     qr=np.array(qG,np.float64)
     x=[]
     qH=[qR[dN]for dN in qA]
     for i,sentence_fixed_tree_constructionorder in yD(qH):
      qI=qs[i]-1
      qQ=(qT-1)-qI
      L2=np.zeros([3,qQ],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     qb=np.array(x)
     qO={dt.yV:qr,dt.batch_real_sentence_length:qs,dt.batch_len:[qL],dt.batch_treeConstructionOrders:qb}
     qn,uA=lq.run([dt.tensorLoss_fixed_tree,dt.batch_sentence_vectors],feed_dict=qO)
     x=[]
     lg=[lp[dN]for dN in qA]
     for i,lV in yD(lg):
      lx=np.sum(lV)
      lV=lV/lx
      lj=qs[i]-1
      qQ=(qT-1)-lj
      L2=np.zeros([qQ],np.int32)
      L=np.concatenate((lV,L2))
      x.append(L)
     lF=np.array(x)
     lw=0
     for dN in qA:
      lk=uA[lw,:,:]
      lV=lF[lw]
      dM=qs[lw]
      lL=2*dM-1
      lA=lk[0:dt.config.embed_size,0:lL]
      lA=lA.astype(np.float32)
      lw=lw+1
      lA=np.transpose(lA)
      lo=yt(lA)
      ls=qK[dN]
      lK[ls]=lo[lL-1]
      ls=qK[dN]
      lT=np.zeros_like(lo[0],np.float32)
      for i in yv(dM,lL):
       lE=lo[i]
       j=i-dM
       ux=lV[j]
       lE=np.multiply(lE,ux)
       lT=np.add(lT,lE)
      li[ls]=lT
     if(lW>70000):
      dt.save_to_pkl(lK,lh)
      dt.save_to_pkl(li,lN)
      lW=0
      del(lK)
      del(li)
      lK={}
      li={}
      la+=1 
      lh='./vector/'+yx(la)+'_fullCorpusLine_Map_Vector_root.xiaojiepkl'
      if os.path.exists(lh): 
       os.remove(lh) 
      else:
       yg('no such file:%s'%lh)
      lN='./vector/'+yx(la)+'_fullCorpusLine_Map_Vector_mean.xiaojiepkl'
      if os.path.exists(lN): 
       os.remove(lN) 
      else:
       yg('no such file:%s'%lN)
     qw=qw+1
     lW=lW+qL
    dt.save_to_pkl(lK,lh)
    dt.save_to_pkl(li,lN)
    del(lK)
    del(li)
    lK={}
    li={}
    la+=1 
    lh='./vector/'+yx(la)+'_fullCorpusLine_Map_Vector_root.xiaojiepkl'
    if os.path.exists(lh): 
     os.remove(lh) 
    else:
     yg('no such file:%s'%lh)
    lN='./vector/'+yx(la)+'_fullCorpusLine_Map_Vector_mean.xiaojiepkl'
    if os.path.exists(lN): 
     os.remove(lN) 
    else:
     yg('no such file:%s'%lN)
    dy.log("再处理较长的句子的语料，每个句子单独处理，开始") 
    qe=yM(qi)
    dC=yt(yv(qe))
    for qk,sentence in yD(qX):
     dy.log("long_setence_index:{}/{}".format(qk,qe))
     qL=1
     qA=dC[qk:qk+qL]
     qo=[qX[dN]for dN in qA]
     qs=[qg[dN]for dN in qA]
     qT=yh(qs)
     x=[]
     for i,sentence in yD(qo):
      qE=qs[i]
      dA=np.ones_like(sentence)
      do=sentence-dA
      L1=ds[:,do]
      qQ=qT-qE
      L2=np.zeros([qj,qQ],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     qG=np.array(x)
     qr=np.array(qG,np.float64)
     x=[]
     qH=[qx[dN]for dN in qA]
     for i,sentence_fixed_tree_constructionorder in yD(qH):
      qI=qs[i]-1
      qQ=(qT-1)-qI
      L2=np.zeros([3,qQ],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     qb=np.array(x) 
     x=[]
     lg=[lR[dN]for dN in qA]
     for i,lV in yD(lg):
      lx=np.sum(lV)
      lV=lV/lx
      lj=qs[i]-1
      qQ=(qT-1)-lj
      L2=np.zeros([qQ],np.int32)
      L=np.concatenate((lV,L2))
      x.append(L)
     lF=np.array(x)
     qO={dt.yV:qr,dt.batch_real_sentence_length:qs,dt.batch_len:[qL],dt.batch_treeConstructionOrders:qb}
     qn,uA=lq.run([dt.tensorLoss_fixed_tree,dt.batch_sentence_vectors],feed_dict=qO)
     lA=uA[0,:,:]
     lV=lF[0]
     dM=qs[0]
     lL=2*dM-1
     lA=lA.astype(np.float32)
     lA=np.transpose(lA)
     lo=yt(lA)
     ls=qi[qk]
     lK[ls]=lo[lL-1]
     ls=qi[qk]
     lT=np.zeros_like(lo[0],np.float32)
     for i in yv(dM,lL):
      lE=lo[i]
      j=i-dM
      ux=lV[j]
      lE=np.multiply(lE,ux)
      lT=np.add(lT,lE)
     li[ls]=lT
  dt.save_to_pkl(lK,lh)
  dt.save_to_pkl(li,lN)
  yg(lh)
  yg(lN)
  del(lK)
  del(li)
  pass
  return 
 def save_to_pkl(dt,lf,pickle_name):
  with yY(pickle_name,'wb')as pickle_f:
   pickle.dump(lf,pickle_f)
 def read_from_pkl(dt,pickle_name):
  with yY(pickle_name,'rb')as pickle_f:
   lf=pickle.load(pickle_f)
  return lf 
 def similarities(dt,qY,qz,qC,lD):
  dy.log('对语料库计算句与句的相似性') 
  dy.log('被相似计算的语料库一共{}个sentence'.format(yM(qY)))
  lC=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
  lS=lC+'.xiaojiepkl'
  if os.path.exists(lS): 
   os.remove(lS) 
  else:
   yg('no such file:%s'%lS)
  ln={}
  with tf.Graph().as_default():
   dt.xiaojie_RvNN_fixed_tree_for_usingmodel()
   qJ=tf.train.Saver()
   ql=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=yN))
   ql.gpu_options.allocator_type='BFC'
   with tf.Session(config=ql)as lq:
    qJ.restore(lq,lD)
    qm=1000; 
    dy.log('设置长短的衡量标准是{}'.format(qm))
    qK=[]
    qi=[]
    for dN,length in yD(qz):
     if length<qm:
      qK.append(dN)
     else:
      qi.append(dN)
    dy.log("较长的句子{}个".format(yM(qi)))
    dy.log("较短的句子{}个".format(yM(qK)))
    qD=[qY[dN]for dN in qK]
    qp=[qz[dN]for dN in qK]
    qR=[qC[dN]for dN in qK]
    qX=[qY[dN]for dN in qi]
    qg=[qz[dN]for dN in qi]
    qx=[qC[dN]for dN in qi]
    dy.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
    dy.log("先处理较短的句子的语料，批处理开始")
    dz=yM(qK)
    ds=np.array(dt.We)
    qj=ds.shape[0]
    dC=yt(yv(dz))
    qF=dt.config.batch_size_using_model_notTrain
    lX=(dz-1)/qF 
    qw=0
    for i in yv(0,dz,qF):
     dy.log("batch_index:{}/{}".format(qw,lX))
     qL=yp(i+qF,dz)-i
     qA=dC[i:i+qL]
     qo=[qD[dN]for dN in qA]
     qs=[qp[dN]for dN in qA]
     qT=yh(qs)
     x=[]
     for i,sentence in yD(qo):
      qE=qs[i]
      dA=np.ones_like(sentence)
      do=sentence-dA
      L1=ds[:,do]
      qQ=qT-qE
      L2=np.zeros([qj,qQ],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     qG=np.array(x)
     qr=np.array(qG,np.float64)
     x=[]
     qH=[qR[dN]for dN in qA]
     for i,sentence_fixed_tree_constructionorder in yD(qH):
      qI=qs[i]-1
      qQ=(qT-1)-qI
      L2=np.zeros([3,qQ],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     qb=np.array(x)
     qO={dt.yV:qr,dt.batch_real_sentence_length:qs,dt.batch_len:[qL],dt.batch_treeConstructionOrders:qb}
     qn,uA=lq.run([dt.tensorLoss_fixed_tree,dt.batch_sentence_vectors],feed_dict=qO)
     lw=0
     for dN in qA:
      lk=uA[lw,:,:]
      dM=qs[lw]
      lL=2*dM-1
      lA=lk[0:dt.config.embed_size,0:lL]
      lA=lA.astype(np.float32)
      lw=lw+1
      lA=np.transpose(lA)
      lo=yt(lA)
      ls=qK[dN]
      ln[ls]=lo 
     qw=qw+1
    dy.log("再处理较长的句子的语料，每个句子单独处理，开始") 
    qe=yM(qi)
    for dN,sentence in yD(qX):
     dy.log("long_setence_index:{}/{}".format(dN,qe))
     lP=qx[dN]
     (_,lA,qv,_)=dt.computelossAndVector_no_tensor_withAST(sentence,lP)
     lo=[]
     for kk in yv(2*qg[dN]-1):
      yd=lA[kk]
      yd=yd[:,0]
      yd=yd.astype(np.float32)
      lo.append(yd)
     ls=qi[dN]
     ln[ls]=lo 
    with yY(lS,'wb')as f:
     pickle.dump(ln,f)
    dy.log('相似性计算结束后，corpus的所有句子的最终向量表示(向量树)存储的位置是为%s'%lS)
def test_RNN():
 yV("开始？")
 dy.log("------------------------------\n程序开始")
 ql=du()
 yu=dl(ql)
 yq='./weights/%s'%yu.config.model_name
 yu.similarities(corpus=yu.fullCorpus,corpus_sentence_length=yu.fullCorpus_sentence_length,weights_path=yq,corpus_fixed_tree_constructionorder=yu.full_corpus_fixed_tree_constructionorder)
 dy.log("程序结束\n------------------------------")
def xiaojie_RNN_1():
 dy.log("------------------------------\n为模型加载训练样本集合，并配置参数")
 dv={}
 dv['label_size']=2
 dv['early_stopping']=2 
 dv['max_epochs']=30
 dv['anneal_threshold']=0.99
 dv['anneal_by']=1.5
 dv['lr']=0.01
 dv['l2']=0.02
 dv['embed_size']=296
 dv['model_name']='weighted_RAE_rnn_embed=%d_l2=%f_lr=%f.weights'%(dv['embed_size'],dv['lr'],dv['l2'])
 dv['IDIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/2word2vecOutData/'
 dv['ODIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/3RvNNoutData/'
 dv['corpus_fixed_tree_constructionorder_file']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/corpus_bcb_reduced.method.AstConstruction.txt'
 dv['MAX_SENTENCE_LENGTH']=1000000
 dv['batch_size']=10 
 dv['batch_size_using_model_notTrain']=400 
 dv['MAX_SENTENCE_LENGTH_for_Bigclonebench']=300 
 dv['corpus_fixed_tree_construction_parentType_weight_file']='./1corpusData/corpus_bcb_reduced.method.AstConstructionParentTypeWeight.txt'
 ql=du(dv)
 yu=dl(ql,experimentID=1)
 yu.using_model_for_BigCloneBench_experimentID_1()
def xiaojie_RNN_2():
 dy.log("------------------------------\n为模型加载训练样本集合，并配置参数")
 dv={}
 dv['label_size']=2
 dv['early_stopping']=2 
 dv['max_epochs']=30
 dv['anneal_threshold']=0.99
 dv['anneal_by']=1.5
 dv['lr']=0.01
 dv['l2']=0.02
 dv['embed_size']=300
 dv['model_name']='experimentID_1_rnn_embed=%d_l2=%f_lr=%f.weights'%(dv['embed_size'],dv['lr'],dv['l2'])
 dv['IDIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/2word2vecOutData/'
 dv['ODIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/3RvNNoutData/'
 dv['corpus_fixed_tree_constructionorder_file']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/corpus_bcb_reduced.method.AstConstruction.txt'
 dv['MAX_SENTENCE_LENGTH']=1000000
 dv['batch_size']=10 
 dv['batch_size_using_model_notTrain']=300 
 dv['MAX_SENTENCE_LENGTH_for_Bigclonebench']=600 
 ql=du(dv)
 yu=dl(ql,experimentID=2)
 yu.using_model_for_BigCloneBench_experimentID_2()
from train_traditional_RAE_configuration import configuration
def xiaojie_RNN_3():
 dy.log("------------------------------\n为模型加载训练样本集合，并配置参数")
 dv=configuration
 yg(dv)
 ql=du(dv)
 yu=dl(ql,experimentID=3)
 yu.using_model_for_BigCloneBench_experimentID_3()
def save_to_pkl(lf,pickle_name):
 with yY(pickle_name,'wb')as pickle_f:
  pickle.dump(lf,pickle_f)
def read_from_pkl(pickle_name):
 with yY(pickle_name,'rb')as pickle_f:
  lf=pickle.load(pickle_f)
 return lf 
if __name__=="__main__":
 xiaojie_RNN_3()
# Created by pyminifier (https://github.com/liftoff/pyminifier)
