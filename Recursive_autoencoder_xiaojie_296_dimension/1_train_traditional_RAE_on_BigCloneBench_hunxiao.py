#!/usr/bin/python
tC=object
tW=list
tu=range
tl=open
tM=id
tz=len
iw=map
iN=max
iG=True
it=dict
iS=None
ia=int
ib=enumerate
iH=min
iY=False
iT=float
io=print
iD=input
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import shutil
import tensorflow as tf
from xiaojie_log import xiaojie_log_class
from processWord2vecOutData import preprocess_withAST,preprocess_withAST_experimentID_1
from scipy.spatial.distance import pdist,squareform
from random import shuffle
import random
import pickle
global wi
wi=xiaojie_log_class()
class wN(tC):
 def __init__(wS,wa):
  wi.log("(1)>>>>  before training RvNN,配置超参数")
  wS.label_size=wa['label_size']
  wS.early_stopping=wa['early_stopping']
  wS.max_epochs=wa['max_epochs']
  wS.anneal_threshold=wa['anneal_threshold']
  wS.anneal_by=wa['anneal_by']
  wS.lr=wa['lr']
  wS.l2=wa['l2']
  wS.embed_size=wa['embed_size']
  wS.model_name=wa['model_name']
  wi.log('模型名称为%s'%wS.model_name)
  wS.IDIR=wa['IDIR']
  wS.ODIR=wa['ODIR']
  wS.corpus_fixed_tree_constructionorder_file=wa['corpus_fixed_tree_constructionorder_file']
  wS.batch_size=wa['batch_size']
  wS.batch_size_using_model_notTrain=wa['batch_size_using_model_notTrain']
  wS.MAX_SENTENCE_LENGTH_for_Bigclonebench=wa['MAX_SENTENCE_LENGTH_for_Bigclonebench']
  wi.log("(1)<<<<  before training RvNN,配置超参数完毕")
class wG(tC):
 def __init__(wS,sentence_length=0):
  wS.sl=wb
  wS.nodeScores=np.zeros((2*wS.sl-1,1),dtype=np.double)
  wS.collapsed_sentence=(tW)(tu(0,wS.sl))
  wS.pp=np.zeros((2*wS.sl-1,1),dtype=np.ia)
class wt():
 def load_data_experimentID_1(wS):
  wi.log("(2)>>>>  加载词向量数据和语料库")
  (wS.trainCorpus,wS.trainCorpus_sentence_length,wS.vocabulary,wS.We,wS.config.max_sentence_length_train_Corpus,wS.train_corpus_fixed_tree_constructionorder)=preprocess_withAST_experimentID_1(wS.config.IDIR,wS.config.ODIR,wS.config.corpus_fixed_tree_constructionorder_file,100000)
  wi.log("------------------------------\n对照BigCloneBench中标注的函数，找出在我们的语料库中的编号位置")
  wY='./SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
  wf=[]
  with tl(wY,'rb')as f:
   wT=pickle.load(f)
   for tM in wT.keys():
    wo=wT[tM]
    if(wo==-1):
     continue
    wf.append(wo)
  wD=tz(wf)
  wi.log('BigCloneBench中有效函数ID多少个，对应的取出我们语料库中的语料多少个.{}个'.format(wD))
  wS.trainCorpus=[wS.trainCorpus[wV]for wV in wf]
  wS.train_corpus_fixed_tree_constructionorder=[wS.train_corpus_fixed_tree_constructionorder[wV]for wV in wf]
  wS.trainCorpus_sentence_length=[wS.trainCorpus_sentence_length[wV]for wV in wf]
  wy=tW(iw(tz,wS.trainCorpus))
  wS.config.max_sentence_length_train_Corpus=iN(wy)
  wL=tz(wS.trainCorpus)
  wq=tW(tu(wL))
  shuffle(wq)
  wn=5000
  wI=wq[0:5000]
  wF=wq[wn:10000]
  wg=[wS.trainCorpus[wV]for wV in wI]
  wm=[wS.train_corpus_fixed_tree_constructionorder[wV]for wV in wI]
  wB=[wS.trainCorpus_sentence_length[wV]for wV in wI]
  wy=tW(iw(tz,wg))
  wS.config.max_sentence_length_train_Corpus=iN(wy)
  wh=tz(wg)
  wS.evalutionCorpus=[wS.trainCorpus[wV]for wV in wF]
  wS.evalution_corpus_fixed_tree_constructionorder=[wS.train_corpus_fixed_tree_constructionorder[wV]for wV in wF]
  wS.evalution_corpus_sentence_length=[wS.trainCorpus_sentence_length[wV]for wV in wF]
  wy=tW(iw(tz,wS.evalutionCorpus))
  wS.config.max_sentence_length_evalution_Corpus=iN(wy)
  wp=tz(wS.evalutionCorpus)
  wS.trainCorpus=wg
  wS.train_corpus_fixed_tree_constructionorder=wm
  wS.trainCorpus_sentence_length=wB
  wi.log('(2)>>>>  对照BigCloneBench中标注的函数,从我们的语料库中抽取语料{}个，进一步划分出训练集样本{}个，验证集样本{}个'.format(wL,wh,wp)) 
 def xiaojie_RvNN_fixed_tree(wS):
  wS.add_placeholders_fixed_tree()
  wS.add_model_vars()
  wS.add_loss_fixed_tree()
  wS.train_op=wS.training(wS.tensorLoss_fixed_tree)
 def xiaojie_RvNN_fixed_tree_for_usingmodel(wS):
  wS.add_placeholders_fixed_tree()
  wS.add_model_vars()
  wS.add_loss_and_batchSentenceNodesVector_fixed_tree()
 def buqi_2DmatrixTensor(wS,ws,NJ,Nc,Nv,Nk):
  ws=tf.pad(ws,[[0,Nv-NJ],[0,Nk-Nc]])
  return ws
 def modify_one_profile(wS,tensor,ws,Nx,NU,Nj,NK):
  ws=tf.expand_dims(ws,axis=0)
  wr=tf.slice(tensor,[0,0,0],[Nx,Nj,NK])
  we=tf.slice(tensor,[Nx+1,0,0],[NU-Nx-1,Nj,NK])
  wQ=tf.concat([wr,ws,we],0)
  return wr,we,wQ
 def delete_one_column(wS,tensor,wV,NO,numcolunms):
  wr=tf.slice(tensor,[0,0],[NO,wV])
  we=tf.slice(tensor,[0,wV+1],[NO,numcolunms-(wV+1)])
  wQ=tf.concat([wr,we],1)
  return wQ
 def modify_one_column(wS,tensor,columnTensor,wV,NO,numcolunms):
  wr=tf.slice(tensor,[0,0],[NO,wV])
  we=tf.slice(tensor,[0,wV+1],[NO,numcolunms-(wV+1)])
  wQ=tf.concat([wr,columnTensor,we],1)
  return wQ
 def computeloss_withAST(wS,sentence,tg):
  with tf.variable_scope('Composition',reuse=iG):
   W1=tf.get_variable("W1",dtype=tf.float64)
   b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=iG):
   U=tf.get_variable("U",dtype=tf.float64)
   bs=tf.get_variable("bs",dtype=tf.float64)
  wJ=np.ones_like(sentence)
  wc=sentence-wJ
  wv=np.array(wS.We)
  L=wv[:,wc]
  sl=L.shape[1]
  wk=it()
  for i in tu(0,sl):
   wk[i]=np.expand_dims(L[:,i],1)
  wx=it()
  if(sl>1):
   for j in tu(0,sl-1):
    wU=W1.eval()
    wj=b1.eval()
    wK=U.eval()
    wE=bs.eval()
    wP=tg[:,j]
    wd=wP[0]-1 
    wO=wk[wd]
    wA=wP[1]-1
    wX=wk[wA] 
    wC=wP[2]-1
    wW=np.concatenate((wO,wX),axis=0)
    p=np.tanh(np.dot(wU,wW)+wj)
    wu=p/(np.linalg.norm(p,axis=0)*np.ones(p.shape))
    wk[wC]=wu
    y=np.tanh(np.dot(wK,wu)+wE)
    [y1,y2]=np.split(y,2,axis=0)
    wl=y1/(np.linalg.norm(y1,axis=0)*np.ones(y1.shape))
    wM=y2/(np.linalg.norm(y2,axis=0)*np.ones(y2.shape))
    wz=1 
    Nw=1
    NG,Nt=wz*(wl-wO),Nw*(wM-wX)
    constructionError=np.sum((NG*(wl-wO)+Nt*(wM-wX)),axis=0)*0.5 
    wx[j]=constructionError
    pass
   pass
  Ni=0
  for(key,value)in wx.items():
   Ni=Ni+value
  Ni=Ni/(sl-1)
  return Ni 
 def add_loss_fixed_tree(wS):
  with tf.variable_scope('Composition',reuse=iG):
   wS.W1=tf.get_variable("W1",dtype=tf.float64)
   wS.b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=iG):
   wS.U=tf.get_variable("U",dtype=tf.float64)
   wS.bs=tf.get_variable("bs",dtype=tf.float64)
  NS=tf.zeros(wS.batch_len,tf.float64)
  NS=tf.expand_dims(NS,0)
  wS.numlines_tensor3=tf.constant(1,dtype=tf.int32)
  wS.numcolunms_tensor3=wS.batch_len[0]
  i=tf.constant(0,dtype=tf.int32)
  Na=wS.batch_len[0]
  Nb=lambda a,b,c:tf.less(a,Na)
  NH=tf.constant(0)
  NY=[i,NS,NH]
  def _recurrence(i,NS,NH):
   wS.sentence_embeddings=tf.gather(wS.iD,i,axis=0)
   wS.sentence_length=wS.batch_real_sentence_length[i]
   wS.treeConstructionOrders=wS.batch_treeConstructionOrders[i]
   NT=2*wS.sentence_length-1
   wk=tf.zeros(NT,tf.float64)
   wk=tf.expand_dims(wk,0)
   wk=tf.tile(wk,(wS.config.embed_size,1))
   wS.numlines_tensor=tf.constant(wS.config.embed_size,dtype=tf.int32)
   wS.numcolunms_tensor=NT
   ii=tf.constant(0,dtype=tf.int32)
   No=lambda a,b:tf.less(a,wS.sentence_length)
   ND=[ii,wk]
   def __recurrence(ii,wk):
    NV=tf.expand_dims(wS.sentence_embeddings[:,ii],1)
    wk=wS.modify_one_column(wk,NV,ii,wS.numlines_tensor,wS.numcolunms_tensor)
    ii=tf.add(ii,1)
    return ii,wk
   ii,wk=tf.while_loop(No,__recurrence,ND,parallel_iterations=1)
   wx=tf.zeros(wS.sentence_length-1,tf.float64)
   wx=tf.expand_dims(wx,0)
   wS.numlines_tensor2=tf.constant(1,dtype=tf.int32)
   wS.numcolunms_tensor2=wS.sentence_length-1
   Ny=tf.constant(0,dtype=tf.int32)
   NL=lambda a,b,c,d:tf.less(a,wS.sentence_length-1)
   Nq=[Ny,wx,wk,NH]
   def ____recurrence(Ny,wx,wk,NH):
    wP=wS.treeConstructionOrders[:,Ny]
    wd=wP[0]-1 
    Nn=wk[:,wd]
    wA=wP[1]-1
    NI=wk[:,wA] 
    wC=wP[2]-1
    NF=tf.concat([Nn,NI],axis=0)
    NF=tf.expand_dims(NF,1)
    Ng=tf.tanh(tf.add(tf.matmul(wS.W1,NF),wS.b1))
    Nm=wS.normalization(Ng)
    wk=wS.modify_one_column(wk,Nm,wC,wS.numlines_tensor,wS.numcolunms_tensor)
    y=tf.matmul(wS.U,Nm)+wS.bs
    NB=y.shape[1].value
    (y1,y2)=wS.split_by_row(y,NB)
    wz=1 
    Nw=1
    Nn=tf.expand_dims(Nn,1)
    NI=tf.expand_dims(NI,1)
    NG=tf.subtract(y1,Nn)
    Nt=tf.subtract(y2,NI) 
    constructionError=wS.constructionError(NG,Nt,wz,Nw)
    constructionError=tf.expand_dims(constructionError,1)
    wx=wS.modify_one_column(wx,constructionError,Ny,wS.numlines_tensor2,wS.numcolunms_tensor2)
    Nh=tf.Print(Ny,[Ny],"\niiii:")
    NH=Nh+NH
    NH=Nh+NH
    Nh=tf.Print(wd,[wd],"\nleftChild_index:",summarize=100)
    NH=tf.to_int32(Nh)+NH
    Nh=tf.Print(wA,[wA],"\nrightChild_index:",summarize=100)
    NH=tf.to_int32(Nh)+NH
    Nh=tf.Print(wC,[wC],"\nparent_index:",summarize=100)
    NH=tf.to_int32(Nh)+NH
    Ny=tf.add(Ny,1)
    return Ny,wx,wk,NH
   Ny,wx,wk,NH=tf.while_loop(NL,____recurrence,Nq,parallel_iterations=1)
   pass
   wS.node_tensors_cost_tensor=tf.identity(wx)
   wS.nodes_tensor=tf.identity(wk)
   Np=tf.reduce_mean(wS.node_tensors_cost_tensor)
   Np=tf.expand_dims(tf.expand_dims(Np,0),1)
   NS=wS.modify_one_column(NS,Np,i,wS.numlines_tensor3,wS.numcolunms_tensor3)
   i=tf.add(i,1)
   return i,NS,NH
  i,NS,NH=tf.while_loop(Nb,_recurrence,NY,parallel_iterations=10)
  wS.tfPrint=NH
  with tf.name_scope('loss'):
   Ns=tf.nn.l2_loss(wS.W1)+tf.nn.l2_loss(wS.U)
   wS.batch_constructionError=tf.reduce_mean(NS)
   wS.tensorLoss_fixed_tree=wS.batch_constructionError+Ns*wS.config.l2
  return wS.tensorLoss_fixed_tree
 def add_loss_and_batchSentenceNodesVector_fixed_tree(wS):
  with tf.variable_scope('Composition',reuse=iG):
   wS.W1=tf.get_variable("W1",dtype=tf.float64)
   wS.b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=iG):
   wS.U=tf.get_variable("U",dtype=tf.float64)
   wS.bs=tf.get_variable("bs",dtype=tf.float64)
  NS=tf.zeros(wS.batch_len,tf.float64)
  NS=tf.expand_dims(NS,0)
  Nr=wS.batch_real_sentence_length[tf.argmax(wS.batch_real_sentence_length)]
  Ne=2*Nr-1
  NQ=wS.batch_len[0]*wS.config.embed_size*Ne
  NR=tf.zeros(NQ,tf.float64)
  NR=tf.reshape(NR,[wS.batch_len[0],wS.config.embed_size,Ne])
  wS.size_firstDimension=wS.batch_len[0]
  wS.size_secondDimension=tf.constant(wS.config.embed_size,dtype=tf.int32)
  wS.size_thirdDimension=Ne
  wS.numlines_tensor3=tf.constant(1,dtype=tf.int32)
  wS.numcolunms_tensor3=wS.batch_len[0]
  wS.numlines_tensor4=tf.constant(wS.config.embed_size,dtype=tf.int32)
  wS.numcolunms_tensor4=wS.batch_len[0]
  i=tf.constant(0,dtype=tf.int32)
  Na=wS.batch_len[0]
  Nb=lambda a,b,c,d:tf.less(a,Na)
  NH=tf.constant(0)
  NY=[i,NS,NH,NR]
  def _recurrence(i,NS,NH,NR):
   wS.sentence_embeddings=tf.gather(wS.iD,i,axis=0)
   wS.sentence_length=wS.batch_real_sentence_length[i]
   wS.treeConstructionOrders=wS.batch_treeConstructionOrders[i]
   NT=2*wS.sentence_length-1
   wk=tf.zeros(NT,tf.float64)
   wk=tf.expand_dims(wk,0)
   wk=tf.tile(wk,(wS.config.embed_size,1))
   wS.numlines_tensor=tf.constant(wS.config.embed_size,dtype=tf.int32)
   wS.numcolunms_tensor=NT
   ii=tf.constant(0,dtype=tf.int32)
   No=lambda a,b:tf.less(a,wS.sentence_length)
   ND=[ii,wk]
   def __recurrence(ii,wk):
    NV=tf.expand_dims(wS.sentence_embeddings[:,ii],1)
    wk=wS.modify_one_column(wk,NV,ii,wS.numlines_tensor,wS.numcolunms_tensor)
    ii=tf.add(ii,1)
    return ii,wk
   ii,wk=tf.while_loop(No,__recurrence,ND,parallel_iterations=1)
   wx=tf.zeros(wS.sentence_length-1,tf.float64)
   wx=tf.expand_dims(wx,0)
   wS.numlines_tensor2=tf.constant(1,dtype=tf.int32)
   wS.numcolunms_tensor2=wS.sentence_length-1
   Ny=tf.constant(0,dtype=tf.int32)
   NL=lambda a,b,c,d:tf.less(a,wS.sentence_length-1)
   Nq=[Ny,wx,wk,NH]
   def ____recurrence(Ny,wx,wk,NH):
    wP=wS.treeConstructionOrders[:,Ny]
    wd=wP[0]-1 
    Nn=wk[:,wd]
    wA=wP[1]-1
    NI=wk[:,wA] 
    wC=wP[2]-1
    NF=tf.concat([Nn,NI],axis=0)
    NF=tf.expand_dims(NF,1)
    Ng=tf.tanh(tf.add(tf.matmul(wS.W1,NF),wS.b1))
    Nm=wS.normalization(Ng)
    wk=wS.modify_one_column(wk,Nm,wC,wS.numlines_tensor,wS.numcolunms_tensor)
    y=(tf.matmul(wS.U,Nm)+wS.bs)
    NB=y.shape[1].value
    (y1,y2)=wS.split_by_row(y,NB)
    wz=1 
    Nw=1
    Nn=tf.expand_dims(Nn,1)
    NI=tf.expand_dims(NI,1)
    NG=tf.subtract(y1,Nn)
    Nt=tf.subtract(y2,NI) 
    constructionError=wS.constructionError(NG,Nt,wz,Nw)
    constructionError=tf.expand_dims(constructionError,1)
    wx=wS.modify_one_column(wx,constructionError,Ny,wS.numlines_tensor2,wS.numcolunms_tensor2)
    Nh=tf.Print(Ny,[Ny],"\niiii:")
    NH=Nh+NH
    NH=Nh+NH
    Nh=tf.Print(wd,[wd],"\nleftChild_index:",summarize=100)
    NH=tf.to_int32(Nh)+NH
    Nh=tf.Print(wA,[wA],"\nrightChild_index:",summarize=100)
    NH=tf.to_int32(Nh)+NH
    Nh=tf.Print(wC,[wC],"\nparent_index:",summarize=100)
    NH=tf.to_int32(Nh)+NH
    Ny=tf.add(Ny,1)
    return Ny,wx,wk,NH
   Ny,wx,wk,NH=tf.while_loop(NL,____recurrence,Nq,parallel_iterations=1)
   pass
   wS.node_tensors_cost_tensor=tf.identity(wx)
   wS.nodes_tensor=tf.identity(wk)
   Np=tf.reduce_mean(wS.node_tensors_cost_tensor)
   Np=tf.expand_dims(tf.expand_dims(Np,0),1)
   NS=wS.modify_one_column(NS,Np,i,wS.numlines_tensor3,wS.numcolunms_tensor3)
   NJ=wS.numlines_tensor 
   Nc=wS.numcolunms_tensor 
   Nv=wS.size_secondDimension 
   Nk=wS.size_thirdDimension
   wk=wS.buqi_2DmatrixTensor(wk,NJ,Nc,Nv,Nk)
   wk=tf.reshape(wk,[wS.config.embed_size,Nk])
   Nx=i 
   NU=wS.size_firstDimension
   Nj=wS.size_secondDimension
   NK=wS.size_thirdDimension
   _,_,NR=wS.modify_one_profile(NR,wk,Nx,NU,Nj,NK)
   i=tf.add(i,1)
   return i,NS,NH,NR
  i,NS,NH,NR=tf.while_loop(Nb,_recurrence,NY,parallel_iterations=10)
  wS.tfPrint=NH
  wS.batch_sentence_vectors=tf.identity(NR)
  with tf.name_scope('loss'):
   Ns=tf.nn.l2_loss(wS.W1)+tf.nn.l2_loss(wS.U)
   wS.batch_constructionError=tf.reduce_mean(NS)
   wS.tensorLoss_fixed_tree=wS.batch_constructionError+Ns*wS.config.l2
  return wS.tensorLoss_fixed_tree,wS.batch_sentence_vectors
 def add_placeholders_fixed_tree(wS):
  NE=wS.config.embed_size
  wS.iD=tf.placeholder(tf.float64,[iS,NE,iS],name='input')
  wS.batch_treeConstructionOrders=tf.placeholder(tf.int32,[iS,3,iS],name='treeConstructionOrders')
  wS.batch_real_sentence_length=tf.placeholder(tf.int32,[iS],name='batch_real_sentence_length')
  wS.batch_len=tf.placeholder(tf.int32,shape=(1,),name='batch_len')
 def add_model_vars(wS):
  with tf.variable_scope('Composition'): 
   tf.get_variable("W1",dtype=tf.float64,shape=[wS.config.embed_size,2*wS.config.embed_size])
   tf.get_variable("b1",dtype=tf.float64,shape=[wS.config.embed_size,1])
  with tf.variable_scope('Projection'):
   tf.get_variable("U",dtype=tf.float64,shape=[2*wS.config.embed_size,wS.config.embed_size])
   tf.get_variable("bs",dtype=tf.float64,shape=[2*wS.config.embed_size,1])
 def normalization(wS,tensor):
  NO=tensor.shape[0].value
  NA=tf.pow(tensor,2)
  NX=tf.reduce_sum(NA,0)
  NC=tf.expand_dims(NX,0)
  NW=tf.tile(tf.sqrt(NC),(NO,1))
  Nu=tf.divide(tensor,NW)
  return Nu
 def split_by_row(wS,tensor,numcolunms):
  NO=tensor.shape[0].value
  Nl=tf.slice(tensor,[0,0],[(ia)(NO/2),numcolunms])
  NM=tf.slice(tensor,[(ia)(NO/2),0],[(ia)(NO/2),numcolunms])
  pass
  return(Nl,NM)
 def constructionError(wS,tensor1,tensor2,wz,Nw):
  Nz=tf.multiply(tf.reduce_sum(tf.pow(tensor1,2),0),wz)
  Gw=tf.multiply(tf.reduce_sum(tf.pow(tensor2,2),0),Nw)
  GN=tf.multiply(tf.add(Nz,Gw),0.5)
  return GN
 def training(wS,GX):
  Gt=iS
  Gi=tf.train.GradientDescentOptimizer(wS.config.lr)
  Gt=Gi.minimize(GX)
  return Gt
 def __init__(wS,GS,experimentID=iS):
  if(experimentID==iS):
   pass
  elif(experimentID==1):
   wS.config=GS
   wS.load_data_experimentID_1()
 def computelossAndVector_no_tensor_withAST(wS,sentence,tg):
  with tf.variable_scope('Composition',reuse=iG):
   W1=tf.get_variable("W1",dtype=tf.float64)
   b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=iG):
   U=tf.get_variable("U",dtype=tf.float64)
   bs=tf.get_variable("bs",dtype=tf.float64)
  wJ=np.ones_like(sentence)
  wc=sentence-wJ
  wv=np.array(wS.We)
  L=wv[:,wc]
  sl=L.shape[1]
  Ga=it()
  for i in tu(0,sl):
   Ga[i]=np.expand_dims(L[:,i],1)
  wx=it()
  Gb=iS
  if(sl>1):
   for j in tu(0,sl-1):
    wU=W1.eval()
    wj=b1.eval()
    wK=U.eval()
    wE=bs.eval()
    wP=tg[:,j]
    wd=wP[0]-1 
    wO=Ga[wd]
    wA=wP[1]-1
    wX=Ga[wA] 
    wC=wP[2]-1
    wW=np.concatenate((wO,wX),axis=0)
    p=np.tanh(np.dot(wU,wW)+wj)
    wu=p/(np.linalg.norm(p,axis=0)*np.ones(p.shape))
    Ga[wC]=wu
    Gb=wu
    y=np.tanh(np.dot(wK,wu)+wE)
    [y1,y2]=np.split(y,2,axis=0)
    wl=y1/(np.linalg.norm(y1,axis=0)*np.ones(y1.shape))
    wM=y2/(np.linalg.norm(y2,axis=0)*np.ones(y2.shape))
    wz=1 
    Nw=1
    NG,Nt=wz*(wl-wO),Nw*(wM-wX)
    constructionError=np.sum((NG*(wl-wO)+Nt*(wM-wX)),axis=0)*0.5 
    wx[j]=constructionError
    pass
   GH=Ga[2*sl-2]
   pass
  Ni=0
  for(key,value)in wx.items():
   Ni=Ni+value
  return(Ni,Ga,GH,Gb)
 def save_to_pkl(wS,GY,pickle_name):
  with tl(pickle_name,'wb')as pickle_f:
   pickle.dump(GY,pickle_f)
 def read_from_pkl(wS,pickle_name):
  with tl(pickle_name,'rb')as pickle_f:
   GY=pickle.load(pickle_f)
  return GY
 def run_epoch_train(wS,ta,GC):
  Gf=[]
  GT=wS.trainCorpus
  Go=wS.trainCorpus_sentence_length
  GD=wS.train_corpus_fixed_tree_constructionorder
  GV=wS.config.max_sentence_length_train_Corpus
  Gy=tz(wS.trainCorpus)
  GL=wS.config.MAX_SENTENCE_LENGTH_for_Bigclonebench
  wi.log('训练过程设置长短的衡量标准是{}，长的单独成batch，短的集合成batch'.format(GL))
  Gq=[]
  Gn=[]
  for wV,length in ib(Go):
   if length<GL:
    Gq.append(wV)
   else:
    Gn.append(wV)
  wi.log("训练集的句子{}个".format(Gy))
  wi.log("较长的句子{}个".format(tz(Gn)))
  wi.log("较短的句子{}个".format(tz(Gq)))
  GI=[GT[wV]for wV in Gq]
  GF=[Go[wV]for wV in Gq]
  Gg=[GD[wV]for wV in Gq]
  Gm=[GT[wV]for wV in Gn]
  GB=[Go[wV]for wV in Gn]
  Gh=[GD[wV]for wV in Gn]
  wi.log("较短的句子，我们走批处理训练网络。长句子，我们单独计算训练网络")
  wi.log("先处理较短的句子的语料，批处理开始")
  Gp=tz(Gq)
  wv=np.array(wS.We)
  Gs=wv.shape[0]
  wq=tW(tu(Gp))
  Gr=wS.config.batch_size
  Ge=0
  for GQ in tu(0,Gp,Gr):
   GR=iH(GQ+Gr,Gp)-GQ
   GJ=wq[GQ:GQ+GR]
   Gc=[GI[wV]for wV in GJ]
   Gv=[GF[wV]for wV in GJ]
   Gk=iN(Gv)
   x=[]
   for i,sentence in ib(Gc):
    Gx=Gv[i]
    wJ=np.ones_like(sentence)
    wc=sentence-wJ
    L1=wv[:,wc]
    GU=Gk-Gx
    L2=np.zeros([Gs,GU],np.float64)
    L=np.concatenate((L1,L2),axis=1)
    x.append(L)
   Gj=np.array(x)
   GK=np.array(Gj,np.float64)
   x=[]
   GE=[Gg[wV]for wV in GJ]
   for i,sentence_fixed_tree_constructionorder in ib(GE):
    GP=Gv[i]-1
    GU=(Gk-1)-GP
    L2=np.zeros([3,GU],np.int32)
    L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
    x.append(L)
   Gd=np.array(x)
   GO={wS.iD:GK,wS.batch_real_sentence_length:Gv,wS.batch_len:[GR],wS.batch_treeConstructionOrders:Gd}
   GA=tf.RunOptions(report_tensor_allocations_upon_oom=iG)
   GX,_=ta.run([wS.tensorLoss_fixed_tree,wS.train_op],feed_dict=GO,options=GA)
   Gf.append(GX)
   wi.log('\repoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(GC,Ge,GR,GX)) 
   wi.log('\repoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(GC,tz(Gf),np.mean(Gf))) 
   Ge=Ge+1
   pass 
  wi.log("保存模型到temp")
  GW=tf.train.Saver()
  if not os.path.exists("./weights"):
   os.makedirs("./weights")
  GW.save(ta,'./weights/%s.temp'%wS.config.model_name)
  return Gf 
 def run_epoch_evaluation(wS,ta,GC):
  Gf=[]
  GT=wS.evalutionCorpus
  Go=wS.evalution_corpus_sentence_length
  GD=wS.evalution_corpus_fixed_tree_constructionorder
  GV=wS.config.max_sentence_length_evalution_Corpus
  Gu=tz(wS.evalutionCorpus)
  GL=600; 
  wi.log('训练过程设置长短的衡量标准是{}，长的单独成batch，短的集合成batch'.format(GL))
  Gq=[]
  Gn=[]
  for wV,length in ib(Go):
   if length<GL:
    Gq.append(wV)
   else:
    Gn.append(wV)
  wi.log("yanzhengji的句子{}个".format(Gu))
  wi.log("较长的句子{}个".format(tz(Gn)))
  wi.log("较短的句子{}个".format(tz(Gq)))
  GI=[GT[wV]for wV in Gq]
  GF=[Go[wV]for wV in Gq]
  Gg=[GD[wV]for wV in Gq]
  Gm=[GT[wV]for wV in Gn]
  GB=[Go[wV]for wV in Gn]
  Gh=[GD[wV]for wV in Gn]
  wi.log("较短的句子，我们走批处理训练网络。长句子，我们单独计算训练网络")
  wi.log("先处理较短的句子的语料，批处理开始")
  Gp=tz(Gq)
  wv=np.array(wS.We)
  Gs=wv.shape[0]
  wq=tW(tu(Gp))
  Gr=wS.config.batch_size_using_model_notTrain 
  Ge=0
  for GQ in tu(0,Gp,Gr):
   GR=iH(GQ+Gr,Gp)-GQ
   GJ=wq[GQ:GQ+GR]
   Gc=[GI[wV]for wV in GJ]
   Gv=[GF[wV]for wV in GJ]
   Gk=iN(Gv)
   x=[]
   for i,sentence in ib(Gc):
    Gx=Gv[i]
    wJ=np.ones_like(sentence)
    wc=sentence-wJ
    L1=wv[:,wc]
    GU=Gk-Gx
    L2=np.zeros([Gs,GU],np.float64)
    L=np.concatenate((L1,L2),axis=1)
    x.append(L)
   Gj=np.array(x)
   GK=np.array(Gj,np.float64)
   x=[]
   GE=[Gg[wV]for wV in GJ]
   for i,sentence_fixed_tree_constructionorder in ib(GE):
    GP=Gv[i]-1
    GU=(Gk-1)-GP
    L2=np.zeros([3,GU],np.int32)
    L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
    x.append(L)
   Gd=np.array(x)
   GO={wS.iD:GK,wS.batch_real_sentence_length:Gv,wS.batch_len:[GR],wS.batch_treeConstructionOrders:Gd}
   GX=ta.run([wS.tensorLoss_fixed_tree],feed_dict=GO)
   Gf.append(GX)
   wi.log('\r验证过程epoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(GC,Ge,GR,GX)) 
   wi.log('\r验证过程epoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(GC,tz(Gf),np.mean(Gf))) 
   Ge=Ge+1
   pass 
  wi.log("再处理较长的句子的语料，每个句子单独处理，开始") 
  Gl=tz(Gn)
  wq=tW(tu(Gl))
  for GQ,sentence in ib(Gm):
   GR=1
   GJ=wq[GQ:GQ+GR]
   Gc=[Gm[wV]for wV in GJ]
   Gv=[GB[wV]for wV in GJ]
   Gk=iN(Gv)
   x=[]
   for i,sentence in ib(Gc):
    Gx=Gv[i]
    wJ=np.ones_like(sentence)
    wc=sentence-wJ
    L1=wv[:,wc]
    GU=Gk-Gx
    L2=np.zeros([Gs,GU],np.float64)
    L=np.concatenate((L1,L2),axis=1)
    x.append(L)
   Gj=np.array(x)
   GK=np.array(Gj,np.float64)
   x=[]
   GE=[Gh[wV]for wV in GJ]
   for i,sentence_fixed_tree_constructionorder in ib(GE):
    GP=Gv[i]-1
    GU=(Gk-1)-GP
    L2=np.zeros([3,GU],np.int32)
    L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
    x.append(L)
   Gd=np.array(x)
   GO={wS.iD:GK,wS.batch_real_sentence_length:Gv,wS.batch_len:[GR],wS.batch_treeConstructionOrders:Gd}
   GX=ta.run([wS.tensorLoss_fixed_tree],feed_dict=GO)
   Gf.append(GX)
   wi.log('\r验证过程epoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(GC,Ge,GR,GX)) 
   wi.log('\r验证过程epoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(GC,tz(Gf),np.mean(Gf))) 
   Ge=Ge+1
   pass
  return Gf 
 def train(wS,restore=iY):
  with tf.Graph().as_default():
   wS.xiaojie_RvNN_fixed_tree()
   GM=tf.initialize_all_variables()
   GW=tf.train.Saver()
   Gz=[]
   tw=[]
   tN=iT('inf')
   tG=iT('inf')
   ti=0 
   tS=-1
   GC=0
   GS=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=iG),allow_soft_placement=iG)
   GS.gpu_options.per_process_gpu_memory_fraction=0.95
   with tf.Session(config=GS)as ta:
    ta.run(GM)
    tb=time.time()
    if restore:GW.restore(ta,'./weights/%s'%wS.config.model_name)
    while GC<wS.config.max_epochs:
     wi.log('epoch %d'%GC)
     Gf=wS.run_epoch_train(ta,GC)
     Gz.extend(Gf)
     tH=wS.run_epoch_evaluation(ta,GC)
     tY=np.mean(tH)
     tw.append(tY)
     wi.log("time per epoch is {} s".format(time.time()-tb))
     tf=tY
     if tf>tN*wS.config.anneal_threshold:
      wS.config.lr/=wS.config.anneal_by
      wi.log('annealed lr to %f'%wS.config.lr)
     tN=tf 
     if tY<tG:
      shutil.copyfile('./weights/%s.temp.data-00000-of-00001'%wS.config.model_name,'./weights/%s.data-00000-of-00001'%wS.config.model_name)
      shutil.copyfile('./weights/%s.temp.index'%wS.config.model_name,'./weights/%s.index'%wS.config.model_name)
      shutil.copyfile('./weights/%s.temp.meta'%wS.config.model_name,'./weights/%s.meta'%wS.config.model_name)
      tG=tY
      ti=GC
     elif GC-ti>=wS.config.early_stopping:
      tS=GC
      break
     GC+=1
     tb=time.time()
     pass
    if(GC<(wS.config.max_epochs-1)):
     wi.log('预定训练{}个epoch,一共训练{}个epoch，在评估集上最优的是第{}个epoch(从0开始计数),最优评估loss是{}'.format(wS.config.max_epochs,tS+1,ti,tG))
    elif(GC==(wS.config.max_epochs-1)):
     wi.log('预定训练{}个epoch,全部轮数达成，在评估集上最优的是第{}个epoch,最优评估loss是{}'.format(wS.config.max_epochs,ti,tG))
    else:
     wi.log('预定训练{}个epoch,全部轮数达成，在评估集上最优的是第{}个epoch,最优评估loss是{}'.format(wS.config.max_epochs,ti,tG))
   return{'complete_loss_history':Gz,'evalution_loss_history':tw,}
 def similarities(wS,GT,Go,GD,weights_path):
  wi.log('对语料库计算句与句的相似性') 
  wi.log('被相似计算的语料库一共{}个sentence'.format(tz(GT)))
  tT=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
  to=tT+'.xiaojiepkl'
  if os.path.exists(to): 
   os.remove(to) 
  else:
   io('no such file:%s'%to)
  tD={}
  with tf.Graph().as_default():
   wS.xiaojie_RvNN_fixed_tree_for_usingmodel()
   GW=tf.train.Saver()
   GS=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=iG))
   GS.gpu_options.allocator_type='BFC'
   with tf.Session(config=GS)as ta:
    GW.restore(ta,weights_path)
    GL=1000; 
    wi.log('设置长短的衡量标准是{}'.format(GL))
    Gq=[]
    Gn=[]
    for wV,length in ib(Go):
     if length<GL:
      Gq.append(wV)
     else:
      Gn.append(wV)
    wi.log("较长的句子{}个".format(tz(Gn)))
    wi.log("较短的句子{}个".format(tz(Gq)))
    GI=[GT[wV]for wV in Gq]
    GF=[Go[wV]for wV in Gq]
    Gg=[GD[wV]for wV in Gq]
    Gm=[GT[wV]for wV in Gn]
    GB=[Go[wV]for wV in Gn]
    Gh=[GD[wV]for wV in Gn]
    wi.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
    wi.log("先处理较短的句子的语料，批处理开始")
    wL=tz(Gq)
    wv=np.array(wS.We)
    Gs=wv.shape[0]
    wq=tW(tu(wL))
    Gr=wS.config.batch_size_using_model_notTrain
    tV=(wL-1)/Gr 
    Ge=0
    for i in tu(0,wL,Gr):
     wi.log("batch_index:{}/{}".format(Ge,tV))
     GR=iH(i+Gr,wL)-i
     GJ=wq[i:i+GR]
     Gc=[GI[wV]for wV in GJ]
     Gv=[GF[wV]for wV in GJ]
     Gk=iN(Gv)
     x=[]
     for i,sentence in ib(Gc):
      Gx=Gv[i]
      wJ=np.ones_like(sentence)
      wc=sentence-wJ
      L1=wv[:,wc]
      GU=Gk-Gx
      L2=np.zeros([Gs,GU],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     Gj=np.array(x)
     GK=np.array(Gj,np.float64)
     x=[]
     GE=[Gg[wV]for wV in GJ]
     for i,sentence_fixed_tree_constructionorder in ib(GE):
      GP=Gv[i]-1
      GU=(Gk-1)-GP
      L2=np.zeros([3,GU],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     Gd=np.array(x)
     GO={wS.iD:GK,wS.batch_real_sentence_length:Gv,wS.batch_len:[GR],wS.batch_treeConstructionOrders:Gd}
     GX,NR=ta.run([wS.tensorLoss_fixed_tree,wS.batch_sentence_vectors],feed_dict=GO)
     ty=0
     for wV in GJ:
      tL=NR[ty,:,:]
      wb=Gv[ty]
      tq=2*wb-1
      tn=tL[0:wS.config.embed_size,0:tq]
      tn=tn.astype(np.float32)
      ty=ty+1
      tn=np.transpose(tn)
      tI=tW(tn)
      tF=Gq[wV]
      tD[tF]=tI 
     Ge=Ge+1
    wi.log("再处理较长的句子的语料，每个句子单独处理，开始") 
    Gl=tz(Gn)
    for wV,sentence in ib(Gm):
     wi.log("long_setence_index:{}/{}".format(wV,Gl))
     tg=Gh[wV]
     (_,tn,GH,_)=wS.computelossAndVector_no_tensor_withAST(sentence,tg)
     tI=[]
     for kk in tu(2*GB[wV]-1):
      tm=tn[kk]
      tm=tm[:,0]
      tm=tm.astype(np.float32)
      tI.append(tm)
     tF=Gn[wV]
     tD[tF]=tI 
    with tl(to,'wb')as f:
     pickle.dump(tD,f)
    wi.log('相似性计算结束后，corpus的所有句子的最终向量表示(向量树)存储的位置是为%s'%to)
def test_RNN():
 iD("开始？")
 wi.log("------------------------------\n程序开始")
 GS=wN()
 tB=wt(GS)
 th='./weights/%s'%tB.config.model_name
 tB.similarities(corpus=tB.fullCorpus,corpus_sentence_length=tB.fullCorpus_sentence_length,weights_path=th,corpus_fixed_tree_constructionorder=tB.full_corpus_fixed_tree_constructionorder)
 wi.log("程序结束\n------------------------------")
from train_traditional_RAE_configuration import configuration
def xiaojie_RNN_1():
 wi.log("------------------------------\n为模型加载训练样本集合，并配置参数")
 wa=configuration
 io(wa) 
 GS=wN(wa)
 tB=wt(GS,experimentID=1)
 wi.log("(3)>>>>  开始训练RvNN")
 tb=time.time()
 tp=tB.train(restore=iY)
 wi.log('Training time: {}'.format(time.time()-tb))
 wi.log("(3)<<<<  训练RvNN结束")
 plt.plot(tp['complete_loss_history'])
 plt.title('Batch Reconstruction Error')
 plt.xlabel('Batch Index')
 plt.ylabel('Reconstruction Error')
 plt.savefig("./3RvNNoutData/TraditionalRAE_Batch_Reconstruction_Error.png")
 plt.show()
 plt.plot(tp['evalution_loss_history'])
 plt.title('Reconstruction Error on Evalution Data')
 plt.xlabel('Epoch')
 plt.ylabel('Reconstruction Error')
 plt.savefig("./3RvNNoutData/TraditionalRAE_Reconstruction_Error_on_Evalution_Data.png")
 plt.show()
 tT=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
 ts='./3RvNNoutData/'+tT+'_TraditionalRAE.xiaojiepkl'
 if os.path.exists(ts): 
  os.remove(ts) 
 else:
  io('no such file:%s'%ts)
 tB.save_to_pkl(tp,ts)
def verification_corpus():
 def linesOfFile(filepath):
  tr=0
  with tl(filepath,'r')as fw:
   for i,te in ib(fw,start=1): 
    tr+=1
  return tr
 tR='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/2word2vecOutData/corpus.int'
 tJ='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/corpus_bcb_reduced.method.txt'
 tc='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/corpus_bcb_reduced.method.AstConstruction.txt'
 tv='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/writerpath_bcb_reduced.method.txt'
 tk=linesOfFile(tR)
 tx=linesOfFile(tJ)
 tU=linesOfFile(tc)
 tj=linesOfFile(tv)
 io(tk,tx,tU,tj)
 with tl(tR,'r')as f1:
  with tl(tJ,'r')as f2:
   with tl(tc,'r')as f3:
    with tl(tv,'r')as f4:
     for i in tu(tk):
      tK=f1.readline()
      tE=f2.readline()
      tP=f3.readline()
      td=f4.readline()
      tO=tK.strip().split()
      tA=tE.strip().split()
      tX=tP.strip('\n').strip(' ').split(' ')
      if((tz(tO))!=(tz(tA))):
       io(td)
       io('在corpus.int中的长度{}，同在txt中的长度{}不一致。'.format(tz(tO),tz(tA)))
       iD()
       return 
      if((tz(tO))!=(1+(tz(tX)))):
       io(td)
       io('句子单词长度{}，不等于构建次数{}+1，'.format(tz(tO),tz(tX)))
       iD()
       return 
      pass
 io('校验完毕，没发现问题')
 return 
if __name__=="__main__":
 xiaojie_RNN_1()
# Created by pyminifier (https://github.com/liftoff/pyminifier)
