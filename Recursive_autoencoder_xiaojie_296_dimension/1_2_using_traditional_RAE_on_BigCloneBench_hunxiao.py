#!/usr/bin/python
vi=object
vg=list
vr=range
vN=len
vz=open
vG=id
ve=map
vR=max
vx=True
vu=dict
vU=None
vY=int
Th=enumerate
TE=min
TJ=False
Tv=float
TO=print
Ts=str
To=input
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
global hT
hT=xiaojie_log_class()
class hE(vi):
 def __init__(hO,hs):
  hT.log("(1)>>>>  before training RvNN,配置超参数")
  hO.label_size=hs['label_size']
  hO.early_stopping=hs['early_stopping']
  hO.max_epochs=hs['max_epochs']
  hO.anneal_threshold=hs['anneal_threshold']
  hO.anneal_by=hs['anneal_by']
  hO.lr=hs['lr']
  hO.l2=hs['l2']
  hO.embed_size=hs['embed_size']
  hO.model_name=hs['model_name']
  hT.log('模型名称为%s'%hO.model_name)
  hO.IDIR=hs['IDIR']
  hO.ODIR=hs['ODIR']
  hO.corpus_fixed_tree_constructionorder_file=hs['corpus_fixed_tree_constructionorder_file']
  hO.MAX_SENTENCE_LENGTH=10000
  hO.batch_size=hs['batch_size']
  hO.batch_size_using_model_notTrain=hs['batch_size_using_model_notTrain']
  hO.MAX_SENTENCE_LENGTH_for_Bigclonebench=hs['MAX_SENTENCE_LENGTH_for_Bigclonebench']
  hT.log("(1)<<<<  before training RvNN,配置超参数完毕")
class hJ(vi):
 def __init__(hO,sentence_length=0):
  hO.sl=ho
  hO.nodeScores=np.zeros((2*hO.sl-1,1),dtype=np.double)
  hO.collapsed_sentence=(vg)(vr(0,hO.sl))
  hO.pp=np.zeros((2*hO.sl-1,1),dtype=np.vY)
class hv():
 def load_data(hO):
  hT.log("(2)>>>>  加载词向量数据和语料库")
  (hO.trainCorpus,hO.fullCorpus,hO.trainCorpus_sentence_length,hO.fullCorpus_sentence_length,hO.vocabulary,hO.We,hO.config.max_sentence_length_train_Corpus,hO.config.max_sentence_length_full_Corpus,hO.train_corpus_fixed_tree_constructionorder,hO.full_corpus_fixed_tree_constructionorder)=preprocess_withAST(hO.config.IDIR,hO.config.ODIR,hO.config.corpus_fixed_tree_constructionorder_file,hO.config.MAX_SENTENCE_LENGTH)
  if vN(hO.trainCorpus)>4000:
   hl=vN(hO.trainCorpus)
   hS=vg(vr(hl))
   shuffle(hS)
   hX=hS[0:4000]
   hO.evalutionCorpus=[hO.trainCorpus[hW]for hW in hX]
   hO.config.max_sentence_length_evalution_Corpus=hO.config.max_sentence_length_train_Corpus 
   hO.evalution_corpus_fixed_tree_constructionorder=[hO.train_corpus_fixed_tree_constructionorder[hW]for hW in hX]
  else:
   hO.evalutionCorpus=hO.trainCorpus
   hO.config.max_sentence_length_evalution_Corpus=hO.config.max_sentence_length_train_Corpus
   hO.evalution_corpus_fixed_tree_constructionorder=hO.train_corpus_fixed_tree_constructionorder
  hT.log("(2)>>>>  加载词向量数据和语料库完毕")
 def load_data_experimentID_1(hO):
  hT.log("(2)>>>>  加载词向量数据和语料库")
  (hO.trainCorpus,hO.trainCorpus_sentence_length,hO.vocabulary,hO.We,hO.config.max_sentence_length_train_Corpus,hO.train_corpus_fixed_tree_constructionorder)=preprocess_withAST_experimentID_1(hO.config.IDIR,hO.config.ODIR,hO.config.corpus_fixed_tree_constructionorder_file,100000)
  hT.log("------------------------------\n对照BigCloneBench中标注的函数，找出在我们的语料库中的编号位置")
  hw='./SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
  hc=[]
  with vz(hw,'rb')as f:
   hQ=pickle.load(f)
   for vG in hQ.keys():
    ht=hQ[vG]
    if(ht==-1):
     continue
    hc.append(ht)
  hO.lines_for_bigcloneBench=hc
  hd=vN(hc)
  hT.log('BigCloneBench中有效函数ID多少个，对应的取出我们语料库中的语料多少个.{}个'.format(hd))
  hO.trainCorpus=[hO.trainCorpus[hW]for hW in hc]
  hO.train_corpus_fixed_tree_constructionorder=[hO.train_corpus_fixed_tree_constructionorder[hW]for hW in hc]
  hO.trainCorpus_sentence_length=[hO.trainCorpus_sentence_length[hW]for hW in hc]
  hI=vg(ve(vN,hO.trainCorpus))
  hO.config.max_sentence_length_train_Corpus=vR(hI)
  hO.bigCloneBench_Corpus=hO.trainCorpus 
  hO.bigCloneBench_Corpus_fixed_tree_constructionorder=hO.train_corpus_fixed_tree_constructionorder
  hO.bigCloneBench_Corpus_sentence_length=hO.trainCorpus_sentence_length
  hO.bigCloneBench_Corpus_max_sentence_length=hO.config.max_sentence_length_train_Corpus
  hl=vN(hO.bigCloneBench_Corpus)
  hT.log('(2)>>>>  对照BigCloneBench中标注的函数,从我们的语料库中抽取语料{}个'.format(hl)) 
 def xiaojie_RvNN_fixed_tree(hO):
  hO.add_placeholders_fixed_tree()
  hO.add_model_vars()
  hO.add_loss_fixed_tree()
  hO.train_op=hO.training(hO.tensorLoss_fixed_tree)
 def xiaojie_RvNN_fixed_tree_for_usingmodel(hO):
  hO.add_placeholders_fixed_tree()
  hO.add_model_vars()
  hO.add_loss_and_batchSentenceNodesVector_fixed_tree()
 def buqi_2DmatrixTensor(hO,hP,Ej,Ek,EH,Ey):
  hP=tf.pad(hP,[[0,EH-Ej],[0,Ey-Ek]])
  return hP
 def modify_one_profile(hO,tensor,hP,EA,EL,EF,En):
  hP=tf.expand_dims(hP,axis=0)
  hV=tf.slice(tensor,[0,0,0],[EA,EF,En])
  hD=tf.slice(tensor,[EA+1,0,0],[EL-EA-1,EF,En])
  ha=tf.concat([hV,hP,hD],0)
  return hV,hD,ha
 def delete_one_column(hO,tensor,hW,Eq,numcolunms):
  hV=tf.slice(tensor,[0,0],[Eq,hW])
  hD=tf.slice(tensor,[0,hW+1],[Eq,numcolunms-(hW+1)])
  ha=tf.concat([hV,hD],1)
  return ha
 def modify_one_column(hO,tensor,columnTensor,hW,Eq,numcolunms):
  hV=tf.slice(tensor,[0,0],[Eq,hW])
  hD=tf.slice(tensor,[0,hW+1],[Eq,numcolunms-(hW+1)])
  ha=tf.concat([hV,columnTensor,hD],1)
  return ha
 def computeloss_withAST(hO,sentence,vq):
  with tf.variable_scope('Composition',reuse=vx):
   W1=tf.get_variable("W1",dtype=tf.float64)
   b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=vx):
   U=tf.get_variable("U",dtype=tf.float64)
   bs=tf.get_variable("bs",dtype=tf.float64)
  hj=np.ones_like(sentence)
  hk=sentence-hj
  hH=np.array(hO.We)
  L=hH[:,hk]
  sl=L.shape[1]
  hy=vu()
  for i in vr(0,sl):
   hy[i]=np.expand_dims(L[:,i],1)
  hA=vu()
  if(sl>1):
   for j in vr(0,sl-1):
    hL=W1.eval()
    hF=b1.eval()
    hn=U.eval()
    hB=bs.eval()
    hC=vq[:,j]
    hK=hC[0]-1 
    hq=hy[hK]
    hb=hC[1]-1
    hm=hy[hb] 
    hM=hC[2]-1
    hi=np.concatenate((hq,hm),axis=0)
    p=np.tanh(np.dot(hL,hi)+hF)
    hg=p/(np.linalg.norm(p,axis=0)*np.ones(p.shape))
    hy[hM]=hg
    y=np.tanh(np.dot(hn,hg)+hB)
    [y1,y2]=np.split(y,2,axis=0)
    hr=y1/(np.linalg.norm(y1,axis=0)*np.ones(y1.shape))
    hN=y2/(np.linalg.norm(y2,axis=0)*np.ones(y2.shape))
    hz=1 
    hG=1
    he,hR=hz*(hr-hq),hG*(hN-hm)
    constructionError=np.sum((he*(hr-hq)+hR*(hN-hm)),axis=0)*0.5 
    hA[j]=constructionError
    pass
   pass
  hx=0
  for(key,value)in hA.items():
   hx=hx+value
  hx=hx/(sl-1)
  return hx 
 def add_loss_fixed_tree(hO):
  with tf.variable_scope('Composition',reuse=vx):
   hO.W1=tf.get_variable("W1",dtype=tf.float64)
   hO.b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=vx):
   hO.U=tf.get_variable("U",dtype=tf.float64)
   hO.bs=tf.get_variable("bs",dtype=tf.float64)
  hu=tf.zeros(hO.batch_len,tf.float64)
  hu=tf.expand_dims(hu,0)
  hO.numlines_tensor3=tf.constant(1,dtype=tf.int32)
  hO.numcolunms_tensor3=hO.batch_len[0]
  i=tf.constant(0,dtype=tf.int32)
  hU=hO.batch_len[0]
  hY=lambda a,b,c:tf.less(a,hU)
  Eh=tf.constant(0)
  EJ=[i,hu,Eh]
  def _recurrence(i,hu,Eh):
   hO.sentence_embeddings=tf.gather(hO.To,i,axis=0)
   hO.sentence_length=hO.batch_real_sentence_length[i]
   hO.treeConstructionOrders=hO.batch_treeConstructionOrders[i]
   ET=2*hO.sentence_length-1
   hy=tf.zeros(ET,tf.float64)
   hy=tf.expand_dims(hy,0)
   hy=tf.tile(hy,(hO.config.embed_size,1))
   hO.numlines_tensor=tf.constant(hO.config.embed_size,dtype=tf.int32)
   hO.numcolunms_tensor=ET
   ii=tf.constant(0,dtype=tf.int32)
   EO=lambda a,b:tf.less(a,hO.sentence_length)
   Es=[ii,hy]
   def __recurrence(ii,hy):
    Eo=tf.expand_dims(hO.sentence_embeddings[:,ii],1)
    hy=hO.modify_one_column(hy,Eo,ii,hO.numlines_tensor,hO.numcolunms_tensor)
    ii=tf.add(ii,1)
    return ii,hy
   ii,hy=tf.while_loop(EO,__recurrence,Es,parallel_iterations=1)
   hA=tf.zeros(hO.sentence_length-1,tf.float64)
   hA=tf.expand_dims(hA,0)
   hO.numlines_tensor2=tf.constant(1,dtype=tf.int32)
   hO.numcolunms_tensor2=hO.sentence_length-1
   Ep=tf.constant(0,dtype=tf.int32)
   El=lambda a,b,c,d:tf.less(a,hO.sentence_length-1)
   ES=[Ep,hA,hy,Eh]
   def ____recurrence(Ep,hA,hy,Eh):
    hC=hO.treeConstructionOrders[:,Ep]
    hK=hC[0]-1 
    EX=hy[:,hK]
    hb=hC[1]-1
    EW=hy[:,hb] 
    hM=hC[2]-1
    Ew=tf.concat([EX,EW],axis=0)
    Ew=tf.expand_dims(Ew,1)
    Ec=tf.tanh(tf.add(tf.matmul(hO.W1,Ew),hO.b1))
    EQ=hO.normalization(Ec)
    hy=hO.modify_one_column(hy,EQ,hM,hO.numlines_tensor,hO.numcolunms_tensor)
    y=(tf.matmul(hO.U,EQ)+hO.bs)
    Et=y.shape[1].value
    (y1,y2)=hO.split_by_row(y,Et)
    hz=1 
    hG=1
    EX=tf.expand_dims(EX,1)
    EW=tf.expand_dims(EW,1)
    he=tf.subtract(y1,EX)
    hR=tf.subtract(y2,EW) 
    constructionError=hO.constructionError(he,hR,hz,hG)
    constructionError=tf.expand_dims(constructionError,1)
    hA=hO.modify_one_column(hA,constructionError,Ep,hO.numlines_tensor2,hO.numcolunms_tensor2)
    Ed=tf.Print(Ep,[Ep],"\niiii:")
    Eh=Ed+Eh
    Eh=Ed+Eh
    Ed=tf.Print(hK,[hK],"\nleftChild_index:",summarize=100)
    Eh=tf.to_int32(Ed)+Eh
    Ed=tf.Print(hb,[hb],"\nrightChild_index:",summarize=100)
    Eh=tf.to_int32(Ed)+Eh
    Ed=tf.Print(hM,[hM],"\nparent_index:",summarize=100)
    Eh=tf.to_int32(Ed)+Eh
    Ep=tf.add(Ep,1)
    return Ep,hA,hy,Eh
   Ep,hA,hy,Eh=tf.while_loop(El,____recurrence,ES,parallel_iterations=1)
   pass
   hO.node_tensors_cost_tensor=tf.identity(hA)
   hO.nodes_tensor=tf.identity(hy)
   EI=tf.reduce_mean(hO.node_tensors_cost_tensor)
   EI=tf.expand_dims(tf.expand_dims(EI,0),1)
   hu=hO.modify_one_column(hu,EI,i,hO.numlines_tensor3,hO.numcolunms_tensor3)
   i=tf.add(i,1)
   return i,hu,Eh
  i,hu,Eh=tf.while_loop(hY,_recurrence,EJ,parallel_iterations=10)
  hO.tfPrint=Eh
  with tf.name_scope('loss'):
   EP=tf.nn.l2_loss(hO.W1)+tf.nn.l2_loss(hO.U)
   hO.batch_constructionError=tf.reduce_mean(hu)
   hO.tensorLoss_fixed_tree=hO.batch_constructionError+EP*hO.config.l2
  return hO.tensorLoss_fixed_tree
 def add_loss_and_batchSentenceNodesVector_fixed_tree(hO):
  with tf.variable_scope('Composition',reuse=vx):
   hO.W1=tf.get_variable("W1",dtype=tf.float64)
   hO.b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=vx):
   hO.U=tf.get_variable("U",dtype=tf.float64)
   hO.bs=tf.get_variable("bs",dtype=tf.float64)
  hu=tf.zeros(hO.batch_len,tf.float64)
  hu=tf.expand_dims(hu,0)
  EV=hO.batch_real_sentence_length[tf.argmax(hO.batch_real_sentence_length)]
  ED=2*EV-1
  Ea=hO.batch_len[0]*hO.config.embed_size*ED
  Ef=tf.zeros(Ea,tf.float64)
  Ef=tf.reshape(Ef,[hO.batch_len[0],hO.config.embed_size,ED])
  hO.size_firstDimension=hO.batch_len[0]
  hO.size_secondDimension=tf.constant(hO.config.embed_size,dtype=tf.int32)
  hO.size_thirdDimension=ED
  hO.numlines_tensor3=tf.constant(1,dtype=tf.int32)
  hO.numcolunms_tensor3=hO.batch_len[0]
  hO.numlines_tensor4=tf.constant(hO.config.embed_size,dtype=tf.int32)
  hO.numcolunms_tensor4=hO.batch_len[0]
  i=tf.constant(0,dtype=tf.int32)
  hU=hO.batch_len[0]
  hY=lambda a,b,c,d:tf.less(a,hU)
  Eh=tf.constant(0)
  EJ=[i,hu,Eh,Ef]
  def _recurrence(i,hu,Eh,Ef):
   hO.sentence_embeddings=tf.gather(hO.To,i,axis=0)
   hO.sentence_length=hO.batch_real_sentence_length[i]
   hO.treeConstructionOrders=hO.batch_treeConstructionOrders[i]
   ET=2*hO.sentence_length-1
   hy=tf.zeros(ET,tf.float64)
   hy=tf.expand_dims(hy,0)
   hy=tf.tile(hy,(hO.config.embed_size,1))
   hO.numlines_tensor=tf.constant(hO.config.embed_size,dtype=tf.int32)
   hO.numcolunms_tensor=ET
   ii=tf.constant(0,dtype=tf.int32)
   EO=lambda a,b:tf.less(a,hO.sentence_length)
   Es=[ii,hy]
   def __recurrence(ii,hy):
    Eo=tf.expand_dims(hO.sentence_embeddings[:,ii],1)
    hy=hO.modify_one_column(hy,Eo,ii,hO.numlines_tensor,hO.numcolunms_tensor)
    ii=tf.add(ii,1)
    return ii,hy
   ii,hy=tf.while_loop(EO,__recurrence,Es,parallel_iterations=1)
   hA=tf.zeros(hO.sentence_length-1,tf.float64)
   hA=tf.expand_dims(hA,0)
   hO.numlines_tensor2=tf.constant(1,dtype=tf.int32)
   hO.numcolunms_tensor2=hO.sentence_length-1
   Ep=tf.constant(0,dtype=tf.int32)
   El=lambda a,b,c,d:tf.less(a,hO.sentence_length-1)
   ES=[Ep,hA,hy,Eh]
   def ____recurrence(Ep,hA,hy,Eh):
    hC=hO.treeConstructionOrders[:,Ep]
    hK=hC[0]-1 
    EX=hy[:,hK]
    hb=hC[1]-1
    EW=hy[:,hb] 
    hM=hC[2]-1
    Ew=tf.concat([EX,EW],axis=0)
    Ew=tf.expand_dims(Ew,1)
    Ec=tf.tanh(tf.add(tf.matmul(hO.W1,Ew),hO.b1))
    EQ=hO.normalization(Ec)
    hy=hO.modify_one_column(hy,EQ,hM,hO.numlines_tensor,hO.numcolunms_tensor)
    y=(tf.matmul(hO.U,EQ)+hO.bs)
    Et=y.shape[1].value
    (y1,y2)=hO.split_by_row(y,Et)
    hz=1 
    hG=1
    EX=tf.expand_dims(EX,1)
    EW=tf.expand_dims(EW,1)
    he=tf.subtract(y1,EX)
    hR=tf.subtract(y2,EW) 
    constructionError=hO.constructionError(he,hR,hz,hG)
    constructionError=tf.expand_dims(constructionError,1)
    hA=hO.modify_one_column(hA,constructionError,Ep,hO.numlines_tensor2,hO.numcolunms_tensor2)
    Ed=tf.Print(Ep,[Ep],"\niiii:")
    Eh=Ed+Eh
    Eh=Ed+Eh
    Ed=tf.Print(hK,[hK],"\nleftChild_index:",summarize=100)
    Eh=tf.to_int32(Ed)+Eh
    Ed=tf.Print(hb,[hb],"\nrightChild_index:",summarize=100)
    Eh=tf.to_int32(Ed)+Eh
    Ed=tf.Print(hM,[hM],"\nparent_index:",summarize=100)
    Eh=tf.to_int32(Ed)+Eh
    Ep=tf.add(Ep,1)
    return Ep,hA,hy,Eh
   Ep,hA,hy,Eh=tf.while_loop(El,____recurrence,ES,parallel_iterations=1)
   pass
   hO.node_tensors_cost_tensor=tf.identity(hA)
   hO.nodes_tensor=tf.identity(hy)
   EI=tf.reduce_mean(hO.node_tensors_cost_tensor)
   EI=tf.expand_dims(tf.expand_dims(EI,0),1)
   hu=hO.modify_one_column(hu,EI,i,hO.numlines_tensor3,hO.numcolunms_tensor3)
   Ej=hO.numlines_tensor 
   Ek=hO.numcolunms_tensor 
   EH=hO.size_secondDimension 
   Ey=hO.size_thirdDimension
   hy=hO.buqi_2DmatrixTensor(hy,Ej,Ek,EH,Ey)
   hy=tf.reshape(hy,[hO.config.embed_size,Ey])
   EA=i 
   EL=hO.size_firstDimension
   EF=hO.size_secondDimension
   En=hO.size_thirdDimension
   _,_,Ef=hO.modify_one_profile(Ef,hy,EA,EL,EF,En)
   i=tf.add(i,1)
   return i,hu,Eh,Ef
  i,hu,Eh,Ef=tf.while_loop(hY,_recurrence,EJ,parallel_iterations=10)
  hO.tfPrint=Eh
  hO.batch_sentence_vectors=tf.identity(Ef)
  with tf.name_scope('loss'):
   EP=tf.nn.l2_loss(hO.W1)+tf.nn.l2_loss(hO.U)
   hO.batch_constructionError=tf.reduce_mean(hu)
   hO.tensorLoss_fixed_tree=hO.batch_constructionError+EP*hO.config.l2
  return hO.tensorLoss_fixed_tree,hO.batch_sentence_vectors
 def add_placeholders_fixed_tree(hO):
  EB=hO.config.embed_size
  hO.To=tf.placeholder(tf.float64,[vU,EB,vU],name='input')
  hO.batch_treeConstructionOrders=tf.placeholder(tf.int32,[vU,3,vU],name='treeConstructionOrders')
  hO.batch_real_sentence_length=tf.placeholder(tf.int32,[vU],name='batch_real_sentence_length')
  hO.batch_len=tf.placeholder(tf.int32,shape=(1,),name='batch_len')
 def add_model_vars(hO):
  with tf.variable_scope('Composition'): 
   tf.get_variable("W1",dtype=tf.float64,shape=[hO.config.embed_size,2*hO.config.embed_size])
   tf.get_variable("b1",dtype=tf.float64,shape=[hO.config.embed_size,1])
  with tf.variable_scope('Projection'):
   tf.get_variable("U",dtype=tf.float64,shape=[2*hO.config.embed_size,hO.config.embed_size])
   tf.get_variable("bs",dtype=tf.float64,shape=[2*hO.config.embed_size,1])
 def normalization(hO,tensor):
  Eq=tensor.shape[0].value
  Eb=tf.pow(tensor,2)
  Em=tf.reduce_sum(Eb,0)
  EM=tf.expand_dims(Em,0)
  Ei=tf.tile(tf.sqrt(EM),(Eq,1))
  Eg=tf.divide(tensor,Ei)
  return Eg
 def split_by_row(hO,tensor,numcolunms):
  Eq=tensor.shape[0].value
  Er=tf.slice(tensor,[0,0],[(vY)(Eq/2),numcolunms])
  EN=tf.slice(tensor,[(vY)(Eq/2),0],[(vY)(Eq/2),numcolunms])
  pass
  return(Er,EN)
 def constructionError(hO,tensor1,tensor2,hz,hG):
  Ez=tf.multiply(tf.reduce_sum(tf.pow(tensor1,2),0),hz)
  EG=tf.multiply(tf.reduce_sum(tf.pow(tensor2,2),0),hG)
  Ee=tf.multiply(tf.add(Ez,EG),0.5)
  return Ee
 def training(hO,Jb):
  ER=vU
  Ex=tf.train.GradientDescentOptimizer(hO.config.lr)
  ER=Ex.minimize(Jb)
  return ER
 def __init__(hO,Eu,experimentID=vU):
  if(experimentID==vU):
   hO.config=Eu
   hO.load_data()
  elif(experimentID==1):
   hO.config=Eu
   hO.load_data_experimentID_1()
  elif(experimentID==2):
   hO.config=Eu
   hO.load_data_experimentID_2()
  elif(experimentID==3):
   hO.config=Eu
   hO.load_data_experimentID_3()
 def computelossAndVector_no_tensor_withAST(hO,sentence,vq):
  with tf.variable_scope('Composition',reuse=vx):
   W1=tf.get_variable("W1",dtype=tf.float64)
   b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=vx):
   U=tf.get_variable("U",dtype=tf.float64)
   bs=tf.get_variable("bs",dtype=tf.float64)
  hj=np.ones_like(sentence)
  hk=sentence-hj
  hH=np.array(hO.We)
  L=hH[:,hk]
  sl=L.shape[1]
  EU=vu()
  for i in vr(0,sl):
   EU[i]=np.expand_dims(L[:,i],1)
  hA=vu()
  EY=vU
  if(sl>1):
   for j in vr(0,sl-1):
    hL=W1.eval()
    hF=b1.eval()
    hn=U.eval()
    hB=bs.eval()
    hC=vq[:,j]
    hK=hC[0]-1 
    hq=EU[hK]
    hb=hC[1]-1
    hm=EU[hb] 
    hM=hC[2]-1
    hi=np.concatenate((hq,hm),axis=0)
    p=np.tanh(np.dot(hL,hi)+hF)
    hg=p/(np.linalg.norm(p,axis=0)*np.ones(p.shape))
    EU[hM]=hg
    EY=hg
    y=np.tanh(np.dot(hn,hg)+hB)
    [y1,y2]=np.split(y,2,axis=0)
    hr=y1/(np.linalg.norm(y1,axis=0)*np.ones(y1.shape))
    hN=y2/(np.linalg.norm(y2,axis=0)*np.ones(y2.shape))
    hz=1 
    hG=1
    he,hR=hz*(hr-hq),hG*(hN-hm)
    constructionError=np.sum((he*(hr-hq)+hR*(hN-hm)),axis=0)*0.5 
    hA[j]=constructionError
    pass
   Jh=EU[2*sl-2]
   pass
  hx=0
  for(key,value)in hA.items():
   hx=hx+value
  return(hx,EU,Jh,EY)
 def run_epoch_train(hO,Ju,Jm):
  JE=[]
  Jv=hO.trainCorpus
  JT=hO.trainCorpus_sentence_length
  JO=hO.train_corpus_fixed_tree_constructionorder
  Js=hO.config.max_sentence_length_train_Corpus
  Jo=vN(hO.trainCorpus)
  Jp=hO.config.MAX_SENTENCE_LENGTH_for_Bigclonebench
  hT.log('训练过程设置长短的衡量标准是{}，长的单独成batch，短的集合成batch'.format(Jp))
  Jl=[]
  JS=[]
  for hW,length in Th(JT):
   if length<Jp:
    Jl.append(hW)
   else:
    JS.append(hW)
  hT.log("训练集的句子{}个".format(Jo))
  hT.log("较长的句子{}个".format(vN(JS)))
  hT.log("较短的句子{}个".format(vN(Jl)))
  JX=[Jv[hW]for hW in Jl]
  JW=[JT[hW]for hW in Jl]
  Jw=[JO[hW]for hW in Jl]
  Jc=[Jv[hW]for hW in JS]
  JQ=[JT[hW]for hW in JS]
  Jt=[JO[hW]for hW in JS]
  hT.log("较短的句子，我们走批处理训练网络。长句子，我们单独计算训练网络")
  hT.log("先处理较短的句子的语料，批处理开始")
  Jd=vN(Jl)
  hH=np.array(hO.We)
  JI=hH.shape[0]
  hS=vg(vr(Jd))
  JP=hO.config.batch_size
  JV=0
  for JD in vr(0,Jd,JP):
   Ja=TE(JD+JP,Jd)-JD
   Jf=hS[JD:JD+Ja]
   Jj=[JX[hW]for hW in Jf]
   Jk=[JW[hW]for hW in Jf]
   JH=vR(Jk)
   x=[]
   for i,sentence in Th(Jj):
    Jy=Jk[i]
    hj=np.ones_like(sentence)
    hk=sentence-hj
    L1=hH[:,hk]
    JA=JH-Jy
    L2=np.zeros([JI,JA],np.float64)
    L=np.concatenate((L1,L2),axis=1)
    x.append(L)
   JL=np.array(x)
   JF=np.array(JL,np.float64)
   x=[]
   Jn=[Jw[hW]for hW in Jf]
   for i,sentence_fixed_tree_constructionorder in Th(Jn):
    JB=Jk[i]-1
    JA=(JH-1)-JB
    L2=np.zeros([3,JA],np.int32)
    L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
    x.append(L)
   JC=np.array(x)
   JK={hO.To:JF,hO.batch_real_sentence_length:Jk,hO.batch_len:[Ja],hO.batch_treeConstructionOrders:JC}
   Jq=tf.RunOptions(report_tensor_allocations_upon_oom=vx)
   Jb,_=Ju.run([hO.tensorLoss_fixed_tree,hO.train_op],feed_dict=JK,options=Jq)
   JE.append(Jb)
   hT.log('\repoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(Jm,JV,Ja,Jb)) 
   hT.log('\repoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(Jm,vN(JE),np.mean(JE))) 
   JV=JV+1
   pass 
  hT.log("保存模型到temp")
  JM=tf.train.Saver()
  if not os.path.exists("./weights"):
   os.makedirs("./weights")
  JM.save(Ju,'./weights/%s.temp'%hO.config.model_name)
  return JE 
 def run_epoch_evaluation(hO,Ju,Jm):
  JE=[]
  Jv=hO.evalutionCorpus
  JT=hO.evalution_corpus_sentence_length
  JO=hO.evalution_corpus_fixed_tree_constructionorder
  Js=hO.config.max_sentence_length_evalution_Corpus
  Ji=vN(hO.evalutionCorpus)
  Jp=1000; 
  hT.log('训练过程设置长短的衡量标准是{}，长的单独成batch，短的集合成batch'.format(Jp))
  Jl=[]
  JS=[]
  for hW,length in Th(JT):
   if length<Jp:
    Jl.append(hW)
   else:
    JS.append(hW)
  hT.log("训练集的句子{}个".format(Jo))
  hT.log("较长的句子{}个".format(vN(JS)))
  hT.log("较短的句子{}个".format(vN(Jl)))
  JX=[Jv[hW]for hW in Jl]
  JW=[JT[hW]for hW in Jl]
  Jw=[JO[hW]for hW in Jl]
  Jc=[Jv[hW]for hW in JS]
  JQ=[JT[hW]for hW in JS]
  Jt=[JO[hW]for hW in JS]
  hT.log("较短的句子，我们走批处理训练网络。长句子，我们单独计算训练网络")
  hT.log("先处理较短的句子的语料，批处理开始")
  Jd=vN(Jl)
  hH=np.array(hO.We)
  JI=hH.shape[0]
  hS=vg(vr(Jd))
  JP=hO.config.batch_size_using_model_notTrain 
  JV=0
  for JD in vr(0,Jd,JP):
   Ja=TE(JD+JP,Jd)-JD
   Jf=hS[JD:JD+Ja]
   Jj=[JX[hW]for hW in Jf]
   Jk=[JW[hW]for hW in Jf]
   JH=vR(Jk)
   x=[]
   for i,sentence in Th(Jj):
    Jy=Jk[i]
    hj=np.ones_like(sentence)
    hk=sentence-hj
    L1=hH[:,hk]
    JA=JH-Jy
    L2=np.zeros([JI,JA],np.float64)
    L=np.concatenate((L1,L2),axis=1)
    x.append(L)
   JL=np.array(x)
   JF=np.array(JL,np.float64)
   x=[]
   Jn=[Jw[hW]for hW in Jf]
   for i,sentence_fixed_tree_constructionorder in Th(Jn):
    JB=Jk[i]-1
    JA=(JH-1)-JB
    L2=np.zeros([3,JA],np.int32)
    L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
    x.append(L)
   JC=np.array(x)
   JK={hO.To:JF,hO.batch_real_sentence_length:Jk,hO.batch_len:[Ja],hO.batch_treeConstructionOrders:JC}
   Jb=Ju.run([hO.tensorLoss_fixed_tree],feed_dict=JK)
   JE.append(Jb)
   hT.log('\r验证过程epoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(Jm,JV,Ja,Jb)) 
   hT.log('\r验证过程epoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(Jm,vN(JE),np.mean(JE))) 
   JV=JV+1
   pass 
  hT.log("再处理较长的句子的语料，每个句子单独处理，开始") 
  Jg=vN(JS)
  hS=vg(vr(Jg))
  for JD,sentence in Th(Jc):
   Ja=1
   Jf=hS[JD:JD+Ja]
   Jj=[Jc[hW]for hW in Jf]
   Jk=[JQ[hW]for hW in Jf]
   JH=vR(Jk)
   x=[]
   for i,sentence in Th(Jj):
    Jy=Jk[i]
    hj=np.ones_like(sentence)
    hk=sentence-hj
    L1=hH[:,hk]
    JA=JH-Jy
    L2=np.zeros([JI,JA],np.float64)
    L=np.concatenate((L1,L2),axis=1)
    x.append(L)
   JL=np.array(x)
   JF=np.array(JL,np.float64)
   x=[]
   Jn=[Jt[hW]for hW in Jf]
   for i,sentence_fixed_tree_constructionorder in Th(Jn):
    JB=Jk[i]-1
    JA=(JH-1)-JB
    L2=np.zeros([3,JA],np.int32)
    L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
    x.append(L)
   JC=np.array(x)
   JK={hO.To:JF,hO.batch_real_sentence_length:Jk,hO.batch_len:[Ja],hO.batch_treeConstructionOrders:JC}
   Jb=Ju.run([hO.tensorLoss_fixed_tree],feed_dict=JK)
   JE.append(Jb)
   hT.log('\r验证过程epoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(Jm,JV,Ja,Jb)) 
   hT.log('\r验证过程epoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(Jm,vN(JE),np.mean(JE))) 
   JV=JV+1
   pass
  return JE 
 def train(hO,restore=TJ):
  with tf.Graph().as_default():
   hO.xiaojie_RvNN_fixed_tree()
   Jr=tf.initialize_all_variables()
   JM=tf.train.Saver()
   JN=[]
   Jz=[]
   JG=Tv('inf')
   Je=Tv('inf')
   JR=0 
   Jx=-1
   Jm=0
   Eu=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=vx),allow_soft_placement=vx)
   Eu.gpu_options.per_process_gpu_memory_fraction=0.95
   with tf.Session(config=Eu)as Ju:
    Ju.run(Jr)
    JU=time.time()
    if restore:JM.restore(Ju,'./weights/%s'%hO.config.model_name)
    while Jm<hO.config.max_epochs:
     hT.log('epoch %d'%Jm)
     JE=hO.run_epoch_train(Ju,Jm)
     JN.extend(JE)
     JY=hO.run_epoch_evaluation(Ju,Jm)
     vh=np.mean(JY)
     Jz.append(vh)
     hT.log("time per epoch is {} s".format(time.time()-JU))
     vE=vh
     if vE>JG*hO.config.anneal_threshold:
      hO.config.lr/=hO.config.anneal_by
      hT.log('annealed lr to %f'%hO.config.lr)
     JG=vE 
     if vh<Je:
      shutil.copyfile('./weights/%s.temp.data-00000-of-00001'%hO.config.model_name,'./weights/%s.data-00000-of-00001'%hO.config.model_name)
      shutil.copyfile('./weights/%s.temp.index'%hO.config.model_name,'./weights/%s.index'%hO.config.model_name)
      shutil.copyfile('./weights/%s.temp.meta'%hO.config.model_name,'./weights/%s.meta'%hO.config.model_name)
      Je=vh
      JR=Jm
     elif Jm-JR>=hO.config.early_stopping:
      Jx=Jm
      break
     Jm+=1
     JU=time.time()
     pass
    if(Jm<(hO.config.max_epochs-1)):
     hT.log('预定训练{}个epoch,一共训练{}个epoch，在评估集上最优的是第{}个epoch(从0开始计数),最优评估loss是{}'.format(hO.config.max_epochs,Jx+1,JR,Je))
    elif(Jm==(hO.config.max_epochs-1)):
     hT.log('预定训练{}个epoch,全部轮数达成，在评估集上最优的是第{}个epoch,最优评估loss是{}'.format(hO.config.max_epochs,JR,Je))
    else:
     hT.log('预定训练{}个epoch,全部轮数达成，在评估集上最优的是第{}个epoch,最优评估loss是{}'.format(hO.config.max_epochs,JR,Je))
   return{'complete_loss_history':JN,'evalution_loss_history':Jz,}
 def using_model_for_BigCloneBench_experimentID_1(hO):
  hT.log("------------------------------\n读取BigCloneBench的所有ID编号")
  hw='./SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
  vJ=[]
  with vz(hw,'rb')as f:
   hQ=pickle.load(f)
   for vT in hQ.keys():
    ht=hQ[vT]
    if(ht==-1):
     continue 
    vJ.append(vT)
  vO=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
  vs='./vector/'+vO+'_BigCloneBench_traditionalRAE_ID_Map_Vector_root.xiaojiepkl'
  if os.path.exists(vs): 
   os.remove(vs) 
  else:
   TO('no such file:%s'%vs)
  vO=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
  vo='./vector/'+vO+'_BigCloneBench_traditionalRAE_ID_Map_Vector_mean.xiaojiepkl'
  if os.path.exists(vo): 
   os.remove(vo) 
  else:
   TO('no such file:%s'%vo)
  Jv=hO.bigCloneBench_Corpus
  JO=hO.bigCloneBench_Corpus_fixed_tree_constructionorder
  JT=hO.bigCloneBench_Corpus_sentence_length
  vp={}
  vl={}
  del(hO.trainCorpus)
  del(hO.trainCorpus_sentence_length)
  del(hO.train_corpus_fixed_tree_constructionorder)
  del(hO.vocabulary)
  with tf.Graph().as_default():
   hO.xiaojie_RvNN_fixed_tree_for_usingmodel() 
   JM=tf.train.Saver()
   Eu=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=vx))
   Eu.gpu_options.allocator_type='BFC'
   with tf.Session(config=Eu)as Ju:
    vS='./weights/%s'%hO.config.model_name
    JM.restore(Ju,vS)
    Jp=300; 
    hT.log('设置长短的衡量标准是{}'.format(Jp))
    Jl=[]
    JS=[]
    for hW,length in Th(JT):
     if length<Jp:
      Jl.append(hW)
     else:
      JS.append(hW)
    hT.log("较长的句子{}个".format(vN(JS)))
    hT.log("较短的句子{}个".format(vN(Jl)))
    JX=[Jv[hW]for hW in Jl]
    JW=[JT[hW]for hW in Jl]
    Jw=[JO[hW]for hW in Jl]
    Jc=[Jv[hW]for hW in JS]
    JQ=[JT[hW]for hW in JS]
    Jt=[JO[hW]for hW in JS]
    hT.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
    hT.log("先处理较短的句子的语料，批处理开始")
    hl=vN(Jl)
    hH=np.array(hO.We)
    del(hO.We)
    JI=hH.shape[0]
    hS=vg(vr(hl))
    JP=hO.config.batch_size_using_model_notTrain
    vX=(hl-1)/JP 
    JV=0
    for JD in vr(0,hl,JP):
     hT.log("batch_index:{}/{}".format(JV,vX))
     Ja=TE(JD+JP,hl)-JD
     Jf=hS[JD:JD+Ja]
     Jj=[JX[hW]for hW in Jf]
     Jk=[JW[hW]for hW in Jf]
     JH=vR(Jk)
     x=[]
     for i,sentence in Th(Jj):
      Jy=Jk[i]
      hj=np.ones_like(sentence)
      hk=sentence-hj
      L1=hH[:,hk]
      JA=JH-Jy
      L2=np.zeros([JI,JA],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     JL=np.array(x)
     JF=np.array(JL,np.float64)
     del(JL)
     x=[]
     Jn=[Jw[hW]for hW in Jf]
     for i,sentence_fixed_tree_constructionorder in Th(Jn):
      JB=Jk[i]-1
      JA=(JH-1)-JB
      L2=np.zeros([3,JA],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     JC=np.array(x)
     JK={hO.To:JF,hO.batch_real_sentence_length:Jk,hO.batch_len:[Ja],hO.batch_treeConstructionOrders:JC}
     Jb,Ef=Ju.run([hO.tensorLoss_fixed_tree,hO.batch_sentence_vectors],feed_dict=JK)
     if JV==0:
      TO(Jb)
     vW=0
     for hW in Jf:
      vw=Ef[vW,:,:]
      ho=Jk[vW]
      vc=2*ho-1
      vQ=vw[0:hO.config.embed_size,0:vc]
      vQ=vQ.astype(np.float32)
      vW=vW+1
      vQ=np.transpose(vQ)
      vt=vg(vQ)
      vd=Jl[hW]
      vp[vd]=vt[vc-1]
     JV=JV+1
    hT.log("再处理较长的句子的语料，每个句子单独处理，开始") 
    Jg=vN(JS)
    hS=vg(vr(Jg))
    for JD,sentence in Th(Jc):
     hT.log("long_setence_index:{}/{}".format(JD,Jg))
     Ja=1
     Jf=hS[JD:JD+Ja]
     Jj=[Jc[hW]for hW in Jf]
     Jk=[JQ[hW]for hW in Jf]
     JH=vR(Jk)
     x=[]
     for i,sentence in Th(Jj):
      Jy=Jk[i]
      hj=np.ones_like(sentence)
      hk=sentence-hj
      L1=hH[:,hk]
      JA=JH-Jy
      L2=np.zeros([JI,JA],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     JL=np.array(x)
     JF=np.array(JL,np.float64)
     x=[]
     Jn=[Jt[hW]for hW in Jf]
     for i,sentence_fixed_tree_constructionorder in Th(Jn):
      JB=Jk[i]-1
      JA=(JH-1)-JB
      L2=np.zeros([3,JA],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     JC=np.array(x)
     JK={hO.To:JF,hO.batch_real_sentence_length:Jk,hO.batch_len:[Ja],hO.batch_treeConstructionOrders:JC}
     Jb,Ef=Ju.run([hO.tensorLoss_fixed_tree,hO.batch_sentence_vectors],feed_dict=JK)
     vQ=Ef[0,:,:]
     ho=Jk[0]
     vc=2*ho-1
     vQ=vQ.astype(np.float32)
     vQ=np.transpose(vQ)
     vt=vg(vQ)
     vd=JS[JD]
     vp[vd]=vt[vc-1]
     pass
  vI={}
  vP={}
  vV={}
  for i,ht in Th(hO.lines_for_bigcloneBench):
   vV[ht]=i
  for vD in vJ:
   ht=hQ[vD]
   va=vV[ht]
   vf=vp[va]
   vI[vD]=vf
  hO.save_to_pkl(vI,vs)
  TO(vs)
  pass
  return 
 def 
 (hO):
  vO=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
  vj='./vectorTree/'+vO+'_vectorTree.xiaojiepkl'
  if os.path.exists(vj): 
   os.remove(vs) 
  else:
   TO('no such file:%s'%vj)
  Jv=hO.need_vectorTree_Corpus
  JO=hO.need_vectorTree_Corpus_fixed_tree_constructionorder
  JT=hO.need_vectorTree_Corpus_sentence_length
  vk={}
  with tf.Graph().as_default():
   hO.xiaojie_RvNN_fixed_tree_for_usingmodel() 
   JM=tf.train.Saver()
   Eu=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=vx))
   Eu.gpu_options.allocator_type='BFC'
   with tf.Session(config=Eu)as Ju:
    vS='./weights/%s'%hO.config.model_name
    JM.restore(Ju,vS)
    Jp=500; 
    hT.log('设置长短的衡量标准是{}'.format(Jp))
    Jl=[]
    JS=[]
    for hW,length in Th(JT):
     if length<Jp:
      Jl.append(hW)
     else:
      JS.append(hW)
    hT.log("较长的句子{}个".format(vN(JS)))
    hT.log("较短的句子{}个".format(vN(Jl)))
    JX=[Jv[hW]for hW in Jl]
    JW=[JT[hW]for hW in Jl]
    Jw=[JO[hW]for hW in Jl]
    Jc=[Jv[hW]for hW in JS]
    JQ=[JT[hW]for hW in JS]
    Jt=[JO[hW]for hW in JS]
    hT.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
    hT.log("先处理较短的句子的语料，批处理开始")
    hl=vN(Jl)
    hH=np.array(hO.We)
    JI=hH.shape[0]
    hS=vg(vr(hl))
    JP=hO.config.batch_size_using_model_notTrain
    vX=(hl-1)/JP 
    JV=0
    for JD in vr(0,hl,JP):
     hT.log("batch_index:{}/{}".format(JV,vX))
     Ja=TE(JD+JP,hl)-JD
     Jf=hS[JD:JD+Ja]
     Jj=[JX[hW]for hW in Jf]
     Jk=[JW[hW]for hW in Jf]
     JH=vR(Jk)
     x=[]
     for i,sentence in Th(Jj):
      Jy=Jk[i]
      hj=np.ones_like(sentence)
      hk=sentence-hj
      L1=hH[:,hk]
      JA=JH-Jy
      L2=np.zeros([JI,JA],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     JL=np.array(x)
     JF=np.array(JL,np.float64)
     x=[]
     Jn=[Jw[hW]for hW in Jf]
     for i,sentence_fixed_tree_constructionorder in Th(Jn):
      JB=Jk[i]-1
      JA=(JH-1)-JB
      L2=np.zeros([3,JA],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     JC=np.array(x)
     JK={hO.To:JF,hO.batch_real_sentence_length:Jk,hO.batch_len:[Ja],hO.batch_treeConstructionOrders:JC}
     Jb,Ef=Ju.run([hO.tensorLoss_fixed_tree,hO.batch_sentence_vectors],feed_dict=JK)
     vW=0
     for hW in Jf:
      vw=Ef[vW,:,:]
      ho=Jk[vW]
      vc=2*ho-1
      vQ=vw[0:hO.config.embed_size,0:vc]
      vQ=vQ.astype(np.float32)
      vW=vW+1
      vQ=np.transpose(vQ)
      vt=vg(vQ)
      vd=Jl[hW]
      vk[vd]=vt
     JV=JV+1
    hT.log("再处理较长的句子的语料，每个句子单独处理，开始") 
    Jg=vN(JS)
    hS=vg(vr(Jg))
    for JD,sentence in Th(Jc):
     hT.log("long_setence_index:{}/{}".format(JD,Jg))
     Ja=1
     Jf=hS[JD:JD+Ja]
     Jj=[Jc[hW]for hW in Jf]
     Jk=[JQ[hW]for hW in Jf]
     JH=vR(Jk)
     x=[]
     for i,sentence in Th(Jj):
      Jy=Jk[i]
      hj=np.ones_like(sentence)
      hk=sentence-hj
      L1=hH[:,hk]
      JA=JH-Jy
      L2=np.zeros([JI,JA],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     JL=np.array(x)
     JF=np.array(JL,np.float64)
     x=[]
     Jn=[Jt[hW]for hW in Jf]
     for i,sentence_fixed_tree_constructionorder in Th(Jn):
      JB=Jk[i]-1
      JA=(JH-1)-JB
      L2=np.zeros([3,JA],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     JC=np.array(x)
     JK={hO.To:JF,hO.batch_real_sentence_length:Jk,hO.batch_len:[Ja],hO.batch_treeConstructionOrders:JC}
     Jb,Ef=Ju.run([hO.tensorLoss_fixed_tree,hO.batch_sentence_vectors],feed_dict=JK)
     vQ=Ef[0,:,:]
     ho=Jk[0]
     vc=2*ho-1
     vQ=vQ.astype(np.float32)
     vQ=np.transpose(vQ)
     vt=vg(vQ)
     vd=JS[JD]
     vk[vd]=vt
  vH={}
  vy={}
  for i,ht in Th(hO.need_vectorTree_lines_for_trainCorpus):
   vy[ht]=i
  for vD in hO.need_vectorTree_ids_for_trainCorpus:
   ht=hO.id_line_dict[vD]
   vA=vy[ht]
   vL=vk[vA]
   vH[vD]=vL
  hO.save_to_pkl(vH,vj)
  TO(vj)
  pass
  return 
 def using_model_for_BigCloneBench_experimentID_3(hO):
  vF=0 
  vs='./vector/'+Ts(vF)+'_fullCorpusLine_Map_Vector_root.xiaojiepkl'
  if os.path.exists(vs): 
   os.remove(vs) 
  else:
   TO('no such file:%s'%vs)
  vo='./vector/'+Ts(vF)+'_fullCorpusLine_Map_Vector_mean.xiaojiepkl'
  if os.path.exists(vo): 
   os.remove(vo) 
  else:
   TO('no such file:%s'%vo)
  vn=0
  Jv=hO.fullCorpus
  JO=hO.full_corpus_fixed_tree_constructionorder
  JT=hO.fullCorpus_sentence_length
  vp={}
  vl={}
  del(hO.fullCorpus)
  del(hO.fullCorpus_sentence_length)
  del(hO.full_corpus_fixed_tree_constructionorder)
  with tf.Graph().as_default():
   hO.xiaojie_RvNN_fixed_tree_for_usingmodel() 
   JM=tf.train.Saver()
   Eu=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=vx))
   Eu.gpu_options.allocator_type='BFC'
   with tf.Session(config=Eu)as Ju:
    vS='./weights/%s'%hO.config.model_name
    JM.restore(Ju,vS)
    Jp=300; 
    hT.log('设置长短的衡量标准是{}'.format(Jp))
    Jl=[]
    JS=[]
    for hW,length in Th(JT):
     if length<Jp:
      Jl.append(hW)
     else:
      JS.append(hW)
    hT.log("较长的句子{}个".format(vN(JS)))
    hT.log("较短的句子{}个".format(vN(Jl)))
    JX=[Jv[hW]for hW in Jl]
    JW=[JT[hW]for hW in Jl]
    Jw=[JO[hW]for hW in Jl]
    Jc=[Jv[hW]for hW in JS]
    JQ=[JT[hW]for hW in JS]
    Jt=[JO[hW]for hW in JS]
    hT.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
    hT.log("先处理较短的句子的语料，批处理开始")
    hl=vN(Jl)
    hH=np.array(hO.We)
    del(hO.We)
    JI=hH.shape[0]
    hS=vg(vr(hl))
    JP=hO.config.batch_size_using_model_notTrain
    vX=(hl-1)/JP 
    JV=0
    for JD in vr(0,hl,JP):
     hT.log("batch_index:{}/{}".format(JV,vX))
     Ja=TE(JD+JP,hl)-JD
     Jf=hS[JD:JD+Ja]
     Jj=[JX[hW]for hW in Jf]
     Jk=[JW[hW]for hW in Jf]
     JH=vR(Jk)
     x=[]
     for i,sentence in Th(Jj):
      Jy=Jk[i]
      hj=np.ones_like(sentence)
      hk=sentence-hj
      L1=hH[:,hk]
      JA=JH-Jy
      L2=np.zeros([JI,JA],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     JL=np.array(x)
     JF=np.array(JL,np.float64)
     x=[]
     Jn=[Jw[hW]for hW in Jf]
     for i,sentence_fixed_tree_constructionorder in Th(Jn):
      JB=Jk[i]-1
      JA=(JH-1)-JB
      L2=np.zeros([3,JA],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     JC=np.array(x)
     JK={hO.To:JF,hO.batch_real_sentence_length:Jk,hO.batch_len:[Ja],hO.batch_treeConstructionOrders:JC}
     Jb,Ef=Ju.run([hO.tensorLoss_fixed_tree,hO.batch_sentence_vectors],feed_dict=JK)
     vW=0
     for hW in Jf:
      vw=Ef[vW,:,:]
      ho=Jk[vW]
      vc=2*ho-1
      vQ=vw[0:hO.config.embed_size,0:vc]
      vQ=vQ.astype(np.float32)
      vW=vW+1
      vQ=np.transpose(vQ)
      vt=vg(vQ)
      vd=Jl[hW]
      vp[vd]=vt[vc-1]
      vB=np.mean(vQ,axis=0)
      vl[vd]=vB
     if(vn>70000):
      hO.save_to_pkl(vp,vs)
      hO.save_to_pkl(vl,vo)
      vn=0
      del(vp)
      del(vl)
      vp={}
      vl={}
      vF+=1 
      vs='./vector/'+Ts(vF)+'_fullCorpusLine_Map_Vector_root.xiaojiepkl'
      if os.path.exists(vs): 
       os.remove(vs) 
      else:
       TO('no such file:%s'%vs)
      vo='./vector/'+Ts(vF)+'_fullCorpusLine_Map_Vector_mean.xiaojiepkl'
      if os.path.exists(vo): 
       os.remove(vo) 
      else:
       TO('no such file:%s'%vo)
     JV=JV+1
     vn=vn+Ja
    hO.save_to_pkl(vp,vs)
    hO.save_to_pkl(vl,vo)
    del(vp)
    del(vl)
    vp={}
    vl={}
    vF+=1 
    vs='./vector/'+Ts(vF)+'_fullCorpusLine_Map_Vector_root.xiaojiepkl'
    if os.path.exists(vs): 
     os.remove(vs) 
    else:
     TO('no such file:%s'%vs)
    vo='./vector/'+Ts(vF)+'_fullCorpusLine_Map_Vector_mean.xiaojiepkl'
    if os.path.exists(vo): 
     os.remove(vo) 
    else:
     TO('no such file:%s'%vo)
    hT.log("再处理较长的句子的语料，每个句子单独处理，开始") 
    Jg=vN(JS)
    hS=vg(vr(Jg))
    for JD,sentence in Th(Jc):
     hT.log("long_setence_index:{}/{}".format(JD,Jg))
     Ja=1
     Jf=hS[JD:JD+Ja]
     Jj=[Jc[hW]for hW in Jf]
     Jk=[JQ[hW]for hW in Jf]
     JH=vR(Jk)
     x=[]
     for i,sentence in Th(Jj):
      Jy=Jk[i]
      hj=np.ones_like(sentence)
      hk=sentence-hj
      L1=hH[:,hk]
      JA=JH-Jy
      L2=np.zeros([JI,JA],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     JL=np.array(x)
     JF=np.array(JL,np.float64)
     x=[]
     Jn=[Jt[hW]for hW in Jf]
     for i,sentence_fixed_tree_constructionorder in Th(Jn):
      JB=Jk[i]-1
      JA=(JH-1)-JB
      L2=np.zeros([3,JA],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     JC=np.array(x)
     JK={hO.To:JF,hO.batch_real_sentence_length:Jk,hO.batch_len:[Ja],hO.batch_treeConstructionOrders:JC}
     Jb,Ef=Ju.run([hO.tensorLoss_fixed_tree,hO.batch_sentence_vectors],feed_dict=JK)
     vQ=Ef[0,:,:]
     ho=Jk[0]
     vc=2*ho-1
     vQ=vQ.astype(np.float32)
     vQ=np.transpose(vQ)
     vt=vg(vQ)
     vd=JS[JD]
     vp[vd]=vt[vc-1]
     vB=np.mean(vQ,axis=0)
     vl[vd]=vB 
  hO.save_to_pkl(vp,vs)
  hO.save_to_pkl(vl,vo)
  TO(vs)
  TO(vo)
  del(vp)
  del(vl)
  pass
  return 
 def save_to_pkl(hO,vC,pickle_name):
  with vz(pickle_name,'wb')as pickle_f:
   pickle.dump(vC,pickle_f)
 def read_from_pkl(hO,pickle_name):
  with vz(pickle_name,'rb')as pickle_f:
   vC=pickle.load(pickle_f)
  return vC 
 def similarities(hO,Jv,JT,JO,vS):
  hT.log('对语料库计算句与句的相似性') 
  hT.log('被相似计算的语料库一共{}个sentence'.format(vN(Jv)))
  vO=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
  vK=vO+'.xiaojiepkl'
  if os.path.exists(vK): 
   os.remove(vK) 
  else:
   TO('no such file:%s'%vK)
  vk={}
  with tf.Graph().as_default():
   hO.xiaojie_RvNN_fixed_tree_for_usingmodel()
   JM=tf.train.Saver()
   Eu=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=vx))
   Eu.gpu_options.allocator_type='BFC'
   with tf.Session(config=Eu)as Ju:
    JM.restore(Ju,vS)
    Jp=1000; 
    hT.log('设置长短的衡量标准是{}'.format(Jp))
    Jl=[]
    JS=[]
    for hW,length in Th(JT):
     if length<Jp:
      Jl.append(hW)
     else:
      JS.append(hW)
    hT.log("较长的句子{}个".format(vN(JS)))
    hT.log("较短的句子{}个".format(vN(Jl)))
    JX=[Jv[hW]for hW in Jl]
    JW=[JT[hW]for hW in Jl]
    Jw=[JO[hW]for hW in Jl]
    Jc=[Jv[hW]for hW in JS]
    JQ=[JT[hW]for hW in JS]
    Jt=[JO[hW]for hW in JS]
    hT.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
    hT.log("先处理较短的句子的语料，批处理开始")
    hl=vN(Jl)
    hH=np.array(hO.We)
    JI=hH.shape[0]
    hS=vg(vr(hl))
    JP=hO.config.batch_size_using_model_notTrain
    vX=(hl-1)/JP 
    JV=0
    for i in vr(0,hl,JP):
     hT.log("batch_index:{}/{}".format(JV,vX))
     Ja=TE(i+JP,hl)-i
     Jf=hS[i:i+Ja]
     Jj=[JX[hW]for hW in Jf]
     Jk=[JW[hW]for hW in Jf]
     JH=vR(Jk)
     x=[]
     for i,sentence in Th(Jj):
      Jy=Jk[i]
      hj=np.ones_like(sentence)
      hk=sentence-hj
      L1=hH[:,hk]
      JA=JH-Jy
      L2=np.zeros([JI,JA],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     JL=np.array(x)
     JF=np.array(JL,np.float64)
     x=[]
     Jn=[Jw[hW]for hW in Jf]
     for i,sentence_fixed_tree_constructionorder in Th(Jn):
      JB=Jk[i]-1
      JA=(JH-1)-JB
      L2=np.zeros([3,JA],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     JC=np.array(x)
     JK={hO.To:JF,hO.batch_real_sentence_length:Jk,hO.batch_len:[Ja],hO.batch_treeConstructionOrders:JC}
     Jb,Ef=Ju.run([hO.tensorLoss_fixed_tree,hO.batch_sentence_vectors],feed_dict=JK)
     vW=0
     for hW in Jf:
      vw=Ef[vW,:,:]
      ho=Jk[vW]
      vc=2*ho-1
      vQ=vw[0:hO.config.embed_size,0:vc]
      vQ=vQ.astype(np.float32)
      vW=vW+1
      vQ=np.transpose(vQ)
      vt=vg(vQ)
      vd=Jl[hW]
      vk[vd]=vt 
     JV=JV+1
    hT.log("再处理较长的句子的语料，每个句子单独处理，开始") 
    Jg=vN(JS)
    for hW,sentence in Th(Jc):
     hT.log("long_setence_index:{}/{}".format(hW,Jg))
     vq=Jt[hW]
     (_,vQ,Jh,_)=hO.computelossAndVector_no_tensor_withAST(sentence,vq)
     vt=[]
     for kk in vr(2*JQ[hW]-1):
      vb=vQ[kk]
      vb=vb[:,0]
      vb=vb.astype(np.float32)
      vt.append(vb)
     vd=JS[hW]
     vk[vd]=vt 
    with vz(vK,'wb')as f:
     pickle.dump(vk,f)
    hT.log('相似性计算结束后，corpus的所有句子的最终向量表示(向量树)存储的位置是为%s'%vK)
def test_RNN():
 To("开始？")
 hT.log("------------------------------\n程序开始")
 Eu=hE()
 vm=hv(Eu)
 vM='./weights/%s'%vm.config.model_name
 vm.similarities(corpus=vm.fullCorpus,corpus_sentence_length=vm.fullCorpus_sentence_length,weights_path=vM,corpus_fixed_tree_constructionorder=vm.full_corpus_fixed_tree_constructionorder)
 hT.log("程序结束\n------------------------------")
from train_traditional_RAE_configuration import configuration 
def xiaojie_RNN_1():
 hT.log("------------------------------\n为模型加载训练样本集合，并配置参数")
 hs=configuration
 TO(hs) 
 Eu=hE(hs)
 vm=hv(Eu,experimentID=1)
 vm.using_model_for_BigCloneBench_experimentID_1()
def save_to_pkl(vC,pickle_name):
 with vz(pickle_name,'wb')as pickle_f:
  pickle.dump(vC,pickle_f)
def read_from_pkl(pickle_name):
 with vz(pickle_name,'rb')as pickle_f:
  vC=pickle.load(pickle_f)
 return vC 
if __name__=="__main__":
 xiaojie_RNN_1()
# Created by pyminifier (https://github.com/liftoff/pyminifier)
