#!/usr/bin/python
Bl=object
BU=list
BI=range
BY=len
BS=open
Bg=id
Fw=map
FP=max
Fk=True
FO=dict
FB=None
FG=int
FQ=enumerate
Fu=min
FN=False
Fc=float
FC=print
FA=sum
Fs=str
Fr=input
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
global wB
wB=xiaojie_log_class()
class wP(Bl):
 def __init__(wF,wG):
  wB.log("(1)>>>>  before training RvNN,配置超参数")
  wF.label_size=wG['label_size']
  wF.early_stopping=wG['early_stopping']
  wF.max_epochs=wG['max_epochs']
  wF.anneal_threshold=wG['anneal_threshold']
  wF.anneal_by=wG['anneal_by']
  wF.lr=wG['lr']
  wF.l2=wG['l2']
  wF.embed_size=wG['embed_size']
  wF.model_name=wG['model_name']
  wB.log('模型名称为%s'%wF.model_name)
  wF.IDIR=wG['IDIR']
  wF.ODIR=wG['ODIR']
  wF.corpus_fixed_tree_constructionorder_file=wG['corpus_fixed_tree_constructionorder_file']
  wF.MAX_SENTENCE_LENGTH=100000
  wF.batch_size=wG['batch_size']
  wF.batch_size_using_model_notTrain=wG['batch_size_using_model_notTrain']
  wF.MAX_SENTENCE_LENGTH_for_Bigclonebench=wG['MAX_SENTENCE_LENGTH_for_Bigclonebench']
  wF.corpus_fixed_tree_construction_parentType_weight_file=wG['corpus_fixed_tree_construction_parentType_weight_file']
  wB.log("(1)<<<<  before training RvNN,配置超参数完毕")
class wk(Bl):
 def __init__(wF,sentence_length=0):
  wF.sl=wQ
  wF.nodeScores=np.zeros((2*wF.sl-1,1),dtype=np.double)
  wF.collapsed_sentence=(BU)(BI(0,wF.sl))
  wF.pp=np.zeros((2*wF.sl-1,1),dtype=np.FG)
class wO():
 def load_data(wF):
  wB.log("(2)>>>>  加载词向量数据和语料库")
  (wF.trainCorpus,wF.fullCorpus,wF.trainCorpus_sentence_length,wF.fullCorpus_sentence_length,wF.vocabulary,wF.We,wF.config.max_sentence_length_train_Corpus,wF.config.max_sentence_length_full_Corpus,wF.train_corpus_fixed_tree_constructionorder,wF.full_corpus_fixed_tree_constructionorder)=preprocess_withAST(wF.config.IDIR,wF.config.ODIR,wF.config.corpus_fixed_tree_constructionorder_file,wF.config.MAX_SENTENCE_LENGTH)
  if BY(wF.trainCorpus)>4000:
   wN=BY(wF.trainCorpus)
   wc=BU(BI(wN))
   shuffle(wc)
   wC=wc[0:4000]
   wF.evalutionCorpus=[wF.trainCorpus[wA]for wA in wC]
   wF.config.max_sentence_length_evalution_Corpus=wF.config.max_sentence_length_train_Corpus 
   wF.evalution_corpus_fixed_tree_constructionorder=[wF.train_corpus_fixed_tree_constructionorder[wA]for wA in wC]
  else:
   wF.evalutionCorpus=wF.trainCorpus
   wF.config.max_sentence_length_evalution_Corpus=wF.config.max_sentence_length_train_Corpus
   wF.evalution_corpus_fixed_tree_constructionorder=wF.train_corpus_fixed_tree_constructionorder
  wB.log("(2)>>>>  加载词向量数据和语料库完毕")
 def load_data_experimentID_1(wF):
  wB.log("(2)>>>>  加载词向量数据和语料库")
  (wF.trainCorpus,wF.trainCorpus_sentence_length,wF.vocabulary,wF.We,wF.config.max_sentence_length_train_Corpus,wF.train_corpus_fixed_tree_constructionorder,wF.train_corpus_fixed_tree_parentType_weight)=preprocess_withAST_experimentID_10(wF.config.IDIR,wF.config.ODIR,wF.config.corpus_fixed_tree_constructionorder_file,wF.config.MAX_SENTENCE_LENGTH,wF.config.corpus_fixed_tree_construction_parentType_weight_file)
  wB.log("------------------------------\n对照BigCloneBench中标注的函数，找出在我们的语料库中的编号位置")
  ws='./SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
  wr=[]
  with BS(ws,'rb')as f:
   wn=pickle.load(f)
   for Bg in wn.keys():
    wy=wn[Bg]
    if(wy==-1):
     continue
    wr.append(wy)
  wF.lines_for_bigcloneBench=wr
  wR=BY(wr)
  wB.log('BigCloneBench中有效函数ID多少个，对应的取出我们语料库中的语料多少个.{}个'.format(wR))
  wF.trainCorpus=[wF.trainCorpus[wA]for wA in wr]
  wF.train_corpus_fixed_tree_constructionorder=[wF.train_corpus_fixed_tree_constructionorder[wA]for wA in wr]
  wF.trainCorpus_sentence_length=[wF.trainCorpus_sentence_length[wA]for wA in wr]
  wF.train_corpus_fixed_tree_parentType_weight=[wF.train_corpus_fixed_tree_parentType_weight[wA]for wA in wr]
  wV=BU(Fw(BY,wF.trainCorpus))
  wF.config.max_sentence_length_train_Corpus=FP(wV)
  wF.bigCloneBench_Corpus=wF.trainCorpus 
  wF.bigCloneBench_Corpus_fixed_tree_constructionorder=wF.train_corpus_fixed_tree_constructionorder
  wF.bigCloneBench_Corpus_sentence_length=wF.trainCorpus_sentence_length
  wF.bigCloneBench_Corpus_max_sentence_length=wF.config.max_sentence_length_train_Corpus
  wF.bigCloneBench_Corpus_fixed_tree_parentType_weight=wF.train_corpus_fixed_tree_parentType_weight
  wB.log('(2)>>>>  对照BigCloneBench中标注的函数,从我们的语料库中抽取语料{}个'.format(BY(wr))) 
 def load_data_experimentID_2(wF):
  wX='./vectorTree/valid_dataset_lst.pkl'
  wL=wF.read_from_pkl(wX)
  wB.log("(2)>>>>  加载词向量数据和语料库")
  (wF.trainCorpus,wF.trainCorpus_sentence_length,wF.vocabulary,wF.We,wF.config.max_sentence_length_train_Corpus,wF.train_corpus_fixed_tree_constructionorder)=preprocess_withAST_experimentID_1(wF.config.IDIR,wF.config.ODIR,wF.config.corpus_fixed_tree_constructionorder_file,wF.config.MAX_SENTENCE_LENGTH)
  wB.log("------------------------------\n对照BigCloneBench中标注的函数，找出在我们的语料库中的编号位置")
  ws='./SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
  wH=[]
  wW=[]
  with BS(ws,'rb')as f:
   wF.id_line_dict=pickle.load(f)
   wA=0
   for Bg in wL:
    wy=wF.id_line_dict[Bg]
    if(wy==-1):
     continue
    wW.append(Bg)
    wH.append(wy)
    wA+=1
  wF.need_vectorTree_lines_for_trainCorpus=wH
  wF.need_vectorTree_ids_for_trainCorpus=wW
  wR=BY(wW)
  wB.log('路哥需要取出{}个ID对应的向量树'.format(wR))
  wF.trainCorpus=[wF.trainCorpus[wA]for wA in wH]
  wF.train_corpus_fixed_tree_constructionorder=[wF.train_corpus_fixed_tree_constructionorder[wA]for wA in wH]
  wF.trainCorpus_sentence_length=[wF.trainCorpus_sentence_length[wA]for wA in wH]
  wV=BU(Fw(BY,wF.trainCorpus))
  wF.config.max_sentence_length_train_Corpus=FP(wV)
  wF.need_vectorTree_Corpus=wF.trainCorpus 
  wF.need_vectorTree_Corpus_fixed_tree_constructionorder=wF.train_corpus_fixed_tree_constructionorder
  wF.need_vectorTree_Corpus_sentence_length=wF.trainCorpus_sentence_length
  wF.need_vectorTree_Corpus_max_sentence_length=wF.config.max_sentence_length_train_Corpus
 def load_data_experimentID_3(wF):
  wB.log("(2)>>>>  加载词向量数据和语料库")
  (wF.fullCorpus,wF.fullCorpus_sentence_length,wF.vocabulary,wF.We,wF.config.max_sentence_length_full_Corpus,wF.full_corpus_fixed_tree_constructionorder,wF.full_corpus_fixed_tree_parentType_weight)=preprocess_withAST_experimentID_10(wF.config.IDIR,wF.config.ODIR,wF.config.corpus_fixed_tree_constructionorder_file,wF.config.MAX_SENTENCE_LENGTH,wF.config.corpus_fixed_tree_construction_parentType_weight_file)
 def xiaojie_RvNN_fixed_tree(wF):
  wF.add_placeholders_fixed_tree()
  wF.add_model_vars()
  wF.add_loss_fixed_tree()
  wF.train_op=wF.training(wF.tensorLoss_fixed_tree)
 def xiaojie_RvNN_fixed_tree_for_usingmodel(wF):
  wF.add_placeholders_fixed_tree()
  wF.add_model_vars()
  wF.add_loss_and_batchSentenceNodesVector_fixed_tree()
 def buqi_2DmatrixTensor(wF,wj,Pe,Pz,Pa,Pp):
  wj=tf.pad(wj,[[0,Pa-Pe],[0,Pp-Pz]])
  return wj
 def modify_one_profile(wF,tensor,wj,Po,Pf,Pd,Pv):
  wj=tf.expand_dims(wj,axis=0)
  wM=tf.slice(tensor,[0,0,0],[Po,Pd,Pv])
  wT=tf.slice(tensor,[Po+1,0,0],[Pf-Po-1,Pd,Pv])
  wJ=tf.concat([wM,wj,wT],0)
  return wM,wT,wJ
 def delete_one_column(wF,tensor,wA,Pb,numcolunms):
  wM=tf.slice(tensor,[0,0],[Pb,wA])
  wT=tf.slice(tensor,[0,wA+1],[Pb,numcolunms-(wA+1)])
  wJ=tf.concat([wM,wT],1)
  return wJ
 def modify_one_column(wF,tensor,columnTensor,wA,Pb,numcolunms):
  wM=tf.slice(tensor,[0,0],[Pb,wA])
  wT=tf.slice(tensor,[0,wA+1],[Pb,numcolunms-(wA+1)])
  wJ=tf.concat([wM,columnTensor,wT],1)
  return wJ
 def computeloss_withAST(wF,sentence,Bj):
  with tf.variable_scope('Composition',reuse=Fk):
   W1=tf.get_variable("W1",dtype=tf.float64)
   b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=Fk):
   U=tf.get_variable("U",dtype=tf.float64)
   bs=tf.get_variable("bs",dtype=tf.float64)
  wq=np.ones_like(sentence)
  wK=sentence-wq
  we=np.array(wF.We)
  L=we[:,wK]
  sl=L.shape[1]
  wz=FO()
  for i in BI(0,sl):
   wz[i]=np.expand_dims(L[:,i],1)
  wa=FO()
  if(sl>1):
   for j in BI(0,sl-1):
    wp=W1.eval()
    wo=b1.eval()
    wf=U.eval()
    wd=bs.eval()
    wv=Bj[:,j]
    wD=wv[0]-1 
    wx=wz[wD]
    wE=wv[1]-1
    wb=wz[wE] 
    wi=wv[2]-1
    wm=np.concatenate((wx,wb),axis=0)
    p=np.tanh(np.dot(wp,wm)+wo)
    wt=p/(np.linalg.norm(p,axis=0)*np.ones(p.shape))
    wz[wi]=wt
    y=np.tanh(np.dot(wf,wt)+wd)
    [y1,y2]=np.split(y,2,axis=0)
    wl=y1/(np.linalg.norm(y1,axis=0)*np.ones(y1.shape))
    wU=y2/(np.linalg.norm(y2,axis=0)*np.ones(y2.shape))
    wI=1 
    wY=1
    wS,wg=wI*(wl-wx),wY*(wU-wb)
    constructionError=np.FA((wS*(wl-wx)+wg*(wU-wb)),axis=0)*0.5 
    wa[j]=constructionError
    pass
   pass
  Pw=0
  for(Oz,value)in wa.items():
   Pw=Pw+value
  Pw=Pw/(sl-1)
  return Pw 
 def add_loss_fixed_tree(wF):
  with tf.variable_scope('Composition',reuse=Fk):
   wF.W1=tf.get_variable("W1",dtype=tf.float64)
   wF.b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=Fk):
   wF.U=tf.get_variable("U",dtype=tf.float64)
   wF.bs=tf.get_variable("bs",dtype=tf.float64)
  Pk=tf.zeros(wF.batch_len,tf.float64)
  Pk=tf.expand_dims(Pk,0)
  wF.numlines_tensor3=tf.constant(1,dtype=tf.int32)
  wF.numcolunms_tensor3=wF.batch_len[0]
  i=tf.constant(0,dtype=tf.int32)
  PO=wF.batch_len[0]
  PB=lambda a,b,c:tf.less(a,PO)
  PF=tf.constant(0)
  PG=[i,Pk,PF]
  def _recurrence(i,Pk,PF):
   wF.sentence_embeddings=tf.gather(wF.Fr,i,axis=0)
   wF.sentence_length=wF.batch_real_sentence_length[i]
   wF.treeConstructionOrders=wF.batch_treeConstructionOrders[i]
   wF.sentence_parentTypes_weight=wF.batch_sentence_parentTypes_weight[i]
   wF.sentence_parentTypes_weight=wF.batch_sentence_parentTypes_weight[i]
   wF.treechildparentweight=wF.batch_childparentweight[i]
   Pu=2*wF.sentence_length-1
   wz=tf.zeros(Pu,tf.float64)
   wz=tf.expand_dims(wz,0)
   wz=tf.tile(wz,(wF.config.embed_size,1))
   wF.numlines_tensor=tf.constant(wF.config.embed_size,dtype=tf.int32)
   wF.numcolunms_tensor=Pu
   ii=tf.constant(0,dtype=tf.int32)
   PN=lambda a,b:tf.less(a,wF.sentence_length)
   Pc=[ii,wz]
   def __recurrence(ii,wz):
    PC=tf.expand_dims(wF.sentence_embeddings[:,ii],1)
    wz=wF.modify_one_column(wz,PC,ii,wF.numlines_tensor,wF.numcolunms_tensor)
    ii=tf.add(ii,1)
    return ii,wz
   ii,wz=tf.while_loop(PN,__recurrence,Pc,parallel_iterations=1)
   wa=tf.zeros(wF.sentence_length-1,tf.float64)
   wa=tf.expand_dims(wa,0)
   wF.numlines_tensor2=tf.constant(1,dtype=tf.int32)
   wF.numcolunms_tensor2=wF.sentence_length-1
   PA=tf.constant(0,dtype=tf.int32)
   Ps=lambda a,b,c,d:tf.less(a,wF.sentence_length-1)
   Pr=[PA,wa,wz,PF]
   def ____recurrence(PA,wa,wz,PF):
    wv=wF.treeConstructionOrders[:,PA]
    wD=wv[0]-1 
    Pn=wz[:,wD]
    wE=wv[1]-1
    Py=wz[:,wE] 
    wi=wv[2]-1
    PR=tf.concat([Pn,Py],axis=0)
    PR=tf.expand_dims(PR,1)
    PV=tf.tanh(tf.add(tf.matmul(wF.W1,PR),wF.b1))
    PX=wF.normalization(PV)
    wz=wF.modify_one_column(wz,PX,wi,wF.numlines_tensor,wF.numcolunms_tensor)
    y=(tf.matmul(wF.U,PX)+wF.bs)
    PL=y.shape[1].value
    (y1,y2)=wF.split_by_row(y,PL)
    wI=1 
    wY=1
    Pn=tf.expand_dims(Pn,1)
    Py=tf.expand_dims(Py,1)
    wS=tf.subtract(y1,Pn)
    wg=tf.subtract(y2,Py) 
    constructionError=wF.constructionError(wS,wg,wI,wY)
    PH=wF.treechildparentweight[:,PA]
    PW=PH[2]
    constructionError=tf.multiply(constructionError,PW)
    constructionError=tf.expand_dims(constructionError,1)
    wa=wF.modify_one_column(wa,constructionError,PA,wF.numlines_tensor2,wF.numcolunms_tensor2)
    Pj=tf.Print(PA,[PA],"\niiii:")
    PF=Pj+PF
    PF=Pj+PF
    Pj=tf.Print(wD,[wD],"\nleftChild_index:",summarize=100)
    PF=tf.to_int32(Pj)+PF
    Pj=tf.Print(wE,[wE],"\nrightChild_index:",summarize=100)
    PF=tf.to_int32(Pj)+PF
    Pj=tf.Print(wi,[wi],"\nparent_index:",summarize=100)
    PF=tf.to_int32(Pj)+PF
    PA=tf.add(PA,1)
    return PA,wa,wz,PF
   PA,wa,wz,PF=tf.while_loop(Ps,____recurrence,Pr,parallel_iterations=1)
   pass
   wF.node_tensors_cost_tensor=tf.identity(wa)
   wF.nodes_tensor=tf.identity(wz)
   PM=tf.reduce_mean(wF.node_tensors_cost_tensor)
   PM=tf.expand_dims(tf.expand_dims(PM,0),1)
   Pk=wF.modify_one_column(Pk,PM,i,wF.numlines_tensor3,wF.numcolunms_tensor3)
   i=tf.add(i,1)
   return i,Pk,PF
  i,Pk,PF=tf.while_loop(PB,_recurrence,PG,parallel_iterations=10)
  wF.tfPrint=PF
  with tf.name_scope('loss'):
   PT=tf.nn.l2_loss(wF.W1)+tf.nn.l2_loss(wF.U)
   wF.batch_constructionError=tf.reduce_mean(Pk)
   wF.tensorLoss_fixed_tree=wF.batch_constructionError+PT*wF.config.l2
  return wF.tensorLoss_fixed_tree
 def add_loss_and_batchSentenceNodesVector_fixed_tree(wF):
  with tf.variable_scope('Composition',reuse=Fk):
   wF.W1=tf.get_variable("W1",dtype=tf.float64)
   wF.b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=Fk):
   wF.U=tf.get_variable("U",dtype=tf.float64)
   wF.bs=tf.get_variable("bs",dtype=tf.float64)
  Pk=tf.zeros(wF.batch_len,tf.float64)
  Pk=tf.expand_dims(Pk,0)
  PJ=wF.batch_real_sentence_length[tf.argmax(wF.batch_real_sentence_length)]
  Ph=2*PJ-1
  Pq=wF.batch_len[0]*wF.config.embed_size*Ph
  PK=tf.zeros(Pq,tf.float64)
  PK=tf.reshape(PK,[wF.batch_len[0],wF.config.embed_size,Ph])
  wF.size_firstDimension=wF.batch_len[0]
  wF.size_secondDimension=tf.constant(wF.config.embed_size,dtype=tf.int32)
  wF.size_thirdDimension=Ph
  wF.numlines_tensor3=tf.constant(1,dtype=tf.int32)
  wF.numcolunms_tensor3=wF.batch_len[0]
  wF.numlines_tensor4=tf.constant(wF.config.embed_size,dtype=tf.int32)
  wF.numcolunms_tensor4=wF.batch_len[0]
  i=tf.constant(0,dtype=tf.int32)
  PO=wF.batch_len[0]
  PB=lambda a,b,c,d:tf.less(a,PO)
  PF=tf.constant(0)
  PG=[i,Pk,PF,PK]
  def _recurrence(i,Pk,PF,PK):
   wF.sentence_embeddings=tf.gather(wF.Fr,i,axis=0)
   wF.sentence_length=wF.batch_real_sentence_length[i]
   wF.treeConstructionOrders=wF.batch_treeConstructionOrders[i]
   wF.sentence_parentTypes_weight=wF.batch_sentence_parentTypes_weight[i]
   wF.treechildparentweight=wF.batch_childparentweight[i]
   Pu=2*wF.sentence_length-1
   wz=tf.zeros(Pu,tf.float64)
   wz=tf.expand_dims(wz,0)
   wz=tf.tile(wz,(wF.config.embed_size,1))
   wF.numlines_tensor=tf.constant(wF.config.embed_size,dtype=tf.int32)
   wF.numcolunms_tensor=Pu
   ii=tf.constant(0,dtype=tf.int32)
   PN=lambda a,b:tf.less(a,wF.sentence_length)
   Pc=[ii,wz]
   def __recurrence(ii,wz):
    PC=tf.expand_dims(wF.sentence_embeddings[:,ii],1)
    wz=wF.modify_one_column(wz,PC,ii,wF.numlines_tensor,wF.numcolunms_tensor)
    ii=tf.add(ii,1)
    return ii,wz
   ii,wz=tf.while_loop(PN,__recurrence,Pc,parallel_iterations=1)
   wa=tf.zeros(wF.sentence_length-1,tf.float64)
   wa=tf.expand_dims(wa,0)
   wF.numlines_tensor2=tf.constant(1,dtype=tf.int32)
   wF.numcolunms_tensor2=wF.sentence_length-1
   PA=tf.constant(0,dtype=tf.int32)
   Ps=lambda a,b,c,d:tf.less(a,wF.sentence_length-1)
   Pr=[PA,wa,wz,PF]
   def ____recurrence(PA,wa,wz,PF):
    wv=wF.treeConstructionOrders[:,PA]
    wD=wv[0]-1 
    Pn=wz[:,wD]
    wE=wv[1]-1
    Py=wz[:,wE] 
    wi=wv[2]-1
    PR=tf.concat([Pn,Py],axis=0)
    PR=tf.expand_dims(PR,1)
    PV=tf.tanh(tf.add(tf.matmul(wF.W1,PR),wF.b1))
    PX=wF.normalization(PV)
    wz=wF.modify_one_column(wz,PX,wi,wF.numlines_tensor,wF.numcolunms_tensor)
    y=(tf.matmul(wF.U,PX)+wF.bs)
    PL=y.shape[1].value
    (y1,y2)=wF.split_by_row(y,PL)
    wI=1 
    wY=1
    Pn=tf.expand_dims(Pn,1)
    Py=tf.expand_dims(Py,1)
    wS=tf.subtract(y1,Pn)
    wg=tf.subtract(y2,Py) 
    constructionError=wF.constructionError(wS,wg,wI,wY)
    PH=wF.treechildparentweight[:,PA]
    PW=PH[2]
    constructionError=tf.multiply(constructionError,PW)
    constructionError=tf.expand_dims(constructionError,1)
    wa=wF.modify_one_column(wa,constructionError,PA,wF.numlines_tensor2,wF.numcolunms_tensor2)
    Pj=tf.Print(PA,[PA],"\niiii:")
    PF=Pj+PF
    PF=Pj+PF
    Pj=tf.Print(wD,[wD],"\nleftChild_index:",summarize=100)
    PF=tf.to_int32(Pj)+PF
    Pj=tf.Print(wE,[wE],"\nrightChild_index:",summarize=100)
    PF=tf.to_int32(Pj)+PF
    Pj=tf.Print(wi,[wi],"\nparent_index:",summarize=100)
    PF=tf.to_int32(Pj)+PF
    PA=tf.add(PA,1)
    return PA,wa,wz,PF
   PA,wa,wz,PF=tf.while_loop(Ps,____recurrence,Pr,parallel_iterations=1)
   pass
   wF.node_tensors_cost_tensor=tf.identity(wa)
   wF.nodes_tensor=tf.identity(wz)
   PM=tf.reduce_mean(wF.node_tensors_cost_tensor)
   PM=tf.expand_dims(tf.expand_dims(PM,0),1)
   Pk=wF.modify_one_column(Pk,PM,i,wF.numlines_tensor3,wF.numcolunms_tensor3)
   Pe=wF.numlines_tensor 
   Pz=wF.numcolunms_tensor 
   Pa=wF.size_secondDimension 
   Pp=wF.size_thirdDimension
   wz=wF.buqi_2DmatrixTensor(wz,Pe,Pz,Pa,Pp)
   wz=tf.reshape(wz,[wF.config.embed_size,Pp])
   Po=i 
   Pf=wF.size_firstDimension
   Pd=wF.size_secondDimension
   Pv=wF.size_thirdDimension
   _,_,PK=wF.modify_one_profile(PK,wz,Po,Pf,Pd,Pv)
   i=tf.add(i,1)
   return i,Pk,PF,PK
  i,Pk,PF,PK=tf.while_loop(PB,_recurrence,PG,parallel_iterations=10)
  wF.tfPrint=PF
  wF.batch_sentence_vectors=tf.identity(PK)
  with tf.name_scope('loss'):
   PT=tf.nn.l2_loss(wF.W1)+tf.nn.l2_loss(wF.U)
   wF.batch_constructionError=tf.reduce_mean(Pk)
   wF.tensorLoss_fixed_tree=wF.batch_constructionError+PT*wF.config.l2
  return wF.tensorLoss_fixed_tree,wF.batch_sentence_vectors
 def add_placeholders_fixed_tree(wF):
  PD=wF.config.embed_size
  wF.Fr=tf.placeholder(tf.float64,[FB,PD,FB],name='input')
  wF.batch_treeConstructionOrders=tf.placeholder(tf.int32,[FB,3,FB],name='treeConstructionOrders')
  wF.batch_real_sentence_length=tf.placeholder(tf.int32,[FB],name='batch_real_sentence_length')
  wF.batch_len=tf.placeholder(tf.int32,shape=(1,),name='batch_len')
  wF.batch_sentence_parentTypes_weight=tf.placeholder(tf.float64,[FB,FB],name='batch_sentence_parentType_weights')
  wF.batch_childparentweight=tf.placeholder(tf.float64,[FB,3,FB],name='treeChildParentWeights')
 def add_model_vars(wF):
  with tf.variable_scope('Composition'): 
   tf.get_variable("W1",dtype=tf.float64,shape=[wF.config.embed_size,2*wF.config.embed_size])
   tf.get_variable("b1",dtype=tf.float64,shape=[wF.config.embed_size,1])
  with tf.variable_scope('Projection'):
   tf.get_variable("U",dtype=tf.float64,shape=[2*wF.config.embed_size,wF.config.embed_size])
   tf.get_variable("bs",dtype=tf.float64,shape=[2*wF.config.embed_size,1])
 def normalization(wF,tensor):
  Pb=tensor.shape[0].value
  Pi=tf.pow(tensor,2)
  Pm=tf.reduce_sum(Pi,0)
  Pt=tf.expand_dims(Pm,0)
  Pl=tf.tile(tf.sqrt(Pt),(Pb,1))
  PU=tf.divide(tensor,Pl)
  return PU
 def split_by_row(wF,tensor,numcolunms):
  Pb=tensor.shape[0].value
  PI=tf.slice(tensor,[0,0],[(FG)(Pb/2),numcolunms])
  PY=tf.slice(tensor,[(FG)(Pb/2),0],[(FG)(Pb/2),numcolunms])
  pass
  return(PI,PY)
 def constructionError(wF,tensor1,tensor2,wI,wY):
  PS=tf.multiply(tf.reduce_sum(tf.pow(tensor1,2),0),wI)
  Pg=tf.multiply(tf.reduce_sum(tf.pow(tensor2,2),0),wY)
  kw=tf.multiply(tf.add(PS,Pg),0.5)
  return kw
 def training(wF,ki):
  kP=FB
  kO=tf.train.GradientDescentOptimizer(wF.config.lr)
  kP=kO.minimize(ki)
  return kP
 def __init__(wF,kB,experimentID=FB):
  if(experimentID==FB):
   wF.config=kB
   wF.load_data()
  elif(experimentID==1):
   wF.config=kB
   wF.load_data_experimentID_1()
  elif(experimentID==2):
   wF.config=kB
   wF.load_data_experimentID_2()
  elif(experimentID==3):
   wF.config=kB
   wF.load_data_experimentID_3()
 def computelossAndVector_no_tensor_withAST(wF,sentence,Bj):
  with tf.variable_scope('Composition',reuse=Fk):
   W1=tf.get_variable("W1",dtype=tf.float64)
   b1=tf.get_variable("b1",dtype=tf.float64)
  with tf.variable_scope('Projection',reuse=Fk):
   U=tf.get_variable("U",dtype=tf.float64)
   bs=tf.get_variable("bs",dtype=tf.float64)
  wq=np.ones_like(sentence)
  wK=sentence-wq
  we=np.array(wF.We)
  L=we[:,wK]
  sl=L.shape[1]
  kF=FO()
  for i in BI(0,sl):
   kF[i]=np.expand_dims(L[:,i],1)
  wa=FO()
  kG=FB
  if(sl>1):
   for j in BI(0,sl-1):
    wp=W1.eval()
    wo=b1.eval()
    wf=U.eval()
    wd=bs.eval()
    wv=Bj[:,j]
    wD=wv[0]-1 
    wx=kF[wD]
    wE=wv[1]-1
    wb=kF[wE] 
    wi=wv[2]-1
    wm=np.concatenate((wx,wb),axis=0)
    p=np.tanh(np.dot(wp,wm)+wo)
    wt=p/(np.linalg.norm(p,axis=0)*np.ones(p.shape))
    kF[wi]=wt
    kG=wt
    y=np.tanh(np.dot(wf,wt)+wd)
    [y1,y2]=np.split(y,2,axis=0)
    wl=y1/(np.linalg.norm(y1,axis=0)*np.ones(y1.shape))
    wU=y2/(np.linalg.norm(y2,axis=0)*np.ones(y2.shape))
    wI=1 
    wY=1
    wS,wg=wI*(wl-wx),wY*(wU-wb)
    constructionError=np.FA((wS*(wl-wx)+wg*(wU-wb)),axis=0)*0.5 
    wa[j]=constructionError
    pass
   kQ=kF[2*sl-2]
   pass
  Pw=0
  for(Oz,value)in wa.items():
   Pw=Pw+value
  return(Pw,kF,kQ,kG)
 def run_epoch_train(wF,OB,km):
  ku=[]
  kN=wF.trainCorpus
  kc=wF.trainCorpus_sentence_length
  kC=wF.train_corpus_fixed_tree_constructionorder
  kA=wF.config.max_sentence_length_train_Corpus
  ks=BY(wF.trainCorpus)
  kr=wF.config.MAX_SENTENCE_LENGTH_for_Bigclonebench
  wB.log('训练过程设置长短的衡量标准是{}，长的单独成batch，短的集合成batch'.format(kr))
  kn=[]
  ky=[]
  for wA,length in FQ(kc):
   if length<kr:
    kn.append(wA)
   else:
    ky.append(wA)
  wB.log("训练集的句子{}个".format(ks))
  wB.log("较长的句子{}个".format(BY(ky)))
  wB.log("较短的句子{}个".format(BY(kn)))
  kR=[kN[wA]for wA in kn]
  kV=[kc[wA]for wA in kn]
  kX=[kC[wA]for wA in kn]
  kL=[kN[wA]for wA in ky]
  kH=[kc[wA]for wA in ky]
  kW=[kC[wA]for wA in ky]
  wB.log("较短的句子，我们走批处理训练网络。长句子，我们单独计算训练网络")
  wB.log("先处理较短的句子的语料，批处理开始")
  kj=BY(kn)
  we=np.array(wF.We)
  kM=we.shape[0]
  wc=BU(BI(kj))
  kT=wF.config.batch_size
  kJ=0
  for kh in BI(0,kj,kT):
   kq=Fu(kh+kT,kj)-kh
   kK=wc[kh:kh+kq]
   ke=[kR[wA]for wA in kK]
   kz=[kV[wA]for wA in kK]
   ka=FP(kz)
   x=[]
   for i,sentence in FQ(ke):
    kp=kz[i]
    wq=np.ones_like(sentence)
    wK=sentence-wq
    L1=we[:,wK]
    ko=ka-kp
    L2=np.zeros([kM,ko],np.float64)
    L=np.concatenate((L1,L2),axis=1)
    x.append(L)
   kf=np.array(x)
   kd=np.array(kf,np.float64)
   x=[]
   kv=[kX[wA]for wA in kK]
   for i,sentence_fixed_tree_constructionorder in FQ(kv):
    kD=kz[i]-1
    ko=(ka-1)-kD
    L2=np.zeros([3,ko],np.int32)
    L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
    x.append(L)
   kx=np.array(x)
   kE={wF.Fr:kd,wF.batch_real_sentence_length:kz,wF.batch_len:[kq],wF.batch_treeConstructionOrders:kx}
   kb=tf.RunOptions(report_tensor_allocations_upon_oom=Fk)
   ki,_=OB.run([wF.tensorLoss_fixed_tree,wF.train_op],feed_dict=kE,options=kb)
   ku.append(ki)
   wB.log('\repoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(km,kJ,kq,ki)) 
   wB.log('\repoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(km,BY(ku),np.mean(ku))) 
   kJ=kJ+1
   pass 
  wB.log("保存模型到temp")
  kt=tf.train.Saver()
  if not os.path.exists("./weights"):
   os.makedirs("./weights")
  kt.save(OB,'./weights/%s.temp'%wF.config.model_name)
  return ku 
 def run_epoch_evaluation(wF,OB,km):
  ku=[]
  kN=wF.evalutionCorpus
  kc=wF.evalution_corpus_sentence_length
  kC=wF.evalution_corpus_fixed_tree_constructionorder
  kA=wF.config.max_sentence_length_evalution_Corpus
  kl=BY(wF.evalutionCorpus)
  kr=1000; 
  wB.log('训练过程设置长短的衡量标准是{}，长的单独成batch，短的集合成batch'.format(kr))
  kn=[]
  ky=[]
  for wA,length in FQ(kc):
   if length<kr:
    kn.append(wA)
   else:
    ky.append(wA)
  wB.log("训练集的句子{}个".format(ks))
  wB.log("较长的句子{}个".format(BY(ky)))
  wB.log("较短的句子{}个".format(BY(kn)))
  kR=[kN[wA]for wA in kn]
  kV=[kc[wA]for wA in kn]
  kX=[kC[wA]for wA in kn]
  kL=[kN[wA]for wA in ky]
  kH=[kc[wA]for wA in ky]
  kW=[kC[wA]for wA in ky]
  wB.log("较短的句子，我们走批处理训练网络。长句子，我们单独计算训练网络")
  wB.log("先处理较短的句子的语料，批处理开始")
  kj=BY(kn)
  we=np.array(wF.We)
  kM=we.shape[0]
  wc=BU(BI(kj))
  kT=wF.config.batch_size_using_model_notTrain 
  kJ=0
  for kh in BI(0,kj,kT):
   kq=Fu(kh+kT,kj)-kh
   kK=wc[kh:kh+kq]
   ke=[kR[wA]for wA in kK]
   kz=[kV[wA]for wA in kK]
   ka=FP(kz)
   x=[]
   for i,sentence in FQ(ke):
    kp=kz[i]
    wq=np.ones_like(sentence)
    wK=sentence-wq
    L1=we[:,wK]
    ko=ka-kp
    L2=np.zeros([kM,ko],np.float64)
    L=np.concatenate((L1,L2),axis=1)
    x.append(L)
   kf=np.array(x)
   kd=np.array(kf,np.float64)
   x=[]
   kv=[kX[wA]for wA in kK]
   for i,sentence_fixed_tree_constructionorder in FQ(kv):
    kD=kz[i]-1
    ko=(ka-1)-kD
    L2=np.zeros([3,ko],np.int32)
    L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
    x.append(L)
   kx=np.array(x)
   kE={wF.Fr:kd,wF.batch_real_sentence_length:kz,wF.batch_len:[kq],wF.batch_treeConstructionOrders:kx}
   ki=OB.run([wF.tensorLoss_fixed_tree],feed_dict=kE)
   ku.append(ki)
   wB.log('\r验证过程epoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(km,kJ,kq,ki)) 
   wB.log('\r验证过程epoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(km,BY(ku),np.mean(ku))) 
   kJ=kJ+1
   pass 
  wB.log("再处理较长的句子的语料，每个句子单独处理，开始") 
  kU=BY(ky)
  wc=BU(BI(kU))
  for kh,sentence in FQ(kL):
   kq=1
   kK=wc[kh:kh+kq]
   ke=[kL[wA]for wA in kK]
   kz=[kH[wA]for wA in kK]
   ka=FP(kz)
   x=[]
   for i,sentence in FQ(ke):
    kp=kz[i]
    wq=np.ones_like(sentence)
    wK=sentence-wq
    L1=we[:,wK]
    ko=ka-kp
    L2=np.zeros([kM,ko],np.float64)
    L=np.concatenate((L1,L2),axis=1)
    x.append(L)
   kf=np.array(x)
   kd=np.array(kf,np.float64)
   x=[]
   kv=[kW[wA]for wA in kK]
   for i,sentence_fixed_tree_constructionorder in FQ(kv):
    kD=kz[i]-1
    ko=(ka-1)-kD
    L2=np.zeros([3,ko],np.int32)
    L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
    x.append(L)
   kx=np.array(x)
   kE={wF.Fr:kd,wF.batch_real_sentence_length:kz,wF.batch_len:[kq],wF.batch_treeConstructionOrders:kx}
   ki=OB.run([wF.tensorLoss_fixed_tree],feed_dict=kE)
   ku.append(ki)
   wB.log('\r验证过程epoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(km,kJ,kq,ki)) 
   wB.log('\r验证过程epoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(km,BY(ku),np.mean(ku))) 
   kJ=kJ+1
   pass
  return ku 
 def train(wF,restore=FN):
  with tf.Graph().as_default():
   wF.xiaojie_RvNN_fixed_tree()
   kI=tf.initialize_all_variables()
   kt=tf.train.Saver()
   kY=[]
   kS=[]
   kg=Fc('inf')
   Ow=Fc('inf')
   OP=0 
   Ok=-1
   km=0
   kB=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=Fk),allow_soft_placement=Fk)
   kB.gpu_options.per_process_gpu_memory_fraction=0.95
   with tf.Session(config=kB)as OB:
    OB.run(kI)
    OF=time.time()
    if restore:kt.restore(OB,'./weights/%s'%wF.config.model_name)
    while km<wF.config.max_epochs:
     wB.log('epoch %d'%km)
     ku=wF.run_epoch_train(OB,km)
     kY.extend(ku)
     OG=wF.run_epoch_evaluation(OB,km)
     OQ=np.mean(OG)
     kS.append(OQ)
     wB.log("time per epoch is {} s".format(time.time()-OF))
     Ou=OQ
     if Ou>kg*wF.config.anneal_threshold:
      wF.config.lr/=wF.config.anneal_by
      wB.log('annealed lr to %f'%wF.config.lr)
     kg=Ou 
     if OQ<Ow:
      shutil.copyfile('./weights/%s.temp.data-00000-of-00001'%wF.config.model_name,'./weights/%s.data-00000-of-00001'%wF.config.model_name)
      shutil.copyfile('./weights/%s.temp.index'%wF.config.model_name,'./weights/%s.index'%wF.config.model_name)
      shutil.copyfile('./weights/%s.temp.meta'%wF.config.model_name,'./weights/%s.meta'%wF.config.model_name)
      Ow=OQ
      OP=km
     elif km-OP>=wF.config.early_stopping:
      Ok=km
      break
     km+=1
     OF=time.time()
     pass
    if(km<(wF.config.max_epochs-1)):
     wB.log('预定训练{}个epoch,一共训练{}个epoch，在评估集上最优的是第{}个epoch(从0开始计数),最优评估loss是{}'.format(wF.config.max_epochs,Ok+1,OP,Ow))
    elif(km==(wF.config.max_epochs-1)):
     wB.log('预定训练{}个epoch,全部轮数达成，在评估集上最优的是第{}个epoch,最优评估loss是{}'.format(wF.config.max_epochs,OP,Ow))
    else:
     wB.log('预定训练{}个epoch,全部轮数达成，在评估集上最优的是第{}个epoch,最优评估loss是{}'.format(wF.config.max_epochs,OP,Ow))
   return{'complete_loss_history':kY,'evalution_loss_history':kS,}
 def using_model_for_BigCloneBench_experimentID_1(wF):
  wB.log("------------------------------\n读取BigCloneBench的所有ID编号")
  ws='./SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
  ON=[]
  with BS(ws,'rb')as f:
   wn=pickle.load(f)
   for Oc in wn.keys():
    wy=wn[Oc]
    if(wy==-1):
     continue 
    ON.append(Oc)
  OC=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
  OA='./vector/childparentweight/'+OC+'_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
  if os.path.exists(OA): 
   os.remove(OA) 
  else:
   FC('no such file:%s'%OA)
  Os='./vector/childparentweight/'+OC+'_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_mean.xiaojiepkl'
  if os.path.exists(Os): 
   os.remove(Os) 
  else:
   FC('no such file:%s'%Os)
  Or='./vector/childparentweight/'+OC+'_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_weighted.xiaojiepkl'
  if os.path.exists(Os): 
   os.remove(Os) 
  else:
   FC('no such file:%s'%Os)
  On='./vector/childparentweight/'+OC+'_using_Weigthed_RAE_BigCloneBenchFunctionID_Map_Vector_meanAndRootMost.xiaojiepkl'
  if os.path.exists(OA): 
   os.remove(OA) 
  else:
   FC('no such file:%s'%OA)
  Oy='./vector/childparentweight/'+OC+'_using_Weigthed_RAE_BigCloneBenchFunctionID_Map_Vector_weightedAndCengci.xiaojiepkl'
  if os.path.exists(OA): 
   os.remove(OA) 
  else:
   FC('no such file:%s'%OA)
  kN=wF.bigCloneBench_Corpus
  kC=wF.bigCloneBench_Corpus_fixed_tree_constructionorder
  kc=wF.bigCloneBench_Corpus_sentence_length
  OR=wF.bigCloneBench_Corpus_fixed_tree_parentType_weight
  OV={}
  OX={}
  OL={}
  OH={}
  OW={}
  del(wF.trainCorpus)
  del(wF.trainCorpus_sentence_length)
  del(wF.train_corpus_fixed_tree_constructionorder)
  del(wF.vocabulary)
  with tf.Graph().as_default():
   wF.xiaojie_RvNN_fixed_tree_for_usingmodel() 
   kt=tf.train.Saver()
   kB=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=Fk))
   kB.gpu_options.allocator_type='BFC'
   with tf.Session(config=kB)as OB:
    Oj='./weights/%s'%wF.config.model_name
    kt.restore(OB,Oj)
    kr=300; 
    wB.log('设置长短的衡量标准是{}'.format(kr))
    kn=[]
    ky=[]
    for wA,length in FQ(kc):
     if length<kr:
      kn.append(wA)
     else:
      ky.append(wA)
    wB.log("较长的句子{}个".format(BY(ky)))
    wB.log("较短的句子{}个".format(BY(kn)))
    kR=[kN[wA]for wA in kn]
    kV=[kc[wA]for wA in kn]
    kX=[kC[wA]for wA in kn]
    OM=[OR[wA]for wA in kn]
    kL=[kN[wA]for wA in ky]
    kH=[kc[wA]for wA in ky]
    kW=[kC[wA]for wA in ky]
    OT=[OR[wA]for wA in ky]
    wB.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
    wB.log("先处理较短的句子的语料，批处理开始")
    wN=BY(kn)
    we=np.array(wF.We)
    del(wF.We)
    kM=we.shape[0]
    wc=BU(BI(wN))
    kT=wF.config.batch_size_using_model_notTrain
    OJ=(wN-1)/kT 
    kJ=0
    for kh in BI(0,wN,kT):
     wB.log("batch_index:{}/{}".format(kJ,OJ))
     kq=Fu(kh+kT,wN)-kh
     kK=wc[kh:kh+kq]
     ke=[kR[wA]for wA in kK]
     kz=[kV[wA]for wA in kK]
     ka=FP(kz)
     x=[]
     for i,sentence in FQ(ke):
      kp=kz[i]
      wq=np.ones_like(sentence)
      wK=sentence-wq
      L1=we[:,wK]
      ko=ka-kp
      L2=np.zeros([kM,ko],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     kf=np.array(x)
     kd=np.array(kf,np.float64)
     del(kf)
     x=[]
     kv=[kX[wA]for wA in kK]
     for i,sentence_fixed_tree_constructionorder in FQ(kv):
      kD=kz[i]-1
      ko=(ka-1)-kD
      L2=np.zeros([3,ko],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     kx=np.array(x)
     Oh=[]
     for i in BI(kq):
      Oq=kv[i]
      kp=kz[i]
      kD=kz[i]-1 
      OK=kp+kD
      Oe={}
      FA=0
      for i in BI(kD+1):
       Oz=i+1
       Oe[Oz]=1
      for i in BI(kD):
       Oa=Oq[0,i]
       Op=Oq[1,i]
       wi=Oq[2,i]
       Oo=Oe[Oa]
       Of=Oe[Op]
       Oe[wi]=Oo+Of 
       FA+=Oe[wi]
      for i in BI(1,OK+1):
       Oe[i]=Oe[i]/(FA)
      Od=[]
      for i in BI(kD):
       Oa=Oq[0,i]
       Op=Oq[1,i]
       wi=Oq[2,i]
       Ov=Oe[Oa]
       OD=Oe[Op]
       Ox=Oe[wi]
       Od.append([Ov,OD,Ox])
      b=np.array(Od)
      b=b.transpose()
      Oh.append(b)
      pass
     x=[]
     for i,Od in FQ(Oh):
      kD=kz[i]-1
      ko=(ka-1)-kD
      L2=np.zeros([3,ko],np.float64)
      L=np.concatenate((Od,L2),axis=1)
      x.append(L)
     Oh=np.array(x)
     x=[]
     OE=[OM[wA]for wA in kK]
     for i,Oi in FQ(OE):
      Ob=np.FA(Oi)
      Oi=Oi/Ob
      Om=kz[i]-1
      ko=(ka-1)-Om
      L2=np.zeros([ko],np.int32)
      L=np.concatenate((Oi,L2))
      x.append(L)
     Ot=np.array(x) 
     kE={wF.Fr:kd,wF.batch_real_sentence_length:kz,wF.batch_len:[kq],wF.batch_treeConstructionOrders:kx,wF.batch_childparentweight:Oh,wF.batch_sentence_parentTypes_weight:Ot}
     ki,PK=OB.run([wF.tensorLoss_fixed_tree,wF.batch_sentence_vectors],feed_dict=kE)
     if kJ==0:
      FC(ki)
     Ol=0
     for wA in kK:
      OU=PK[Ol,:,:]
      Oi=Ot[Ol]
      wQ=kz[Ol]
      OK=2*wQ-1
      OI=OU[0:wF.config.embed_size,0:OK]
      OI=OI.astype(np.float32)
      Oq=kx[Ol,:,:]
      kD=kz[Ol]-1
      Ol=Ol+1
      OI=np.transpose(OI)
      OY=BU(OI)
      OS=kn[wA]
      OV[OS]=OY[OK-1]
      OS=kn[wA]
      Og=np.zeros_like(OY[0],np.float32)
      for i in BI(wQ,OK):
       Bw=OY[i]
       j=i-wQ
       BP=Oi[j]
       Bw=np.multiply(Bw,BP)
       Og=np.add(Og,Bw)
      OL[OS]=Og
     kJ=kJ+1
    wB.log("再处理较长的句子的语料，每个句子单独处理，开始") 
    kU=BY(ky)
    wc=BU(BI(kU))
    for kh,sentence in FQ(kL):
     wB.log("long_setence_index:{}/{}".format(kh,kU))
     kq=1
     kK=wc[kh:kh+kq]
     ke=[kL[wA]for wA in kK]
     kz=[kH[wA]for wA in kK]
     ka=FP(kz)
     x=[]
     for i,sentence in FQ(ke):
      kp=kz[i]
      wq=np.ones_like(sentence)
      wK=sentence-wq
      L1=we[:,wK]
      ko=ka-kp
      L2=np.zeros([kM,ko],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     kf=np.array(x)
     kd=np.array(kf,np.float64)
     x=[]
     kv=[kW[wA]for wA in kK]
     for i,sentence_fixed_tree_constructionorder in FQ(kv):
      kD=kz[i]-1
      ko=(ka-1)-kD
      L2=np.zeros([3,ko],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     kx=np.array(x)
     Oh=[]
     for i in BI(kq):
      Oq=kv[i]
      kp=kz[i]
      kD=kz[i]-1 
      OK=kp+kD
      Oe={}
      FA=0
      for i in BI(kD+1):
       Oz=i+1
       Oe[Oz]=1
      for i in BI(kD):
       Oa=Oq[0,i]
       Op=Oq[1,i]
       wi=Oq[2,i]
       Oo=Oe[Oa]
       Of=Oe[Op]
       Oe[wi]=Oo+Of 
       FA+=Oe[wi]
      for i in BI(1,OK+1):
       Oe[i]=Oe[i]/(FA)
      Od=[]
      for i in BI(kD):
       Oa=Oq[0,i]
       Op=Oq[1,i]
       wi=Oq[2,i]
       Ov=Oe[Oa]
       OD=Oe[Op]
       Ox=Oe[wi]
       Od.append([Ov,OD,Ox])
      b=np.array(Od)
      b=b.transpose()
      Oh.append(b)
      pass
     x=[]
     for i,Od in FQ(Oh):
      kD=kz[i]-1
      ko=(ka-1)-kD
      L2=np.zeros([3,ko],np.float64)
      L=np.concatenate((Od,L2),axis=1)
      x.append(L)
     Oh=np.array(x)
     x=[]
     OE=[OT[wA]for wA in kK]
     for i,Oi in FQ(OE):
      Ob=np.FA(Oi)
      Oi=Oi/Ob
      Om=kz[i]-1
      ko=(ka-1)-Om
      L2=np.zeros([ko],np.int32)
      L=np.concatenate((Oi,L2))
      x.append(L)
     Ot=np.array(x)
     kE={wF.Fr:kd,wF.batch_real_sentence_length:kz,wF.batch_len:[kq],wF.batch_treeConstructionOrders:kx,wF.batch_childparentweight:Oh,wF.batch_sentence_parentTypes_weight:Ot}
     ki,PK=OB.run([wF.tensorLoss_fixed_tree,wF.batch_sentence_vectors],feed_dict=kE)
     OI=PK[0,:,:]
     Oi=Ot[0]
     wQ=kz[0]
     OK=2*wQ-1
     OI=OI.astype(np.float32)
     OI=np.transpose(OI)
     OY=BU(OI)
     OS=ky[kh]
     OV[OS]=OY[OK-1]
     OS=ky[kh]
     Og=np.zeros_like(OY[0],np.float32)
     for i in BI(wQ,OK):
      Bw=OY[i]
      j=i-wQ
      BP=Oi[j]
      Bw=np.multiply(Bw,BP)
      Og=np.add(Og,Bw)
     OL[OS]=Og
     pass
  Bk={}
  BO={}
  BF={}
  BG={}
  BQ={}
  Bu={}
  for i,wy in FQ(wF.lines_for_bigcloneBench):
   Bu[wy]=i
  for BN in ON:
   wy=wn[BN]
   Bc=Bu[wy]
   BC=OV[Bc]
   Bk[BN]=BC
   BA=OL[Bc]
   BF[BN]=BA
  wF.save_to_pkl(Bk,OA)
  wF.save_to_pkl(BO,Os)
  wF.save_to_pkl(BF,Or)
  wF.save_to_pkl(BG,On)
  wF.save_to_pkl(BQ,Oy)
  FC(OA)
  FC(Or)
  pass
  return 
 def using_model_for_BigCloneBench_experimentID_2(wF):
  OC=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
  Bs='./vectorTree/'+OC+'_vectorTree.xiaojiepkl'
  if os.path.exists(Bs): 
   os.remove(OA) 
  else:
   FC('no such file:%s'%Bs)
  kN=wF.need_vectorTree_Corpus
  kC=wF.need_vectorTree_Corpus_fixed_tree_constructionorder
  kc=wF.need_vectorTree_Corpus_sentence_length
  Br={}
  with tf.Graph().as_default():
   wF.xiaojie_RvNN_fixed_tree_for_usingmodel() 
   kt=tf.train.Saver()
   kB=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=Fk))
   kB.gpu_options.allocator_type='BFC'
   with tf.Session(config=kB)as OB:
    Oj='./weights/%s'%wF.config.model_name
    kt.restore(OB,Oj)
    kr=500; 
    wB.log('设置长短的衡量标准是{}'.format(kr))
    kn=[]
    ky=[]
    for wA,length in FQ(kc):
     if length<kr:
      kn.append(wA)
     else:
      ky.append(wA)
    wB.log("较长的句子{}个".format(BY(ky)))
    wB.log("较短的句子{}个".format(BY(kn)))
    kR=[kN[wA]for wA in kn]
    kV=[kc[wA]for wA in kn]
    kX=[kC[wA]for wA in kn]
    kL=[kN[wA]for wA in ky]
    kH=[kc[wA]for wA in ky]
    kW=[kC[wA]for wA in ky]
    wB.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
    wB.log("先处理较短的句子的语料，批处理开始")
    wN=BY(kn)
    we=np.array(wF.We)
    kM=we.shape[0]
    wc=BU(BI(wN))
    kT=wF.config.batch_size_using_model_notTrain
    OJ=(wN-1)/kT 
    kJ=0
    for kh in BI(0,wN,kT):
     wB.log("batch_index:{}/{}".format(kJ,OJ))
     kq=Fu(kh+kT,wN)-kh
     kK=wc[kh:kh+kq]
     ke=[kR[wA]for wA in kK]
     kz=[kV[wA]for wA in kK]
     ka=FP(kz)
     x=[]
     for i,sentence in FQ(ke):
      kp=kz[i]
      wq=np.ones_like(sentence)
      wK=sentence-wq
      L1=we[:,wK]
      ko=ka-kp
      L2=np.zeros([kM,ko],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     kf=np.array(x)
     kd=np.array(kf,np.float64)
     x=[]
     kv=[kX[wA]for wA in kK]
     for i,sentence_fixed_tree_constructionorder in FQ(kv):
      kD=kz[i]-1
      ko=(ka-1)-kD
      L2=np.zeros([3,ko],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     kx=np.array(x)
     kE={wF.Fr:kd,wF.batch_real_sentence_length:kz,wF.batch_len:[kq],wF.batch_treeConstructionOrders:kx}
     ki,PK=OB.run([wF.tensorLoss_fixed_tree,wF.batch_sentence_vectors],feed_dict=kE)
     Ol=0
     for wA in kK:
      OU=PK[Ol,:,:]
      wQ=kz[Ol]
      OK=2*wQ-1
      OI=OU[0:wF.config.embed_size,0:OK]
      OI=OI.astype(np.float32)
      Ol=Ol+1
      OI=np.transpose(OI)
      OY=BU(OI)
      OS=kn[wA]
      Br[OS]=OY
     kJ=kJ+1
    wB.log("再处理较长的句子的语料，每个句子单独处理，开始") 
    kU=BY(ky)
    wc=BU(BI(kU))
    for kh,sentence in FQ(kL):
     wB.log("long_setence_index:{}/{}".format(kh,kU))
     kq=1
     kK=wc[kh:kh+kq]
     ke=[kL[wA]for wA in kK]
     kz=[kH[wA]for wA in kK]
     ka=FP(kz)
     x=[]
     for i,sentence in FQ(ke):
      kp=kz[i]
      wq=np.ones_like(sentence)
      wK=sentence-wq
      L1=we[:,wK]
      ko=ka-kp
      L2=np.zeros([kM,ko],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     kf=np.array(x)
     kd=np.array(kf,np.float64)
     x=[]
     kv=[kW[wA]for wA in kK]
     for i,sentence_fixed_tree_constructionorder in FQ(kv):
      kD=kz[i]-1
      ko=(ka-1)-kD
      L2=np.zeros([3,ko],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     kx=np.array(x)
     kE={wF.Fr:kd,wF.batch_real_sentence_length:kz,wF.batch_len:[kq],wF.batch_treeConstructionOrders:kx}
     ki,PK=OB.run([wF.tensorLoss_fixed_tree,wF.batch_sentence_vectors],feed_dict=kE)
     OI=PK[0,:,:]
     wQ=kz[0]
     OK=2*wQ-1
     OI=OI.astype(np.float32)
     OI=np.transpose(OI)
     OY=BU(OI)
     OS=ky[kh]
     Br[OS]=OY
  Bn={}
  By={}
  for i,wy in FQ(wF.need_vectorTree_lines_for_trainCorpus):
   By[wy]=i
  for BN in wF.need_vectorTree_ids_for_trainCorpus:
   wy=wF.id_line_dict[BN]
   BR=By[wy]
   BV=Br[BR]
   Bn[BN]=BV
  wF.save_to_pkl(Bn,Bs)
  FC(Bs)
  pass
  return 
 def using_model_for_BigCloneBench_experimentID_3(wF):
  BX=0 
  OA='./vector/'+Fs(BX)+'_using_Weigthed_RAE_fullCorpusLine_Map_Vector_root.xiaojiepkl'
  if os.path.exists(OA): 
   os.remove(OA) 
  else:
   FC('no such file:%s'%OA)
  Os='./vector/'+Fs(BX)+'_using_Weigthed_RAE_fullCorpusLine_Map_Vector_mean.xiaojiepkl'
  if os.path.exists(Os): 
   os.remove(Os) 
  else:
   FC('no such file:%s'%Os)
  BL=0
  kN=wF.fullCorpus
  kC=wF.full_corpus_fixed_tree_constructionorder
  kc=wF.fullCorpus_sentence_length
  OR=wF.full_corpus_fixed_tree_parentType_weight
  OV={}
  OX={}
  del(wF.fullCorpus)
  del(wF.fullCorpus_sentence_length)
  del(wF.full_corpus_fixed_tree_constructionorder)
  with tf.Graph().as_default():
   wF.xiaojie_RvNN_fixed_tree_for_usingmodel() 
   kt=tf.train.Saver()
   kB=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=Fk))
   kB.gpu_options.allocator_type='BFC'
   with tf.Session(config=kB)as OB:
    Oj='./weights/%s'%wF.config.model_name
    kt.restore(OB,Oj)
    kr=300; 
    wB.log('设置长短的衡量标准是{}'.format(kr))
    kn=[]
    ky=[]
    for wA,length in FQ(kc):
     if length<kr:
      kn.append(wA)
     else:
      ky.append(wA)
    wB.log("较长的句子{}个".format(BY(ky)))
    wB.log("较短的句子{}个".format(BY(kn)))
    kR=[kN[wA]for wA in kn]
    kV=[kc[wA]for wA in kn]
    kX=[kC[wA]for wA in kn]
    OM=[OR[wA]for wA in kn]
    kL=[kN[wA]for wA in ky]
    kH=[kc[wA]for wA in ky]
    kW=[kC[wA]for wA in ky]
    OT=[OR[wA]for wA in ky]
    wB.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
    wB.log("先处理较短的句子的语料，批处理开始")
    wN=BY(kn)
    we=np.array(wF.We)
    del(wF.We)
    kM=we.shape[0]
    wc=BU(BI(wN))
    kT=wF.config.batch_size_using_model_notTrain
    OJ=(wN-1)/kT 
    kJ=0
    for kh in BI(0,wN,kT):
     wB.log("batch_index:{}/{}".format(kJ,OJ))
     kq=Fu(kh+kT,wN)-kh
     kK=wc[kh:kh+kq]
     ke=[kR[wA]for wA in kK]
     kz=[kV[wA]for wA in kK]
     ka=FP(kz)
     x=[]
     for i,sentence in FQ(ke):
      kp=kz[i]
      wq=np.ones_like(sentence)
      wK=sentence-wq
      L1=we[:,wK]
      ko=ka-kp
      L2=np.zeros([kM,ko],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     kf=np.array(x)
     kd=np.array(kf,np.float64)
     x=[]
     kv=[kX[wA]for wA in kK]
     for i,sentence_fixed_tree_constructionorder in FQ(kv):
      kD=kz[i]-1
      ko=(ka-1)-kD
      L2=np.zeros([3,ko],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     kx=np.array(x)
     kE={wF.Fr:kd,wF.batch_real_sentence_length:kz,wF.batch_len:[kq],wF.batch_treeConstructionOrders:kx}
     ki,PK=OB.run([wF.tensorLoss_fixed_tree,wF.batch_sentence_vectors],feed_dict=kE)
     x=[]
     OE=[OM[wA]for wA in kK]
     for i,Oi in FQ(OE):
      Ob=np.FA(Oi)
      Oi=Oi/Ob
      Om=kz[i]-1
      ko=(ka-1)-Om
      L2=np.zeros([ko],np.int32)
      L=np.concatenate((Oi,L2))
      x.append(L)
     Ot=np.array(x)
     Ol=0
     for wA in kK:
      OU=PK[Ol,:,:]
      Oi=Ot[Ol]
      wQ=kz[Ol]
      OK=2*wQ-1
      OI=OU[0:wF.config.embed_size,0:OK]
      OI=OI.astype(np.float32)
      Ol=Ol+1
      OI=np.transpose(OI)
      OY=BU(OI)
      OS=kn[wA]
      OV[OS]=OY[OK-1]
      OS=kn[wA]
      Og=np.zeros_like(OY[0],np.float32)
      for i in BI(wQ,OK):
       Bw=OY[i]
       j=i-wQ
       BP=Oi[j]
       Bw=np.multiply(Bw,BP)
       Og=np.add(Og,Bw)
      OX[OS]=Og
     if(BL>70000):
      wF.save_to_pkl(OV,OA)
      wF.save_to_pkl(OX,Os)
      BL=0
      del(OV)
      del(OX)
      OV={}
      OX={}
      BX+=1 
      OA='./vector/'+Fs(BX)+'_fullCorpusLine_Map_Vector_root.xiaojiepkl'
      if os.path.exists(OA): 
       os.remove(OA) 
      else:
       FC('no such file:%s'%OA)
      Os='./vector/'+Fs(BX)+'_fullCorpusLine_Map_Vector_mean.xiaojiepkl'
      if os.path.exists(Os): 
       os.remove(Os) 
      else:
       FC('no such file:%s'%Os)
     kJ=kJ+1
     BL=BL+kq
    wF.save_to_pkl(OV,OA)
    wF.save_to_pkl(OX,Os)
    del(OV)
    del(OX)
    OV={}
    OX={}
    BX+=1 
    OA='./vector/'+Fs(BX)+'_fullCorpusLine_Map_Vector_root.xiaojiepkl'
    if os.path.exists(OA): 
     os.remove(OA) 
    else:
     FC('no such file:%s'%OA)
    Os='./vector/'+Fs(BX)+'_fullCorpusLine_Map_Vector_mean.xiaojiepkl'
    if os.path.exists(Os): 
     os.remove(Os) 
    else:
     FC('no such file:%s'%Os)
    wB.log("再处理较长的句子的语料，每个句子单独处理，开始") 
    kU=BY(ky)
    wc=BU(BI(kU))
    for kh,sentence in FQ(kL):
     wB.log("long_setence_index:{}/{}".format(kh,kU))
     kq=1
     kK=wc[kh:kh+kq]
     ke=[kL[wA]for wA in kK]
     kz=[kH[wA]for wA in kK]
     ka=FP(kz)
     x=[]
     for i,sentence in FQ(ke):
      kp=kz[i]
      wq=np.ones_like(sentence)
      wK=sentence-wq
      L1=we[:,wK]
      ko=ka-kp
      L2=np.zeros([kM,ko],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     kf=np.array(x)
     kd=np.array(kf,np.float64)
     x=[]
     kv=[kW[wA]for wA in kK]
     x=[]
     OE=[OT[wA]for wA in kK]
     for i,Oi in FQ(OE):
      Ob=np.FA(Oi)
      Oi=Oi/Ob
      Om=kz[i]-1
      ko=(ka-1)-Om
      L2=np.zeros([ko],np.int32)
      L=np.concatenate((Oi,L2))
      x.append(L)
     Ot=np.array(x)
     for i,sentence_fixed_tree_constructionorder in FQ(kv):
      kD=kz[i]-1
      ko=(ka-1)-kD
      L2=np.zeros([3,ko],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     kx=np.array(x)
     kE={wF.Fr:kd,wF.batch_real_sentence_length:kz,wF.batch_len:[kq],wF.batch_treeConstructionOrders:kx}
     ki,PK=OB.run([wF.tensorLoss_fixed_tree,wF.batch_sentence_vectors],feed_dict=kE)
     OI=PK[0,:,:]
     Oi=Ot[0]
     wQ=kz[0]
     OK=2*wQ-1
     OI=OI.astype(np.float32)
     OI=np.transpose(OI)
     OY=BU(OI)
     OS=ky[kh]
     OV[OS]=OY[OK-1]
     OS=kn[wA]
     Og=np.zeros_like(OY[0],np.float32)
     for i in BI(wQ,OK):
      Bw=OY[i]
      j=i-wQ
      BP=Oi[j]
      Bw=np.multiply(Bw,BP)
      Og=np.add(Og,Bw)
     OX[OS]=Og
  wF.save_to_pkl(OV,OA)
  wF.save_to_pkl(OX,Os)
  FC(OA)
  FC(Os)
  del(OV)
  del(OX)
  pass
  return 
 def save_to_pkl(wF,BH,pickle_name):
  with BS(pickle_name,'wb')as pickle_f:
   pickle.dump(BH,pickle_f)
 def read_from_pkl(wF,pickle_name):
  with BS(pickle_name,'rb')as pickle_f:
   BH=pickle.load(pickle_f)
  return BH 
 def similarities(wF,kN,kc,kC,Oj):
  wB.log('对语料库计算句与句的相似性') 
  wB.log('被相似计算的语料库一共{}个sentence'.format(BY(kN)))
  OC=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
  BW=OC+'.xiaojiepkl'
  if os.path.exists(BW): 
   os.remove(BW) 
  else:
   FC('no such file:%s'%BW)
  Br={}
  with tf.Graph().as_default():
   wF.xiaojie_RvNN_fixed_tree_for_usingmodel()
   kt=tf.train.Saver()
   kB=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=Fk))
   kB.gpu_options.allocator_type='BFC'
   with tf.Session(config=kB)as OB:
    kt.restore(OB,Oj)
    kr=1000; 
    wB.log('设置长短的衡量标准是{}'.format(kr))
    kn=[]
    ky=[]
    for wA,length in FQ(kc):
     if length<kr:
      kn.append(wA)
     else:
      ky.append(wA)
    wB.log("较长的句子{}个".format(BY(ky)))
    wB.log("较短的句子{}个".format(BY(kn)))
    kR=[kN[wA]for wA in kn]
    kV=[kc[wA]for wA in kn]
    kX=[kC[wA]for wA in kn]
    kL=[kN[wA]for wA in ky]
    kH=[kc[wA]for wA in ky]
    kW=[kC[wA]for wA in ky]
    wB.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
    wB.log("先处理较短的句子的语料，批处理开始")
    wN=BY(kn)
    we=np.array(wF.We)
    kM=we.shape[0]
    wc=BU(BI(wN))
    kT=wF.config.batch_size_using_model_notTrain
    OJ=(wN-1)/kT 
    kJ=0
    for i in BI(0,wN,kT):
     wB.log("batch_index:{}/{}".format(kJ,OJ))
     kq=Fu(i+kT,wN)-i
     kK=wc[i:i+kq]
     ke=[kR[wA]for wA in kK]
     kz=[kV[wA]for wA in kK]
     ka=FP(kz)
     x=[]
     for i,sentence in FQ(ke):
      kp=kz[i]
      wq=np.ones_like(sentence)
      wK=sentence-wq
      L1=we[:,wK]
      ko=ka-kp
      L2=np.zeros([kM,ko],np.float64)
      L=np.concatenate((L1,L2),axis=1)
      x.append(L)
     kf=np.array(x)
     kd=np.array(kf,np.float64)
     x=[]
     kv=[kX[wA]for wA in kK]
     for i,sentence_fixed_tree_constructionorder in FQ(kv):
      kD=kz[i]-1
      ko=(ka-1)-kD
      L2=np.zeros([3,ko],np.int32)
      L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
      x.append(L)
     kx=np.array(x)
     kE={wF.Fr:kd,wF.batch_real_sentence_length:kz,wF.batch_len:[kq],wF.batch_treeConstructionOrders:kx}
     ki,PK=OB.run([wF.tensorLoss_fixed_tree,wF.batch_sentence_vectors],feed_dict=kE)
     Ol=0
     for wA in kK:
      OU=PK[Ol,:,:]
      wQ=kz[Ol]
      OK=2*wQ-1
      OI=OU[0:wF.config.embed_size,0:OK]
      OI=OI.astype(np.float32)
      Ol=Ol+1
      OI=np.transpose(OI)
      OY=BU(OI)
      OS=kn[wA]
      Br[OS]=OY 
     kJ=kJ+1
    wB.log("再处理较长的句子的语料，每个句子单独处理，开始") 
    kU=BY(ky)
    for wA,sentence in FQ(kL):
     wB.log("long_setence_index:{}/{}".format(wA,kU))
     Bj=kW[wA]
     (_,OI,kQ,_)=wF.computelossAndVector_no_tensor_withAST(sentence,Bj)
     OY=[]
     for kk in BI(2*kH[wA]-1):
      BM=OI[kk]
      BM=BM[:,0]
      BM=BM.astype(np.float32)
      OY.append(BM)
     OS=ky[wA]
     Br[OS]=OY 
    with BS(BW,'wb')as f:
     pickle.dump(Br,f)
    wB.log('相似性计算结束后，corpus的所有句子的最终向量表示(向量树)存储的位置是为%s'%BW)
def test_RNN():
 Fr("开始？")
 wB.log("------------------------------\n程序开始")
 kB=wP()
 BT=wO(kB)
 BJ='./weights/%s'%BT.config.model_name
 BT.similarities(corpus=BT.fullCorpus,corpus_sentence_length=BT.fullCorpus_sentence_length,weights_path=BJ,corpus_fixed_tree_constructionorder=BT.full_corpus_fixed_tree_constructionorder)
 wB.log("程序结束\n------------------------------")
from train_traditional_RAE_configuration import configuration
def xiaojie_RNN_1():
 wB.log("------------------------------\n为模型加载训练样本集合，并配置参数")
 wG=configuration
 FC(wG) 
 kB=wP(wG)
 BT=wO(kB,experimentID=1)
 BT.using_model_for_BigCloneBench_experimentID_1()
def xiaojie_RNN_2():
 wB.log("------------------------------\n为模型加载训练样本集合，并配置参数")
 wG={}
 wG['label_size']=2
 wG['early_stopping']=2 
 wG['max_epochs']=30
 wG['anneal_threshold']=0.99
 wG['anneal_by']=1.5
 wG['lr']=0.01
 wG['l2']=0.02
 wG['embed_size']=300
 wG['model_name']='experimentID_1_rnn_embed=%d_l2=%f_lr=%f.weights'%(wG['embed_size'],wG['lr'],wG['l2'])
 wG['IDIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/2word2vecOutData/'
 wG['ODIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/3RvNNoutData/'
 wG['corpus_fixed_tree_constructionorder_file']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/corpus_bcb_reduced.method.AstConstruction.txt'
 wG['MAX_SENTENCE_LENGTH']=1000000
 wG['batch_size']=10 
 wG['batch_size_using_model_notTrain']=300 
 wG['MAX_SENTENCE_LENGTH_for_Bigclonebench']=600 
 kB=wP(wG)
 BT=wO(kB,experimentID=2)
 BT.using_model_for_BigCloneBench_experimentID_2()
def xiaojie_RNN_3():
 wB.log("------------------------------\n为模型加载训练样本集合，并配置参数")
 wG={}
 wG['label_size']=2
 wG['early_stopping']=2 
 wG['max_epochs']=30
 wG['anneal_threshold']=0.99
 wG['anneal_by']=1.5
 wG['lr']=0.01
 wG['l2']=0.02
 wG['embed_size']=296
 wG['model_name']='weighted_RAE_rnn_embed=%d_l2=%f_lr=%f.weights'%(wG['embed_size'],wG['lr'],wG['l2'])
 wG['IDIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/2word2vecOutData/'
 wG['ODIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/3RvNNoutData/'
 wG['corpus_fixed_tree_constructionorder_file']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/corpus_bcb_reduced.method.AstConstruction.txt'
 wG['MAX_SENTENCE_LENGTH']=1000000
 wG['batch_size']=10 
 wG['batch_size_using_model_notTrain']=400 
 wG['MAX_SENTENCE_LENGTH_for_Bigclonebench']=300 
 wG['corpus_fixed_tree_construction_parentType_weight_file']='./1corpusData/corpus_bcb_reduced.method.AstConstructionParentTypeWeight.txt'
 kB=wP(wG)
 BT=wO(kB,experimentID=3)
 BT.using_model_for_BigCloneBench_experimentID_3()
def verification_corpus():
 def linesOfFile(filepath):
  Bh=0
  with BS(filepath,'r')as fw:
   for i,Bq in FQ(fw,start=1): 
    Bh+=1
  return Bh
 Be='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/2word2vecOutData/corpus.int'
 Bz='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/corpus_bcb_reduced.method.txt'
 Ba='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/corpus_bcb_reduced.method.AstConstruction.txt'
 Bp='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/writerpath_bcb_reduced.method.txt'
 Bo=linesOfFile(Be)
 Bf=linesOfFile(Bz)
 Bd=linesOfFile(Ba)
 Bv=linesOfFile(Bp)
 FC(Bo,Bf,Bd,Bv)
 with BS(Be,'r')as f1:
  with BS(Bz,'r')as f2:
   with BS(Ba,'r')as f3:
    with BS(Bp,'r')as f4:
     for i in BI(Bo):
      BD=f1.readline()
      Bx=f2.readline()
      BE=f3.readline()
      Bb=f4.readline()
      Bi=BD.strip().split()
      Bm=Bx.strip().split()
      Bt=BE.strip('\n').strip(' ').split(' ')
      if((BY(Bi))!=(BY(Bm))):
       FC(Bb)
       FC('在corpus.int中的长度{}，同在txt中的长度{}不一致。'.format(BY(Bi),BY(Bm)))
       Fr()
       return 
      if((BY(Bi))!=(1+(BY(Bt)))):
       FC(Bb)
       FC('句子单词长度{}，不等于构建次数{}+1，'.format(BY(Bi),BY(Bt)))
       Fr()
       return 
      pass
 FC('校验完毕，没发现问题')
 return 
def save_to_pkl(BH,pickle_name):
 with BS(pickle_name,'wb')as pickle_f:
  pickle.dump(BH,pickle_f)
def read_from_pkl(pickle_name):
 with BS(pickle_name,'rb')as pickle_f:
  BH=pickle.load(pickle_f)
 return BH 
if __name__=="__main__":
 xiaojie_RNN_1()
# Created by pyminifier (https://github.com/liftoff/pyminifier)
