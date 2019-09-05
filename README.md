# Fast Clone Detection Based on Weighted Recursive Autoencoders

use Weighted Recursive Autoencoders to learn the target software system, model all functions to vectors, and at last use NSG algorithm to detect clone Pair. ***Some core codes are obscured***. Every question if you encounter, you can directyly concat my email zyj183247166@qq.com. 

Because the GitHub has storage space constraints, the data after the step 1-3 below are shared at the Baidu Netdisk with the link and password as:
LINK：https://pan.baidu.com/s/1zaX1YMLmLsr3EKXK8nrOjA 
PASSWORD：tzib 

And, the database of BigCloneBench is processed and the data about the TrueClonePairs, FalseClonePairs and CloneTypoes are shared at the Baidu Netdisk with the link and password as:

LINK：https://pan.baidu.com/s/1S7iQnsjJgHnsh5NzHc196Q 
PASSWORD：whtc 

After download the two dataset in the above links, there are two folders named as "1corpusData" and "SplitDataSet". Please copy them to the folder "Recursive_autoencoder_xiaojie_296_dimension" as two subfolders. Otherwise, some programs cannot be runned correctly without these data.

# 0 Acknowledgement

I very appreciate Martin White,  Cong Fu and Jeffrey Svajlenko for offering much help about ast2bin, NSG and  BigCloneBench respectively. I email a lot to them for some help. 
ast2bin:[https://github.com/micheletufano/ast2bin](https://github.com/micheletufano/ast2bin)
NSG:https://github.com/ZJULearning/nsg
BigCloneBench:[https://github.com/jeffsvajlenko/BigCloneEval](https://github.com/jeffsvajlenko/BigCloneEval)

# 1 Hardware and software environment configuration

## Hardware 

 1. Intel(R) Core(TM) i7-8700K CPU
 2. NVIDIA GeForce GTX 1080 Ti独立显卡
 3. RAM 32.00GB
	RAM should be no less than 32.00GB, otherwise the experiment will collapse.

## software environment configuration
![image](https://github.com/zyj183247166/Recursive_autoencoder_xiaojie/blob/master/1.PNG)

# 2 Steps of experiment

workspace is: Recursive_autoencoder_xiaojie_256_dimension


## Step 1: process the ASTs of programs, get full binary trees and corpus of sentences at function granularity

 1. the ast_bin_xiaojie_generating_train_corpus.jar is an executalbe jar. it  improves a lot on the basis of [ast2bin](https://github.com/micheletufano/ast2bin). The improvements include: adding seven binary grammar, store the architecture of full binary trees in specific format, extract the functions' locations consisting of the startline and endline in their Java Files.
 2. run the following command:
	> java -Xmx20000m -Xss1024m -jar ast_bin_xiaojie_generating_train_corpus.jar -t generate-method-corpus-ast-xiaojie E:\bcb_reduced\ G:\data_of_zengjie\analysisBigCloneBench\method\corpus G:\data_of_zengjie\analysisBigCloneBench\method\writerpath 10 0 100000
 3. about the args
	 1. E:\bcb_reduced\ is directory of the source  repository of [BigCloneBench](https://github.com/jeffsvajlenko/BigCloneEval)
	 2. 'G:\data_of_zengjie\analysisBigCloneBench\method\corpus' specify the following:
	
		-the generated corpus of sentences are stored in G:\data_of_zengjie\analysisBigCloneBench\method\
		-the file storing generated corpus of sentences has a filename with 'corpus' as the prefix
	 3. 'G:\data_of_zengjie\analysisBigCloneBench\method\writerpath' specify the following:
		
		-the file recording all functions' locations are stored in G:\data_of_zengjie\analysisBigCloneBench\method\
		-the file has a filename with 'writerpath' as the prefix
	4. 0 100000 respectively denotes that the analysed function should have lens larger than 0 and lower than 100000. Of course, they can be modified.
 4. about the output 
		under the directory will exist the following files.
	![image](https://github.com/zyj183247166/Recursive_autoencoder_xiaojie/blob/master/2.png)
	 - *corpus_bcb_reduced.method.txt* records the corpus of sentences of which
	   every sentence are extracted from a corresponding function.
	 - *corpus_bcb_reduced.method.AstConstruction*  records all sentences' full binary trees transformed from the corresponding functions.
	 - *notProcessedToCorpus_files_bcb_reduced.method* records all functions which are not processed and the reasons why they are not processed.
	 - *writerpath_bcb_reduced.method* record every function's location in corresponding Java File.
 5. matters  needing  attention
	
	the folder *G:\data_of_zengjie\analysisBigCloneBench\method\* must be built  in advance, otherwise it will prompt the error of not finding the path.
## Step 2: copy the following files above to  ***\Recursive_autoencoder_xiaojie_256_dimension\1corpusData***
 1. corpus_bcb_reduced.method.txt
 2. corpus_bcb_reduced.method.AstConstruction
 3. notProcessedToCorpus_files_bcb_reduced.method

## Step 3: computing all TF-IDFs of those nonTerminal Types.

 run AnalyseTFIDFofNonTerminalType.py. Another two files will be produced in ***\Recursive_autoencoder_xiaojie_256_dimension\1corpusData***
![image](https://github.com/zyj183247166/Recursive_autoencoder_xiaojie/blob/master/3.png)

 1. *Corpus_bcb_reduced.Method.AstConstructionParentType.TXT* record all types of nonterminal nodes corresponding to every function's full
    binary tree. (in post-order traversal)
 2. *corpus_bcb_reduced.method.AstConstructionParentTypeWeight.txt* record all TF-IDFs of  nonterminal types corresponding to every
        function's full binary tree.(in post-order traversal)

## Step 4: training word2vec on the corpus.

1. Install cygwin on windows10. Pay attention to install 'make' packages during the installation process

2. start cygwin and install word2vec.
	> cd '\Recursive_autoencoder_xiaojie_256_dimension\word2vec'
	make
3. start cygwin and train word2vec on the corpus.
	> ./run_word2vec.sh ./1corpusData/corpus_bcb_reduced.method.txt ./2word2vecOutData 296

	296 represents the dimensions of the word vector.The output is '*./2 word2vecoutdata/word2vec.Out*'
	![image](https://github.com/zyj183247166/Recursive_autoencoder_xiaojie/blob/master/4.png)
	![image](https://github.com/zyj183247166/Recursive_autoencoder_xiaojie/blob/master/5.png)
## Step 5: process word2vec.Out and the corpus file.
1. Start terminal from anaconda
2. run the command 

> cd \Recursive_autoencoder_xiaojie_256_dimension\
> python ./run_postprocess.py --w2v ./2word2vecOutData/word2vec.out  --src ./1corpusData/corpus_bcb_reduced.method.txt
3. the another  three files will be produced:
	![image](https://github.com/zyj183247166/Recursive_autoencoder_xiaojie/blob/master/6.png)
## Step 6: train Recursive AutoEncoders.
	
1. configure in **train_traditional_RAE_configuration.py**
> configuration={}
configuration['label_size']=2
configuration['early_stopping']=5
configuration['max_epochs']=30
configuration['anneal_threshold']=0.99
configuration['anneal_by']=1.5
configuration['lr']=0.01
configuration['l2']=0.02
configuration['embed_size']=296
configuration['model_name']='traditional_RAE_embed=%d_l2=%f_lr=%f.weights'%(configuration['embed_size'], configuration['lr'], configuration['l2'])
configuration['IDIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie_296_dimension/2word2vecOutData/'
configuration['ODIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie_296_dimension/3RvNNoutData/'
configuration['corpus_fixed_tree_constructionorder_file']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie_296_dimension/1corpusData/corpus_bcb_reduced.method.AstConstruction.txt'
configuration['batch_size']=10 
configuration['batch_size_using_model_notTrain']=400 
configuration['MAX_SENTENCE_LENGTH_for_Bigclonebench']=600
configuration['corpus_fixed_tree_construction_parentType_weight_file']='./1corpusData/corpus_bcb_reduced.method.AstConstructionParentTypeWeight.txt'
2. run the training program
> python ./1_train_traditional_RAE_on_BigCloneBench_hunxiao.py
3. The training results

![image](https://github.com/zyj183247166/Recursive_autoencoder_xiaojie/blob/master/7.png)
![image](https://github.com/zyj183247166/Recursive_autoencoder_xiaojie/blob/master/8.png)
## Step 7: compare the traditional and weighted RAE on the BigCloneBench data set
1. dataset preparation
	
	I preprocessed BigCloneBench , mainly analyzes the CLONES table and FALSEPOSITIVES table in this database.The former stores positive clones, while the latter stores negative clones.
Through the analysis , we found that 25 functions in BigCloneBench were incorrectly marked. See ***functions_with_wrong_location_bigclonebench.txt*** for details.
	In addition, we removed duplicate clone pairs marked by BigCloneBench. See ***Duplicate_clone_pair_record.txt*** file.
Finally, the remaining BigCloneBench clone pairs (positive or negative labels)  are stored into ***all_pairs_id_xiaoje.pkl***.
	We store each clone pair's corresponding clone type in the ***all_clone_id_pair_clonetype_xiaojie.pkl*** file.
	Most importantly, the functions' Numbers marked in BigCloneBench are inconsistent with those in our corpus ***corpus_bcb_reduce.method.txt***. We map the function Numbers in BigCloneBench to our function Numbers (lines in the txt) in ***corpus_bcb_reduce.method.txt*** and storing them in the ***all_idmapline_xiaojie.pkl*** file.

2. using two different models to obtain the vector representation for each function in BigCloneBench respectively.
	In Anaconda, starts the python3.6.5 environment and then runs the command
	>     python. /1 _2_using_traditional_rae_on_bigclonebench_hunxiao.py
	The results calculated for each vector by **unweighted RAE** are stored in:

	    ./vector/tkqnm_BigCloneBench_traditionalRAE_ID_Map_Vector_root. Xiaojiepkl

	You can query with function ID in BigCloneBench as the key value. ***'TKQNM'*** is a random name that changes with each execution.
		then runs the command
     `python. /1_3_using_weighted_rae_on_bigclonebench_hunxiao.py`
	The results calculated for each vector by **weighted RAE** are stored in:
***`./vector/childparentweight/enoyz_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_weighted.xiaojiepkl`***
'enoyz' is a random name that changes with each execution.
3. getting the indicators two models.
run the ***3_evaluate_on_BigCloneBench.py*** file
the evaluation results of two models set with different distance thresholds are saved into: ***./result/unweighted_BigCloneBench_traditionalRAE_metrics_root.xiaojiepkl***
and 
***./result/weighted_BigCloneBench_traditionalRAE_metrics_root_TF-IDF.xiaojiepkl***
4. visualize the indicators of two models.

	Let's go ahead and run

    `python 4_visualizeTheIndicatorsOnBigCloneBench.py`
## Step 8: directly apply the weighted RAE to *bcd_reduced* source code library
1. Generates a vector representation for each function in the bcd_reduced source library
		
	run the below program.
	> python 1_4_using_weighted_rae_on_bcd_reduced.py
	
	As the program is run with too many parameters, it is impossible to save all the vectors of functions in one time. 					
	We will save multiple PKL files into the vector directory in batches, and the naming rule is

	***0_fullcorpusline_map_vector_mean***

	***1_fullcorpusline_map_vector_mean***

	....

	A series of PKL files are generated.We then run the file ***6_mergepkl.py***
	we then Merge all PKL files and generate one:
	***using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.xiaojiepkl*** and along with 	***Using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.npy***
	The row number of the matrix corresponds to the function number in the whole corpus. There are 785438 lines in total, and each line of vector represents the corresponding function.

## Step 9: use NSG algorithm for cloning detection and report the results, , evaluating its scalability and geting results.

We copy ***using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted. npy*** to the vmware virtual machine (Ubuntu 18 64bit) with an NSG project.
1. convert the npy file to fvec file using yael's fvecs_fwrite function
	***using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.fvec***
2. build the KNN diagram

> /home/xiaojie/Desktop/efanna_graph-master/tests/test_nndescent
> ./using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.fvecs
> ./using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.200NN.graph
> 200 200 8 10 100

3. transform KNN graph into NSG graph

> /home/xiaojie/Desktop/nsg/build/tests/test_nsg_index ./using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.fvecs ./using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.200NN.graph 40 300 500 ./using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.nsg

4. retrieves in the NSG graph and returns the 200 nearest neighbors of each vector.

> /home/xiaojie/Desktop/nsg/build/tests/test_nsg_optimized_search
> ./using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.fvecs
> ./using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.fvecs
> ./using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.nsg 201 200
> ./RESULT_using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.ivecs
5. copy RESULT_using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted. ivecs back to the nsg_result directory in the host's Windows system.
	The detection results under different thresholds are returned here. It's going to be a lot of computation and it's going to take some time.
6. run ***7_processing_nsgresult_tocloneresult_by_therold.py***
	it will produce some files.
	添加一些图片
	Each file records the detection results under the specified threshold.
7. run ***8_evaluate_cloneresultbyNSG_onBigCloneBench.py***
	the results will be stored into ***metrices_onFullCorpusWithNSG_WeightedRAE_differentTherolds.xiaojiepkl***
	Then you can compare it with directly using weighted RAE on tagged clone pairs in BigCloneBench by executing the ***experimentID_6()*** in this python file.
	It can be seen that, with the increase of threshold value, due to the limitation of the number of nearest neighbors searched by NSG, the recall rate of the algorithm using NSG in the later stage is far less than that of the algorithm without NSG. But we donot need to set the distance threshold larger because it will also produce more fasle positives.
## END (the experiments in the corresponding paper).
