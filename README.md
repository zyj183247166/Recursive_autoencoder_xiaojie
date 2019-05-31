# Fast Clone Detection Based on Weighted Recursive Autoencoders

use Weighted Recursive Autoencoders to learn the target software system, model all functions to vectors, and at last use NSG algorithm to detect clone Pair.

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

添加一张图片

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
		添加一张图片
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
 添加一张图片

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
	添加两张图片
	添加两张图片
## Step 5: process word2vec.Out and the corpus file.
1. Start terminal from anaconda
2. run the command 
> cd \Recursive_autoencoder_xiaojie_256_dimension\
> python ./run_postprocess.py --w2v ./2word2vecOutData/word2vec.out  --src ./1corpusData/corpus_bcb_reduced.method.txt
3. the another  three files will be produced:
	添加一张图片
## Step 6: train traditional Recursive AutoEncoders.
	
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
2. run the training program
> python ./1_train_traditional_RAE_on_BigCloneBench_hunxiao.py

## Step 7: train weighted Recursive AutoEncoders.
1. configure in  **train_weighted_RAE_configuration.py**
 one more configuration than Step 6:
>**configuration['corpus_fixed_tree_construction_parentType_weight_file']='./1corpusData/corpus_bcb_reduced.method.AstConstructionParentTypeWeight.txt'**
2. run the training program
> python ./2_train_weighted_RAE_on_BigCloneBench_hunxiao.py
