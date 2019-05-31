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

 - the ast_bin_xiaojie_generating_train_corpus.jar is an executalbe jar. it  improves a lot on the basis of [ast2bin](https://github.com/micheletufano/ast2bin). The improvements include: adding seven binary grammar, store the architecture of full binary trees in specific format, extract the functions' locations consisting of the startline and endline in their Java Files.
 - run the following command:
	> java -Xmx20000m -Xss1024m -jar ast_bin_xiaojie_generating_train_corpus.jar -t generate-method-corpus-ast-xiaojie E:\bcb_reduced\ G:\data_of_zengjie\analysisBigCloneBench\method\corpus G:\data_of_zengjie\analysisBigCloneBench\method\writerpath 10 0 100000
 - about the args
	 1. E:\bcb_reduced\ is directory of the source  repository of [BigCloneBench](https://github.com/jeffsvajlenko/BigCloneEval)
	 2. 'G:\data_of_zengjie\analysisBigCloneBench\method\corpus' specify the following:
		-the generated corpus of sentences are stored in G:\data_of_zengjie\analysisBigCloneBench\method\
		-the file storing generated corpus of sentences has a filename with 'corpus' as the prefix
	 3. 'G:\data_of_zengjie\analysisBigCloneBench\method\writerpath' specify the following:
		-the file recording all functions' locations are stored in G:\data_of_zengjie\analysisBigCloneBench\method\
		-the file has a filename with 'writerpath' as the prefix
	4. 0 100000 respectively denotes that the analysed function should have lens larger than 0 and lower than 100000. Of course, they can be modified.
 - about the output 
		under the directory will exist the following files.
		添加一张图片
	 - *corpus_bcb_reduced.method.txt* records the corpus of sentences of which
	   every sentence are extracted from a corresponding function.
	 - *corpus_bcb_reduced.method.AstConstruction*  records all sentences' full binary trees transformed from the corresponding functions.
	 - *notProcessedToCorpus_files_bcb_reduced.method* records all functions which are not processed and the reasons why they are not processed.
	 - *writerpath_bcb_reduced.method* record every function's location in corresponding Java File.
