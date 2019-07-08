# -*- coding: utf-8 -*-
"""
@author: 曾杰
"""
corpus_fixed_tree_constructionorder_file='./1corpusData/corpus_bcb_reduced.method.AstConstruction.txt'
corpus_fixed_tree_construction_parentType_file='./1corpusData/corpus_bcb_reduced.method.AstConstructionParentType.txt'
def get_all_sentence_parentTypes(corpus_fixed_tree_constructionorder_file,corpus_fixed_tree_construction_parentType_file):
    with open(corpus_fixed_tree_constructionorder_file, 'r') as fw:
        with open(corpus_fixed_tree_construction_parentType_file, 'w') as fv:
            for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
                sentence_fixed_tree_constructionorder_list=item.strip('\n').strip(' ').split(' ')
                sentence_fixed_tree__parentTypes=[]
                for j in range(len(sentence_fixed_tree_constructionorder_list)):
                    one_time_construction_order=sentence_fixed_tree_constructionorder_list[j];
                    one_time_construction_order=one_time_construction_order.strip('(')
                    one_time_construction_order=one_time_construction_order.strip(')')
                    one_time_construction_order_list=one_time_construction_order.split(',')

                    parentType=one_time_construction_order_list[5]
                    sentence_fixed_tree__parentTypes.append(parentType)
                    pass
                #将非叶子节点类型写入文件
                for parentType in sentence_fixed_tree__parentTypes:
                    fv.write(parentType)
                    fv.write(' ')
                fv.write('\n')
                pass
                if(i%10000==0):
                    print("index:%d"%i)
    pass
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer

corpus=[]
with open(corpus_fixed_tree_construction_parentType_file, 'r') as fw:   
    for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
        corpus.append(item)
        if(i%10000==0):
            print("index:%d"%i)

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(corpus)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
feature_name = vectorizer.get_feature_names() 
feature_name_dict={}
index=0
for i in range(len(feature_name)):
    feature=feature_name[i]
    feature_name_dict[feature]=index
    index=index+1

corpus_fixed_tree_construction_parentType_weight_file='./1corpusData/corpus_bcb_reduced.method.AstConstructionParentTypeWeight.txt'
with open(corpus_fixed_tree_construction_parentType_weight_file,'w') as fv:  
    with open(corpus_fixed_tree_construction_parentType_file, 'r') as fw:   
        for i, item in enumerate(fw, start=0):  # MATLAB is 1-indexed
            sentence_parent_type_list=item.strip('\n').strip(' ').split(' ')

            for j in range(len(sentence_parent_type_list)):
                word=sentence_parent_type_list[j]
                word_lower=word.lower()#为什么要进行小写转换，就是因为前面计算tf-idf的时候，将所有的词都转换为了小写
                index=feature_name_dict[word_lower]

                weight=(tfidf[i,index])

                fv.write(str(weight)+' ')
                pass
            fv.write('\n')
            pass



