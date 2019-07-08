#!/usr/bin/python

import optparse
import os

class Word2vecOutput(object):
    def __init__(self, path):
        self._path = path
        self._vocab = {}
        zero_vector_flag=0
        zero_vector=None
        zero_vector_len=None
        with open(self._path) as fw:
            with open(self.DIR + 'vocab.txt', 'w') as fv:
                with open(self.DIR + 'embed.txt', 'w') as fe:
                    next(fw)  # Retrieve header
                    for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
                        #print (i,item)
                        word, embedding = item.strip().split(' ', 1)
                        if zero_vector_flag==0:
                            zero_vector_len=len(embedding.strip().split(' '))
                            empty=[]
                            for i in range(zero_vector_len):
                                empty.append('0.0')
                            zero_vector=(' '.join(empty))
                            zero_vector=zero_vector.strip()
                            zero_vector_flag=1
                        fv.write(word + '\n')
                        fe.write(embedding + '\n')
                        self._vocab[word] = i  # Map words to ints
                    fv.write('unknow_xiaojie'+'\n')
                    fe.write(zero_vector + '\n')
                    
    @property
    def DIR(self):
        return os.path.dirname(self._path) + os.sep

    @property
    def vocab(self):
        return self._vocab

class Corpus(object):
    def __init__(self, src_dir, int_dir, granularities=['corpus']):
        self._src_dir = src_dir
        self._int_dir = int_dir
        self._granularities = granularities

    def transform(self, vocab):
        for granularity in self._granularities:
            #src_path = self._src_dir + os.sep + granularity + '.src'
            src_path = self._src_dir
            int_path = self._int_dir + os.sep + granularity + '.int'
            with open(src_path) as fi:
                with open(int_path, 'w') as fo:
                    for line in fi:
                        words = line.strip().split()
#                        fo.write(' '.join(str(vocab[w]) for w in words) + '\n')
                        for i in range(len(words)):
                            word=words[i]    
                            if word not in vocab:
                                print ('word2vec没有训练出词向量的单词:')
                                print (word)
                                #对于这些word2vec没有训练出词向量的单词，我们直接用单词编号为0.后期取单词向量的时候，我们判定是否为0，如果为0，直接取0向量
                                
                                fo.write((str(0)))
                                fo.write(' ')
                                #后续取语料的时候，我们就可以直接用下标减去1去取词向量。当下标为0，减去1的时候，就是取列表的最后一个元素，此时，我们就在embedding和vocab的两个文件中
                                #认为添加零向量和单词"unknow"作为末尾。就能保证整个系统仍然能够正确运行。
                                
                            else:
                                fo.write(str(vocab[word]))
                                fo.write(' ')
                        fo.write('\n')
if __name__ == '__main__':
    # Build *.int
    parser = optparse.OptionParser()
    parser.add_option('--w2v', help='/path/to/word2vec.out')
    parser.add_option('--src', help='/path/to/*.src')
    (options, args) = parser.parse_args()

    word2vec_output = Word2vecOutput(options.w2v)

    corpus = Corpus(options.src, word2vec_output.DIR)
    corpus.transform(word2vec_output.vocab)

