#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import csv, json
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import get_file
import codecs
import utils
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np
import json

'''
	训练数据及标签的准备
	vocab  字典  vacab['人']=0
	chars = ['人','民',...]
	tags = [0,1,2,3,...]------['B', 'M', 'E', 'S']
	charVec = [0,5,23,2,...]
	X = [[5170,5170,5170,45,5170,5170,5170],[ ],...]
	y = [0,1,2,3,...]------>[0,0,0,1]
'''

# 1 Initialize global variables
tags = ['B', 'M', 'E', 'S']
tagSize = len(tags)
base = './model/'
train_file = 'data/msr_training.utf8'
test_file = 'data/msr_test.utf8'
emb_file = 'embedding/vectors-300-Chinese.txt'
info_file = base + 'cws.info'
data_file = base + 'cws.data'
train_data_file = base + 'train.npy'
label_data_file = base + 'label.npy'
emb_matrix_file = base + 'word_embedding_matrix.npy'
nb_words_file = base + 'nb_words.json'
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 5
EMBEDDING_DIM = 300

start_C = [0 for i in range(tagSize)]  # 开始概率
trans_C = [[0 for j in range(tagSize)] for k in range(tagSize)]  # 转移概率


# 2 Build tokenized word index (建立字符索引字典)
def getDict(train_file, test_file, delimiters):
	print('\n获取索引字典......')
	vocab = dict()
	indexVocab = []
	# 一次性读入文件，注意内存
	with codecs.open(train_file, encoding='utf-8') as train_f:
		with codecs.open(test_file, encoding='utf-8') as test_f:
			data = train_f.read()
			data += test_f.read()
	for char in data:
		if char not in delimiters:
			if char in vocab:
				vocab[char] += 1
				indexVocab.append(char)
			else:
				vocab[char] = 1
		else:
			continue

	# 根据词频来确定每个词的索引
	wcounts = list(vocab.items())
	wcounts.sort(key=lambda x: x[1], reverse=True)
	sorted_voc = [wc[0] for wc in wcounts]
	# note that index 0 is reserved, never assigned to an existing word
	vocab = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

	# 加入未登陆新词和填充词
	vocab['retain-padding'] = 0
	vocab['retain-unknown'] = len(vocab)
	indexVocab.append('retain-unknown')
	indexVocab.append('retain-padding')
	print("Words in index: %d\n" % len(vocab))  # 5183
	return vocab, indexVocab


# def getDict2(fname):
# 	# '''利用 Tokenizer 将样本转化为神经网络训练所用的张量'''
# 	texts = []
# 	with open(fname) as f:
# 		texts.append(f.read())
# 	tokenizer = Tokenizer()
# 	tokenizer.fit_on_texts(texts)
#
# 	word_index = tokenizer.word_index
# 	print('Found %s unique tokens.' % len(word_index)) # 88111: 分词结果


# 3 process GloVe embeddings
def read_vec(GLOVE_FILE):
	print("Processing ", GLOVE_FILE)

	embeddings_index = dict()
	with codecs.open(GLOVE_FILE, encoding='utf-8') as f:
		for line in f:
			values = line.strip().split()
			if len(values) > 2:
				word = values[0]
				embedding = np.asarray(values[1:], dtype='float32')
				embeddings_index[word] = embedding
	print('Word embeddings: %d' % len(embeddings_index))
	return embeddings_index


# 4 Prepare word embedding matrix
def get_embedding_matrix(vocab, EMBEDDING_FILE):
	print('\nPreparing embedding matrix.')

	embeddings_index = read_vec(EMBEDDING_FILE)
	nb_words = min(MAX_NB_WORDS, len(vocab))
	print('\nnb_word: %d' % nb_words)  # 5183
	# 词向量矩阵随机初始化
	embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
	for word, i in vocab.items():
		if i == 0: continue
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		else:
			vec = np.random.uniform(-1, 1, size=EMBEDDING_DIM)  # 随机初始化
			embedding_matrix[i] = vec
	return nb_words, embedding_matrix


'''
将一行转换为转化为词索引序列
在分词中，一个词的label受上下文影响很大，因此我们将一个长度为n个字符的输入文本处理成n个长度为k的向量，k为奇数。
举例来说，当k=7时，表示考虑了一个字前3个字和后三个字的上下文，将这个七个字作为一个输入，输出就是这个字的label类型（BEMS）。
'''
def sent2vec2(sent, vocab, ctxWindows):
	X = []
	charVec = []
	for char in sent:
		if char in vocab:
			charVec.append(vocab[char])
		else:
			charVec.append(vocab['retain-unknown'])
	num = len(charVec)
	pad = int((ctxWindows - 1) / 2)
	# 首尾 padding
	for i in range(pad):
		charVec.insert(0, vocab['retain-padding'])
		charVec.append(vocab['retain-padding'])
	for i in range(num):
		X.append(charVec[i:i + ctxWindows])
	return X


# 5、将文本转化为词索引序列
def doc2vec(train_file, vocab, ctxWindows=5):
	print('\nPreparing data tensors')
	line_num = -1
	x_train, y_train = [], []
	# train_label = []
	# 读入文件，注意内存
	with codecs.open(train_file, 'r', 'utf-8') as fd:
		lines = fd.readlines()
	for line in lines:
		line_num += 1
		if line_num % 10000 == 0:
			print(line_num)

		data_line = []
		label_line = []
		split_line = line.split('  ')
		for word in split_line:  # 遍历词
			if len(word) > 1:
				# 词的首字
				data_line.append(word[0])
				label_line.append(tags.index('B'))  # 0
				# train_label.append(0)
				# 词中间的字
				for char in word[1:(len(word) - 1)]:
					data_line.append(char)
					label_line.append(tags.index('M'))  # 1
					# train_label.append(1)
				# 词的尾字
				data_line.append(word[-1])
				label_line.append(tags.index('E'))  # 2
				# train_label.append(2)
			else:  # 单字词
				data_line.append(word)
				label_line.append(tags.index('S'))  # 3
				# train_label.append(3)
		index_line = sent2vec2(data_line, vocab, ctxWindows)
		x_train.extend(index_line)
		y_train.extend(label_line)

		# 计算概率 A B
		if len(data_line) != len(label_line):
			print("[line_num = %d][line = %s]" % (line_num, line))
		else:
			for k in range(len(label_line)):
				if k > 0:
					trans_C[label_line[k - 1]][label_line[k]] += 1
				start_C[label_line[k]] += 1  # 4种状态的出现次数

	start_P = [0.76898, 0.000005, 0.000005, 0.23101]
	# 转移概率A: Count(Ci,Cj) / Count(Ci)
	trans_P = []
	for i in range(tagSize):
		p = []
		for j in range(tagSize):
			p.append(trans_C[i][j] / float(start_C[i]))
		trans_P.append(p)

	x_train = pad_sequences(x_train, maxlen=ctxWindows)
	y_train = to_categorical(np.asarray(y_train), tagSize)
	print('Shape of data  tensor:', x_train.shape)
	print('Shape of label tensor:', y_train.shape)
	return x_train, y_train, start_P, trans_P


# def cal_A_B(train_label):
# 	lastTag = -1
# 	for tag in train_label:
# 		# 统计 tag 频次
# 		start_C[tag] += 1
# 		# 统计 tag 转移频次
# 		if lastTag != -1:
# 			trans_C[lastTag][tag] += 1
# 		lastTag = tag  # 暂存上一次结果
#
# 	# 转移总频次
# 	tranCnt = [sum(tag) for tag in trans_C]
# 	print('\n转移总频次: ', tranCnt)  # [1341391, 514352, 1341390, 1029583]
# 	# 初始概率
# 	start_P = [0.76898, 0.000005, 0.000005, 0.23101]
# 	# for i in range(tagSize):
# 	# 	start_P.append(start_C[i] / float(total))
# 	# 转移概率
# 	trans_P = []
# 	for i in range(tagSize):
# 		p = []
# 		for j in range(tagSize):
# 			p.append(trans_C[i][j] / float(tranCnt[i]))
# 		trans_P.append(p)
# 	return start_P, trans_P


if __name__ == '__main__':
	vocab, indexVocab = getDict(train_file, test_file, [' ', '\n'])  # 获取字典
	nb_words, embedding_matrix = get_embedding_matrix(vocab, emb_file)
	x_train, y_train, start_P, trans_P = doc2vec(train_file, vocab, ctxWindows=5)
	x_test, y_test, s1, t1 = doc2vec(test_file, vocab, ctxWindows=5)
	# start_P, trans_P = cal_A_B(train_label)

	utils.saveCwsInfo(info_file, ((start_P, trans_P), (vocab, indexVocab)))
	utils.saveCwsData(data_file, ((x_train, y_train),(x_test, y_test)))
	np.save(open(train_data_file, 'wb'), x_train)
	np.save(open(label_data_file, 'wb'), y_train)
	np.save(open(emb_matrix_file, 'wb'), embedding_matrix)
	with open(nb_words_file, 'w') as f:
		json.dump({'nb_words': nb_words}, f)



