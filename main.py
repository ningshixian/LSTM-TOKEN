#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import codecs
import json
import numpy as np
from keras.layers import Bidirectional
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
import acc_loss
import preprocess as pp
import utils
import viterbi
from callback import checkpoint

# Initialize global variables
tags = ['B', 'M', 'E', 'S']
tagSize = len(tags)
prob_file = r'predict_result'
prf_file = 'prf.txt'
base = './model/'
best_model_file = base + "best_model.h5"
info_file = base + 'cws.info'
data_file = base + 'cws.data'
train_data_file = base + 'train.npy'
label_data_file = base + 'label.npy'
emb_matrix_file = base + 'word_embedding_matrix.npy'
nb_words_file = base + 'nb_words.json'
MAX_NB_WORDS = 300000  # 字典大小
max_len = 5
EMBEDDING_DIM = 300
BATCH_SIZE = 128
NB_EPOCHS = 10


# Load the dataset, embedding matrix and word count
def load_data():
	print('数据加载中......')
	cwsInfo = utils.loadCwsInfo('./model/cws.info')
	# cwsData = utils.loadCwsData('./model/cws.data')
	# (x_train, y_train), (x_test, y_test) = cwsData
	x_train = np.load(open(train_data_file, 'rb'))
	y_train = np.load(open(label_data_file, 'rb'))
	embedding_matrix = np.load(open(emb_matrix_file, 'rb'))
	with open(nb_words_file, 'r') as f:
		nb_words = json.load(f)['nb_words']
	print('Done!\n')
	return x_train, y_train, embedding_matrix, nb_words, cwsInfo


# Define the model
def train(x_train, y_train, nb_words, embedding_matrix):  # cwsInfo, cwsData, modelPath, weightPath
	print('开始建立模型/....')
	model = Sequential()
	model.add(Embedding(input_dim=nb_words,
						output_dim=EMBEDDING_DIM,
						weights=[embedding_matrix],
						input_length=max_len,
						mask_zero=True))
	# model.add(LSTM(output_dim=300, return_sequences=True, consume_less='gpu'))
	# model.add(LSTM(output_dim=300, return_sequences=False, consume_less='gpu'))
	model.add(Bidirectional(LSTM(output_dim=512, consume_less='gpu'), merge_mode='concat'))
	model.add(Dropout(0.2))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(tagSize, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',  # rmsprop
				  metrics=['accuracy'])
	model.summary()  # 打印出模型概况

	#############################################  Train Model  ############################################################

	# 该回调函数将在每个迭代后保存的最好模型
	check_call = checkpoint(best_model_file)

	# IndexError: One of the index value is out of bound. Error code: 65535.\n
	# exception IndexError :Raised when a sequence subscript is out of range.
	history = model.fit(x_train, y_train,
						batch_size=BATCH_SIZE,
						nb_epoch=NB_EPOCHS,
						callbacks=[check_call.check()],
						validation_split=0.2)
	# model.save(best_model_file)

	#############################################  善后处理  ############################################################

	# 打印最好的迭代结果
	max_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
	print('Minimum loss at epoch', '{:d}'.format(idx + 1), '=', '{:.4f}'.format(max_val_loss))
	with open(prf_file, 'a') as f:
		f.write('\nMinimum loss at epoch: ' + str(idx + 1) + ' =' + str(max_val_loss) + '\n')

	# plot the result
	acc_loss.plot(history)
	return model


def cwsFile(fname, dstname, model, cwsInfo):
	with codecs.open(fname, 'r', 'utf-8') as fd:
		lines = fd.readlines()

	with codecs.open(dstname, 'w', 'utf-8') as fd:
		for line in lines:
			rst = cwsSent(line.strip(), model, cwsInfo)
			fd.write(rst + '\n')


def cwsSent(sent, model, cwsInfo):
	(initProb, tranProb), (vocab, indexVocab) = cwsInfo
	vec = pp.sent2vec2(sent, vocab, ctxWindows=5)
	vec = np.array(vec)
	# 根据输入得到 [B,E,M,S] 标注推断
	# 按batch产生输入数据属于各个类别的概率
	emit_P = model.predict_proba(vec)
	# 按batch产生输入数据的类别预测结果
	classes = model.predict_classes(vec)
	# print(classes)  # [3 0 2 3 0 1 2 3 0 1 2 3]

	prob, path = viterbi.viterbi(vec, tags, initProb, tranProb, emit_P.transpose())
	assert len(path)==len(sent)

	result = ''
	for i, t in enumerate(path):
		if tags[t] == 'B':
			result += sent[i]
		elif tags[t] == 'M':
			result += sent[i]
		elif tags[t] == 'E':
			result += sent[i] + ' '
		else:
			result += sent[i] + ' '
	return result.strip()


if __name__ == '__main__':
	x_train, y_train, embedding_matrix, nb_words, cwsInfo = load_data()

	print('Loading model...')
	# model = train(x_train, y_train, nb_words, embedding_matrix)
	model = load_model(best_model_file)
	print('Done!\n')

	print('-------------start predict----------------')
	sen = u'国务院中央军委'
	print(cwsSent(sen, model, cwsInfo))
	cwsFile('data/msr_test.utf8', 'data/result_cws.txt', model, cwsInfo)