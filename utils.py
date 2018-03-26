#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import json
import h5py
import codecs
import numpy as np


def saveCwsInfo(path, cwsInfo):
	'''保存分词训练数据字典和概率'''
	print('save cws info to %s' % path)
	fd = open(path, 'w')
	(initProb, tranProb), (vocab, indexVocab) = cwsInfo
	j = json.dumps((initProb, tranProb))
	fd.write(j + '\n')
	for char in vocab:
		fd.write(char.encode('utf-8') + '\t' + str(vocab[char]) + '\n')
	fd.close()


def loadCwsInfo(path):
	'''载入分词训练数据字典和概率'''
	print('load cws info from %s' % path)
	fd = open(path, 'r')
	line = fd.readline()
	j = json.loads(line.strip())
	initProb, tranProb = j[0], j[1]
	lines = fd.readlines()
	fd.close()
	vocab = {}
	indexVocab = [0 for i in range(len(lines))]
	for line in lines:
		rst = line.strip().split('\t')
		if len(rst) < 2: continue
		char, index = rst[0].decode('utf-8'), int(rst[1])
		vocab[char] = index
		indexVocab[index] = char
	return (initProb, tranProb), (vocab, indexVocab)


def saveCwsData(path, cwsData):
	'''保存分词训练输入样本'''
	print('save cws data to %s' % path)
	# 采用hdf5保存大矩阵效率最高
	fd = h5py.File(path, 'w')
	(x_train, y_train), (x_test, y_test) = cwsData
	fd.create_dataset('x_train', data=x_train)
	fd.create_dataset('y_train', data=y_train)
	fd.create_dataset('x_test', data=x_test)
	fd.create_dataset('y_test', data=y_test)
	fd.close()


def loadCwsData(path):
	'''载入分词训练输入样本'''
	print('load cws data from %s' % path)
	fd = h5py.File(path, 'r')
	x_train = fd['x_train'][:]
	y_train = fd['y_train'][:]
	x_test = fd['x_test'][:]
	y_test = fd['y_test'][:]
	fd.close()
	return (x_train, y_train), (x_test, y_test)