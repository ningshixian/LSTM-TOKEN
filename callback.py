# encoding=utf-8
from keras.callbacks import Callback, ModelCheckpoint
import prf

# 该回调函数将在每个迭代后保存的最好模型
class checkpoint():
	def __init__(self, best_model_file):
		self.model_file = best_model_file

	def check(self):
		checkpoint = ModelCheckpoint(filepath=self.model_file, monitor='val_loss',
									 verbose=1, save_best_only=True, mode='min')
		return checkpoint
