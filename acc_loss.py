import matplotlib.pyplot as plt
import pandas as pd


def plot(history):
	plt.figure(figsize=(16,7))
	plt.subplot(121)
	plt.xlabel('epoch')
	plt.ylabel('acc')
	plt.plot(history.epoch, history.history['acc'], 'b', label="acc")
	plt.plot(history.epoch, history.history['val_acc'], 'r', label="val_acc")
	plt.scatter(history.epoch, history.history['acc'], marker='*')
	plt.scatter(history.epoch, history.history['val_acc'])
	plt.legend(loc='lower right')

	plt.subplot(122)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.plot(history.epoch, history.history['loss'], 'b', label="loss")
	plt.plot(history.epoch, history.history['val_loss'], 'r', label="val_loss")
	plt.scatter(history.epoch, history.history['loss'], marker='*')
	plt.scatter(history.epoch, history.history['val_loss'], marker='*')
	plt.legend(loc='lower right')
	plt.show()