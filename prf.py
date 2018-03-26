from __future__ import print_function
import numpy as np


def calculate(predictions, test_label, RESULT_FILE, epoch, prf_file):
	num = len(predictions)
	with open(RESULT_FILE, 'w') as f:
		for i in range(num):
			f.write(str(predictions[i][0]) + ' ' + str(predictions[i][1]) + '\n')


	TP = len([1 for i in range(num) if
			  predictions[i][1] > predictions[i][0] and (test_label[i] == np.asarray([0, 1])).all()])
	FP = len([1 for i in range(num) if
			  predictions[i][1] > predictions[i][0] and (test_label[i] == np.asarray([1, 0])).all()])
	FN = len([1 for i in range(num) if
			  predictions[i][1] < predictions[i][0] and (test_label[i] == np.asarray([0, 1])).all()])
	TN = len([1 for i in range(num) if
			  predictions[i][1] < predictions[i][0] and (test_label[i] == np.asarray([1, 0])).all()])

	precision = recall = Fscore = 0, 0, 0
	try:
		precision = TP / (float)(TP + FP)  # ZeroDivisionError: float division by zero
		recall = TP / (float)(TP + FN)
		Fscore = (2 * precision * recall) / (precision + recall)
	except ZeroDivisionError as exc:
		print(exc.message)

	print(">> Report the result ...")
	print("-1 --> ", len([1 for i in range(num) if predictions[i][1] < predictions[i][0]]))
	print("+1 --> ", len([1 for i in range(num) if predictions[i][1] > predictions[i][0]]))
	print("TP=", TP, "  FP=", FP, " FN=", FN, " TN=", TN)
	print('\n')
	print("precision= ", precision)
	print("recall= ", recall)
	print("Fscore= ", Fscore)

	with open(prf_file, 'a') as pf:
		print('write prf...... ')
		pf.write("epoch= " + str(epoch+1) + '\n')
		pf.write("precision= " + str(precision) + '\n')
		pf.write("recall= " + str(recall) + '\n')
		pf.write("Fscore= " + str(Fscore) + '\n\n')

