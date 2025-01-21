#Start working from the lowest bops models and work your way up to find the best model that we have achieved from our design space exploration.

import torch
import math
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from brevitas.nn import QuantLinear, QuantReLU
from brevitas.quant import SignedBinaryWeightPerTensorConst
import torch.nn as nn
import itertools
import time
import os
from scipy.spatial.distance import hamming
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve

##------Setting the training device-------------------------

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
device = torch.device('cpu')
print(device)

class canTestDataset(Dataset):
    def __init__(self):
        #will be mostly used for dataloading
        x_load = np.loadtxt('./dos_fuzzy_rpm_test_x.txt',delimiter = ",",dtype=np.float32)
        y_load = np.loadtxt('./dos_fuzzy_rpm_test_y_v1.txt',delimiter = ",",dtype=np.float32)
        self.x = torch.from_numpy(x_load)
        self.y = torch.from_numpy(y_load)
        self.n_samples = x_load.shape[0]
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
        
    def __len__(self):
        #Will allow us to get the length of our dataset
        return self.n_samples


testDataset = canTestDataset()
first_data = testDataset[0]
features,labels = first_data
print(features,labels)
samples = len(testDataset)

testloader = DataLoader(dataset=testDataset, batch_size=1800, shuffle=False)#Works for a batch size of 1. 
#Working with a large batch size makes the process very fast.
epochs = 200
torch.manual_seed(0)
a1 = 4
a2 = 4
a3 = 4
a4 = 4
a5 = 4
# model_folder = "./dos_fuzzy_v4/model"#This version for the epoch indexed with 20 gave 99.3% accuracy....5,4,5,4,5 bit width and normal model
# model_folder = "./dos_fuzzy_v5/model" #This version was trained without dropout and the 5,4,5,4,5 config...99.93% accuracy on the 49th epoch.
# model_folder = "./dos_fuzzy_v7/model" #This version was trained without dropout for 100 epoch and the config was 4,4,4,4,4 and the BCE loss function.
# model_folder = "./dos_fuzzy_v8/model" #This version was trained without dropout for 100 epoch and the config was 2,2,2,2,2 and the BCE loss function.
# model_folder = "./dos_fuzzy_v9/model" #This version was trained without dropout for 100 epoch and the config was 3,3,3,3,3 and the BCE loss function.
# model_folder = "./dos_fuzzy_rpm_v1_2bit/model" #This version was trained without dropout for 200 epoch and the config was 2,2,2,2,2 and the BCE loss function for DoS, Fuzzing and RPM-Spoof where each block is considered an attack block if it has even one attack message.
# model_folder = "./dos_fuzzy_rpm_v1_3bit/model" #This version was trained without dropout for 200 epoch and the config was 2,2,2,2,2 and the BCE loss function.
model_folder = "./dos_fuzzy_rpm_v1_4bit/model" #This version was trained without dropout for 200 epoch and the config was 2,2,2,2,2 and the BCE loss function.
model = nn.Sequential(
              QuantLinear(40,256,bias=True, weight_bit_width=int(a1)),
              nn.BatchNorm1d(256),
              #nn.Dropout(0.2),
              QuantReLU(bit_width=int(a1)),
              QuantLinear(256, 128,bias=True, weight_bit_width=int(a2)),
              nn.BatchNorm1d(128),
              #nn.Dropout(0.4),
              QuantReLU(bit_width=int(a2)),
              QuantLinear(128, 64,bias=True, weight_bit_width=int(a3)),
              nn.BatchNorm1d(64),
              #nn.Dropout(0.2),
              QuantReLU(bit_width=int(a3)),
              QuantLinear(64, 32,bias=True, weight_bit_width=int(a4)),
              nn.BatchNorm1d(32),
              #nn.Dropout(0.5),
              QuantReLU(bit_width=int(a4)),
              QuantLinear(32,4,bias=True, weight_bit_width=int(a5)),
              nn.Softmax(dim=1)#The dimension along which the softmax will be computed and the sum along that dimension would be one.
            )
model = model.float()
max_accuracy = 0
max_index = 200
acc = 0
count_n = 0
count_n_acc = 0
count_d = 0
count_d_acc = 0
count_f = 0
count_f_acc = 0
count_r = 0
count_r_acc = 0
full_test_label = []
full_pred_label = []
roc_true = []
roc_predict = []
box_normal = np.zeros(103012)
box_dos = np.zeros(23555)
box_fuzzy = np.zeros(27859)
box_rpm = np.zeros(25040)
for j in range(200):
	path = model_folder+'/modelModel_'+str(j)+'.pt'
	model.load_state_dict(torch.load(path,map_location=device))#When trying to load on the 'cpu' which is trained on a GPU use the map_location arguement with the torch.load function.
	count = np.zeros(7)
	model.eval()
	t1 = 0
	t2 = 0
	t3 = 0
	acc = 0
	count_n = 0
	count_n_acc = 0
	count_d = 0
	count_d_acc = 0
	count_f = 0
	count_f_acc = 0
	count_r = 0
	count_r_acc = 0
	print_count = 0
	accuracy = 0 # We report the accuracy of the model here.
	for l,(test_inputs,test_labels) in enumerate(testloader):
		t1 = t1 + time.time()
		outputs = model(test_inputs.float())
		a = outputs.detach().numpy()
		# print(a[0])
		b = test_labels.detach().numpy() #In this and the above steps the tensors are converted into a numpy array to allow us to process stuff.
		roc_true.append(b)
		roc_predict.append(a)
		max_idx_pred = a.argmax(axis=1)
		max_idx_test = b.argmax(axis=1)
		full_test_label.append(max_idx_test)
		full_pred_label.append(max_idx_pred)
		for i in range(1800):
			if(max_idx_test[i] == max_idx_pred[i]):
				acc = acc+1
			if(max_idx_test[i] == 0):
				count_n = count_n+1
				if(max_idx_pred[i] == 0):
					box_normal[count_n_acc] = a[i][0]
					# print(a[i][0])
					count_n_acc = count_n_acc + 1 
			if(max_idx_test[i] == 1):
				count_d = count_d+1
				if(max_idx_pred[i] == 1):
					box_dos[count_d_acc] = a[i][1]
					count_d_acc = count_d_acc + 1
					# print(a[i][1])
			if(max_idx_test[i] == 2):
				count_f = count_f+1
				# print_count+=1
				# if(print_count%50 == 0):
				# 	print(a[i][2])
				if(max_idx_pred[i] == 2):
					box_fuzzy[count_f_acc] = a[i][2]
					count_f_acc = count_f_acc + 1
					print_count+=1
					# print(a[i][2])
			if(max_idx_test[i] == 3):
				count_r = count_r+1
				# print_count+=1
				# if(print_count%50 == 0):
				# 	print(a[i][2])
				if(max_idx_pred[i] == 3):
					box_rpm[count_r_acc] = a[i][3]
					count_r_acc = count_r_acc + 1
					# print_count+=1
					# print(a[i][3])
		# a = np.where(a >= 0.5, 1, 0) #This step applies kind of a step function to the output array. Where value below 0.5 are converted to 0 and the ones greater than 0.5 are converted to 1.
		# print(a)
		accuracy = accuracy + (1800-hamming(max_idx_pred,max_idx_test)*len(max_idx_test))# We then calculate the hamming distance of the output and the actual output arrays..subtract it from the batch size and add it to the final accuracy of the model.
		t2 = t2 + time.time()
		#This 'if' statement notes the max accuracy and the index for which we get that accuracy so that we know which is the best model among the top 50 for a given variation
	if(accuracy > max_accuracy):
		max_accuracy = accuracy
		max_index = j
	t3 = t3+t2-t1
	print('Total messages =',samples,'Overall accuracy =',accuracy,'Misclassifications = ',(samples-accuracy),'Percentage accuracy =',(accuracy/samples)*100,'Epoch =',int(j),'\n') # Print the accuracy and the percentage accuracy here in this statement.
	print('Total Normal =',count_n,' Correct normal =',count_n_acc,'Misclassifications = ',(count_n-count_n_acc),'\n')
	print('Total DoS =',count_d,' Correct DoS =',count_d_acc,'Misclassifications = ',(count_d-count_d_acc),'\n')
	print('Total Fuzzing =',count_f,' Correct Fuzzy =',count_f_acc,'Misclassifications = ',(count_f-count_f_acc),'\n')
	print('Total RPM =',count_r,' Correct RPM =',count_r_acc,'Misclassifications = ',(count_r-count_r_acc),'\n')
	print('---------------------------')
	#The below snippet gives the confusion matrix of the multi-class classification model.
	# y_true = np.array(full_test_label)
	# y_true = y_true.reshape(180000)
	# y_pred = np.array(full_pred_label)
	# y_pred = y_pred.reshape(180000)
	# print(classification_report(y_true, y_pred,output_dict=True))#This function only works when the input is a numpy array and it is flattened...Otherwise if a numpy array has shape (100,1200)...This will throw up an multiclass output not supported error.
	# print(confusion_matrix(y_true,y_pred))
	# fig = plt.figure(figsize =(20, 15))
	# Creating plot
	# dfa = ['Benign','DoS','Fuzzing','RPM-Spoof']
	# plt.boxplot([box_normal,box_dos,box_fuzzy,box_rpm],labels=dfa)
	# plt.boxplot(box_dos)
	# plt.boxplot(box_fuzzy)
	# plt.boxplot(box_rpm)
	# show plot
	# plt.semilogy()
	# plt.show()
	#ROC-Curve plots
	# roc_true_plot = np.array(roc_true)
	# roc_true_plot = roc_true_plot.reshape(180000,4)
	# roc_predict_plot = np.array(roc_predict)
	# roc_predict_plot = roc_predict_plot.reshape(180000,4)
	# fpr, tpr, thresholds = roc_curve(roc_true_plot[:, 3], roc_predict_plot[:, 3])
	# display = RocCurveDisplay(fpr=fpr, tpr=tpr)
	# display.plot()
	# plt.show()
	# Plotting
	# plt.plot(fpr, tpr, marker='.')
	# plt.ylabel('True Positive Rate')
	# plt.xlabel('False Positive Rate' )
	# plt.show()
	# RocCurveDisplay.from_predictions(
 #    roc_true_plot[:, 2],
 #    roc_predict_plot[:, 2],
 #    name="Fuzzing  vs the rest",
 #    color="darkorange",
	# )
	# plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
	# plt.axis("square")
	# plt.xlabel("False Positive Rate")
	# plt.ylabel("True Positive Rate")
	# plt.title("One-vs-Rest ROC curves:\nFuzzing vs (Normal, DoS & RPM)")
	# plt.legend()
	# plt.show()
print("Maximum accuracy index = ",max_index,"Max accuracy = ",(max_accuracy/samples)*100)