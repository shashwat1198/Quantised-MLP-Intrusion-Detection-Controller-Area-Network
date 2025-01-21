#Training file for multiclass-classification for the 4-bit MLP model.
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

##------Setting the training device-------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

##------ Dataset loader---------------------------------------

class canTrainDataset(Dataset):
    def __init__(self):
        #will be mostly used for dataloading
        x_load = np.loadtxt('./dos_fuzzy_rpm_train_X.txt',delimiter = ",",dtype=np.float32)
        y_load = np.loadtxt('./dos_fuzzy_rpm_train_Y_v1.txt',delimiter = ",",dtype=np.float32)
        self.x = torch.from_numpy(x_load)
        self.y = torch.from_numpy(y_load)
        self.n_samples = x_load.shape[0]
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
        
    def __len__(self):
        #Will allow us to get the length of our dataset
        return self.n_samples

class canValDataset(Dataset):
    def __init__(self):
        #will be mostly used for dataloading
        x_load = np.loadtxt('./dos_fuzzy_rpm_val_X.txt',delimiter = ",",dtype=np.float32)
        y_load = np.loadtxt('./dos_fuzzy_rpm_val_Y_v1.txt',delimiter = ",",dtype=np.float32)
        self.x = torch.from_numpy(x_load)
        self.y = torch.from_numpy(y_load)
        self.n_samples = x_load.shape[0]
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
        
    def __len__(self):
        #Will allow us to get the length of our dataset
        return self.n_samples

trainDataset = canTrainDataset()
valDataset = canValDataset()
first_data = trainDataset[0]
features,labels = first_data
print(features,labels)
first_data = valDataset[0]
features,labels = first_data
print(features,labels)

trainloader = DataLoader(dataset=trainDataset, batch_size=128, shuffle=True)#We train the model by setting shuffle as true to make batch training stable.
validationloader = DataLoader(dataset=valDataset, batch_size=1000, shuffle=False)
num_epochs = 200
train_batch_size = 128
val_batch_size = 1000
total_samples = len(trainDataset)
batches = math.ceil(total_samples/train_batch_size)
print(total_samples,batches)
a1 = 4
a2 = 4
a3 = 4
a4 = 4
a5 = 4
directory = "./dos_fuzzy_rpm_v1_4bit"
data_folder = "./dos_fuzzy_rpm_v1_4bit/data"    
model_folder = "./dos_fuzzy_rpm_v1_4bit/model" 
os.mkdir(directory)
os.mkdir(data_folder)
os.mkdir(model_folder)
model = nn.Sequential(
              QuantLinear(40,256,bias=True, weight_bit_width=int(a1)),
              nn.BatchNorm1d(256),
              # nn.Dropout(0.2),
              QuantReLU(bit_width=int(a1)),
              QuantLinear(256, 128,bias=True, weight_bit_width=int(a2)),
              nn.BatchNorm1d(128),
              # nn.Dropout(0.4),
              QuantReLU(bit_width=int(a2)),
              QuantLinear(128, 64,bias=True, weight_bit_width=int(a3)),
              nn.BatchNorm1d(64),
              # nn.Dropout(0.2),
              QuantReLU(bit_width=int(a3)),
              QuantLinear(64, 32,bias=True, weight_bit_width=int(a4)),
              nn.BatchNorm1d(32),
              # nn.Dropout(0.5),
              QuantReLU(bit_width=int(a4)),
              QuantLinear(32,4,bias=True, weight_bit_width=int(a5)),
              nn.Softmax(dim=1)#The dimension along which the softmax will be computed and the sum along that dimension would be one.
            )
model = model.float()
model.to(device)
criterion = nn.BCELoss()
#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)
print("No. of parameters in the model = ",sum(p.numel() for p in model.parameters() if p.requires_grad)) #Number of parameters in the model.
# loss criterion and optimizer
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
running_loss = 0.0
epoch = 0
max_acc = 0
epoch_time = 0
min_val_loss = 1000000
valid_loss = np.zeros(num_epochs)
avg_valid_loss = np.zeros(num_epochs)
epoch_loss = np.zeros(num_epochs)
avg_epoch_loss_per_batch = np.zeros(num_epochs)
start = time.time()
for j in range(num_epochs):  
    running_loss = 0.0
    running_val_loss = 0.0
    count = 0
    epoch = 0
    count_test = 0
    model.train()
    epoch_time_start = time.time()
    for k, (inputs,labels) in enumerate(trainloader):
    #Zero the computed gradients every batch
        inputs = inputs.to(device)
        optimizer.zero_grad()
    #inputs = inputs.to(device)
    # Make predictions
        outputs = model(inputs)
    #Compute the loss for the batch
        labels = labels.reshape([train_batch_size,4])
        loss = criterion(outputs.cpu(), labels)
    #Compute the gradients...derviate of the loss w.r.t the inputs
        loss.backward()
    #Adjust the gradients.
        optimizer.step()
        count = count+1
        running_loss += loss.item()
#Save the model for this particular epoch after training.
    model.eval()
    for l,(val_inputs,val_labels) in enumerate(validationloader): #The validation loop for the models to be trained here.
        val_inputs = val_inputs.to(device)
        val_outputs = model(val_inputs)
        val_labels = val_labels.reshape([val_batch_size,4])
        val_loss =  criterion(val_outputs.cpu(), val_labels)
        running_val_loss += val_loss.item()
    epoch = epoch+1
    epoch_loss[j] = running_loss
    avg_epoch_loss_per_batch[j] = running_loss/batches
    valid_loss[j] = running_val_loss
    avg_valid_loss[j] = running_val_loss/val_batch_size
    print('Epoch Number = ',j,'Epoch loss = ',running_loss,'Average Epoch loss = ',running_loss/batches)
    print('Epoch Number = ',j,'Epoch_Val loss = ',running_val_loss,'Average Epoch_Val loss = ',running_val_loss/val_batch_size)
    epoch_time_end = time.time()
    total_epoch_time = epoch_time_end - epoch_time_start
    print("Total epoch time = "+str(total_epoch_time))
    print('--------------------------')    
    path3 = model_folder+'Model_'+str(j)+'.pt'
    torch.save(model.state_dict(), path3) #Saving the model after every epoch
# torch.save(model.state_dict(), './fuzzy_dse_models/fuzzy_'+str(lbw[i][0])+'_'+str(lbw[i][1])+'_'+str(lbw[i][2])+'_'+str(lbw[i][3])+'_'+str(lbw[i][4])+'_'+str(epoch)+'.pt')
#Saving the entire (normal and average) epoch and validation loss for all the epoch of a particular model here.
path1 = data_folder+"/avg_el.txt"
file = open(path1, "w+") 
content = str(avg_epoch_loss_per_batch)
file.write(content)
file.close()
path2 = data_folder+"/avg_vl.txt"
file = open(path2, "w+")
content = str(avg_valid_loss)
file.write(content)
file.close()
path4 = data_folder+"/el.txt"
file = open(path4, "w+")
content = str(epoch_loss)
file.write(content)
file.close()
path5 = data_folder+"/vl.txt"
file = open(path5, "w+")
content = str(valid_loss)
file.write(content)
file.close()
        
end = time.time()
total_time = end-start
print("Total time for training = ",total_time)