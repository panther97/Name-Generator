#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange
import sys
import string


# In[2]:


def char_to_int(c_dict):
    c_int_dict={}
    for key, value in c_dict.items():
        c_int_dict[value]=key
    return c_int_dict


# In[3]:


names_list=[]
f=open("C:/Users/slamba6/Desktop/names.txt","r") #reading file
for l in f:
    l = l.strip('\n')
    l = l.lower()
    names_list.append(l) #making lower case 


# In[4]:


en=[]
OHE_dict={}
letters = ':' + string.ascii_lowercase
character_dict = dict(enumerate(letters))
for i in range(27):
    en.append(i)
oh=F.one_hot(torch.tensor(en), num_classes=27) #one hot encoding
for alphabet, om in zip(letters,oh):
    OHE_dict[alphabet]=om


# In[5]:


break_list=[]

for name in names_list:
    b_list = []
    count =0
    for char in name:
        b_list.append(char)
        count+=1
    for i in range(11-count):
        b_list.append(':')
    break_list.append(b_list)


# In[6]:


char_int_dict=char_to_int(character_dict)


# In[7]:


def names_generator(break_list):
    names_tensor=torch.zeros((11,27))
    s =[]
    for c in break_list:
        x = OHE_dict[c]
        s.append(x)
    for i in range(11):
        for j in range(27):
            names_tensor[i][j]=s[i][j]
    return names_tensor


# In[8]:


def make_tensor(break_list):
    cd=char_to_int(character_dict)
    y_data=[]
    s_y=[]
    y=break_list[1:]
    y.append(":")
    for c_y in y:
        s_y.append(cd[c_y])
    labels_tensor=torch.zeros((11))
    for m in range(11):
        labels_tensor[m]=s_y[m]
    names_tensor = names_generator(break_list)
    return names_tensor,labels_tensor


# In[9]:


class name_dataset(torch.utils.data.Dataset):
    def __init__(self,break_list):
        self.break_list=break_list
        
    def __len__(self) -> int:
        return len(self.break_list)

    def __getitem__(self, idx: int):
        return make_tensor(break_list[idx])


# In[10]:


class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.i_size = 27
        self.h_size = 54
        self.n_layers = 4
        self.lstm1 = nn.LSTM(27, 54, 4, batch_first=True)   #lstm layer 1
        self.fc2 = nn.Linear(54, 27) #fully connected layer 1 
        self.fc4 = nn.Linear(27, 27)   #fully connected layer 3
        
    def forward(self, inp, states):
        ht, ct = states
        batch_size = inp.size(0)
        output, (ht, ct) = self.lstm1(inp, (ht, ct))
        output = F.relu(self.fc2(output))
        output = self.fc4(output)
        return output, (ht, ct) 


# In[13]:


def train(inp, out):
    inp = inp.to(device)
    out = out.to(device)
    epoch_loss = 0
        
    hidden_state = torch.zeros((4, inp.size(0), 54)).to(device)
    cell_state = torch.zeros((4, inp.size(0), 54)).to(device)
    optimizer.zero_grad()
    out_pred_logits, (hidden_state, cell_state) = model(inp, (hidden_state, cell_state))
    out_pred_logits = out_pred_logits.transpose(1, 2) 
    loss = loss_function(out_pred_logits, out.long())
    loss.backward(retain_graph=True)

    epoch_loss = loss.item()
    return epoch_loss


# In[15]:


model = Model()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
model = model.to(device)
name_data = name_dataset(break_list)
loss_function = nn.CrossEntropyLoss(reduction='mean')
train_loader = torch.utils.data.DataLoader(name_data, batch_size=20, shuffle=True)
training_iterations = iter(train_loader)
inp, out = training_iterations.next()
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.97)
epoch_losses = []
for epoch in range(1, 151): #number of epochs = 150
    epoch_loss = 0
    for i, (inp, out) in enumerate(train_loader, 1):
        e = train(inp, out)
        epoch_loss += e
        optimizer.step()
        scheduler.step()
    epoch_loss /= len(train_loader)  #loss per epoch
    epoch_losses.append(epoch_loss)

    print("Epoch:{}    Loss:{}   ".format(epoch, epoch_loss))


# In[16]:


torch.torch.save(model.state_dict(), "0702-668186281-Lamba.pt")


# In[17]:


epoch = np.arange(150)
plt.plot(epoch, epoch_losses)
plt.title("Epoch vs Loss")
plt.xlabel("Epochs", c = "red")
plt.ylabel("Loss", c = "blue")
plt.show()


# In[ ]:




