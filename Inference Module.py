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


letters = ':' + string.ascii_lowercase
character_dict = dict(enumerate(letters))
ehi=char_to_int(character_dict)


# In[4]:


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


# In[29]:


pretrained_model = Model()
pretrained_model.load_state_dict(torch.load('C:/Users/slamba6/Desktop/0702-668186281-Lamba.pt')) #Location of pretrained model 
pretrained_model.eval()


# In[30]:


device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


# In[31]:


def lstm_sampler(model, start='p', top=2):
    Names_lstm = [start]
    counter = 0
    while counter <= 10:
        with torch.no_grad():
            hidden_state = torch.zeros((4, 1, 54)).to(device) #hidden state
            cell_state = torch.zeros((4, 1, 54)).to(device)    #cell state
            t = 0  
            name = start
            for char in start:
                inp= torch.zeros((1, 1, 27)) 
                inp[0, 0, ehi[char]] = 1
                output, (hidden_state, cell_state) = model(inp, (hidden_state, cell_state))  #retrieving output from LSTM
                t += 1
            vals, idxs = torch.topk(output[0], top)        #top values from lstm
            idx = np.random.choice(idxs.cpu().numpy()[0], p = [0.6,0.4]) 
            character = character_dict[idx]
            name += character
            if name[-1] != ":":
                name += ":"
        if character==':':
            break
        else:
            start=start+character
            Names_lstm.append(start)
    return Names_lstm[-1]


# In[32]:


names_generated = []
alphabet = input("enter the first alphabet of name to be generated")
for i in range(20):
    generated_names = lstm_sampler(pretrained_model, start=alphabet, top=2)
    names_generated.append(generated_names)
print(names_generated)


# In[ ]:




