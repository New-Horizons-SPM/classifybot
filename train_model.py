# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:03:49 2022

@author: Maxwell West & Julian Ceddia
"""

import os
import pickle
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter

class ConvNet(nn.Module):
    def __init__(self, load_model=""):
        """
        Simple convolutional neural network with two convolutional layers, one
        hidden layer, and one binary output layer.

        Parameters
        ----------
        load_model : Path to the model to be loaded

        """
        super(ConvNet, self).__init__()
        """
        /Replace this code with best architecture
        """
        self.conv1 = nn.Conv2d( 3,  5, 3)                                       # 3 input channels, 5 output channels and a kernel size of 3
        self.conv2 = nn.Conv2d( 5,  5, 3)                                       # 5 input channels, 5 output channels and a kernel size of 3
        
        self.pool = nn.MaxPool2d(2, 2)                                          # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html : Applies a 2D max pooling over an input signal composed of several input planes.

        self.fc1 = nn.Linear(14045, 128)                                        # 14,045 just comes from the amount of neurons the convolutional layers happened to finish with. 128 output neurons in this hidden layer is completely arbitrary 
        self.fc3 = nn.Linear(128, 2)                                            # Output layer with binary classification to begin with
        
        self.opt = optim.Adam(self.parameters(), lr=0.001)                      # lr is another thing which can be played with... https://pythonguides.com/adam-optimizer-pytorch/

        """
        Replace this code with best architecture/
        """
        
        if load_model:
            load_model = "saved_nets/" + load_model + ".pt"
            ckpt = torch.load(load_model)
            
            if "state_dict" in ckpt.keys():
                self.load_state_dict(ckpt['state_dict'])
            
            else:
                self.load_state_dict(ckpt)
    
    def forward(self, x):
        """
        Function that's called when a prediction is to be made. call like:
        net = ConvNet()...
        ...
        prediction = net(data)

        Parameters
        ----------
        x : Data/image to predict on

        Returns
        -------
        x : Prediction tensor where each element is a label. Take the highest

        """
        """
        /Replace this code with the best architecture
        """
        x = self.pool(F.relu(self.conv1(x)))                                    # The first convolutional layer
        x = self.pool(F.relu(self.conv2(x)))                                    # Second convolutional layer
        
        x = x.view(x.size(0), -1)                                               # Returns a new tensor with the same data as the x-tensor but of a different shape.
        x = F.relu(self.fc1(x))                                                 # Hidden layer
        x = self.fc3(x)                                                         # Output layer
        """
        Replace this code with the best architecture/
        """
        return x
    
    def train(self, x_train, y_train, x_test, y_test, name, epochs=1, batch_size=64):
        criterion = nn.CrossEntropyLoss()
        best = 0.0                                                              # Keep track of the best performing model
        print('Start training...')
        print('------------------------------------------------')
        print(' Train  Acc | Test Acc | Best Test Acc |  Loss')
        print('------------------------------------------------')
        for epoch in range(epochs):
            running_loss = 0.0
            for i in range(x_train.size(0) // batch_size):
                inputs = x_train[i * batch_size : (i+1) * batch_size]
                labels = y_train[i * batch_size : (i+1) * batch_size]
    
                self.opt.zero_grad()
    
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.opt.step()
    
                running_loss += float(loss.detach())
    
                if i and not(i % 10) or True:
    
                    with torch.no_grad():
                        
                        train_pred = self(x_train)
                        train_acc  = (torch.sum(torch.argmax(train_pred, axis=1) == y_train) / y_train.size(0)).item()
                        
                        test_pred = self(x_test)
                        test_acc  = (torch.sum(torch.argmax(test_pred, axis=1) == y_test) / y_test.size(0)).item()
                        
    
                    if test_acc > best:
                        best = test_acc
                        torch.save(self.state_dict(), "saved_nets/" + name + ".pt")
                        
                    print(f'    {train_acc:.3f}   |   {test_acc:.3f}  |     {best:.3f}     |  {running_loss:.3f}')
    				
                    running_loss = 0.0
                
        print('Done training')
        return name

def trainNewCNN(runName, targetLabel, pklPath, augmentData):
    x = []                                                                          # The images will go here
    y = []                                                                          # The labels will go here
    allLabels = []                                                                  # This will keep count of all the labels we see
    if(not pklPath.endswith('/')): pklPath += '/'
    pklPath += runName + "/labelled/"
    pklFiles = os.listdir(pklPath)
    for pklFile in pklFiles:
        batchDict = pickle.load(open(pklPath + pklFile,'rb'))
        batchData = batchDict['data']
        print(pklFile)
        for key, value in batchData.items():
            im     = np.array(value[0]/np.max(value[0]),dtype=np.float32)           # Normalise the data and force to be float32
            labels = value[1]
            if(im.shape != (221, 221, 4)): continue                                 # Skip images that are the wrong size
            x.append(np.transpose(im[:,:,:3], (2,0,1)))                             # Append the image to x
            y.append(int(targetLabel in labels))                                    # At the moment I'm assuming the labels are just 0 or 1 depending on whether the target label is present
            
            if augmentData:                                                         # Optionally add images which are just reflections of existing images
                x.append(np.transpose(im[:,::-1,:3], (2,0,1)))                      # Reflect in x
                y.append(int(targetLabel in labels))
                
                x.append(np.transpose(im[::-1,:,:3], (2,0,1)))                      # Reflect in y
                y.append(int(targetLabel in labels))
                
                x.append(np.transpose(im[::-1,::-1,:3], (2,0,1)))                   # Reflect in xy
                y.append(int(targetLabel in labels))
    
            allLabels.extend(labels * (4 if augmentData else 1))                    # Keeping count of all labels we've seen
        
    x = np.array(x)                                                                 # Convert to numpy array
    y = np.array(y)                                                                 # Convert to numpy array
    
    print('Total number of images: ', len(x))
    print('Summary of labels:')
    label_freq = sorted(zip(Counter(allLabels).keys(), Counter(allLabels).values()), key=lambda x: -x[1])
    print(*label_freq,sep='\n')
    
    shuffle = np.random.permutation(range(x.shape[0]))                              # Shuffle the data
    
    x = torch.tensor(x[shuffle])                                                    # Convert to a pytorch compatible tensor
    y = torch.tensor(y[shuffle], dtype=torch.long)
    
    cutoff = int(0.9 * x.shape[0])                                                  # Split into training and testing/validation subsets
    x_train, x_test = x[:cutoff], x[cutoff:]                                        # Image data
    y_train, y_test = y[:cutoff], y[cutoff:]                                        # Labels
    
    net = ConvNet()                                                                 # Instantiate the CNN
    modelPath = net.train(x_train, y_train, x_test, y_test, name=runName, epochs=2, batch_size=64) # Train the CNN
    
    return modelPath