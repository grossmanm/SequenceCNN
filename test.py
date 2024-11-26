import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

class DeliveryNotificationCNN(nn.Module):
    def __init__(self, input_features, sequence_length, num_classes):
        super(DeliveryNotificationCNN, self).__init__()
        
        # CNN layer with 32 filters
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # transform encoder
        #encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4)
        #self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * sequence_length, 50)
        self.fc2 = nn.Linear(50, 50)
        self.output = nn.Linear(50, num_classes)
        
    def forward(self, x):
        # Convolution + Activation + Dropout
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Flatten for fully connected layers
        #x = x.permute(2,0,1)

        #x = self.transformer(x)
        x = x.view(x.size(0), -1)


        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        # Output layer
        x = self.output(x)
        
        return x

# load model
model = torch.load('model1.pth')

messages = [
    {'sTime': 7.648223, 'mTime':7.740022,'rTime':8.606829}
]

rtt_times = []

for message in messages:
    rtt_server = message['mTime']-message['sTime']
    rtt_receiver = message['rTime']-message['sTime']
    rtt_diff = rtt_receiver-rtt_server
    for i in range(5):
        rtt_times.append([rtt_server, rtt_receiver, rtt_diff])

rtt_times = np.array([rtt_times])
rtt_times = torch.tensor(rtt_times, dtype=torch.float32).transpose(1,2)
with torch.no_grad():
    outputs = model(rtt_times)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)



