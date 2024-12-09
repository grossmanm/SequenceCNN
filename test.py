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
test_data = 'test_data_d_01.csv'


data = pd.read_csv(test_data)

def create_sequences(data,sequence_length,features,labels):
    sequences = []
    sequence_labels = []
    for i in range(len(data)-sequence_length):
        list_finished = True
        cur_loc = data['sequence'][i]
        seq = np.zeros((sequence_length,3))

        for j in range(sequence_length):
            if cur_loc == data['sequence'][i+j]:
                seq[j] = data[features].iloc[i+j].values 
            else:
                list_finished = False
        if list_finished:
            sequences.append(seq)
            sequence_labels.append(labels[i])
    return np.array(sequences), np.array(sequence_labels)

labels = data['label'].values
sequences, sequence_labels = create_sequences(data, 5, ['mTime','rTime','rtt_diff'], labels)
print(sequences.shape)
X_test = torch.tensor(sequences, dtype=torch.float32).transpose(1,2)
y_test = torch.tensor(sequence_labels, dtype=torch.long)
batch_size = 1
dataset = TensorDataset(X_test,y_test)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
total = 0
correct = 0
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
print(f'Accuracy of the network: {100 * correct // total} %')


