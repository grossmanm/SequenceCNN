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


model = DeliveryNotificationCNN(input_features=3, sequence_length=5, num_classes=10)

optimizer_type = 'Adam'
learning_rate = 0.001

if optimizer_type == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
elif optimizer_type == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer_type == 'RMSProp':
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
else:
    raise ValueError("Unsupported optimizer type")


num_epochs = 60
loss_function = nn.CrossEntropyLoss()

def train_model(model, optimizer, loss_function, dataloader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")


data = pd.read_csv('data/r1_timings_distance.csv')

sequence_length = 5
features = ['rtt_server', 'rtt_receiver', 'rtt_diff'] #features

# normalization
#data[features] = (data[features]-data[features].mean())/data[features].std()

# label encoder
label_encoder = LabelEncoder()
data['SR_Distance'] = label_encoder.fit_transform(data['SR_Distance'])

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(label_mapping)

labels = data['SR_Distance'].values
# create sequences
def create_sequences(data,sequence_length,features,labels):
    sequences = []
    sequence_labels = []
    for i in range(len(data)-sequence_length):
        list_finished = True
        cur_loc = data['receiver_device_id'][i]
        seq = np.zeros((sequence_length,3))

        for j in range(sequence_length):
            if cur_loc == data['receiver_device_id'][i+j]:
                seq[j] = data[features].iloc[i+j].values 
            else:
                list_finished = False
        if list_finished:
            sequences.append(seq)
            sequence_labels.append(labels[i])
    return np.array(sequences), np.array(sequence_labels)
    

print("Creating Dataset")       
        
sequences, sequence_labels = create_sequences(data, sequence_length, features, labels)
train_sequences = sequences[int(len(sequences)*.8):]
train_labels = sequence_labels[int(len(sequence_labels)*.8):]
test_sequences = sequences[:int(len(sequences)*.8)]
test_labels = sequence_labels[:int(len(sequence_labels)*.8)]
print(train_labels)

X_train = torch.tensor(train_sequences, dtype=torch.float32).transpose(1,2)
y_train = torch.tensor(train_labels, dtype=torch.long)
X_test = torch.tensor(test_sequences, dtype=torch.float32).transpose(1,2)
y_test = torch.tensor(test_labels, dtype=torch.long)


batch_size = 2
dataset = TensorDataset(X_train, y_train)

print("Dataset creation. Initializing Data Loader")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Data Loader initialized. Begin Training")

train_model(model, optimizer, loss_function, dataloader, num_epochs)

print("Saving Trained Model")
torch.save(model, "model1.pth")
# evaluation 
print("Begin Evaluation")
dataset_test = TensorDataset(X_test, y_test)

total = 0
correct = 0
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
with torch.no_grad():
    for inputs, labels in dataloader_test:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

print(f'Accuracy of the network: {100 * correct // total} %')