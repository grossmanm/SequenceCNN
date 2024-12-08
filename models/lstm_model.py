import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder

from data_processing.data_sequence import create_dataset

torch.manual_seed(1)

class LSTMPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dimension):
        super(LSTMPredictor, self).__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.distancelayer = nn.Linear(hidden_dim, output_dimension)

    def forward(self, sequence):
        lstm_out, hidden = self.lstm(sequence)
        lstm_out = lstm_out[-1, :, :] # select the final output

        out = self.distancelayer(lstm_out)
        
        return out
    
    def predict(self, sequence):
        raw_predictions = self.forward(sequence)
        return torch.argmax(raw_predictions)

class SequenceDataset(Dataset):
    def __init__(self, data_file, train=True, data_percent=80, transform=None, target_transform=None):
        data = pd.read_csv(data_file)

        if train:
            data = data.head(int(len(data)* data_percent/100))
        else:
            test_percent = 100 - data_percent
            data = data.tail(int(len(data)* test_percent/100))

        self.sequences, self.labels = create_dataset(data)

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]

        if self.transform:
            sequence = self.transform(sequence)
        if self.target_transform:
            label = self.target_transform(label)
        return sequence, label
    

"""
Data and Model Setup
"""
model = LSTMPredictor(embedding_dim=3, hidden_dim=3, output_dimension=10)

print("Loading Data")
percent = 100
train_dataset = SequenceDataset('data/r1_timings_distance.csv', train=True, data_percent=percent)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = SequenceDataset('data/r1_timings_distance.csv', train=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

print(f"Training Data: {len(train_dataset)}")


example, label = train_dataset[0]
print(f"Sample: {example}")
print(f"Shape: {example.shape}")
print(f"Label: {label}")

# demonstrates a calculation with the model
example = example.type(torch.float32)[:, None, :]
out = model.forward(example)
print(f"Model Output: {out}")
model.zero_grad()


"""
Training
"""
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 20
loss_function = nn.CrossEntropyLoss()

def train_model(model, optimizer, loss_function, dataloader, num_epochs):
    model.train()
    for epoch in range(num_epochs):

        for inputs, labels in dataloader:
            optimizer.zero_grad()

            inputs = inputs.swapaxes(0, 1) # switch batches and sequence

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if epoch % (num_epochs / 10) == 0:
            print(f"Epoch: {epoch} Loss: {loss}")

print("Training the Model")
train_model(model, optimizer,loss_function, train_dataloader, num_epochs)


"""
Testing
"""
def test_model(model: nn.Module, loss_function, dataloader, dataset):
    model.eval()
    total_correct = 0
    total_loss = 0

    n_batches = 0
    for inputs, labels in dataloader:
        inputs = inputs.swapaxes(0, 1) # switch batches and sequence

        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        n_correct = torch.sum(torch.argmax(outputs, dim=1) == labels)

        total_loss += loss
        total_correct += n_correct
        n_batches += 1
    
    n_sequences = len(dataset)
    print(f"Accuracy: {total_correct/n_sequences}")
    print(f"Average Loss: {total_loss/n_batches}")

print(f"Testing the Model")
test_model(model, loss_function, train_dataloader, train_dataset)