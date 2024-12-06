import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class LSTMPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dimension):
        super(LSTMPredictor, self).__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.distancelayer = nn.Linear(hidden_dim, output_dimension)

    def forward(self, sequence):

        lstm_out, hidden = self.lstm(sequence.view(len(sequence), 1, -1))

        out = self.distancelayer(lstm_out.view(len(sequence), -1))

        return out

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
