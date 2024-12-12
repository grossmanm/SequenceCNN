"""
Tests the LSTM model on the various timing datasets
"""
import torch
import pandas as pd

from lstm_model import LSTMPredictor

model = LSTMPredictor(embedding_dim=3, hidden_dim=3, output_dimension=10)
model.load_state_dict(torch.load("lstm_model.pth", weights_only=True))
model.eval()

data_files = ["test_data_d_1.csv", "test_data_d_01.csv", "test_data_d_001.csv", "test_data_d_0001.csv", "test_data_d_10.csv", "test_data.csv", "vpn_test_data.csv"]

def parse_sequences(data: pd.DataFrame):
    sequences = []
    labels = []

    features = ['rtt_server', 'rtt_receiver', 'rtt_diff']

    current_sequence = []
    current_sequence_number = None

    for i, row in data.iterrows():
        if current_sequence_number is None:
            current_sequence_number = row['sequence']
        
        current_sequence.append(torch.tensor(row[features].values.astype(float)))

        if current_sequence_number != row['sequence'] or i == len(data) - 1:
            current_sequence_number = row['sequence']
            
            tensor = torch.stack(current_sequence).type(torch.float32)
            sequences.append(tensor)
            labels.append(row['SR_Distance'])

            current_sequence = []
    
    return sequences, labels

data = pd.read_csv(data_files[0])

sequences, labels = parse_sequences(data)
print(labels)