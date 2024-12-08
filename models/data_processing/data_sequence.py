import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


# Creates a workable dataset from the raw data
# Uses the default of 5 packets
def create_dataset(data: pd.DataFrame, validate_timestamps=False):
    sequences = []
    sequence_timestamps = []
    labels = []

    label_encoder = LabelEncoder()
    data['SR_Distance'] = label_encoder.fit_transform(data['SR_Distance'])
    # label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    features = ['rtt_server', 'rtt_receiver', 'rtt_diff']

    current_sequence = []
    current_timestamp_sequence = []
    sequence_timestamp = None

    # Gather sequences of five consequetive packets
    for i, row in data.iterrows():
        if sequence_timestamp is None:
            sequence_timestamp = row['timestamp']
        else:
            if not sequence_timestamp == row['timestamp']:
                current_sequence = []
                current_timestamp_sequence = []
                sequence_timestamp = row['timestamp']
        
        current_sequence.append(torch.tensor(row[features].values.astype(float)))
        current_timestamp_sequence.append(row['timestamp']) 

        if len(current_sequence) == 5:
            tensor = torch.stack(current_sequence).type(torch.float32)
            sequences.append(tensor)

            """
            if tensor.shape != torch.Size([5, 3]):
                print("Size error!")
                print(tensor.shape)
                exit()
            """
            
            
            sequence_timestamps.append(current_timestamp_sequence)
            labels.append(row['SR_Distance'])

            current_sequence = []
            current_timestamp_sequence = []
            sequence_timestamp = None

    # Code to validate the collected sequences
    if validate_timestamps:
        validated_timestamps = True
        for i, timestamp_sequence in enumerate(sequence_timestamps):
            if len(timestamp_sequence) == 5:
                timestamp = timestamp_sequence[0]
                for ts in timestamp_sequence:
                    if not ts == timestamp:
                        print(f"Invalid timestamp sequence!")
                        validated_timestamps = False
                        break
            else:
                print(f"Invalid number of timestamps {len(timestamp_sequence)} not 5")
                validated_timestamps = False
                break

            if not validated_timestamps:
                print(f"Found at index: {i}")
                break

    return sequences, labels


if __name__ == "__main__":
    data = pd.read_csv('data/r1_timings_distance.csv')

    # data = data.head(2000)
    print(data)
    # Create the dataset
    sequences, labels = create_dataset(data)
    print(len(sequences))
