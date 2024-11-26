Python Version: Python 3.12.3

Setup:
pip -r requirements.txt

Data:
The data used to train this model should be in *.csv format and contain columns
rtt_server, rtt_receiver, rtt_diff, receiver_device_id, sender_cc, receiver_cc 
rtt_server: numerical value indicating the round trip time between sender and messaging server
rtt_receiver: numerical value indicating the round trip time between sender and receiver
rtt_diff: numerical value of rtt_receiver-rtt_server
receiver_device_id: key value indicating the id of the receiving device
sender_cc: two letter country code id of the country the sender is in
receiver_cc: two letter country code id of the country the receiver is in

Data Preprocessing:
Our process uses distance values instead of country codes in an attempt to make results more transferable between regions.
Run this before training to obtain the data that will be used in train.py
data_processing.py
NOTE: This process reduces overall model accuracy


To train:
train.py

To test:
train.py