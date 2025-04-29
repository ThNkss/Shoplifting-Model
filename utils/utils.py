import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded_sequences, labels, lengths
