import torch
import torch.optim as optim
import os
import json
from torch import nn
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from utils.utils import custom_collate_fn
from torchvision import transforms
from dataset.ucf_dataset import UCFCrimeDataset
from models.lstm_module import LSTMModule
from models.model import VideoClassifier
from torch.utils.data import random_split
import yaml

# Load config.yaml
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


# Hyperparameters
batch_size = config['dataset']['batch_size']
learning_rate = config['training']['learning_rate']
num_epochs = config['training']['num_epochs']
weight_decay = config['training']['weight_decay']
sequence_length = config['dataset']['sequence_length']
image_size = config['dataset']['image_size']
num_workers = config['dataset']['num_workers']

# Model Config
hidden_dim = config['model']['hidden_dim']
num_layers = config['model']['num_layers']
dropout = config['model']['dropout']
bidirectional = config['model']['bidirectional']
cnn_type = config['model']['cnn_type']
pretrained = config['model']['pretrained']

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using " + torch.cuda.get_device_name(0))


# Transformations
transform = transforms.Compose([
    transforms.Resize((image_size + 32, image_size + 32)),  # Resize a bit larger
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load full dataset
with open("data/labels.json", "r") as f:
    label_dict = json.load(f)

full_dataset = UCFCrimeDataset(
    root_dir=config['dataset']['root_dir'],
    label_dict=json.load(open(config['dataset']['label_file'], 'r')),
    sequence_length=sequence_length,
    transform=transform
)

# Calculate split sizes
val_split = 0.2  # 20% for validation
train_size = int((1 - val_split) * len(full_dataset))
val_size = len(full_dataset) - train_size


# Split datasets
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=num_workers,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,  # no shuffle for validation
    collate_fn=custom_collate_fn,
    num_workers=num_workers,
    pin_memory=True
)

# Model initialization
if cnn_type == "efficientnet_b0":
    from efficientnet_pytorch import EfficientNet
    cnn = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
    cnn._fc = nn.Identity()
else:
    raise NotImplementedError(f"CNN model {cnn_type} not supported yet.")

rnn = LSTMModule(
    input_dim=1280,  # EfficientNet-b0 output feature size
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
    bidirectional=bidirectional
).to(device)

model = VideoClassifier(cnn, rnn).to(device)

# Freezing CNN
model.freeze_cnn()

# Optimizer setup
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

criterion = torch.nn.BCEWithLogitsLoss()

if __name__ == '__main__':
# Training loop
    for epoch in range(num_epochs):
        # ----------------- TRAINING -----------------
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for sequences, labels, lengths in train_loader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.float().to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            outputs = model(sequences, lengths)

            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds.squeeze() == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = train_correct / train_total
        train_loss = train_loss / len(train_loader)

        # ----------------- VALIDATION -----------------
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sequences, labels, lengths in val_loader:
                sequences = sequences.to(device, non_blocking=True)
                labels = labels.float().to(device)
                lengths = lengths.to(device)

                outputs = model(sequences, lengths)
                loss = criterion(outputs.view(-1), labels)

                val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds.squeeze() == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        # ----------------- PRINT RESULTS -----------------
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # ----------------- SAVE CHECKPOINT -----------------
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        }, f'checkpoints/epoch_{epoch+1}.pth')

print("Training completed!")