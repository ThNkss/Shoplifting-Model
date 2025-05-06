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
else:
    print("using cpu")

# Transformations
transform = transforms.Compose([
    transforms.Resize((image_size + 32, image_size + 32)),  # Resize a bit larger
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load labels
with open("data/labels.json", "r") as f:
    label_dict = json.load(f)


# Load training and validation datasets
train_dataset = UCFCrimeDataset(
    root_dir=config['dataset']['train_root_dir'],
    label_dict=label_dict,
    sequence_length=sequence_length,
    transform=transform
)

val_dataset = UCFCrimeDataset(
    root_dir=config['dataset']['val_root_dir'],
    label_dict=label_dict,
    sequence_length=sequence_length,
    transform=transform
)

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

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# Model initialization
if cnn_type == "efficientnet_b0":
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

# === CHECKPOINT PATH AND RESUME SETUP ===
checkpoint_dir = 'checkpoints'
checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
epoch_data_dir = 'epoch_data'  # Directory to save epoch data
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(epoch_data_dir, exist_ok=True)

# Resume from checkpoint if exists
start_epoch = 1
if os.path.exists(checkpoint_path):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Continue from next epoch
    print(f"Resumed from epoch {start_epoch}")

# Training loop
if __name__ == '__main__':
    print(f"Training for {num_epochs} epochs...")

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Starting epoch [{epoch}/{num_epochs}]")

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
            classification, _ = model(sequences, lengths)
            loss = criterion(classification.view(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.sigmoid(classification) > 0.5
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

                classification, _ = model(sequences, lengths)
                loss = criterion(classification.view(-1), labels)

                val_loss += loss.item()
                preds = torch.sigmoid(classification) > 0.5
                val_correct += (preds.squeeze() == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        # ----------------- PRINT RESULTS -----------------
        print(f"Epoch [{epoch}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # ----------------- SAVE EPOCH DATA -----------------
        epoch_data = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }
        torch.save(epoch_data, os.path.join(epoch_data_dir, f'epoch_{epoch}.pth'))

        # ----------------- SAVE CHECKPOINT -----------------
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

    print("Training completed!")
