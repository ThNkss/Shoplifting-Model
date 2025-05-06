import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UCFCrimeDataset(Dataset):
    def __init__(self, root_dir, label_dict, sequence_length, transform=None):
        self.root_dir = root_dir  # Path to the directory containing video folders
        self.label_dict = label_dict
        self.sequence_length = sequence_length
        self.transform = transform
        self.video_folders = [
            folder for folder in os.listdir(self.root_dir)
            if folder in self.label_dict
        ]

    def __getitem__(self, index):
        print(f"Fetching item {index}")
        try:
            video_folder = self.video_folders[index]
            video_path = os.path.join(self.root_dir, video_folder)
            
            # Extract frames from video
            frames = extract_frames_from_video(video_path)

            # Handle case where no frames are found
            if len(frames) == 0:
                print(f"Warning: No frames found for video {video_folder}. Skipping.")
                
                # If sequence_length is None, set it to a default value (e.g., 16)
                if self.sequence_length is None:
                    self.sequence_length = 16  # Default sequence length

                # Return a tensor of zeros with the default sequence length and other dimensions
                return torch.zeros(self.sequence_length, 3, 224, 224, dtype=torch.float32), 0, self.sequence_length

            # Retrieve the label for this video folder from the label_dict
            label = self.label_dict.get(video_folder, None)
            
            # If label is None, handle it (e.g., raise an exception or use a default value)
            if label is None:
                raise ValueError(f"Label for video {video_folder} not found in label_dict.")

            # Convert frames to tensor and permute to (seq_len, C, H, W)
            frames_tensor = torch.tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2)
            
            # Return frames_tensor, label, and the length of the sequence (number of frames)
            return frames_tensor, label, len(frames_tensor)
        except Exception as e:
            print(f"❌ Failed to load index {index}: {e}")
            return None  # BAD: this may silently be dropped if collate_fn can't handle it


    def __len__(self):
        # The length of the dataset is the number of video folders
        return len(self.video_folders)


def extract_frames_from_video(video_path, max_frames=1000):
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return frames

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))  # Resize for consistency
        frame = frame[:, :, ::-1]  # BGR to RGB
        frame = frame / 255.0      # Normalize
        frames.append(frame.astype(np.float32))
        frame_count += 1

    cap.release()
    return frames