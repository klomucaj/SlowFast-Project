import os
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):  # Inherit from Dataset
    def _init_(self, video_dir, annotation_dir, transform=None):
        self.video_files = sorted(os.listdir(video_dir))
        self.annotation_files = sorted(os.listdir(annotation_dir))
        self.video_dir = video_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.action_to_index = self._load_action_mapping()

    def _load_action_mapping(self):
        return {"action1": 0, "action2": 1, "action3": 2}

    def _getitem_(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])
        video = self._load_video(video_path)
        label = self._load_annotation(annotation_path)

        if self.transform:
            video = self.transform(video)

        return video, label

    def _load_video(self, video_path):
        # Example: Load video frames as tensors
        return torch.randn(3, 16, 224, 224)  # Dummy video data (C, T, H, W)

    def _load_annotation(self, annotation_path):
        with open(annotation_path, 'r') as f:
            return self.action_to_index[f.read().strip()]

    def _len_(self):
        return len(self.video_files)