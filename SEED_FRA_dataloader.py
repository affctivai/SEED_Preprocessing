import os
import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

class SEEDFRADataset(Dataset):
    def __init__(self, directory_path):
        self.data, self.labels, self.subjects = self.load_data(directory_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'label': self.labels[idx],
            'subject': self.subjects[idx]
        }

    def load_data(self, directory_path):
        all_data = {}
        for filename in os.listdir(directory_path):
            if filename.endswith('.mat') and 'label' not in filename:
                subject_id = int(filename.split('_')[0])  # 파일 이름에서 subject ID 추출
                input_file_path = os.path.join(directory_path, filename)
                mat_data = loadmat(input_file_path)
                eeg_keys = [key for key in mat_data.keys() if 'eeg' in key]
                file_data = [mat_data[key] for key in eeg_keys]
                all_data[subject_id] = file_data

        all_cov_matrices = {}
        for subject_id, file_data in all_data.items():
            file_cov_matrices = [np.cov(matrix, rowvar=True) for matrix in file_data]
            all_cov_matrices[subject_id] = file_cov_matrices

        labels = [1, -1, 0, -1, 1, 0, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, -1, 0, 1]  # SEED_FRA 레이블

        SEED_data, SEED_label, SEED_subject = [], [], []
        for subject_id, file_cov_matrices in all_cov_matrices.items():
            for cov in file_cov_matrices:
                SEED_data.append(cov)
                SEED_label.append(labels)
                SEED_subject.append(subject_id)

        return np.array(SEED_data), np.array(SEED_label), np.array(SEED_subject)

SEED_FRA_directory = '/home/isaac/data/SEED_FRA/French/Preprocessed'
dataset_FRA = SEEDFRADataset(SEED_FRA_directory)
torch.save(dataset_FRA, '/home/isaac/Research/SEED_FRA/SEED_FRA_dataset.pth')
print('SEED FRA save complete')
