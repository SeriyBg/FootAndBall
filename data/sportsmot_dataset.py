import torch


class SportsmotDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform, splits, ann_file_path):
        self.dataset_path = dataset_path
        self.transform = transform
        self.splits = splits
        self.ann_file_path = ann_file_path
        self.image_list = []

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, ndx):
        pass
