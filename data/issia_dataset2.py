import os

import data.augmentation as augmentation
from data.issia_dataset import IssiaDataset


def create_issia_dataset(dataset_path, cameras, mode, only_ball_frames=False):
    # Get ISSIA datasets for multiple cameras
    assert mode == 'train' or mode == 'val'
    assert os.path.exists(dataset_path), 'Cannot find dataset: ' + str(dataset_path)

    train_image_size = (720, 1280)
    val_image_size = (1080, 1920)
    if mode == 'train':
        transform = augmentation.TrainAugmentation2((1080, 1920))
    elif mode == 'val':
        transform = augmentation.NoAugmentation(size=val_image_size)

    dataset = IssiaDataset(dataset_path, cameras, transform, only_ball_frames=only_ball_frames)
    return dataset
