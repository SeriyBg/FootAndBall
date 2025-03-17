# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

import random

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Sampler, DataLoader, ConcatDataset

import data.augmentation as augmentation
from data.batch_augmentation import apply_all_transformations
from data.issia_dataset import IssiaDataset
from data.issia_dataset import create_issia_dataset
from data.sampler import SlidingWindowSampler
from data.spd_bmvc2017_dataset import create_spd_dataset
from misc.config import Params


def make_dataloaders(params: Params):
    if params.issia_path is None:
        train_issia_dataset = None
    else:
        train_issia_dataset = create_issia_dataset(params.issia_path, params.issia_train_cameras, mode='train', only_ball_frames=False)
        if len(params.issia_val_cameras) == 0:
            val_issia_dataset = None
        else:
            val_issia_dataset = create_issia_dataset(params.issia_path, params.issia_val_cameras, mode='val',
                                                     only_ball_frames=True)

    if params.spd_path is None:
        train_spd_dataset = None
    else:
        train_spd_dataset = create_spd_dataset(params.spd_path, params.spd_set, mode='train')

    dataloaders = {}
    if val_issia_dataset is not None:
        dataloaders['val'] = DataLoader(val_issia_dataset, batch_size=params.batch_size, num_workers=params.num_workers,
                                        pin_memory=True, collate_fn=my_collate)

    if train_spd_dataset is None:
        train_dataset = ConcatDataset([train_issia_dataset])
    else:
        train_dataset = ConcatDataset([train_issia_dataset, train_spd_dataset])
    batch_sampler = BalancedSampler(train_dataset)
    dataloaders['train'] = DataLoader(train_dataset,
                                      sampler=batch_sampler,
                                      batch_size=params.batch_size,
                                      num_workers=params.num_workers, pin_memory=True, collate_fn=my_collate)
    # batch_sampler = SlidingWindowSampler(train_dataset, params.batch_size, step=params.sliding_window_stride)
    # dataloaders['train'] = DataLoader(train_dataset,
    #                                   batch_sampler=batch_sampler,
    #                                   num_workers=params.num_workers, pin_memory=True, collate_fn=transform_collate)

    return dataloaders


def my_collate(batch):
    images = torch.stack([e[0] for e in batch], dim=0)
    # older_images = images
    boxes = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    # visualize_batch(older_images, images)
    return images, boxes, labels


class WrappedDataLoader(DataLoader):

    def __init__(self, dataloader, print_every=100):
        self.dataloader = dataloader  # Store the original DataLoader
        self.print_every = print_every
        self.counter = 0  # Batch counter

    def __iter__(self):
        self.counter = 0  # Reset counter for new epoch
        for batch in iter(self.dataloader):
            self.counter += 1
            if self.counter % self.print_every == 0:
                print(f"Accessed {self.counter} batches.")
            yield batch

    def __getattr__(self, name):
        """Delegate attribute access to the original DataLoader."""
        return getattr(self.dataloader, name)


def transform_collate(batch):
    images, boxes, labels = zip(*batch)
    # old_images = images

    # Get image dimensions (assuming all images have the same size)
    height, width = images[0].shape[1], images[0].shape[2]

    # Initialize the affine transformation **once per batch**
    affine_params = augmentation.RandomAffine(degrees=5, scale=(0.8, 1.2), p_hflip=0.5).get_params(height, width)
    flip = torch.rand(1).item() < 0.5

    # Initialize the affine transformation **once per batch**
    # train_image_size = (720, 1280)
    # train_image_size = (1080, 1920)
    # crop_params = augmentation.RandomCrop(train_image_size).get_params(height, width)

    # Initialize color jitter transformation **once per batch**
    # brightness_factor = torch.empty(1).uniform_(0.8, 1.2).item()
    # contrast_factor = torch.empty(1).uniform_(0.8, 1.2).item()
    # saturation_factor = torch.empty(1).uniform_(0.8, 1.2).item()
    # hue_factor = torch.empty(1).uniform_(-0.05, 0.05).item()
    # jitter_params = (brightness_factor, contrast_factor, saturation_factor, hue_factor)

    # Apply the same transformation to all images in the batch
    transformed_batch = [
        apply_all_transformations(img, b, l,
                                  (*affine_params, flip),
                                  None, #(*crop_params, train_image_size[0], train_image_size[1]),
                                  None, #jitter_params,
                                  (height, width))
        for img, b, l in batch]

    # Unpack transformed images, boxes, and labels
    images, boxes, labels = zip(*transformed_batch)

    # Convert back to the correct format. Stack transformed images into a single tensor
    images = torch.stack(images, dim=0)

    # visualize_batch(old_images[:5], images[:5])

    return images, boxes, labels


def visualize_batch(old_images, images):
    batch_size = len(old_images)
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))

    if batch_size == 1:  # Ensure iterable for single image batch
        axes = [[axes[0]], [axes[1]]]

    for i in range(batch_size):
        # Convert tensors to NumPy arrays for visualization
        old_img_np = old_images[i].permute(1, 2, 0).cpu().numpy()
        new_img_np = images[i].permute(1, 2, 0).cpu().numpy()

        # Top row: Original images
        axes[0][i].imshow(old_img_np)
        axes[0][i].axis("off")
        axes[0][i].set_title(f"O{i}")

        # Bottom row: Transformed images
        axes[1][i].imshow(new_img_np)
        axes[1][i].axis("off")
        axes[1][i].set_title(f"T{i}")

    plt.tight_layout()
    plt.show()

class BalancedSampler(Sampler):
    # Sampler sampling the same number of frames with and without the ball
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        self.sample_ndx = []
        self.generate_samples()

    def generate_samples(self):
        # Sample generation function expects concatenation of 2 datasets: one is ISSIA CNR and the other is SPD
        # or only one ISSIA CNR dataset.
        #assert len(self.data_source.datasets) <= 2
        issia_dataset_ndx = None
        spd_dataset_ndx = None
        for ndx, ds in enumerate(self.data_source.datasets):
            if isinstance(ds, IssiaDataset):
                issia_dataset_ndx = ndx
            else:
                spd_dataset_ndx = ndx

        assert issia_dataset_ndx is not None, 'Training data must contain ISSIA CNR dataset.'

        issia_ds = self.data_source.datasets[issia_dataset_ndx]
        n_ball_images = len(issia_ds.ball_images_ndx)
        # no_ball_images = 0.5 * ball_images
        n_no_ball_images = min(len(issia_ds.no_ball_images_ndx), int(0.5 * n_ball_images))
        issia_samples_ndx = list(issia_ds.ball_images_ndx) + random.sample(list(issia_ds.no_ball_images_ndx),
                                                                           n_no_ball_images)
        if issia_dataset_ndx > 0:
            # Add sizes of previous datasets to create cummulative indexes
            issia_samples_ndx = [e + self.data_source.cumulative_sizes[issia_dataset_ndx-1] for e in issia_samples_ndx]

        if spd_dataset_ndx is not None:
            spd_dataset = self.data_source.datasets[spd_dataset_ndx]
            n_spd_images = min(len(spd_dataset), int(0.5 * n_ball_images))
            spd_samples_ndx = random.sample(range(len(spd_dataset)), k=n_spd_images)
            if spd_dataset_ndx > 0:
                # Add sizes of previous datasets to create cummulative indexes
                spd_samples_ndx = [e + self.data_source.cumulative_sizes[spd_dataset_ndx - 1] for e in spd_samples_ndx]
        else:
            n_spd_images = 0
            spd_samples_ndx = []

        self.sample_ndx = issia_samples_ndx + spd_samples_ndx
        random.shuffle(self.sample_ndx)

    def __iter__(self):
        self.generate_samples()         # Re-generate samples every epoch
        for ndx in self.sample_ndx:
            yield ndx

    def __len(self):
        return len(self.sample_ndx)


def collate_fn(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    images, targets = zip(*batch)
    return torch.stack(images, 0), targets
