import random

from torch.utils.data import Sampler

from data.issia_dataset import IssiaDataset
from data.issia_dataset2 import IssiaDataset as IssiaDataset2


class SlidingWindowSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.sample_ndx = []
        self.no_ball_images_ndx = []
        self.camera_groups = {}  # Stores indices grouped by camera_id
        self.generate_samples()

    def generate_samples(self):
        issia_dataset_ndx = None
        for ndx, ds in enumerate(self.data_source.datasets):
            if isinstance(ds, IssiaDataset) or isinstance(ds, IssiaDataset2):
                issia_dataset_ndx = ndx

        issia_ds = self.data_source.datasets[issia_dataset_ndx]
        self.sample_ndx = issia_ds.image_list  # List of (image_path, camera_id, image_id)
        self.no_ball_images_ndx = issia_ds.no_ball_images_ndx

        # Group indices by camera_id
        self.camera_groups = {}
        for idx, (_, camera_id, _) in enumerate(self.sample_ndx):
            if camera_id not in self.camera_groups:
                self.camera_groups[camera_id] = []
            self.camera_groups[camera_id].append(idx)

    def __iter__(self):
        all_batches = []

        for camera_id, indices in self.camera_groups.items():
            num_samples = len(indices)

            # Generate sliding window batches per camera
            batches = [indices[i: i + self.batch_size] for i in range(num_samples - self.batch_size + 1)]

            # Filter out "no ball" sequences based on probability
            filtered_batches = [
                batch for batch in batches
                if not (all(idx in self.no_ball_images_ndx for idx in batch) and random.random() < 0.7)
            ]

            all_batches.extend(filtered_batches)

        random.shuffle(all_batches)  # Shuffle batches, not indices inside batches
        return iter(all_batches)

    def __len__(self):
        return sum(len(indices) - self.batch_size + 1 for indices in self.camera_groups.values())