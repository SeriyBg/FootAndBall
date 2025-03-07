import numpy as np
import torch
import torchvision.transforms.functional as F

from data.augmentation import apply_transform_and_clip


def apply_affine_to_tensor(image, boxes, labels, angle, translate, scale, shear, img_shape):
    """
    Apply affine transformation directly to a tensor image and corresponding bounding boxes.
    """
    height, width = img_shape  # Get image dimensions

    # Define the affine transformation matrix
    center = (width * 0.5, height * 0.5)
    image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=F.InterpolationMode.BILINEAR)

    # Create affine transformation matrix manually (to apply on bounding boxes)
    affine_matrix = F._get_inverse_affine_matrix(center, angle, translate, scale, shear)
    inverse_affine_matrix = torch.tensor(affine_matrix).reshape(2, 3)

    # Apply transformation to bounding boxes
    boxes, labels = apply_transform_and_clip(boxes, labels, inverse_affine_matrix.numpy(), (width, height))
    if isinstance(boxes, np.ndarray):
        boxes = torch.tensor(boxes, dtype=torch.float32)

    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, dtype=torch.int64)

    return image, boxes, labels


def clip(boxes, labels, shape):
    """
    Clip bounding boxes to ensure they stay within image boundaries.

    :param boxes: Tensor of shape (N, 4) with (x1, y1, x2, y2) coordinates.
    :param labels: Tensor of shape (N,) with class labels.
    :param shape: (width, height) tuple.
    :return: Filtered boxes and labels within image bounds.
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    mask = (x1 >= 0) & (y1 >= 0) & (x2 < shape[0]) & (y2 < shape[1])  # Keep valid boxes

    return boxes[mask], labels[mask]  # Return only valid boxes & labels


def apply_crop_to_tensor(image, boxes, labels, i, j, out_h, out_w):
    """
    Crop the image and adjust bounding boxes accordingly.

    :param image: Torch tensor of shape (C, H, W).
    :param boxes: Tensor of shape (N, 4) with (x1, y1, x2, y2) coordinates.
    :param labels: Tensor of shape (N,) with class labels.
    :param i: Top-left y-coordinate for cropping.
    :param j: Top-left x-coordinate for cropping.
    :param out_h: Height of cropped region.
    :param out_w: Width of cropped region.
    :return: Cropped image, adjusted boxes, filtered labels.
    """
    # Crop the image
    image = F.crop(image, top=i, left=j, height=out_h, width=out_w)

    # Adjust bounding boxes
    boxes[:, :2] -= torch.tensor([j, i], device=boxes.device)
    boxes[:, 2:4] -= torch.tensor([j, i], device=boxes.device)

    # Clip boxes to remain inside the cropped region
    boxes, labels = clip(boxes, labels, (out_w, out_h))

    return image, boxes, labels
