import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms._functional_tensor as F_t
from torchvision.transforms import transforms

from data.augmentation import NORMALIZATION_MEAN, NORMALIZATION_STD

normalize = transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)

def apply_all_transformations(image, boxes, labels, affine_params, crop_params, jitter_params, img_shape):
    """
    Apply Affine, Crop, and Color Jitter transformations in one step to reduce memory usage.
    """
    height, width = img_shape  # Image dimensions

    ## APPLY AFFINE TRANSFORMATION ##
    # Apply affine transformation to the image
    angle, translate, scale, shear, flip = affine_params
    center = (width * 0.5, height * 0.5)
    image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=shear,
                     interpolation=F.InterpolationMode.BILINEAR)

    # Compute affine transformation matrix manually for bounding boxes
    affine_matrix = F._get_inverse_affine_matrix(center, angle, translate, scale, shear)
    inverse_affine_matrix = torch.tensor(affine_matrix).reshape(2, 3)

    # Apply transformation to bounding boxes
    boxes, labels = apply_transform_and_clip(boxes, labels, inverse_affine_matrix, (width, height))

    # Apply Horizontal Flip
    if flip:
        image = F.hflip(image)
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]  # Flip x-coordinates

    ## APPLY CROP ##
    if crop_params is not None:
        crop_i, crop_j, crop_h, crop_w = crop_params
        image = F.crop(image, top=crop_i, left=crop_j, height=crop_h, width=crop_w)
        boxes[:, :2] -= torch.tensor([crop_j, crop_i], device=boxes.device)  # Adjust top-left
        boxes[:, 2:4] -= torch.tensor([crop_j, crop_i], device=boxes.device)  # Adjust bottom-right
        boxes, labels = clip(boxes, labels, (crop_w, crop_h))

    ## APPLY COLOR JITTER ##
    brightness, contrast, saturation, hue = jitter_params
    image = F_t.adjust_brightness(image, brightness)
    image = F_t.adjust_contrast(image, contrast)
    image = F_t.adjust_saturation(image, saturation)
    image = F_t.adjust_hue(image, hue)

    # Normalize after transformations
    image = normalize(image)

    return image, boxes, labels


def apply_affine_to_tensor(image, boxes, labels, angle, translate, scale, shear, flip, img_shape):
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
    boxes, labels = apply_transform_and_clip(boxes, labels, inverse_affine_matrix, (width, height))
    if isinstance(boxes, np.ndarray):
        boxes = torch.tensor(boxes, dtype=torch.float32)

    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, dtype=torch.int64)

    if flip:
        image = F.hflip(image)
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

    return image, boxes, labels


def apply_transform_and_clip(boxes, labels, M, shape):
    """
    Apply an affine transformation to bounding boxes and filter out boxes that fall outside image boundaries.

    :param boxes: Tensor of shape (N, 4) with (x1, y1, x2, y2) coordinates.
    :param labels: Tensor of shape (N,) with class labels.
    :param M: Affine transformation matrix of shape (3, 3).
    :param shape: (width, height) tuple defining the image size.
    :return: Filtered transformed boxes and corresponding labels.
    """
    assert len(boxes) == len(labels)

    # Add ones for affine transformation
    ones = torch.ones((len(boxes), 1), device=boxes.device)
    ext_pts1 = torch.cat((boxes[:, :2], ones), dim=1).T  # Upper-left corner
    ext_pts2 = torch.cat((boxes[:, 2:4], ones), dim=1).T  # Lower-right corner

    # Apply affine transformation
    transformed_pts1 = torch.mm(M[:2], ext_pts1).T
    transformed_pts2 = torch.mm(M[:2], ext_pts2).T

    # Determine min/max coordinates after transformation
    transformed_boxes = torch.zeros_like(boxes)
    transformed_boxes[:, 0] = torch.min(transformed_pts1[:, 0], transformed_pts2[:, 0])
    transformed_boxes[:, 1] = torch.min(transformed_pts1[:, 1], transformed_pts2[:, 1])
    transformed_boxes[:, 2] = torch.max(transformed_pts1[:, 0], transformed_pts2[:, 0])
    transformed_boxes[:, 3] = torch.max(transformed_pts1[:, 1], transformed_pts2[:, 1])

    # Use your filtering-only `clip` function
    return clip(transformed_boxes, labels, shape)


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


def apply_color_jitter_to_tensor(image, boxes, labels, brightness_factor=0, contrast_factor=0, saturation_factor=0, hue_factor=0):
    """
    Apply ColorJitter transformation (brightness, contrast, saturation, hue) to a tensor image.

    :param image: Tensor of shape (C, H, W)
    :param brightness_factor: Brightness factor (0 gives a black image, 1 gives the original, >1 brightens)
    :param contrast_factor: Contrast factor (0 gives a solid gray image, 1 gives the original, >1 increases contrast)
    :param saturation_factor: Saturation factor (0 removes color, 1 is the original, >1 increases saturation)
    :param hue_factor: Hue shift factor (-0.5 to 0.5, where 0 keeps the original hue)
    :return: Transformed image
    """
    image = F_t.adjust_brightness(image, brightness_factor)
    image = F_t.adjust_contrast(image, contrast_factor)
    image = F_t.adjust_saturation(image, saturation_factor)
    image = F_t.adjust_hue(image, hue_factor)

    norm = transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
    image = norm(image)

    return image, boxes, labels
