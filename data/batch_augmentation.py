import torch
from PIL import Image
import numpy as np
import numbers
import random

from data.augmentation import apply_transform_and_clip, image2tensor, numpy2tensor, tensor2image, clip
import torchvision.transforms.functional as F


class RandomAffineBatch:
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, p_hflip=0.5):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                       (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        self.p_hflip = p_hflip  # Horizontal mirror probability

    def get_params(self, h, w):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if self.translate is not None:
            max_dx = self.translate[0] * w
            max_dy = self.translate[1] * h
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if self.scale is not None:
            scale = random.uniform(self.scale[0], self.scale[1])
        else:
            scale = 1.0

        if self.shear is not None:
            if len(self.shear) == 2:
                shear = [random.uniform(self.shear[0], self.shear[1]), 0.]
            elif len(self.shear) == 4:
                shear = [random.uniform(self.shear[0], self.shear[1]), random.uniform(self.shear[2], self.shear[3])]
            else:
                assert NotImplementedError('Incorrect shear: {}'.format(self.shear))
        else:
            shear = [0., 0.]

        return angle, translations, scale, shear

    def __call__(self, sample, angle, translate, scale, shear):
        image, boxes, labels = sample
        _, height, width = image.shape


        if isinstance(image, torch.Tensor):
            # image = tensor2image(image)
            image = F.to_pil_image(image)

        center = (width * 0.5 + 0.5, height * 0.5 + 0.5)
        coeffs = F._get_inverse_affine_matrix(center, angle, translate, scale, shear)
        inverse_affine_matrix = np.eye(3)
        inverse_affine_matrix[:2] = np.array(coeffs).reshape(2, 3)

        if np.random.rand() < self.p_hflip:
            # Post-apply horizontal flip
            # Pre-multiply by [ [-1, 0, width], [0, 1, 0], [0, 0, 1] ] matrix
            flip_matrix = np.eye(3)
            flip_matrix[0, 0] = -1
            flip_matrix[0, 2] = width - 1
            # For inverse affine matrix, pre-multiply by a inverse flip matrix (which is the same as a flip matrix)
            inverse_affine_matrix = flip_matrix @ inverse_affine_matrix

        image = image.transform((width, height), Image.AFFINE, inverse_affine_matrix[:2].reshape(6), Image.BILINEAR)

        if not isinstance(image, torch.Tensor):
            image = image2tensor(image)

        # Compute affine transform matrix and apply it to keypoints
        affine_matrix = np.linalg.pinv(inverse_affine_matrix)
        boxes, labels = apply_transform_and_clip(boxes, labels, affine_matrix, (width, height))

        return image, boxes, labels


class RandomCropBatch:
    """
    Crop the given PIL Image at a random location.

    Args:
        size: Desired output size of the crop (height,width)
    """
    def __init__(self, size):
        self.out_h, self.out_w = size

    def get_params(self, h, w):
        if w == self.out_w and h == self.out_h:
            return 0, 0
        i = random.randint(0, h - self.out_h)
        j = random.randint(0, w - self.out_w)
        return i, j

    def __call__(self, sample, i , j):
        image, boxes, labels = sample

        if isinstance(image, torch.Tensor):
            # image = tensor2image(image)
            image = F.to_pil_image(image)

        image = F.crop(image, i, j, self.out_h, self.out_w)

        if not isinstance(image, torch.Tensor):
            image = image2tensor(image)

        boxes[:, :2] -= (j, i)
        boxes[:, 2:4] -= (j, i)
        boxes, labels = clip(boxes, labels, (self.out_w, self.out_h))
        return image, boxes, labels