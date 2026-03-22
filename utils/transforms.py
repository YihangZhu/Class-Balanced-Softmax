# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray, return_array=True) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        output_image = resize(to_pil_image(image), target_size)
        if return_array:
            return np.array(output_image)
        else:
            return output_image

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
            self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
            self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


def transform_coords(coords: torch.Tensor, normalize=False, orig_hw=None, resolution=None) -> torch.Tensor:
    """
    Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
    If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

    Returns
        Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 model.
    """
    if normalize:
        assert orig_hw is not None
        h, w = orig_hw
        coords = coords.clone()
        coords[..., 0] = coords[..., 0] / w
        coords[..., 1] = coords[..., 1] / h

    coords = coords * resolution  # unnormalize coords
    return coords


def transform_boxes(boxes: torch.Tensor, normalize=False, orig_hw=None, resolution=None) -> torch.Tensor:
    """
    Expects a tensor of shape Bx4. The coordinates can be in absolute image or normalized coordinates,
    if the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
    """
    boxes = transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw, resolution)
    return boxes


from PIL.Image import Image
from torchvision.transforms import functional as TF


def get_h_w_img(x):
    if isinstance(x, Image):
        w, h = x.size
    elif isinstance(x, np.ndarray):
        h, w = x.shape[:2]
    elif isinstance(x, torch.Tensor):
        h, w = x.shape[-2:]
    else:
        assert False
    return w, h


def get_padding(w, h, target_size):
    w_diff = target_size - w
    left = w_diff // 2
    right = w_diff - left
    h_diff = target_size - h
    top = h_diff // 2
    bottom = h_diff - top

    return left, top, right, bottom


def pad_image2square(x):
    w, h = get_h_w_img(x)
    if w == h:
        return x
    target_size = max(h, w)
    left, top, right, bottom = get_padding(w, h, target_size)

    if isinstance(x, np.ndarray):
        x = to_pil_image(x)
    x = TF.pad(x, [left, top, right, bottom])
    return x


def unpad_resize(x, orig_img):
    w, h = get_h_w_img(orig_img)
    target_size = max(h, w)
    left, top, right, bottom = get_padding(w, h, target_size)

    if isinstance(x, np.ndarray):
        x = torch.tensor(x)

    # Use torch.nn.functional.interpolate for tensor resizing
    if x.ndim == 2:  # single-channel HxW
        x = x.unsqueeze(0).unsqueeze(0)  # -> (1,1,H,W)
    elif x.ndim == 3:  # (C,H,W)
        x = x.unsqueeze(0)  # -> (1,C,H,W)
    # Now resize with interpolate
    x = F.interpolate(x.float(), size=(target_size, target_size), mode="bilinear", align_corners=False)
    x = x.squeeze(0)  # remove batch dim

    return x[..., top:target_size - bottom, left:target_size - right]


def crop_black(img):
    zero_max = 0
    zero_min = img.shape[0]
    one_max = 0
    one_min = img.shape[1]

    for zero in range(img.shape[0]):
        zero_min = zero
        if np.sum(img[zero, :]) > 0:
            break
    for zero in range(img.shape[0] - 1, 0, -1):
        zero_max = zero
        if np.sum(img[zero, :]) > 0:
            break
    for one in range(img.shape[1]):
        one_min = one
        if np.sum(img[:, one]) > 0:
            break
    for one in range(img.shape[1] - 1, 0, -1):
        one_max = one
        if np.sum(img[:, one]) > 0:
            break
    # max value is exclusive, therefore add one.
    return zero_min, zero_max + 1, one_min, one_max + 1


def crop_img_mask(img, mask):
    zero_min, zero_max, one_min, one_max = crop_black(mask)
    img = img[zero_min: zero_max, one_min: one_max, :]
    mask = mask[zero_min: zero_max, one_min: one_max]
    return img, mask


def xyhw2xyxy(xywh):
    one_min, zero_min, one_length, zero_length = xywh
    return [one_min, zero_min, one_min + one_length, zero_min + zero_length]


def crop_img_bbox_lvis(img, xywh, square):
    one_min, zero_min, one_length, zero_length = xywh
    if square:
        max_length = max(one_length, zero_length)
        one_length = max_length
        zero_length = max_length
    one_max = min(int(np.ceil(one_length + one_min)), img.shape[1])
    zero_max = min(int(np.ceil(zero_length + zero_min)), img.shape[0])

    one_min = int(np.floor(one_min))
    zero_min = int(np.floor(zero_min))
    img = img[zero_min:zero_max, one_min:one_max]
    return img


def crop_img_bbox(img, xyxy=None, xywh=None, square=False, pad=0, return_img=False, xyxy_inclusive=False):
    '''
    bbox is in the format xywh
    img in the shape of (h,w,c)
    '''
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if square:
        one_min, zero_min, one_max, zero_max = square_bbox(xywh=xywh, xyxy=xyxy, pad=pad, img_shape=img.shape)
    else:
        if xywh is not None:
            one_min, zero_min, one_length, zero_length = xywh
            one_max = one_length + one_min
            zero_max = zero_length + zero_min
        else:
            assert xyxy is not None
            one_min, zero_min, one_max, zero_max = xyxy

    if xyxy_inclusive:
        zero_max += 1
        one_max += 1

    if pad > 0:
        one_pad = int((one_max - one_min) * pad)
        one_min -= one_pad
        one_max += one_pad

        zero_pad = int((zero_max - zero_min) * pad)
        zero_min -= zero_pad
        zero_max += zero_pad

        one_min = max(one_min, 0)
        zero_min = max(zero_min, 0)
        zero_max = min(zero_max, img.shape[0])
        one_max = min(one_max, img.shape[1])

    img = img[zero_min:zero_max, one_min:one_max]
    if return_img:
        img = to_pil_image(img)

    return img


def mask_img_func(img, mask):
    # mask = np.expand_dims(mask, axis=2)
    # mask = np.repeat(mask, axis=2, repeats=3)
    # new_img = np.zeros_like(img)
    img[mask <= 0, :] = 0
    # img = img * mask
    return img


def masks_sample_points(masks: torch.Tensor, k=10):
    """Sample points on mask
    """
    if masks.numel() == 0:
        return torch.zeros((0, 2), device=masks.device)

    len_x, len_y = masks.shape[-2:]
    x = torch.arange(0, len_x, dtype=torch.float)
    y = torch.arange(0, len_y, dtype=torch.float)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    bool_mask = masks > 0
    selected_x = torch.masked_select(grid_x, bool_mask)
    selected_y = torch.masked_select(grid_y, bool_mask)
    perm = torch.randperm(selected_x.size(0))
    selected_x = selected_x[perm[:k]]
    selected_y = selected_y[perm[:k]]

    points = torch.cat((selected_y[:, None], selected_x[:, None]), dim=1)
    # for point in points:  # check the generated points
    #     point = point.numpy().astype(int)
    #     check = bool_mask[point[1], point[0]]
    #     assert check
    return points


def mask_to_bbox_torch(mask: torch.Tensor, square: bool = False):
    """
    Convert a binary mask tensor to bounding box coordinates.

    Args:
        mask (torch.Tensor): 2D binary mask (H, W) with 1 for object, 0 for background.
        square (bool): If True, return a square bounding box enclosing the mask.

    Returns:
        bbox (tuple): (x_min, y_min, x_max, y_max) of bounding box enclosing mask.
                      Returns None if mask is empty.
    """

    ys, xs = torch.nonzero(mask, as_tuple=True)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min().item(), xs.max().item()
    y_min, y_max = ys.min().item(), ys.max().item()

    if square:
        x_min, x_max, y_min, y_max = square_bbox(x_min, x_max, y_min, y_max, img=mask)

    return [x_min, y_min, x_max, y_max]


def square_bbox(xyxy=None, xywh=None, img_shape=None, pad=0):
    """
    the h, w is the highth and wideth for the bounding box instead of the image.
    """
    if xyxy is not None:
        if isinstance(xyxy, list):
            xyxy = torch.tensor(xyxy)
        x_min, y_min, x_max, y_max = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
        w = x_max - x_min
        h = y_max - y_min
    else:
        assert xywh is not None
        if isinstance(xywh, list):
            xywh = torch.tensor(xywh)
        x_min, y_min, w, h = xywh[..., 0], xywh[..., 1], xywh[..., 2], xywh[..., 3]
        x_max = x_min + w
        y_max = y_min + h

    side = torch.stack((w, h), dim=-1)
    side = torch.max(side, dim=-1).values
    if pad > 0:
        side += pad

    return update_xyxy_with_wh(x_min, y_min, x_max, y_max, side, side, img_shape=img_shape)


def update_xyxy_with_wh(x_min, y_min, x_max, y_max, w, h, img_shape):
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    x_min = x_center - w // 2
    y_min = y_center - h // 2
    x_max = x_min + w
    y_max = y_min + h

    x_min = max(torch.floor(x_min).item(), 0)
    y_min = max(torch.floor(y_min).item(), 0)
    x_max = min(torch.ceil(x_max).item(), img_shape[1] - 1)
    y_max = min(torch.ceil(y_max).item(), img_shape[0] - 1)

    return x_min, y_min, x_max, y_max
