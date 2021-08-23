import torch
import math
import numpy as np
import torchvision.transforms as transforms
from pytracking import TensorDict
import ltr.data.processing_utils as prutils


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), train_transform=None, test_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if train_transform or
                                test_transform is None.
            train_transform - The set of transformations to be applied on the train images. If None, the 'transform'
                                argument is used instead.
            test_transform  - The set of transformations to be applied on the test images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the train and test images.  For
                                example, it can be used to convert both test and train images to grayscale.
        """
        self.transform = {'train': transform if train_transform is None else train_transform,
                          'test':  transform if test_transform is None else test_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class SegProcessing(BaseProcessing):
    """ The processing class used for training JOINT. The images are processed in the following way.
    First, the target bounding box (computed using the segmentation mask)is jittered by adding some noise.
    Next, a rectangular region (called search region ) centered at the jittered target center, and of area
    search_area_factor^2 times the area of the jittered box is cropped from the image.
    The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. The argument 'crop_type' determines how out-of-frame regions are handled when cropping the
    search region. For instance, if crop_type == 'replicate', the boundary pixels are replicated in case the search
    region crop goes out of frame. If crop_type == 'inside_major', the search region crop is shifted/shrunk to fit
    completely inside one axis of the image.
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair', new_roll=False, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - The size (width, height) to which the search region is resized. The aspect ratio is always
                        preserved when resizing the search region
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - Determines how out-of-frame regions are handled when cropping the search region.
                        If 'replicate', the boundary pixels are replicated in case the search region crop goes out of
                                        image.
                        If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
                        If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis
                        of the image.
            max_scale_change - Maximum allowed scale change when shrinking the search region to fit the image
                               (only applicable to 'inside' and 'inside_major' cropping modes). In case the desired
                               shrink factor exceeds the max_scale_change, the search region is only shrunk to the
                               factor max_scale_change. Out-of-frame regions are then handled by replicating the
                               boundary pixels. If max_scale_change is set to None, unbounded shrinking is allowed.

            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            new_roll - Whether to use the same random roll values for train and test frames when applying the joint
                       transformation. If True, a new random roll is performed for the test frame transformations. Thus,
                       if performing random flips, the set of train frames and the set of test frames will be flipped
                       independently.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode
        self.max_scale_change = max_scale_change

        self.new_roll = new_roll

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        if self.scale_jitter_factor.get('mode', 'gauss') == 'gauss':
            jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        elif self.scale_jitter_factor.get('mode', 'gauss') == 'uniform':
            jittered_size = box[2:4] * torch.exp(torch.FloatTensor(2).uniform_(-self.scale_jitter_factor[mode],
                                                                               self.scale_jitter_factor[mode]))
        else:
            raise Exception

        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode])).float()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        # Apply joint transformations. i.e. All train/test frames in a sequence are applied the transformation with the
        # same parameters
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'], data['train_masks'] = self.transform['joint'](
                image=data['train_images'], bbox=data['train_anno'], mask=data['train_masks'])
            data['test_images'], data['test_anno'], data['test_masks'] = self.transform['joint'](
                image=data['test_images'], bbox=data['test_anno'], mask=data['test_masks'], new_roll=self.new_roll)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
            orig_anno = data[s + '_anno']

            # Extract a crop containing the target
            crops, boxes, mask_crops = prutils.target_image_crop(data[s + '_images'], jittered_anno,
                                                                 data[s + '_anno'], self.search_area_factor,
                                                                 self.output_sz, mode=self.crop_type,
                                                                 max_scale_change=self.max_scale_change,
                                                                 masks=data[s + '_masks'])

            # Apply independent transformations to each image
            data[s + '_images'], data[s + '_anno'], data[s + '_masks'] = self.transform[s](image=crops, bbox=boxes, mask=mask_crops, joint=False)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
