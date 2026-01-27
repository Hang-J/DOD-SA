# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class PairedTransformHelper(object):
    """
    Utility that records the affine mapping between the original image
    coordinate space and the current augmented space for paired data.
    The internal state is kept as a numpy array for framework compliance.
    """

    _STATE_KEY = 'paired_state'  # stored as [sx, sy, tx, ty, curr_w, curr_h, orig_w, orig_h]

    @classmethod
    def reset(cls, sample, shape=None):
        """
        Initialize the transform state with identity mapping.
        """
        if shape is None:
            vis_im = sample.get('vis_image', None)
            if vis_im is None:
                return
            shape = vis_im.shape[:2]
        h, w = float(shape[0]), float(shape[1])
        state = np.array(
            [1.0, 1.0, 0.0, 0.0, w, h, w, h], dtype=np.float32)
        sample[cls._STATE_KEY] = state
        cls._export(sample)

    @classmethod
    def _get_state(cls, sample):
        """
        Retrieve the state array, creating one if necessary.
        """
        if cls._STATE_KEY not in sample:
            cls.reset(sample)
        return sample.get(cls._STATE_KEY, None)

    @classmethod
    def translate(cls, sample, offset_x, offset_y, new_size=None):
        """
        Apply translation to the transform state. `new_size` can be a tuple
        of (w, h) representing the new canvas size after translation.
        """
        state = cls._get_state(sample)
        if state is None:
            return
        state = state.copy()
        state[2] += float(offset_x)
        state[3] += float(offset_y)
        if new_size is not None:
            new_w, new_h = new_size
            state[4] = float(new_w)
            state[5] = float(new_h)
        sample[cls._STATE_KEY] = state
        cls._export(sample)

    @classmethod
    def scale(cls, sample, scale_x, scale_y, new_size=None):
        """
        Apply scaling to the transform state. When `new_size` is provided,
        it is treated as the authoritative canvas size after scaling.
        """
        state = cls._get_state(sample)
        if state is None:
            return
        sx = float(scale_x)
        sy = float(scale_y)
        state = state.copy()
        state[0] *= sx
        state[1] *= sy
        state[2] *= sx
        state[3] *= sy
        if new_size is not None:
            new_w, new_h = new_size
            state[4] = float(new_w)
            state[5] = float(new_h)
        else:
            state[4] *= abs(sx)
            state[5] *= abs(sy)
        sample[cls._STATE_KEY] = state
        cls._export(sample)

    @classmethod
    def flip_horizontal(cls, sample):
        """
        Update transform state for a left-right flip.
        """
        state = cls._get_state(sample)
        if state is None:
            return
        state = state.copy()
        width = state[4]
        state[0] = -state[0]
        state[2] = width - state[2]
        sample[cls._STATE_KEY] = state
        cls._export(sample)

    @classmethod
    def set_current_size(cls, sample, width, height):
        """
        Record the latest canvas size without changing the affine mapping.
        """
        state = cls._get_state(sample)
        if state is None:
            return
        state = state.copy()
        state[4] = float(width)
        state[5] = float(height)
        sample[cls._STATE_KEY] = state
        cls._export(sample)

    @classmethod
    def _export(cls, sample):
        """
        Write the current affine parameters and original shape to the sample
        so that downstream consumers can recover the mapping.
        """
        state = sample.get(cls._STATE_KEY, None)
        if state is None:
            return
        sample['paired_affine'] = state[:4].astype(np.float32)
        sample['paired_ori_shape'] = np.array(
            [state[7], state[6]], dtype=np.float32)
        sample['paired_curr_shape'] = np.array(
            [state[5], state[4]], dtype=np.float32)
