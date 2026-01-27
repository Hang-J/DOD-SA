# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import math
import paddle
import weakref
from copy import deepcopy

from .utils import get_bn_running_state_names

__all__ = ['ModelEMA', 'SimpleModelEMA']


class ModelEMA(object):
    """
    Exponential Weighted Average for Deep Neutal Networks
    Args:
        model (nn.Layer): Detector of model.
        decay (int):  The decay used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `ema_param = decay * ema_param + (1 - decay) * cur_param`.
            Defaults is 0.9998.
        ema_decay_type (str): type in ['threshold', 'normal', 'exponential'],
            'threshold' as default.
        cycle_epoch (int): The epoch of interval to reset ema_param and
            step. Defaults is -1, which means not reset. Its function is to
            add a regular effect to ema, which is set according to experience
            and is effective when the total training epoch is large.
        ema_black_list (set|list|tuple, optional): The custom EMA black_list.
            Blacklist of weight names that will not participate in EMA
            calculation. Default: None.
    """

    def __init__(self,
                 model,
                 decay=0.9998,
                 ema_decay_type='threshold',
                 cycle_epoch=-1,
                 ema_black_list=None,
                 ema_filter_no_grad=False):
        self.step = 0
        self.epoch = 0
        self.decay = decay
        self._decay = decay
        self.ema_decay_type = ema_decay_type
        self.cycle_epoch = cycle_epoch
        self.ema_black_list = self._match_ema_black_list(
            model.state_dict().keys(), ema_black_list)
        bn_states_names = get_bn_running_state_names(model)
        if ema_filter_no_grad:
            for n, p in model.named_parameters():
                if p.stop_gradient and n not in bn_states_names:
                    self.ema_black_list.add(n)

        self.state_dict = dict()
        for k, v in model.state_dict().items():
            if k in self.ema_black_list:
                self.state_dict[k] = v
            else:
                self.state_dict[k] = paddle.zeros_like(v)

        self._model_state = {
            k: weakref.ref(p)
            for k, p in model.state_dict().items()
        }
        # Maintain a detached teacher copy so callers can use self.ema.model.
        self.model = deepcopy(model)
        self.model.eval()
        for param in self.model.parameters():
            param.stop_gradient = True
        self._ema_model_state = {
            k: weakref.ref(p)
            for k, p in self.model.state_dict().items()
        }
        self._load_state_into_ema_model(model.state_dict())

    def reset(self):
        self.step = 0
        self.epoch = 0
        for k, v in self.state_dict.items():
            if k in self.ema_black_list:
                self.state_dict[k] = v
            else:
                self.state_dict[k] = paddle.zeros_like(v)
        self._sync_model_from_student()

    def resume(self, state_dict, step=0):
        for k, v in state_dict.items():
            if k in self.state_dict:
                if self.state_dict[k].dtype == v.dtype:
                    self.state_dict[k] = v
                else:
                    self.state_dict[k] = v.astype(self.state_dict[k].dtype)
        self.step = step
        self._load_state_into_ema_model(state_dict)

    def update(self, model=None, decay=None):
        if decay is not None:
            decay_value = decay
        elif self.ema_decay_type == 'threshold':
            decay_value = min(self.decay, (1 + self.step) / (10 + self.step))
        elif self.ema_decay_type == 'exponential':
            decay_value = self.decay * (1 - math.exp(-(self.step + 1) / 2000))
        else:
            decay_value = self.decay
        self._decay = decay_value

        if model is not None:
            model_dict = model.state_dict()
        else:
            model_dict = {k: p() for k, p in self._model_state.items()}
            assert all(
                [v is not None for _, v in model_dict.items()]), 'python gc.'

        for k, v in self.state_dict.items():
            if k not in self.ema_black_list:
                v = decay_value * v + (1 - decay_value) * model_dict[k]
                v.stop_gradient = True
                self.state_dict[k] = v
        self.step += 1
        self._update_ema_model(model_dict, decay_value)
        self._sync_ema_model()

    def apply(self):
        if self.step == 0:
            return self.state_dict
        correct_bias = self.ema_decay_type != 'exponential'
        state_dict = self._build_ema_state(correct_bias=correct_bias)
        self.epoch += 1
        if self.cycle_epoch > 0 and self.epoch == self.cycle_epoch:
            self.reset()

        return state_dict

    def _match_ema_black_list(self, weight_name, ema_black_list=None):
        out_list = set()
        if ema_black_list:
            for name in weight_name:
                for key in ema_black_list:
                    if key in name:
                        out_list.add(name)
        return out_list

    def _build_ema_state(self, correct_bias=True):
        state_dict = dict()
        for k, v in self.state_dict.items():
            tensor = v
            if k not in self.ema_black_list and paddle.is_floating_point(v):
                if correct_bias and self.step > 0 and self.ema_decay_type != 'exponential':
                    denom = 1 - self._decay**self.step
                    if denom != 0:
                        tensor = tensor / denom
            if isinstance(tensor, paddle.Tensor):
                tensor = tensor.detach()
                tensor.stop_gradient = True
            state_dict[k] = tensor
        return state_dict

    def _sync_ema_model(self):
        if not hasattr(self, '_ema_model_state'):
            return
        if self.step == 0:
            self._sync_model_from_student()
            return
        correct_bias = self.ema_decay_type != 'exponential'
        state_dict = self._build_ema_state(correct_bias=correct_bias)
        self._load_state_into_ema_model(state_dict)

    def _update_ema_model(self, model_dict, decay):
        if not hasattr(self, '_ema_model_state'):
            return
        ema_state = {
            k: ref()
            for k, ref in self._ema_model_state.items()
        }
        assert all(
            [v is not None for _, v in ema_state.items()]), 'python gc.'
        for k, ema_tensor in ema_state.items():
            if k not in model_dict:
                continue
            model_tensor = model_dict[k]
            if isinstance(model_tensor, paddle.Tensor):
                model_tensor = model_tensor.detach()
            if not paddle.is_floating_point(model_tensor):
                new_tensor = model_tensor
            elif k in self.ema_black_list:
                new_tensor = model_tensor
            else:
                new_tensor = decay * ema_tensor + (1 - decay) * model_tensor
            if isinstance(new_tensor, paddle.Tensor):
                new_tensor = new_tensor.detach()
            new_tensor.stop_gradient = True
            if ema_tensor.dtype != new_tensor.dtype:
                new_tensor = new_tensor.astype(ema_tensor.dtype)
            ema_tensor.set_value(new_tensor)

    def _sync_model_from_student(self):
        student_state = {
            k: p()
            for k, p in self._model_state.items()
        }
        assert all(
            [v is not None for _, v in student_state.items()]), 'python gc.'
        self._load_state_into_ema_model(student_state)

    def _load_state_into_ema_model(self, state_dict):
        if not hasattr(self, '_ema_model_state'):
            return
        for k, ref in self._ema_model_state.items():
            ema_tensor = ref()
            if ema_tensor is None or k not in state_dict:
                continue
            tensor = state_dict[k]
            if not isinstance(tensor, paddle.Tensor):
                tensor = paddle.to_tensor(tensor)
            tensor = tensor.detach()
            tensor.stop_gradient = True
            if ema_tensor.dtype != tensor.dtype:
                tensor = tensor.astype(ema_tensor.dtype)
            ema_tensor.set_value(tensor)


class SimpleModelEMA(object):
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model=None, decay=0.9996):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
        """
        self.model = deepcopy(model)
        self.decay = decay

    def update(self, model, decay=None):
        if decay is None:
            decay = self.decay

        with paddle.no_grad():
            state = {}
            msd = model.state_dict()
            for k, v in self.model.state_dict().items():
                if paddle.is_floating_point(v):
                    v *= decay
                    v += (1.0 - decay) * msd[k].detach()
                state[k] = v
            self.model.set_state_dict(state)

    def resume(self, state_dict, step=0):
        state = {}
        msd = state_dict
        for k, v in self.model.state_dict().items():
            if paddle.is_floating_point(v):
                v = msd[k].detach()
            state[k] = v
        self.model.set_state_dict(state)
        self.step = step
