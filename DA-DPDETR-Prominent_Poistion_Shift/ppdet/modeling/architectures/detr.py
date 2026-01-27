# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['DETR', 'DETR_Rotate', 'DETR_Rotate_DAOD']
# Deformable DETR, DINO use the same architecture as DETR


@register
class DETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck=None,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck = neck
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # neck
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs) if cfg['neck'] else None

        # transformer
        if neck is not None:
            kwargs = {'input_shape': neck.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck": neck
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # Neck
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        out_transformer = self.transformer(body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, body_feats,
                                         self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

@register
class DETR_Rotate(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck=None,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(DETR_Rotate, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck = neck
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # neck
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs) if cfg['neck'] else None

        # transformer
        if neck is not None:
            kwargs = {'input_shape': neck.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck": neck
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)
        if self.training:
            b,l,_ = self.inputs['gt_bbox'].shape
            new_gt_bbox = []
            new_gt_rbox = []
            new_gt_poly = []
            new_gt_class = []
            new_is_crowd = []
            # record_idx = [l] * b
            # for i in range(b):
            #     for m in range(l):
            #         if sum(self.inputs['gt_bbox'][i][m]) == 0:
            #             record_idx[i] = m
            #             break
            #     if m != l-1:
            #         new_gt_bbox.append(self.inputs['gt_bbox'][i][:m][:])
            #         new_gt_class.append(self.inputs['gt_class'][i][:m][:])
            #         new_gt_rbox.append(self.inputs['gt_rbox'][i][:m][:])
            #         new_gt_poly.append(self.inputs['gt_poly'][i][:m][:])
            #         new_is_crowd.append(self.inputs['is_crowd'][i][:m][:])
            #     else:
            #         new_gt_bbox.append(self.inputs['gt_bbox'][i][:][:])
            #         new_gt_class.append(self.inputs['gt_class'][i][:][:])
            #         new_gt_rbox.append(self.inputs['gt_rbox'][i][:][:])
            #         new_gt_poly.append(self.inputs['gt_poly'][i][:][:])
            #         new_is_crowd.append(self.inputs['is_crowd'][i][:][:])
            gt_bbox_sum = paddle.sum(self.inputs['gt_bbox'], axis=-1)
            for i in range(b):
                nonzero_indices = paddle.where(gt_bbox_sum[i] != 0)[0][-1]
                new_gt_bbox.append(self.inputs['gt_bbox'][i,:nonzero_indices+1,:])
                new_gt_class.append(self.inputs['gt_class'][i,:nonzero_indices+1,:])
                new_gt_rbox.append(self.inputs['gt_rbox'][i,:nonzero_indices+1,:])
                new_gt_poly.append(self.inputs['gt_poly'][i,:nonzero_indices+1,:])
                new_is_crowd.append(self.inputs['is_crowd'][i,:nonzero_indices+1,:])


            self.inputs['gt_bbox'] = new_gt_bbox
            self.inputs['gt_class'] = new_gt_class
            self.inputs['gt_rbox'] = new_gt_rbox
            self.inputs['gt_poly'] = new_gt_poly
            self.inputs['is_crowd'] = new_is_crowd

        # Neck
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        out_transformer = self.transformer(body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, body_feats,
                                         self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()


@register
class DETR_Rotate_DAOD(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck=None,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(DETR_Rotate_DAOD, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck = neck
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # neck
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs) if cfg['neck'] else None

        # transformer
        if neck is not None:
            kwargs = {'input_shape': neck.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone': backbone,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck": neck
        }

    def _forward(self, ):
        # Backbone
        body_feats = self.backbone(self.inputs)
        if self.training and self.inputs['flag'] == 3:
            b, l, _ = self.inputs['gt_bbox_vis'].shape
            # b_c, l_c, _ = self.inputs['gt_bbox_vis_cat'].shape
            new_gt_bbox_vis = []
            new_gt_rbox_vis = []
            new_gt_poly_vis = []

            new_gt_bbox_vis_cat = []
            new_gt_rbox_vis_cat = []
            new_gt_poly_vis_cat = []

            new_gt_bbox_ir = []
            new_gt_rbox_ir = []
            new_gt_poly_ir = []
            new_gt_class = []
            new_is_crowd = []
            gt_bbox_sum = paddle.sum(self.inputs['gt_bbox_vis'], axis=-1)
            for i in range(b):
                nonzero_indices = paddle.where(gt_bbox_sum[i] != 0)[0][-1]
                new_gt_bbox_vis.append(self.inputs['gt_bbox_vis'][i, :nonzero_indices + 1, :])
                new_gt_bbox_ir.append(self.inputs['gt_bbox_ir'][i, :nonzero_indices + 1, :])
                new_gt_class.append(self.inputs['gt_class'][i, :nonzero_indices + 1, :])
                new_gt_rbox_vis.append(self.inputs['gt_rbox_vis'][i, :nonzero_indices + 1, :])
                new_gt_rbox_ir.append(self.inputs['gt_rbox_ir'][i, :nonzero_indices + 1, :])
                new_gt_poly_vis.append(self.inputs['gt_poly_vis'][i, :nonzero_indices + 1, :])
                new_gt_poly_ir.append(self.inputs['gt_poly_ir'][i, :nonzero_indices + 1, :])
                new_is_crowd.append(self.inputs['is_crowd'][i, :nonzero_indices + 1, :])

            self.inputs['gt_bbox_vis'] = new_gt_bbox_vis
            self.inputs['gt_bbox_vis_cat'] = new_gt_bbox_vis + new_gt_bbox_vis
            self.inputs['gt_bbox_ir'] = new_gt_bbox_ir
            self.inputs['gt_class'] = new_gt_class
            self.inputs['gt_class_cat'] = new_gt_class + new_gt_class
            self.inputs['gt_rbox_vis'] = new_gt_rbox_vis
            self.inputs['gt_poly_vis'] = new_gt_poly_vis
            self.inputs['gt_rbox_vis_cat'] = new_gt_rbox_vis + new_gt_rbox_vis
            self.inputs['gt_poly_vis_cat'] = new_gt_poly_vis + new_gt_poly_vis
            self.inputs['gt_rbox_ir'] = new_gt_rbox_ir
            self.inputs['gt_poly_ir'] = new_gt_poly_ir
            self.inputs['is_crowd'] = new_is_crowd

        # Neck
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        out_transformer = self.transformer(body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training and self.inputs['flag']!=4:
            detr_losses = self.detr_head(out_transformer, body_feats,
                                         self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses
        elif self.training and self.inputs['flag']==4:
            preds = self.detr_head(out_transformer, body_feats, self.inputs)
            return preds

        else:
            preds = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                if self.inputs['flag'] == 5 or self.inputs['flag'] == 6:
                    bbox, bbox_num, mask = self.post_process(
                        preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                        paddle.shape(self.inputs['ir_image'])[2:], flag=1)
                else:
                    bbox, bbox_num, mask = self.post_process(
                        preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                        paddle.shape(self.inputs['ir_image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_ssod_loss(self, input):
        detr_losses = self.detr_head(input, None,
                                     None, flag='DAOD')
        detr_losses.update({
            'loss': paddle.add_n(
                [v for k, v in detr_losses.items() if 'log' not in k])
        })
        return detr_losses

    def get_pred(self):
        return self._forward()