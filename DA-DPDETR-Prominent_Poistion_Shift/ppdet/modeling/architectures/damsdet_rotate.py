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

__all__ = ['DAMSDET_Rotate','DAMSDET_Rotate_Paired','DAMSDET_Rotate_baseline', 'DAMSDET_Rotate_Paired_DAOD']
# My multispectral DETR


@register
class DAMSDET_Rotate(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone_vis,
                 backbone_ir,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck_vis=None,
                 neck_ir=None,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(DAMSDET_Rotate, self).__init__()
        self.backbone_vis = backbone_vis
        self.backbone_ir = backbone_ir
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck_vis = neck_vis
        self.neck_ir = neck_ir
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone_vis
        backbone_vis = create(cfg['backbone_vis'])
        # backbone_ir
        backbone_ir = create(cfg['backbone_ir'])
        # neck
        kwargs = {'input_shape': backbone_vis.out_shape}
        neck_vis = create(cfg['neck_vis'], **kwargs) if cfg['neck_vis'] else None
        neck_ir = create(cfg['neck_ir'], **kwargs) if cfg['neck_ir'] else None

        # transformer
        if neck_vis is not None:
            kwargs = {'input_shape': neck_vis.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone_vis.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone_vis': backbone_vis,
            'backbone_ir': backbone_ir,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck_vis": neck_vis,
            "neck_ir": neck_ir
        }

    def _forward(self):
        # Backbone
        vis_body_feats = self.backbone_vis(self.inputs,1)
        ir_body_feats = self.backbone_ir(self.inputs,2)
        # body_feats = []
        # for ii in range(len(vis_body_feats)):
        #     body_feats.append(vis_body_feats[ii]+ir_body_feats[ii])
        if self.training:
            b, l, _ = self.inputs['gt_bbox'].shape
            new_gt_bbox = []
            new_gt_rbox = []
            new_gt_poly = []
            new_gt_class = []
            new_is_crowd = []
            gt_bbox_sum = paddle.sum(self.inputs['gt_bbox'], axis=-1)
            for i in range(b):
                nonzero_indices = paddle.where(gt_bbox_sum[i] != 0)[0][-1]
                new_gt_bbox.append(self.inputs['gt_bbox'][i, :nonzero_indices + 1, :])
                new_gt_class.append(self.inputs['gt_class'][i, :nonzero_indices + 1, :])
                new_gt_rbox.append(self.inputs['gt_rbox'][i, :nonzero_indices + 1, :])
                new_gt_poly.append(self.inputs['gt_poly'][i, :nonzero_indices + 1, :])
                new_is_crowd.append(self.inputs['is_crowd'][i, :nonzero_indices + 1, :])

            self.inputs['gt_bbox'] = new_gt_bbox
            self.inputs['gt_class'] = new_gt_class
            self.inputs['gt_rbox'] = new_gt_rbox
            self.inputs['gt_poly'] = new_gt_poly
            self.inputs['is_crowd'] = new_is_crowd

        # Neck
        if self.neck_vis is not None:
            #body_feats = self.neck(body_feats)
            vis_body_feats = self.neck_vis(vis_body_feats)
            ir_body_feats = self.neck_ir(ir_body_feats)

        # body_feats = []
        # for ii in range(len(vis_body_feats)):
        #     body_feats.append(vis_body_feats[ii]+ir_body_feats[ii])
        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        #(out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,dn_meta)
        out_transformer = self.transformer(None,vis_body_feats, ir_body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, None,
                                         self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses
        else:
            #normal
            preds = self.detr_head(out_transformer, None)
            preds = (preds[0][:,:,:],preds[1][:,:,:],preds[2])
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['vis_image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

            # #paired infer
            # preds = self.detr_head(out_transformer, None)
            # preds = (preds[0][:, :, :], preds[1][:, :, :],preds[2][:, :, :], preds[3])
            # if self.exclude_post_process:
            #     bbox, bbox_num, mask = preds
            # else:
            #     bbox_vis,bbox_ir, bbox_num, mask = self.post_process(
            #         preds, self.inputs['im_shape'], self.inputs['scale_factor'],
            #         paddle.shape(self.inputs['vis_image'])[2:])
            #
            # output = {'bbox_vis': bbox_vis, 'bbox_ir':bbox_ir, 'bbox_num': bbox_num}
            # if self.with_mask:
            #     output['mask'] = mask
            # return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

@register
class DAMSDET_Rotate_baseline(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone_vis,
                 backbone_ir,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck=None,
                 #neck_ir=None,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(DAMSDET_Rotate_baseline, self).__init__()
        self.backbone_vis = backbone_vis
        self.backbone_ir = backbone_ir
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck = neck
        #self.neck_ir = neck_ir
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone_vis
        backbone_vis = create(cfg['backbone_vis'])
        # backbone_ir
        backbone_ir = create(cfg['backbone_ir'])
        # neck
        kwargs = {'input_shape': backbone_vis.out_shape}
        neck = create(cfg['neck'], **kwargs) if cfg['neck'] else None
        # neck_ir = create(cfg['neck_ir'], **kwargs) if cfg['neck_ir'] else None

        # transformer
        if neck is not None:
            kwargs = {'input_shape': neck.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone_vis.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone_vis': backbone_vis,
            'backbone_ir': backbone_ir,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck": neck,
            # "neck_ir": neck_ir
        }

    def _forward(self):
        # Backbone
        vis_body_feats = self.backbone_vis(self.inputs,1)
        ir_body_feats = self.backbone_ir(self.inputs,2)
        body_feats = []
        for ii in range(len(vis_body_feats)):
            body_feats.append(vis_body_feats[ii]+ir_body_feats[ii])
        if self.training:
            b, l, _ = self.inputs['gt_bbox'].shape
            new_gt_bbox = []
            new_gt_rbox = []
            new_gt_poly = []
            new_gt_class = []
            new_is_crowd = []
            gt_bbox_sum = paddle.sum(self.inputs['gt_bbox'], axis=-1)
            for i in range(b):
                nonzero_indices = paddle.where(gt_bbox_sum[i] != 0)[0][-1]
                new_gt_bbox.append(self.inputs['gt_bbox'][i, :nonzero_indices + 1, :])
                new_gt_class.append(self.inputs['gt_class'][i, :nonzero_indices + 1, :])
                new_gt_rbox.append(self.inputs['gt_rbox'][i, :nonzero_indices + 1, :])
                new_gt_poly.append(self.inputs['gt_poly'][i, :nonzero_indices + 1, :])
                new_is_crowd.append(self.inputs['is_crowd'][i, :nonzero_indices + 1, :])

            self.inputs['gt_bbox'] = new_gt_bbox
            self.inputs['gt_class'] = new_gt_class
            self.inputs['gt_rbox'] = new_gt_rbox
            self.inputs['gt_poly'] = new_gt_poly
            self.inputs['is_crowd'] = new_is_crowd

        # Neck
        if self.neck is not None:
            body_feats = self.neck(body_feats)
            # vis_body_feats = self.neck_vis(vis_body_feats)
            # ir_body_feats = self.neck_ir(ir_body_feats)

        # body_feats = []
        # for ii in range(len(vis_body_feats)):
        #     body_feats.append(vis_body_feats[ii]+ir_body_feats[ii])
        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        #(out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,dn_meta)
        out_transformer = self.transformer(body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, None,
                                         self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses
        else:
            #normal
            preds = self.detr_head(out_transformer, None)
            #preds = (preds[0][:,:,:],preds[1][:,:,:],preds[2])
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['vis_image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

            # #paired infer
            # preds = self.detr_head(out_transformer, None)
            # preds = (preds[0][:, :, :], preds[1][:, :, :],preds[2][:, :, :], preds[3])
            # if self.exclude_post_process:
            #     bbox, bbox_num, mask = preds
            # else:
            #     bbox_vis,bbox_ir, bbox_num, mask = self.post_process(
            #         preds, self.inputs['im_shape'], self.inputs['scale_factor'],
            #         paddle.shape(self.inputs['vis_image'])[2:])
            #
            # output = {'bbox_vis': bbox_vis, 'bbox_ir':bbox_ir, 'bbox_num': bbox_num}
            # if self.with_mask:
            #     output['mask'] = mask
            # return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

@register
class DAMSDET_Rotate_Paired(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone_vis,
                 backbone_ir,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck_vis=None,
                 neck_ir=None,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(DAMSDET_Rotate_Paired, self).__init__()
        self.backbone_vis = backbone_vis
        self.backbone_ir = backbone_ir
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck_vis = neck_vis
        self.neck_ir = neck_ir
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone_vis
        backbone_vis = create(cfg['backbone_vis'])
        # backbone_ir
        backbone_ir = create(cfg['backbone_ir'])
        # neck
        kwargs = {'input_shape': backbone_vis.out_shape}
        neck_vis = create(cfg['neck_vis'], **kwargs) if cfg['neck_vis'] else None
        neck_ir = create(cfg['neck_ir'], **kwargs) if cfg['neck_ir'] else None

        # transformer
        if neck_vis is not None:
            kwargs = {'input_shape': neck_vis.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone_vis.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone_vis': backbone_vis,
            'backbone_ir': backbone_ir,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck_vis": neck_vis,
            "neck_ir": neck_ir
        }

    def _forward(self):
        # Backbone
        vis_body_feats = self.backbone_vis(self.inputs,1)
        ir_body_feats = self.backbone_ir(self.inputs,2)
        # body_feats = []
        # for ii in range(len(vis_body_feats)):
        #     body_feats.append(vis_body_feats[ii]+ir_body_feats[ii])
        if self.training:
            b, l, _ = self.inputs['gt_bbox_ir'].shape
            new_gt_bbox_vis = []
            new_gt_rbox_vis = []
            new_gt_poly_vis = []
            new_gt_bbox_ir = []
            new_gt_rbox_ir = []
            new_gt_poly_ir = []
            new_gt_class = []
            new_is_crowd = []
            gt_bbox_sum = paddle.sum(self.inputs['gt_bbox_ir'], axis=-1)
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
            self.inputs['gt_bbox_ir'] = new_gt_bbox_ir
            self.inputs['gt_class'] = new_gt_class
            self.inputs['gt_rbox_vis'] = new_gt_rbox_vis
            self.inputs['gt_poly_vis'] = new_gt_poly_vis
            self.inputs['gt_rbox_ir'] = new_gt_rbox_ir
            self.inputs['gt_poly_ir'] = new_gt_poly_ir
            self.inputs['is_crowd'] = new_is_crowd

        # Neck
        if self.neck_vis is not None:
            #body_feats = self.neck(body_feats)
            vis_body_feats = self.neck_vis(vis_body_feats)
            ir_body_feats = self.neck_ir(ir_body_feats)

        # body_feats = []
        # for ii in range(len(vis_body_feats)):
        #     body_feats.append(vis_body_feats[ii]+ir_body_feats[ii])
        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        #(out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,dn_meta)
        out_transformer = self.transformer(None,vis_body_feats, ir_body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, None,
                                         self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses
        else:
            #normal
            preds = self.detr_head(out_transformer, None)
            preds = (preds[0][:,:,:],preds[1][:,:,:],preds[2])
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['vis_image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

            # #paired infer
            # preds = self.detr_head(out_transformer, None)
            # preds = (preds[0][:, :, :], preds[1][:, :, :],preds[2][:, :, :], preds[3])
            # if self.exclude_post_process:
            #     bbox, bbox_num, mask = preds
            # else:
            #     bbox_vis,bbox_ir, bbox_num, mask = self.post_process(
            #         preds, self.inputs['im_shape'], self.inputs['scale_factor'],
            #         paddle.shape(self.inputs['vis_image'])[2:])
            #
            # output = {'bbox_vis': bbox_vis, 'bbox_ir':bbox_ir, 'bbox_num': bbox_num}
            # if self.with_mask:
            #     output['mask'] = mask
            # return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()


@register
class DAMSDET_Rotate_Paired_DAOD(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process','post_process_paired']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone_stream1,
                 backbone_stream2,
                 transformer_single='DETRTransformer',
                 transformer_DPDETR='DETRTransformer',
                 detr_head_single='DETRHead',
                 detr_head_DPDETR='DETRHead',
                 neck_stream1=None,
                 neck_stream2=None,
                 post_process='DETRPostProcess',
                 post_process_paired = 'DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(DAMSDET_Rotate_Paired_DAOD, self).__init__()
        self.backbone_stream1 = backbone_stream1
        self.backbone_stream2 = backbone_stream2
        self.transformer_single = transformer_single
        self.transformer_DPDETR = transformer_DPDETR
        self.detr_head_single = detr_head_single
        self.detr_head_DPDETR = detr_head_DPDETR
        self.neck_stream1 = neck_stream1
        self.neck_stream2 = neck_stream2
        self.post_process = post_process
        self.post_process_paired = post_process_paired
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone_stream1
        backbone_stream1 = create(cfg['backbone_stream1'])
        # backbone_stream2
        backbone_stream2 = create(cfg['backbone_stream2'])
        # neck
        kwargs = {'input_shape': backbone_stream1.out_shape}
        neck_stream1 = create(cfg['neck_stream1'], **kwargs) if cfg['neck_stream1'] else None
        neck_stream2 = create(cfg['neck_stream2'], **kwargs) if cfg['neck_stream2'] else None

        # transformer
        if neck_stream1 is not None:
            kwargs = {'input_shape': neck_stream1.out_shape}
        transformer_single = create(cfg['transformer_single'], **kwargs)
        transformer_DPDETR = create(cfg['transformer_DPDETR'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer_single.hidden_dim,
            'nhead': transformer_single.nhead,
            'input_shape': backbone_stream1.out_shape
        }
        detr_head_single = create(cfg['detr_head_single'], **kwargs)
        detr_head_DPDETR = create(cfg['detr_head_DPDETR'], **kwargs)

        return {
            'backbone_stream1': backbone_stream1,
            'backbone_stream2': backbone_stream2,
            'transformer_single': transformer_single,
            'transformer_DPDETR': transformer_DPDETR,
            "detr_head_single": detr_head_single,
            "detr_head_DPDETR": detr_head_DPDETR,
            "neck_stream1": neck_stream1,
            "neck_stream2": neck_stream2
        }

    def _forward(self,):
        # Backbone
        # vis_body_feats = self.backbone_vis(self.inputs,1)
        # ir_body_feats = self.backbone_ir(self.inputs,2)

        b, l,_,_ = self.inputs['ir_image'].shape
        # body_feats = []
        # for ii in range(len(vis_body_feats)):
        #     body_feats.append(vis_body_feats[ii]+ir_body_feats[ii])
        if (self.training and (self.inputs['flag'] == 3 or self.inputs['flag'] == 7)) or self.inputs['flag'] == 9:
            
            # b_c, l_c, _ = self.inputs['gt_bbox_vis_cat'].shape
            self.new_gt_bbox_vis = []
            self.new_gt_rbox_vis = []
            self.new_gt_poly_vis = []

            self.new_gt_bbox_ir_cat = []
            self.new_gt_rbox_ir_cat = []
            self.new_gt_poly_ir_cat = []

            self.new_gt_bbox_ir = []
            self.new_gt_rbox_ir = []
            self.new_gt_poly_ir = []
            self.new_gt_class = []
            self.new_is_crowd = []
            gt_bbox_sum = paddle.sum(self.inputs['gt_bbox_ir'], axis=-1)
            for i in range(b):
                nonzero_indices = paddle.where(gt_bbox_sum[i] != 0)[0][-1]
                self.new_gt_bbox_vis.append(self.inputs['gt_bbox_vis'][i, :nonzero_indices + 1, :])
                self.new_gt_bbox_ir.append(self.inputs['gt_bbox_ir'][i, :nonzero_indices + 1, :])
                self.new_gt_class.append(self.inputs['gt_class'][i, :nonzero_indices + 1, :])
                self.new_gt_rbox_vis.append(self.inputs['gt_rbox_vis'][i, :nonzero_indices + 1, :])
                self.new_gt_rbox_ir.append(self.inputs['gt_rbox_ir'][i, :nonzero_indices + 1, :])
                self.new_gt_poly_vis.append(self.inputs['gt_poly_vis'][i, :nonzero_indices + 1, :])
                self.new_gt_poly_ir.append(self.inputs['gt_poly_ir'][i, :nonzero_indices + 1, :])
                self.new_is_crowd.append(self.inputs['is_crowd'][i, :nonzero_indices + 1, :])

            self.inputs['gt_bbox_vis'] = self.new_gt_bbox_vis
            self.inputs['gt_bbox_ir_cat'] = self.new_gt_bbox_ir + self.new_gt_bbox_ir
            self.inputs['gt_bbox_ir'] = self.new_gt_bbox_ir
            self.inputs['gt_class'] = self.new_gt_class
            self.inputs['gt_class_cat'] = self.new_gt_class + self.new_gt_class
            self.inputs['gt_rbox_vis'] = self.new_gt_rbox_vis
            self.inputs['gt_poly_vis'] = self.new_gt_poly_vis
            self.inputs['gt_rbox_ir_cat'] = self.new_gt_rbox_ir + self.new_gt_rbox_ir
            self.inputs['gt_poly_ir_cat'] = self.new_gt_poly_ir + self.new_gt_poly_ir
            self.inputs['gt_rbox_ir'] = self.new_gt_rbox_ir
            self.inputs['gt_poly_ir'] = self.new_gt_poly_ir
            self.inputs['is_crowd'] = self.new_is_crowd



        # stage 1
        if self.inputs['flag'] == 3 or self.inputs['flag'] == 4 or self.inputs['flag'] == 5 or self.inputs['flag'] == 'eval_stage1' or self.inputs['flag'] == 'eval_stage1_2':  # do sup_cat
            # backbone

            body_feats = self.backbone_stream1(self.inputs)

            # Neck
            if self.neck_stream1 is not None:
                body_feats = self.neck_stream1(body_feats)

            if self.inputs['flag'] == 3:
                self.inputs['gt_rbox_dn'] = self.new_gt_rbox_ir + self.new_gt_rbox_ir
                self.inputs['gt_class_dn'] = self.new_gt_class + self.new_gt_class
                self.inputs['im_shape']= paddle.tile(self.inputs['im_shape'][0], repeat_times=[2*b, 1])

            if self.inputs['flag'] == 4:
                self.inputs['gt_rbox_dn'] = self.inputs['teacher_gt_rbox_vis']
                self.inputs['gt_class_dn'] = self.inputs['teacher_gt_class']
                self.inputs['im_shape']= paddle.tile(self.inputs['im_shape'][0], repeat_times=[b, 1])

            # Transformer
            pad_mask = self.inputs.get('pad_mask', None)
            out_transformer = self.transformer_single(body_feats, pad_mask, self.inputs)

            # DETR Head
            if self.training and self.inputs['flag'] != 4:
                detr_losses = self.detr_head_single(out_transformer, body_feats,
                                                    self.inputs)
                detr_losses.update({
                    'loss': paddle.add_n(
                        [v for k, v in detr_losses.items() if 'log' not in k])
                })
                return detr_losses
            elif self.training and self.inputs['flag'] == 4:
                preds = self.detr_head_single(out_transformer, body_feats, self.inputs)
                return preds

            else:
                preds = self.detr_head_single(out_transformer, body_feats)
                if self.exclude_post_process:
                    bbox, bbox_num, mask = preds
                else:
                    if self.inputs['flag'] == 5 or self.inputs['flag'] == 6:
                        if self.inputs['flag'] == 5:
                            self.inputs['im_shape']= paddle.tile(self.inputs['im_shape'][0], repeat_times=[b, 1])
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


        # stage 2 ----single branch
        if self.inputs['flag'] == 7:  # do sup
            # backbone
            body_feats = self.backbone_stream1(self.inputs)

            # Neck
            if self.neck_stream1 is not None:
                body_feats = self.neck_stream1(body_feats)

            if self.inputs['flag'] == 7:
                self.inputs['gt_rbox_dn'] = self.new_gt_rbox_ir
                self.inputs['gt_class_dn'] = self.new_gt_class
                self.inputs['im_shape']= paddle.tile(self.inputs['im_shape'][0], repeat_times=[b, 1])

            # Transformer
            pad_mask = self.inputs.get('pad_mask', None)
            out_transformer = self.transformer_single(body_feats, pad_mask, self.inputs)

            # DETR Head
            if self.training and self.inputs['flag'] != 4:
                detr_losses = self.detr_head_single(out_transformer, body_feats,
                                                    self.inputs)
                detr_losses.update({
                    'loss': paddle.add_n(
                        [v for k, v in detr_losses.items() if 'log' not in k])
                })
                return detr_losses
            elif self.training and self.inputs['flag'] == 4:
                preds = self.detr_head_single(out_transformer, body_feats, self.inputs)
                return preds

            else:
                preds = self.detr_head_single(out_transformer, body_feats)
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

        # stage 2 ----multi branch
        if self.inputs['flag'] == 8 or self.inputs['flag'] == 10 or self.inputs['flag'] == 'eval_stage2':  # do multi branch
            # backbone
            ir_body_feats = self.backbone_stream1(self.inputs,1)
            vis_body_feats = self.backbone_stream2(self.inputs, 2)

            # Neck
            if self.neck_stream1 is not None:
                ir_body_feats = self.neck_stream1(ir_body_feats)
                vis_body_feats = self.neck_stream2(vis_body_feats)

            # if self.inputs['flag'] == 8:
            #     self.inputs['gt_rbox_ir_dn'] = self.inputs['paired_gt_rbox_ir']
            #     self.inputs['gt_rbox_vis_dn'] = self.inputs['paired_teacher_rbox_vis']
            #     self.inputs['gt_class_dn'] = self.inputs['paired_gt_class']
            #     self.inputs['im_shape']= paddle.tile(self.inputs['im_shape'][0], repeat_times=[b, 1])

            if self.inputs['flag'] == 8 or self.inputs['flag'] == 10:    
                paired_gt_rbox_ir=[]
                paired_teacher_rbox_vis=[]
                for t1, t2 in zip(self.inputs['paired_gt_rbox_ir_match'], self.inputs['paired_gt_rbox_ir_unmatch']):
                    combined_tensor = paddle.concat((t1, t2), axis=0)
                    paired_gt_rbox_ir.append(combined_tensor)
                for t1, t2 in zip(self.inputs['paired_teacher_bbox_match'], self.inputs['paired_teacher_bbox_unmatch']):
                    combined_tensor = paddle.concat((t1, t2), axis=0)
                    paired_teacher_rbox_vis.append(combined_tensor)
                self.inputs['gt_rbox_ir_dn'] = paired_gt_rbox_ir
                self.inputs['gt_rbox_vis_dn'] = paired_teacher_rbox_vis
                self.inputs['gt_class_dn'] = self.inputs['paired_gt_class']
                self.inputs['im_shape']= paddle.tile(self.inputs['im_shape'][0], repeat_times=[b, 1])                    
            # Transformer
            pad_mask = self.inputs.get('pad_mask', None)
            out_transformer = self.transformer_DPDETR(None,vis_body_feats, ir_body_feats, pad_mask, self.inputs)

            if self.training:
                if self.inputs['flag'] == 10 or self.inputs['flag'] == 8:
                    detr_losses = self.detr_head_DPDETR(out_transformer, None,
                                                self.inputs)
                    detr_losses.update({
                        'loss': paddle.add_n(
                            [v for k, v in detr_losses.items() if 'log' not in k])
                    })      
                    return detr_losses
                else:
                    detr_losses = self.detr_head_DPDETR(out_transformer, None,
                                                self.inputs)
                    detr_losses.update({
                        'loss': paddle.add_n(
                            [v for k, v in detr_losses.items() if 'log' not in k])
                    })                    
                    return detr_losses
                # preds = self.detr_head_DPDETR(out_transformer, None, self.inputs)
                # return preds
            else:
                #normal
                preds = self.detr_head_DPDETR(out_transformer, None, self.inputs)
                preds = (preds[0][:,:,:],preds[1][:,:,:],preds[2])
                if self.exclude_post_process:
                    bbox, bbox_num, mask = preds
                else:
                    bbox, bbox_num, mask = self.post_process(
                        preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                        paddle.shape(self.inputs['vis_image'])[2:])

                output = {'bbox': bbox, 'bbox_num': bbox_num}
                if self.with_mask:
                    output['mask'] = mask
                return output

        # stage 3
        if self.inputs['flag'] == 9 or self.inputs['flag'] == 'eval_stage3':  # do multi branch
            # backbone
            ir_body_feats = self.backbone_stream1(self.inputs, 1)
            vis_body_feats = self.backbone_stream2(self.inputs, 2)

            # Neck
            if self.neck_stream1 is not None:
                ir_body_feats = self.neck_stream1(ir_body_feats)
                vis_body_feats = self.neck_stream2(vis_body_feats)
            if self.inputs['flag'] == 9:
                self.inputs['im_shape']= paddle.tile(self.inputs['im_shape'][0], repeat_times=[b, 1])
            # Transformer
            pad_mask = self.inputs.get('pad_mask', None)
            out_transformer = self.transformer_DPDETR(None, vis_body_feats, ir_body_feats, pad_mask,
                                                      self.inputs)

            if self.training:
                detr_losses = self.detr_head_DPDETR(out_transformer, None,
                                             self.inputs)
                detr_losses.update({
                    'loss': paddle.add_n(
                        [v for k, v in detr_losses.items() if 'log' not in k])
                })
                return detr_losses

                # preds = self.detr_head_DPDETR(out_transformer, None, self.inputs)
                # return preds
            else:
                if self.inputs['flag'] == 'eval_stage3':
                    # normal
                    preds = self.detr_head_DPDETR(out_transformer, None, self.inputs)
                    preds = (preds[0][:, :, :], preds[1][:, :, :], preds[2])
                    if self.exclude_post_process:
                        bbox, bbox_num, mask = preds
                    else:
                        bbox, bbox_num, mask = self.post_process(
                            preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                            paddle.shape(self.inputs['vis_image'])[2:])

                    output = {'bbox': bbox, 'bbox_num': bbox_num}
                    if self.with_mask:
                        output['mask'] = mask
                    return output
                elif self.inputs['flag'] != 'eval_stage3':
                    #paired infer
                    preds = self.detr_head_DPDETR(out_transformer, None, self.inputs)
                    preds = (preds[0][:, :, :], preds[1][:, :, :],preds[2][:, :, :], preds[3])
                    if self.exclude_post_process:
                        bbox, bbox_num, mask = preds
                    else:
                        bbox_vis,bbox_ir, bbox_num, mask = self.post_process_paired(
                            preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                            paddle.shape(self.inputs['vis_image'])[2:],flag=1)

                    output = {'bbox_vis': bbox_vis, 'bbox_ir':bbox_ir, 'bbox_num': bbox_num}
                    if self.with_mask:
                        output['mask'] = mask
                    return output


        #multi infer paired
        if self.inputs['flag'] == 'infer_paired':
            # backbone
            ir_body_feats = self.backbone_stream1(self.inputs, 1)
            vis_body_feats = self.backbone_stream2(self.inputs, 2)

            # Neck
            if self.neck_stream1 is not None:
                ir_body_feats = self.neck_stream1(ir_body_feats)
                vis_body_feats = self.neck_stream2(vis_body_feats)

            # Transformer
            pad_mask = self.inputs.get('pad_mask', None)
            out_transformer = self.transformer_DPDETR(None, vis_body_feats, ir_body_feats, pad_mask,
                                                      self.inputs)

            # paired infer
            preds = self.detr_head_DPDETR(out_transformer, None, self.inputs)
            preds = (preds[0][:, :, :], preds[1][:, :, :], preds[2][:, :, :], preds[3])
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox_vis, bbox_ir, bbox_num, mask = self.post_process_paired(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['vis_image'])[2:])

            output = {'bbox_vis': bbox_vis, 'bbox_ir': bbox_ir, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output



        # Neck
        # if self.neck_vis is not None:
        #     #body_feats = self.neck(body_feats)
        #     vis_body_feats = self.neck_vis(vis_body_feats)
        #     ir_body_feats = self.neck_ir(ir_body_feats)

        # body_feats = []
        # for ii in range(len(vis_body_feats)):
        #     body_feats.append(vis_body_feats[ii]+ir_body_feats[ii])
        # Transformer
        # pad_mask = self.inputs.get('pad_mask', None)
        # #(out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,dn_meta)
        # out_transformer = self.transformer(None,vis_body_feats, ir_body_feats, pad_mask, self.inputs)

        # # DETR Head
        # if self.training:
        #     detr_losses = self.detr_head(out_transformer, None,
        #                                  self.inputs)
        #     detr_losses.update({
        #         'loss': paddle.add_n(
        #             [v for k, v in detr_losses.items() if 'log' not in k])
        #     })
        #     return detr_losses
        # else:
        #     #normal
        #     preds = self.detr_head(out_transformer, None)
        #     preds = (preds[0][:,:,:],preds[1][:,:,:],preds[2])
        #     if self.exclude_post_process:
        #         bbox, bbox_num, mask = preds
        #     else:
        #         bbox, bbox_num, mask = self.post_process(
        #             preds, self.inputs['im_shape'], self.inputs['scale_factor'],
        #             paddle.shape(self.inputs['vis_image'])[2:])
        #
        #     output = {'bbox': bbox, 'bbox_num': bbox_num}
        #     if self.with_mask:
        #         output['mask'] = mask
        #     return output

            # #paired infer
            # preds = self.detr_head(out_transformer, None)
            # preds = (preds[0][:, :, :], preds[1][:, :, :],preds[2][:, :, :], preds[3])
            # if self.exclude_post_process:
            #     bbox, bbox_num, mask = preds
            # else:
            #     bbox_vis,bbox_ir, bbox_num, mask = self.post_process(
            #         preds, self.inputs['im_shape'], self.inputs['scale_factor'],
            #         paddle.shape(self.inputs['vis_image'])[2:])
            #
            # output = {'bbox_vis': bbox_vis, 'bbox_ir':bbox_ir, 'bbox_num': bbox_num}
            # if self.with_mask:
            #     output['mask'] = mask
            # return output

    def get_loss(self):
        return self._forward()

    def get_ssod_loss_single(self, input):
        detr_losses = self.detr_head_single(input, None,
                                     None, flag='DAOD')
        detr_losses.update({
            'loss': paddle.add_n(
                [v for k, v in detr_losses.items() if 'log' not in k])
        })
        return detr_losses

    def get_pred(self):
        return self._forward()