# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import copy
import time
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import typing
from PIL import Image, ImageOps, ImageFile
import copy
import math
ImageFile.LOAD_TRUNCATED_IMAGES = True
    
from scipy.optimize import linear_sum_assignment
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.static import InputSpec
from ppdet.optimizer import ModelEMA, SimpleModelEMA

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.utils.visualizer import visualize_results, save_result, visualize_results_paired
from ppdet.metrics import Metric, COCOMetric, VOCMetric, WiderFaceMetric, get_infer_results
from ppdet.metrics import RBoxMetric, SNIPERCOCOMetric
from ppdet.data.source.sniper_coco import SniperCOCODataSet
from ppdet.data.source.category import get_categories
import ppdet.utils.stats as stats
from ppdet.utils.fuse_utils import fuse_conv_bn
from ppdet.utils import profiler
from ppdet.modeling.post_process import multiclass_nms

from .callbacks import Callback, ComposeCallback, LogPrinter, Checkpointer, WiferFaceEval, VisualDLWriter, SniperProposalsGenerator, WandbCallback
from .export_utils import _dump_infer_config, _prune_input_spec, apply_to_static

from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

from ..modeling.losses.probiou_loss import ProbIoULoss
import cv2

from ppdet.utils.logger import setup_logger
logger = setup_logger('ppdet.engine')

__all__ = ['Trainer_DAOD_no_flip_supression']

MOT_ARCH = ['JDE', 'FairMOT', 'DeepSORT', 'ByteTrack', 'CenterTrack']


class Trainer_DAOD_no_flip_supression(object):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg.copy()
        assert mode.lower() in ['train', 'eval', 'test'], \
                "mode should be 'train', 'eval' or 'test'"
        self.mode = mode.lower()
        self.optimizer = None
        self.is_loaded_weights = False
        self.use_amp = self.cfg.get('amp', False)
        self.amp_level = self.cfg.get('amp_level', 'O1')
        self.custom_white_list = self.cfg.get('custom_white_list', None)
        self.custom_black_list = self.cfg.get('custom_black_list', None)
        if 'slim' in cfg and cfg['slim_type'] == 'PTQ':
            self.cfg['TestDataset'] = create('TestDataset')()

        # build data loader
        capital_mode = self.mode.capitalize()
        if cfg.architecture in MOT_ARCH and self.mode in [
                'eval', 'test'
        ] and cfg.metric not in ['COCO', 'VOC']:
            self.dataset = self.cfg['{}MOTDataset'.format(
                capital_mode)] = create('{}MOTDataset'.format(capital_mode))()
        else:
            self.dataset = self.cfg['{}Dataset'.format(capital_mode)] = create(
                '{}Dataset'.format(capital_mode))()

        if cfg.architecture == 'DeepSORT' and self.mode == 'train':
            logger.error('DeepSORT has no need of training on mot dataset.')
            sys.exit(1)

        if cfg.architecture == 'FairMOT' and self.mode == 'eval':
            images = self.parse_mot_images(cfg)
            self.dataset.set_images(images)

        if self.mode == 'train':
            self.loader = create('{}Reader'.format(capital_mode))(
                self.dataset, cfg.worker_num)

        if cfg.architecture == 'JDE' and self.mode == 'train':
            self.cfg['JDEEmbeddingHead'][
                'num_identities'] = self.dataset.num_identities_dict[0]
            # JDE only support single class MOT now.

        if cfg.architecture == 'FairMOT' and self.mode == 'train':
            self.cfg['FairMOTEmbeddingHead'][
                'num_identities_dict'] = self.dataset.num_identities_dict
            # FairMOT support single class and multi-class MOT now.

        # build model
        if 'model' not in self.cfg:
            self.model = create(cfg.architecture)

            #conduct flops
            #paddle.flops(self.model, input_size=[1,3,640,640],print_detail=True)
        else:
            self.model = self.cfg.model
            self.is_loaded_weights = True

        if cfg.architecture == 'YOLOX':
            for k, m in self.model.named_sublayers():
                if isinstance(m, nn.BatchNorm2D):
                    m._epsilon = 1e-3  # for amp(fp16)
                    m._momentum = 0.97  # 0.03 in pytorch

        #normalize params for deploy
        if 'slim' in cfg and cfg['slim_type'] == 'OFA':
            self.model.model.load_meanstd(cfg['TestReader'][
                'sample_transforms'])
        elif 'slim' in cfg and cfg['slim_type'] == 'Distill':
            self.model.student_model.load_meanstd(cfg['TestReader'][
                'sample_transforms'])
        elif 'slim' in cfg and cfg[
                'slim_type'] == 'DistillPrune' and self.mode == 'train':
            self.model.student_model.load_meanstd(cfg['TestReader'][
                'sample_transforms'])
        else:
            self.model.load_meanstd(cfg['TestReader']['sample_transforms'])

        # EvalDataset build with BatchSampler to evaluate in single device
        # TODO: multi-device evaluate
        if self.mode == 'eval':
            if cfg.architecture == 'FairMOT':
                self.loader = create('EvalMOTReader')(self.dataset, 0)
            elif cfg.architecture == "METRO_Body":
                reader_name = '{}Reader'.format(self.mode.capitalize())
                self.loader = create(reader_name)(self.dataset, cfg.worker_num)
            else:
                self._eval_batch_sampler = paddle.io.BatchSampler(
                    self.dataset, batch_size=self.cfg.EvalReader['batch_size'])
                reader_name = '{}Reader'.format(self.mode.capitalize())
                # If metric is VOC, need to be set collate_batch=False.
                if cfg.metric == 'VOC':
                    self.cfg[reader_name]['collate_batch'] = False
                self.loader = create(reader_name)(self.dataset, cfg.worker_num,
                                                  self._eval_batch_sampler)
        # TestDataset build after user set images, skip loader creation here

        # get Params
        print_params = self.cfg.get('print_params', False)
        if print_params:
            params = sum([
                p.numel() for n, p in self.model.named_parameters()
                if all([x not in n for x in ['_mean', '_variance', 'aux_']])
            ])  # exclude BatchNorm running status
            logger.info('Model Params : {} M.'.format((params / 1e6).numpy()[
                0]))

        # build optimizer in train mode
        if self.mode == 'train':
            steps_per_epoch = len(self.loader)
            if steps_per_epoch < 1:
                logger.warning(
                    "Samples in dataset are less than batch_size, please set smaller batch_size in TrainReader."
                )
            self.lr = create('LearningRate')(steps_per_epoch)
            self.optimizer = create('OptimizerBuilder')(self.lr, self.model)

            # Unstructured pruner is only enabled in the train mode.
            if self.cfg.get('unstructured_prune'):
                self.pruner = create('UnstructuredPruner')(self.model,
                                                           steps_per_epoch)
        if self.use_amp and self.amp_level == 'O2':
            self.model, self.optimizer = paddle.amp.decorate(
                models=self.model,
                optimizers=self.optimizer,
                level=self.amp_level)
        self.use_ema = ('use_ema' in cfg and cfg['use_ema'])
        if self.use_ema:
            ema_decay = self.cfg.get('ema_decay', 0.9998)
            ema_decay_type = self.cfg.get('ema_decay_type', 'threshold')
            cycle_epoch = self.cfg.get('cycle_epoch', -1)
            ema_black_list = self.cfg.get('ema_black_list', None)
            ema_filter_no_grad = self.cfg.get('ema_filter_no_grad', False)
            self.ema = ModelEMA(
                self.model,
                decay=ema_decay,
                ema_decay_type=ema_decay_type,
                cycle_epoch=cycle_epoch,
                ema_black_list=ema_black_list,
                ema_filter_no_grad=ema_filter_no_grad)

        # simple_ema for SSOD
        self.use_simple_ema = ('use_simple_ema' in cfg and
                               cfg['use_simple_ema'])
        if self.use_simple_ema:
            self.use_ema = True
            ema_decay = self.cfg.get('ema_decay', 0.9996)
            self.ema = SimpleModelEMA(self.model, decay=ema_decay)
            self.ema_start_epochs = self.cfg.get('ema_start_epochs', 0)

        self._nranks = dist.get_world_size()
        self._local_rank = dist.get_rank()

        self.status = {}

        self.start_epoch = 0
        self.end_epoch = 0 if 'epoch' not in cfg else cfg.epoch

        # initial default callbacks
        self._init_callbacks()

        # initial default metrics
        self._init_metrics()
        self._reset_metrics()

    def _init_callbacks(self):
        if self.mode == 'train':
            self._callbacks = [LogPrinter(self), Checkpointer(self)]
            if self.cfg.get('use_vdl', False):
                self._callbacks.append(VisualDLWriter(self))
            if self.cfg.get('save_proposals', False):
                self._callbacks.append(SniperProposalsGenerator(self))
            if self.cfg.get('use_wandb', False) or 'wandb' in self.cfg:
                self._callbacks.append(WandbCallback(self))
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'eval':
            self._callbacks = [LogPrinter(self)]
            if self.cfg.metric == 'WiderFace':
                self._callbacks.append(WiferFaceEval(self))
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'test' and self.cfg.get('use_vdl', False):
            self._callbacks = [VisualDLWriter(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        else:
            self._callbacks = []
            self._compose_callback = None

    def _init_metrics(self, validate=False):
        if self.mode == 'test' or (self.mode == 'train' and not validate):
            self._metrics = []
            return
        classwise = self.cfg['classwise'] if 'classwise' in self.cfg else False
        if self.cfg.metric == 'COCO' or self.cfg.metric == "SNIPERCOCO":
            # TODO: bias should be unified
            bias = 1 if self.cfg.get('bias', False) else 0
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)

            # pass clsid2catid info to metric instance to avoid multiple loading
            # annotation file
            clsid2catid = {v: k for k, v in self.dataset.catid2clsid.items()} \
                                if self.mode == 'eval' else None

            # when do validation in train, annotation file should be get from
            # EvalReader instead of self.dataset(which is TrainReader)
            if self.mode == 'train' and validate:
                eval_dataset = self.cfg['EvalDataset']
                eval_dataset.check_or_download_dataset()
                anno_file = eval_dataset.get_anno()
                dataset = eval_dataset
            else:
                dataset = self.dataset
                anno_file = dataset.get_anno()

            IouType = self.cfg['IouType'] if 'IouType' in self.cfg else 'bbox'
            if self.cfg.metric == "COCO":
                self._metrics = [
                    COCOMetric(
                        anno_file=anno_file,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IouType=IouType,
                        save_prediction_only=save_prediction_only)
                ]
            elif self.cfg.metric == "SNIPERCOCO":  # sniper
                self._metrics = [
                    SNIPERCOCOMetric(
                        anno_file=anno_file,
                        dataset=dataset,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IouType=IouType,
                        save_prediction_only=save_prediction_only)
                ]
        elif self.cfg.metric == 'RBOX':
            # TODO: bias should be unified
            bias = self.cfg['bias'] if 'bias' in self.cfg else 0
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            imid2path = self.cfg.get('imid2path', None)

            # when do validation in train, annotation file should be get from
            # EvalReader instead of self.dataset(which is TrainReader)
            anno_file = self.dataset.get_anno()
            if self.mode == 'train' and validate:
                eval_dataset = self.cfg['EvalDataset']
                eval_dataset.check_or_download_dataset()
                anno_file = eval_dataset.get_anno()

            self._metrics = [
                RBoxMetric(
                    anno_file=anno_file,
                    classwise=classwise,
                    output_eval=output_eval,
                    bias=bias,
                    save_prediction_only=save_prediction_only,
                    imid2path=imid2path)
            ]
        elif self.cfg.metric == 'VOC':
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)

            self._metrics = [
                VOCMetric(
                    label_list=self.dataset.get_label_list(),
                    class_num=self.cfg.num_classes,
                    map_type=self.cfg.map_type,
                    classwise=classwise,
                    output_eval=output_eval,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg.metric == 'WiderFace':
            multi_scale = self.cfg.multi_scale_eval if 'multi_scale_eval' in self.cfg else True
            self._metrics = [
                WiderFaceMetric(
                    image_dir=os.path.join(self.dataset.dataset_dir,
                                           self.dataset.image_dir),
                    anno_file=self.dataset.get_anno(),
                    multi_scale=multi_scale)
            ]
        else:
            logger.warning("Metric not support for metric type {}".format(
                self.cfg.metric))
            self._metrics = []

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def register_callbacks(self, callbacks):
        callbacks = [c for c in list(callbacks) if c is not None]
        for c in callbacks:
            assert isinstance(c, Callback), \
                    "metrics shoule be instances of subclass of Metric"
        self._callbacks.extend(callbacks)
        self._compose_callback = ComposeCallback(self._callbacks)

    def register_metrics(self, metrics):
        metrics = [m for m in list(metrics) if m is not None]
        for m in metrics:
            assert isinstance(m, Metric), \
                    "metrics shoule be instances of subclass of Metric"
        self._metrics.extend(metrics)

    def load_weights(self, weights, ARSL_eval=False):
        if self.is_loaded_weights:
            return
        self.start_epoch = 0
        load_pretrain_weight(self.model, weights, ARSL_eval, mode = 'multi_DAOD')
        logger.debug("Load weights {} to start training".format(weights))

    def load_weights_sde(self, det_weights, reid_weights):
        if self.model.detector:
            load_weight(self.model.detector, det_weights)
            if self.model.reid:
                load_weight(self.model.reid, reid_weights)
        else:
            load_weight(self.model.reid, reid_weights)

    def resume_weights(self, weights):
        # support Distill resume weights
        if hasattr(self.model, 'student_model'):
            self.start_epoch = load_weight(self.model.student_model, weights,
                                           self.optimizer)
        else:
            self.start_epoch = load_weight(self.model, weights, self.optimizer,
                                           self.ema if self.use_ema else None)
        vis_gt_data_path=os.path.join(os.path.dirname(weights),f'vis_gt_data_{self.start_epoch-1}.pdparams')
        if os.path.exists(vis_gt_data_path):
            self.vis_gt_data = paddle.load(vis_gt_data_path)
        logger.debug("Resume weights of epoch {}".format(self.start_epoch))

    def train(self, validate=False):
        assert self.mode == 'train', "Model not in 'train' mode"
        Init_mark = False
        
        self.semi_start_epochs = self.cfg.get('semi_start_epochs', 30)
        self.psudo_labels_threhold = self.cfg.get('psudo_label_threhold', 0.4)
        self.stage2_start_epochs = self.cfg.get('stage2_start_epochs', 12)
        self.stage3_start_epochs = self.cfg.get('stage3_start_epochs', 18)
        stage2_iter = self.stage2_start_epochs * len(self.loader)
        st_iter = self.semi_start_epochs * len(self.loader)
        stage3_iter = self.stage3_start_epochs * len(self.loader)
        
        if validate:
            self.cfg['EvalDataset'] = self.cfg.EvalDataset = create(
                "EvalDataset")()

        train_cfg = self.cfg['train_cfg']

        #model = self.model
        if self.cfg.get('to_static', False):
            self.model = apply_to_static(self.cfg, self.model)
        sync_bn = (
            getattr(self.cfg, 'norm_type', None) == 'sync_bn' and
            (self.cfg.use_gpu or self.cfg.use_npu or self.cfg.use_mlu) and
            self._nranks > 1)
        if sync_bn:
            self.model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # enabel auto mixed precision mode
        if self.use_amp:
            scaler = paddle.amp.GradScaler(
                enable=self.cfg.use_gpu or self.cfg.use_npu or self.cfg.use_mlu,
                init_loss_scaling=self.cfg.get('init_loss_scaling', 1024))
        # get distributed model
        if self.cfg.get('fleet', False):
            self.model = fleet.distributed_model(self.model)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
        elif self._nranks > 1:
            find_unused_parameters = self.cfg[
                'find_unused_parameters'] if 'find_unused_parameters' in self.cfg else False
            self.model = paddle.DataParallel(
                self.model, find_unused_parameters=find_unused_parameters)

        self.status.update({
            'epoch_id': self.start_epoch,
            'step_id': 0,
            'steps_per_epoch': len(self.loader)
        })

        self.status['batch_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg.log_iter)

        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg.worker_num)
            self._flops(flops_loader)
        profiler_options = self.cfg.get('profiler_options', None)

        self._compose_callback.on_train_begin(self.status)

        use_fused_allreduce_gradients = self.cfg[
            'use_fused_allreduce_gradients'] if 'use_fused_allreduce_gradients' in self.cfg else False

        stage_single_param_prefixes = (
            'backbone_stream1',
            'neck_stream1',
            'transformer_single',
            'detr_head_single',
        )
        stage_multi_param_prefixes = (
            'backbone_stream1',
            'backbone_stream2',
            'neck_stream1',
            'neck_stream2',
            'transformer_DPDETR',
            'detr_head_DPDETR',
        )
        stage2_param_prefixes = stage_multi_param_prefixes + (
            'transformer_single',
            'detr_head_single',
        )

        self._stage_fps_stats = {'name': None, 'time': 0.0, 'images': 0}

        for param in self.ema.model.parameters():
            param.stop_gradient = True

        #self.iou_matrix = ProbIoULoss()

        for epoch_id in range(self.start_epoch, self.cfg.epoch):
            self.status['mode'] = 'train'
            self.status['epoch_id'] = epoch_id
            self.epoch=epoch_id
            self._compose_callback.on_epoch_begin(self.status)
            self.loader.dataset.set_epoch(epoch_id)
            #model.train()
            iter_tic = time.time()

            loss_dict = {
                'loss': paddle.to_tensor([0]),
                'loss_sup_sum': paddle.to_tensor([0]),
                'loss_unsup_sum': paddle.to_tensor([0]),
                'fg_sum': paddle.to_tensor([0]),
            }
            self.model.train()
            self.ema.model.eval()
            for step_id, data in enumerate(self.loader):
                self.step = step_id
                bs=data['im_id'].shape[0]
                self.status['data_time'].update(time.time() - iter_tic)
                self.status['step_id'] = step_id
                profiler.add_profiler_step(profiler_options)
                self._compose_callback.on_step_begin(self.status)
                data['epoch_id'] = epoch_id
                if self.cfg.get('to_static',
                                False) and 'image_file' in data.keys():
                    data.pop('image_file')

                if self.use_amp:
                    if isinstance(
                            self.model, paddle.
                            DataParallel) and use_fused_allreduce_gradients:
                        with self.model.no_sync():
                            with paddle.amp.auto_cast(
                                    enable=self.cfg.use_gpu or
                                    self.cfg.use_npu or self.cfg.use_mlu,
                                    custom_white_list=self.custom_white_list,
                                    custom_black_list=self.custom_black_list,
                                    level=self.amp_level):
                                # model forward
                                outputs = self.model(data)
                                loss = outputs['loss']
                            # model backward
                            scaled_loss = scaler.scale(loss)
                            scaled_loss.backward()
                        fused_allreduce_gradients(
                            list(self.model.parameters()), None)
                    else:
                        with paddle.amp.auto_cast(
                                enable=self.cfg.use_gpu or self.cfg.use_npu or
                                self.cfg.use_mlu,
                                custom_white_list=self.custom_white_list,
                                custom_black_list=self.custom_black_list,
                                level=self.amp_level):
                            # model forward
                            outputs = self.model(data)
                            loss = outputs['loss']
                        # model backward
                        scaled_loss = scaler.scale(loss)
                        scaled_loss.backward()
                    # in dygraph mode, optimizer.minimize is equal to optimizer.step
                    scaler.minimize(self.optimizer, scaled_loss)
                else:
                    if isinstance(
                            self.model, paddle.
                            DataParallel) and use_fused_allreduce_gradients:
                        with self.model.no_sync():
                            # model forward
                            outputs = self.model(data)
                            loss = outputs['loss']
                            # model backward
                            loss.backward()
                        fused_allreduce_gradients(
                            list(self.model.parameters()), None)
                    else:

                        curr_iter = len(self.loader) * epoch_id + step_id
###############################################################################################################STAGE 1
##################################sup- 真实标签监督   unsup- 伪标签监督   multi- DP-MSDecoder###################
                        if curr_iter < stage2_iter:

                            if curr_iter == 0:
                                logger.info("***" * 30)
                                logger.info('STAGE 1 ...')
                                logger.info("***" * 30)
                                if self.cfg.get('print_flops', False):
                                    single_params, single_flops = self._log_stage_flops_and_params(
                                        data,
                                        stage_name='stage1_single_base',
                                        stage_flag='eval_stage1',
                                        param_prefixes=stage_single_param_prefixes)
                                    stage1_has_unsup = st_iter < stage2_iter
                                    if single_params is not None:
                                        logger.info(
                                            "Stage stage1 Params (student) : {:.3f} M.".format(
                                                single_params / 1e6))
                                        if self.use_ema:
                                            logger.info(
                                                "Stage stage1 Params (student+teacher) : {:.3f} M.".format(
                                                    (single_params * 2) / 1e6))
                                    if single_flops is not None:
                                        sup_only = single_flops * 2
                                        logger.info(
                                            "Stage stage1 FLOPs (student sup only) : {:.6f} G.".format(
                                                sup_only))
                                        if stage1_has_unsup:
                                            student_total = single_flops * 3
                                            logger.info(
                                                "Stage stage1 FLOPs (student sup+unsup) : {:.6f} G.".format(
                                                    student_total))
                                            if self.use_ema:
                                                logger.info(
                                                    "Stage stage1 FLOPs (student+teacher sup+unsup) : {:.6f} G.".format(
                                                        student_total +
                                                        single_flops))

                            self.model.train()
                            self.ema.model.eval()

                            # concat sup data (vis, vis_aug)
                            #data['vis_image_cat'] = paddle.concat([data['vis_image_aug'],data['vis_image']])
                            data['ir_image_cat'] = paddle.concat([data['ir_image_aug'],data['ir_image']])
                            # # concat sup label
                            # data['gt_bbox_vis_cat'] = paddle.concat([data['gt_bbox_vis'],data['gt_bbox_vis']])
                            # data['gt_poly_vis_cat'] = paddle.concat([data['gt_poly_vis'],data['gt_poly_vis']])
                            # data['gt_rbox_vis_cat'] = paddle.concat([data['gt_rbox_vis'],data['gt_rbox_vis']])

                            # model forward
                            data['flag'] = 3
                            outputs = self.model(data) #flag=3 means do sup_concat train
                            loss_sup = outputs['loss'] * self.cfg['sup_weight']
                            # model backward
                            loss_sup.backward()
                            losses = loss_sup.detach()

                            #st_iter = 3
                            #stage2_iter = 5
                            #stage3_iter = 10
                            #self.semi_start_epochs = -1
                            # self.ema_start_iter = 1
                            if curr_iter == st_iter:
                                self.vis_gt_data=dict()
                                logger.info("***" * 30)
                                logger.info('Semi starting ...')
                                logger.info("***" * 30)
                            if curr_iter >= st_iter:

                                unsup_weight = train_cfg['unsup_weight']
                                # if train_cfg['suppress'] == 'linear':
                                #     tar_iter = st_iter * 2
                                #     if curr_iter <= tar_iter:
                                #         unsup_weight *= (curr_iter - st_iter) / st_iter
                                # elif train_cfg['suppress'] == 'exp':
                                #     tar_iter = st_iter + 2000
                                #     if curr_iter <= tar_iter:
                                #         scale = np.exp((curr_iter - tar_iter) / 1000)
                                #         unsup_weight *= scale
                                # elif train_cfg['suppress'] == 'step':
                                #     tar_iter = st_iter * 2
                                #     if curr_iter <= tar_iter:
                                #         unsup_weight *= 0.25
                                # else:
                                #     raise ValueError
                                if train_cfg['suppress'] == 'linear':
                                    tar_iter=int(st_iter+(stage2_iter-st_iter)*2/3)
                                    if curr_iter <= tar_iter:
                                        unsup_weight *= 2*(curr_iter - st_iter) / (stage2_iter-st_iter)
                                elif train_cfg['suppress'] == 'exp':
                                    tar_iter = st_iter + 2000
                                    if curr_iter <= tar_iter:
                                        scale = np.exp((curr_iter - tar_iter) / 1000)
                                        unsup_weight *= scale
                                elif train_cfg['suppress'] == 'step':
                                    tar_iter = st_iter * 2
                                    if curr_iter <= tar_iter:
                                        unsup_weight *= 0.25
                                else:
                                    raise ValueError
                                # if data_unsup_w['image'].shape != data_unsup_s[
                                #     'image'].shape:
                                #     data_unsup_w, data_unsup_s = align_weak_strong_shape(
                                #         data_unsup_w, data_unsup_s)

                                # data_unsup_w['epoch_id'] = epoch_id
                                # data_unsup_s['epoch_id'] = epoch_id
                                #
                                # data_unsup_s['get_data'] = True
                                 #4 means do student preds unsup_s
                                
                                # _,bs,_,_ = student_preds[0].shape
                                #H = data['im_shape'][0,0]
                                W = data['im_shape'][0, 1]
                                with paddle.no_grad():
                                    #data_unsup_w['is_teacher'] = True
                                    data['flag'] = 5 #5 means do teacher preds  unsup_w
                                    teacher_preds = self.ema.model(data)
                                    #fliter high socre preds as persudo labels
                                    # mean = teacher_preds['bbox'][:,1][teacher_preds['bbox'][:,1]>0.2].mean().item()
                                    # std = teacher_preds['bbox'][:,1][teacher_preds['bbox'][:,1]>0.2].std().item()
                                    bbox_y = teacher_preds['bbox'][:, 1]
                                    valid_y = bbox_y[bbox_y > 0.2]
                                    if valid_y.shape[0] > 0:
                                        mean = valid_y.mean().item()
                                        std = valid_y.std().item()
                                    else:
                                        mean = 10.0  # 或者你可以选择别的默认值，比如 None
                                        std = 0.0

                                    teacher_preds['bbox'] = paddle.split(teacher_preds['bbox'],bs,0)
                                    teacher_preds['class'] = []
                                    teacher_preds['score'] = []
                                    # filterd_teacher_preds = dict()
                                    # filterd_teacher_preds['bbox'] = []
                                    # filterd_teacher_preds['class'] = []
                                    for xx in range(len(teacher_preds['bbox'])) :
                                        teacher_preds['bbox'][xx] = teacher_preds['bbox'][xx][teacher_preds['bbox'][xx][:,1] > (mean-std)]#self.psudo_labels_threhold
                                        teacher_preds['bbox_num'][xx] = len(teacher_preds['bbox'][xx])

                                        teacher_preds['class'].append(teacher_preds['bbox'][xx][:,0].unsqueeze(1).astype('int32'))
                                        teacher_preds['score'].append(teacher_preds['bbox'][xx][:,1].unsqueeze(1))
                                        teacher_preds['bbox'][xx] = teacher_preds['bbox'][xx][:, 2:]
                                if epoch_id==self.semi_start_epochs:###记得保存模型的时候也得把self.vis_gt_data保存下来，这个框的大小对应544*640的
                                    if 'im_id' not in self.vis_gt_data and 'gt_rbox_vis' not in self.vis_gt_data:
                                        self.vis_gt_data = {
                                            'im_id': data['im_id'].clone(),  # 或 paddle.to_tensor(data['im_id'].numpy())
                                            'gt_rbox_vis': [t.clone() for t in data['gt_rbox_ir']],
                                            'gt_class': [t.clone() for t in data['gt_class']]
                                        }
                                    else:
                                        self.vis_gt_data['im_id'] = paddle.concat(
                                            [self.vis_gt_data['im_id'], data['im_id'].clone()], axis=0
                                        )
                                        self.vis_gt_data['gt_rbox_vis']+= copy.deepcopy(data['gt_rbox_ir'])
                                        self.vis_gt_data['gt_class'] += copy.deepcopy(data['gt_class'])
                                paired_gt_rbox_ir_match, paired_gt_rbox_ir_unmatch, paired_gt_class, \
                                    paired_teacher_bbox_match, paired_teacher_bbox_unmatch= self.paired_label_processing(data,teacher_preds['bbox'],teacher_preds['class'],teacher_preds['score'])
                                class_ir_match=[]
                                class_ir_unmatch=[]
                                for i,value in enumerate(paired_gt_rbox_ir_match):
                                    num=(value.shape)[0]
                                    class_ir_match.append(paired_gt_class[i][:num])
                                    class_ir_unmatch.append(paired_gt_class[i][num:])                           
                                self.paired_label_visual(data['im_id'],data['ir_image'],data['vis_image'],{'bbox': data['gt_rbox_ir'],'class':data['gt_class']},teacher_preds)
                                # if self.epoch >= self.semi_start_epochs+4:
                                #     self.paired_label_visual_shiftcorrectshow(data['im_id'],data['ir_image'],data['vis_image'],{'bbox': paired_gt_rbox_ir_match,'class':paired_gt_class},paired_teacher_bbox_match) 
                                for i in range(len(paired_teacher_bbox_match)):
                                    tensor = paired_teacher_bbox_match[i]
                                    if list(tensor.shape) == [0]:
                                        # 替换为 [0, 5] 的空 tensor
                                        paired_teacher_bbox_match[i] = paddle.empty([0, 5], dtype=tensor.dtype)
                                
                                data['teacher_gt_class']=class_ir_match
                                data['teacher_gt_rbox_vis']=paired_teacher_bbox_match
                                data['flag'] = 4
                                student_preds = self.model(data)

                                train_cfg['curr_iter'] = curr_iter
                                train_cfg['st_iter'] = st_iter

                                # loss_dict_unsup = self.model.get_ssod_loss_single((student_preds[0], student_preds[1], student_preds[2],
                                #                                             teacher_preds['bbox'], teacher_preds['class'], data['im_shape'],
                                #                                             None,None,None,None))
                                loss_dict_unsup = self.model.get_ssod_loss_single((student_preds[0], student_preds[1], student_preds[2],
                                                                            paired_teacher_bbox_match, class_ir_match, data['im_shape'],
                                                                            student_preds[3],student_preds[4],student_preds[5],student_preds[6]))                                

                                # fg_num = loss_dict_unsup["fg_sum"]
                                # del loss_dict_unsup["fg_sum"]
                                # distill_weights = train_cfg['loss_weight']
                                # loss_dict_unsup = {
                                #     k: v * distill_weights[k]
                                #     for k, v in loss_dict_unsup.items()
                                # }

                                losses_unsup = loss_dict_unsup['loss'] * unsup_weight
                                losses_unsup.backward()

                                #loss_dict.update(loss_dict_unsup)
                                loss_dict.update({'loss_unsup_sum': losses_unsup})
                                losses += losses_unsup.detach()
                                #loss_dict.update({"fg_sum": fg_num})
                                loss_dict['loss'] = losses

###############################################################################################################STAGE 2
##################################sup- 真实标签监督   unsup- 伪标签监督   multi- DP-MSDecoder####################
                        if curr_iter >= stage2_iter and curr_iter < stage3_iter: #STAGE 2
                            if curr_iter == stage2_iter:
                                logger.info("***" * 30)
                                logger.info('STAGE 2 ...')
                                logger.info("***" * 30)

                                # #
                                # new_items = {}
                                # for key in self.status['training_staus'].meters.keys():
                                #     if '_dn' not in key:
                                #         new_items[key+'_ir'] = 0.0
                                #         new_items[key + '_vis'] = 0.0
                                #     else:
                                #         temp_key = key.replace('_dn','')
                                #         new_items[temp_key+'_ir_dn'] = 0.0
                                #         new_items[temp_key + '_vis_dn'] = 0.0
                                # self.status['training_staus'].meters.update(new_items)
                                # del self.status['training_staus'].meters['loss_class_vis']
                                # del self.status['training_staus'].meters['loss_class_ir']
                                # del self.status['training_staus'].meters['loss_vis']
                                # del self.status['training_staus'].meters['loss_ir']
                                # #

                                logger.info('copy weights to multi branch ......')
                                # copy weights to multi branch
                                model_params = self.model.state_dict()
                                for name, param in model_params.items():
                                    if '_stream2' in name:
                                        stream1_name = name.replace('_stream2','_stream1')
                                        param.set_value(model_params[stream1_name])
                                logger.info('copy done !!!')
                                if self.cfg.get('print_flops', False):
                                    single_params, single_flops = self._log_stage_flops_and_params(
                                        data,
                                        stage_name='stage2_single',
                                        stage_flag='eval_stage1',
                                        param_prefixes=stage_single_param_prefixes)
                                    multi_params, multi_flops = self._log_stage_flops_and_params(
                                        data,
                                        stage_name='stage2_multi',
                                        stage_flag='eval_stage2',
                                        param_prefixes=stage_multi_param_prefixes)
                                    stage2_params, stage2_matched = self._count_params_by_prefix(
                                        self._get_base_model(),
                                        stage2_param_prefixes)
                                    if not stage2_matched:
                                        stage2_params = self._count_total_params(
                                            self._get_base_model())
                                    logger.info(
                                        "Stage stage2 Params (student) : {:.3f} M.".format(
                                            stage2_params / 1e6))
                                    if self.use_ema:
                                        logger.info(
                                            "Stage stage2 Params (student+teacher) : {:.3f} M.".format(
                                                (stage2_params * 2) / 1e6))
                                    if single_flops is not None and multi_flops is not None:
                                        student_total = single_flops * 2 + multi_flops
                                        logger.info(
                                            "Stage stage2 FLOPs (student total) : {:.6f} G.".format(
                                                student_total))
                                        if self.use_ema:
                                            logger.info(
                                                "Stage stage2 FLOPs (student+teacher total) : {:.6f} G.".format(
                                                    student_total +
                                                    single_flops))


                            ########STAGE 2 ---- SINGLE BRANCH
                            # model single forward
                            data['flag'] = 7
                            outputs = self.model(data)  # flag=7 means do sup train
                            loss_sup = outputs['loss'] * self.cfg['sup_weight']
                            # model backward
                            loss_sup.backward()
                            losses = loss_sup.detach()

                            #do unsup--single

                            # _, bs, _, _ = student_preds[0].shape
                            # H = data['im_shape'][0,0]
                            W = data['im_shape'][0, 1]
                            with paddle.no_grad():
                                # data_unsup_w['is_teacher'] = True
                                data['flag'] = 5  # 5 means do teacher preds  unsup_w
                                teacher_preds = self.ema.model(data)
                                bbox_y = teacher_preds['bbox'][:, 1]
                                valid_y = bbox_y[bbox_y > 0.2]
                                if valid_y.shape[0] > 0:
                                    mean = valid_y.mean().item()
                                    std = valid_y.std().item()
                                else:
                                    mean = 10.0  # 或者你可以选择别的默认值，比如 None
                                    std = 0.0
                                # fliter high socre preds as persudo labels
                                teacher_preds['bbox'] = paddle.split(teacher_preds['bbox'], bs, 0)
                                teacher_preds['class'] = []
                                teacher_preds['score'] = []
                                # filterd_teacher_preds = dict()
                                # filterd_teacher_preds['bbox'] = []
                                # filterd_teacher_preds['class'] = []
                                for xx in range(len(teacher_preds['bbox'])):
                                    teacher_preds['bbox'][xx] = teacher_preds['bbox'][xx][
                                        teacher_preds['bbox'][xx][:, 1] > (mean-std)]#self.psudo_labels_threhold
                                    teacher_preds['bbox_num'][xx] = len(teacher_preds['bbox'][xx])
                                    teacher_preds['class'].append(
                                        teacher_preds['bbox'][xx][:, 0].unsqueeze(1).astype('int32'))
                                    teacher_preds['score'].append(teacher_preds['bbox'][xx][:,1].unsqueeze(1))
                                    teacher_preds['bbox'][xx] = teacher_preds['bbox'][xx][:, 2:]
                            paired_gt_rbox_ir_match, paired_gt_rbox_ir_unmatch, paired_gt_class, \
                                    paired_teacher_bbox_match, paired_teacher_bbox_unmatch= self.paired_label_processing(data,teacher_preds['bbox'],teacher_preds['class'],teacher_preds['score'])
                            class_ir_match=[]
                            class_ir_unmatch=[]
                            for i,value in enumerate(paired_gt_rbox_ir_match):
                                num=(value.shape)[0]
                                class_ir_match.append(paired_gt_class[i][:num])
                                class_ir_unmatch.append(paired_gt_class[i][num:])
                            self.paired_label_visual(data['im_id'],data['ir_image'],data['vis_image'],{'bbox': data['gt_rbox_ir'],'class':data['gt_class']},teacher_preds)

                            data['teacher_gt_class']=class_ir_match
                            data['teacher_gt_rbox_vis']=paired_teacher_bbox_match           
                            data['flag'] = 4  # 4 means do student preds unsup_s
                            student_preds = self.model(data)
                            train_cfg['curr_iter'] = curr_iter
                            train_cfg['st_iter'] = st_iter
                            
                            loss_dict_unsup = self.model.get_ssod_loss_single((student_preds[0], student_preds[1], student_preds[2],
                                                                        paired_teacher_bbox_match, class_ir_match, data['im_shape'],
                                                                        student_preds[3],student_preds[4],student_preds[5],student_preds[6]))                                

                            losses_unsup = loss_dict_unsup['loss'] * 1 #
                            losses_unsup.backward()

                            # loss_dict.update(loss_dict_unsup)
                            loss_dict.update({'loss_unsup_sum': losses_unsup})
                            losses += losses_unsup.detach()
                            # loss_dict.update({"fg_sum": fg_num})
                            loss_dict['loss'] = losses

                            #############STAGE 2 ---- MULTI BRANCH
                                 #paired label processing
                            # paired_gt_rbox_vis, paired_gt_class, paired_teacher_rbox_ir = self.paired_label_processing(data['gt_rbox_vis'],
                            #                                                                                  data['gt_class'],
                            #                                                                                  teacher_preds['bbox'],
                            #                                                                                  teacher_preds['class'])
                            # data['paired_gt_rbox_vis'] = paired_gt_rbox_vis
                            # data['paired_gt_class'] = paired_gt_class
                            # data['paired_teacher_rbox_ir'] = paired_teacher_rbox_ir

                            #self.paired_label_visual_shiftcorrectshow(data['im_id'],data['ir_image'],data['vis_image'],{'bbox': paired_gt_rbox_ir_match,'class':paired_gt_class},paired_teacher_bbox_match) 
                            data['paired_gt_rbox_ir_match'] = paired_gt_rbox_ir_match
                            data['paired_gt_rbox_ir_unmatch'] = paired_gt_rbox_ir_unmatch
                            data['paired_gt_class'] = paired_gt_class
                            data['paired_teacher_bbox_match'] = paired_teacher_bbox_match 
                            data['paired_teacher_bbox_unmatch'] = paired_teacher_bbox_unmatch
        
                                #student do multi branch
                            data['flag'] = 8
                            outputs = self.model(data)  # flag=8 means do multi branch train
                            loss_multi = outputs['loss'] * 1
                            # model backward
                            loss_multi.backward()
                            losses += loss_multi.detach()
###############################################################################################################STAGE 3
##################################sup- 真实标签监督   unsup- 伪标签监督   multi- DP-MSDecoder####################
                        if curr_iter >= stage3_iter: #STAGE 3
                            if curr_iter == stage3_iter:
                                logger.info("***" * 30)
                                logger.info('STAGE 3 ...')
                                logger.info("***" * 30)
                                if self.cfg.get('print_flops', False):
                                    multi_params, multi_flops = self._log_stage_flops_and_params(
                                        data,
                                        stage_name='stage3',
                                        stage_flag='eval_stage3',
                                        param_prefixes=stage_multi_param_prefixes)
                                    if multi_params is not None:
                                        logger.info(
                                            "Stage stage3 Params (student) : {:.3f} M.".format(
                                                multi_params / 1e6))
                                        if self.use_ema:
                                            logger.info(
                                                "Stage stage3 Params (student+teacher) : {:.3f} M.".format(
                                                    (multi_params * 2) / 1e6))
                                    if multi_flops is not None:
                                        student_total = multi_flops
                                        logger.info(
                                            "Stage stage3 FLOPs (student total) : {:.6f} G.".format(
                                                student_total))
                                        if self.use_ema:
                                            logger.info(
                                                "Stage stage3 FLOPs (student+teacher total) : {:.6f} G.".format(
                                                    student_total * 2))
                            # do teacher multi
                            with paddle.no_grad():
                                # data_unsup_w['is_teacher'] = True
                                data['flag'] = 9  # 9 means do teacher multi preds  unsup_w
                                teacher_preds = self.ema.model(data)
                                bbox_y = teacher_preds['bbox_vis'][:, 1]
                                valid_y = bbox_y[bbox_y > 0.2]
                                if valid_y.shape[0] > 0:
                                    mean = valid_y.mean().item()
                                    std = valid_y.std().item()
                                else:
                                    mean = 10.0  # 或者你可以选择别的默认值，比如 None
                                    std = 0.0
                                # fliter high socre preds as persudo labels
                                teacher_preds['bbox'] = paddle.split(teacher_preds['bbox_vis'], bs, 0)
                                teacher_preds['class'] = []
                                teacher_preds['score'] = []

                                for xx in range(len(teacher_preds['bbox'])):
                                    teacher_preds['bbox'][xx] = teacher_preds['bbox'][xx][
                                        teacher_preds['bbox'][xx][:, 1] > (mean-std)]
                                    teacher_preds['bbox_num'][xx] = len(teacher_preds['bbox'][xx])
                                    teacher_preds['score'].append(teacher_preds['bbox'][xx][:,1].unsqueeze(1))
                                    teacher_preds['class'].append(
                                        teacher_preds['bbox'][xx][:, 0].unsqueeze(1).astype('int32'))
                                    teacher_preds['bbox'][xx] = teacher_preds['bbox'][xx][:, 2:]

                            train_cfg['curr_iter'] = curr_iter
                            train_cfg['st_iter'] = st_iter


                            # paired_gt_rbox_ir_match, paired_gt_rbox_ir_unmatch, paired_gt_class, \
                            #     paired_teacher_bbox_match, paired_teacher_bbox_unmatch = self.paired_label_processing(
                            #         data['gt_rbox_ir'],
                            #         data['gt_class'],
                            #         teacher_preds['bbox'],
                            #         teacher_preds['class']
                            #     )
                            paired_gt_rbox_ir_match, paired_gt_rbox_ir_unmatch, paired_gt_class, \
                                    paired_teacher_bbox_match, paired_teacher_bbox_unmatch= self.paired_label_processing(data,teacher_preds['bbox'],teacher_preds['class'],teacher_preds['score'])
                            class_ir_match=[]
                            class_ir_unmatch=[]
                            for i,value in enumerate(paired_gt_rbox_ir_match):
                                num=(value.shape)[0]
                                class_ir_match.append(paired_gt_class[i][:num])
                                class_ir_unmatch.append(paired_gt_class[i][num:])
                            self.paired_label_visual(data['im_id'],data['ir_image'],data['vis_image'],{'bbox': data['gt_rbox_ir'],'class':data['gt_class']},teacher_preds)
                            #self.paired_label_visual_shiftcorrectshow(data['im_id'],data['ir_image'],data['vis_image'],{'bbox': paired_gt_rbox_ir_match,'class':paired_gt_class},paired_teacher_bbox_match) 
                            data['paired_gt_rbox_ir_match'] = paired_gt_rbox_ir_match
                            data['paired_gt_rbox_ir_unmatch'] = paired_gt_rbox_ir_unmatch
                            data['paired_gt_class'] = paired_gt_class
                            data['paired_teacher_bbox_match'] = paired_teacher_bbox_match 
                            data['paired_teacher_bbox_unmatch'] = paired_teacher_bbox_unmatch
                            # student do multi branch
                            data['flag'] = 10
                            outputs = self.model(data)  # flag=8 means do multi branch train
                            loss_multi = outputs['loss'] * 1
                            # model backward
                            loss_multi.backward()
                            losses = loss_multi.detach()
                    self.optimizer.step()
                curr_lr = self.optimizer.get_lr()
                self.lr.step()
                if self.cfg.get('unstructured_prune'):
                    self.pruner.step()
                self.optimizer.clear_grad()
                self.status['learning_rate'] = curr_lr

                if self._nranks < 2 or self._local_rank == 0:
                    if curr_iter >= stage2_iter and curr_iter <= stage3_iter:
                        self.status['training_staus'].update(outputs, flag=1)
                    else:
                        self.status['training_staus'].update(outputs)

                iter_cost = time.time() - iter_tic
                self.status['batch_time'].update(iter_cost)
                if self.cfg.get('print_fps', False):
                    self._update_stage_fps(curr_iter, stage2_iter, stage3_iter,
                                           st_iter, data, iter_cost)
                self._compose_callback.on_step_end(self.status)

                #curr_iter = len(self.loader) * epoch_id + step_id
                self.ema_start_iter = self.ema_start_epochs * len(self.loader)
                #self.ema_start_iter = 1
                # Note: ema_start_iters
                if self.use_ema and curr_iter == self.ema_start_iter:
                    logger.info("***" * 30)
                    logger.info('EMA starting ...')
                    logger.info("***" * 30)
                    self.ema.update(self.model, decay=0)
                elif self.use_ema and curr_iter > self.ema_start_iter:
                    self.ema.update(self.model)
                iter_tic = time.time()
            if self.cfg.get('unstructured_prune'):
                self.pruner.update_params()

            is_snapshot = (self._nranks < 2 or (self._local_rank == 0 or self.cfg.metric == "Pose3DEval")) \
                       and ((epoch_id + 1) % self.cfg.snapshot_epoch == 0 or epoch_id == self.end_epoch - 1)
            if is_snapshot and self.use_ema:

                # apply ema weight on model
                weight = copy.deepcopy(self.ema.model.state_dict())
                for k, v in weight.items():
                    if paddle.is_floating_point(v):
                        weight[k].stop_gradient = True
                self.status['weight'] = weight
            if hasattr(self, 'vis_gt_data'):
                self.status['vis_gt_data']= copy.deepcopy(self.vis_gt_data)
            self._compose_callback.on_epoch_end(self.status)

            if validate and is_snapshot:
                if not hasattr(self, '_eval_loader'):
                    # build evaluation dataset and loader
                    self._eval_dataset = self.cfg.EvalDataset
                    self._eval_batch_sampler = \
                        paddle.io.BatchSampler(
                            self._eval_dataset,
                            batch_size=self.cfg.EvalReader['batch_size'])
                    # If metric is VOC, need to be set collate_batch=False.
                    if self.cfg.metric == 'VOC':
                        self.cfg['EvalReader']['collate_batch'] = False
                    if self.cfg.metric == "Pose3DEval":
                        self._eval_loader = create('EvalReader')(
                            self._eval_dataset, self.cfg.worker_num)
                    else:
                        self._eval_loader = create('EvalReader')(
                            self._eval_dataset,
                            self.cfg.worker_num,
                            batch_sampler=self._eval_batch_sampler)
                # if validation in training is enabled, metrics should be re-init
                # Init_mark makes sure this code will only execute once
                if validate and Init_mark == False:
                    Init_mark = True
                    self._init_metrics(validate=validate)
                    self._reset_metrics()

                with paddle.no_grad():
                    self.status['save_best_model'] = True
                    if epoch_id > self.semi_start_epochs:
                        eval_flag = 1
                    else:
                        eval_flag = 0
                    if curr_iter <= stage2_iter:
                        self._eval_with_loader(self._eval_loader, eval_flag, my_eval_flag='eval_stage1')
                    elif curr_iter > stage2_iter and curr_iter <= stage3_iter:
                        self._eval_with_loader(self._eval_loader, eval_flag, my_eval_flag='eval_stage2')
                    elif curr_iter > stage3_iter:
                        self._eval_with_loader(self._eval_loader, eval_flag, my_eval_flag='eval_stage3')

            if is_snapshot and self.use_ema:
                self.status.pop('weight')

        if self.cfg.get('print_fps', False):
            self._log_stage_fps(self._stage_fps_stats)
        self._compose_callback.on_train_end(self.status)
    def paired_label_visual(self, im_id, ir_image, vis_image, bboxes_ir, bboxes_vis):
        def box2corners(pred_bboxes):
            x, y, w, h, angle = pred_bboxes
            cos_a_half = math.cos(angle) * 0.5
            sin_a_half = math.sin(angle) * 0.5
            w_x = cos_a_half * w
            w_y = sin_a_half * w
            h_x = -sin_a_half * h
            h_y = cos_a_half * h
            return np.array([x + w_x + h_x, y + w_y + h_y, x - w_x + h_x, y - w_y + h_y,x - w_x - h_x, y - w_y - h_y, x + w_x - h_x, y + w_y - h_y])
        def draw_bbox_paired(image, bboxes,i):
            """
            Draw bbox on image
            """
            bboxes=copy.deepcopy(bboxes)
            draw = ImageDraw.Draw(image)
            count= 0
            catid2color = {}
            catid2name={0: 'car', 1: 'truck', 2: 'bus', 3: 'van', 4: 'feright'}
            for key, value in bboxes.items():
                for k,value1 in enumerate(bboxes[key]):
                    bboxes[key][k]=value1.numpy()
            for j,_ in enumerate(bboxes['bbox'][i]):
                if bboxes.get('score') is not None:
                    catid, bbox, score = bboxes['class'][i][j], bboxes['bbox'][i][j], bboxes['score'][i][j]
                else:
                    catid, bbox= bboxes['class'][i][j], bboxes['bbox'][i][j]
                # if score < threshold:
                #     continue
                catid=catid[0]
                if catid not in catid2color:
                    if catid == 0:
                        catid2color[catid] = np.array([16, 215, 248])# 255 0 0
                    elif catid == 1:
                        catid2color[catid] = np.array([0, 0, 255])
                    elif catid == 2:
                        catid2color[catid] = np.array([0, 255, 0])
                    elif catid == 3:
                        catid2color[catid] = np.array([255, 165, 0])
                    elif catid == 4:
                        catid2color[catid] = np.array([160, 32, 240])
                        # idx = np.random.randint(len(color_list))
                        # catid2color[catid] = color_list[idx]
                color = tuple(catid2color[catid]) 
                x1, y1, x2, y2, x3, y3, x4, y4 = box2corners(bbox)#*840/640
                draw.line(
                    [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                    width=2,
                    fill=color)
                xmin = min(x1, x2, x3, x4)
                ymin = min(y1, y2, y3, y4)
                xmax = max(x1, x2, x3, x4)
                ymax = max(y1, y2, y3, y4)
                if bboxes.get('score') is not None:
                    text = "{}|{}|{:.2f}".format(count,catid2name[catid],score[0])
                else:
                    text = "{}|{}".format(count,catid2name[catid])
                count+=1
                bbox = draw.textbbox((0,0), text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.rectangle(
                    [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
                draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
            return image
            
        def tensor_to_pil_images(tensor):
            tensor=copy.deepcopy(tensor)
            # 反标准化参数（ImageNet）
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            
            # 将 Tensor 数据移动到 CPU 并转换为 numpy 数组
            tensor_np = tensor.numpy()  # 假设已处理过梯度或位于 CPU
            
            images = []
            for batch in range(tensor_np.shape[0]):
                # 提取单张图像数据 [3, H, W]
                img_data = tensor_np[batch]
                
                # 反标准化
                img_data = img_data * std + mean
                img_data = np.clip(img_data, 0, 1)  # 限制数值范围
                
                # 转换为 HWC 格式并调整到 [0, 255]
                img_data = np.transpose(img_data, (1, 2, 0))
                img_data = (img_data * 255).astype(np.uint8)
                
                # 生成 PIL 图像
                images.append(Image.fromarray(img_data))
            return images
        folder = f'paired_label_visual2/{self.epoch}'
        # 创建文件夹（如果不存在，且支持递归创建多级目录）
        os.makedirs(folder, exist_ok=True)
        ir_images=tensor_to_pil_images(ir_image)
        vis_images=tensor_to_pil_images(vis_image)
        im_id=copy.deepcopy(im_id)
        imid2savepath={}
        im_id=im_id.numpy()+1
        for i,id in enumerate(im_id):
            if os.path.exists(f'{self.cfg.TrainDataset.dataset_dir}/train_1000/trainimg/{(id[0]):05d}.jpg'):
                id_str_ir = '{:05d}'.format(id[0])+'_ir.jpg'
                id_str_vis = '{:05d}'.format(id[0])+'_vis.jpg'
                ir_image_savepath = f'{folder}/{id_str_ir}'
                vis_image_savepath = f'{folder}/{id_str_vis}'
                vis_image=vis_images[i]
                ir_image=ir_images[i]
                vis_image=draw_bbox_paired(vis_image,bboxes_vis,i)
                ir_image=draw_bbox_paired(ir_image,bboxes_ir,i) 
                #logger.info("Detection bbox results save in {}".format(ir_image_savepath+' '+vis_image_savepath))
                vis_image.save(vis_image_savepath, quality=100)
                ir_image.save(ir_image_savepath, quality=100)  
    def paired_label_visual_shiftcorrectshow(self, im_id, ir_image, vis_image, bboxes_ir, bboxes_vis):
        def box2corners(pred_bboxes):
            x, y, w, h, angle = pred_bboxes
            cos_a_half = math.cos(angle) * 0.5
            sin_a_half = math.sin(angle) * 0.5
            w_x = cos_a_half * w
            w_y = sin_a_half * w
            h_x = -sin_a_half * h
            h_y = cos_a_half * h
            return np.array([x + w_x + h_x, y + w_y + h_y, x - w_x + h_x, y - w_y + h_y,x - w_x - h_x, y - w_y - h_y, x + w_x - h_x, y + w_y - h_y])
        def draw_bbox_paired_ir(image, bboxes_ir,i):
            """
            Draw bbox on image
            """
            bboxes_ir=copy.deepcopy(bboxes_ir)
            draw = ImageDraw.Draw(image)
            count= 0
            for j,_ in enumerate(bboxes_ir['bbox'][i]):
                bbox= bboxes_ir['bbox'][i][j]
                x1, y1, x2, y2, x3, y3, x4, y4 = box2corners(bbox)#*840/640
                draw.line(
                    [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                    width=1,
                    fill=(255,0,0))
            return image
        def draw_bbox_paired_vis(image, bboxes_vis,i):
            """
            Draw bbox on image
            """
            bboxes_vis=copy.deepcopy(bboxes_vis)
            draw = ImageDraw.Draw(image)
            count= 0
            for j,_ in enumerate(bboxes_vis[i]):
                bbox= bboxes_vis[i][j]
                x1, y1, x2, y2, x3, y3, x4, y4 = box2corners(bbox)#*840/640
                draw.line(
                    [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                    width=1,
                    fill=(0,255,0))
            return image            
        def tensor_to_pil_images(tensor):
            tensor=copy.deepcopy(tensor)
            # 反标准化参数（ImageNet）
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            
            # 将 Tensor 数据移动到 CPU 并转换为 numpy 数组
            tensor_np = tensor.numpy()  # 假设已处理过梯度或位于 CPU
            
            images = []
            for batch in range(tensor_np.shape[0]):
                # 提取单张图像数据 [3, H, W]
                img_data = tensor_np[batch]
                
                # 反标准化
                img_data = img_data * std + mean
                img_data = np.clip(img_data, 0, 1)  # 限制数值范围
                
                # 转换为 HWC 格式并调整到 [0, 255]
                img_data = np.transpose(img_data, (1, 2, 0))
                img_data = (img_data * 255).astype(np.uint8)
                
                # 生成 PIL 图像
                images.append(Image.fromarray(img_data))
            return images
        folder = f'paired_label_visual_shiftcorrectshow2/{self.epoch}'
        # 创建文件夹（如果不存在，且支持递归创建多级目录）
        os.makedirs(folder, exist_ok=True)
        ir_images=tensor_to_pil_images(ir_image)
        vis_images=tensor_to_pil_images(vis_image)
        im_id=copy.deepcopy(im_id)
        imid2savepath={}
        im_id=im_id.numpy()+1
        for i,id in enumerate(im_id):
            if os.path.exists(f'{self.cfg.TrainDataset.dataset_dir}/train_1000/trainimg/{(id[0]):05d}.jpg'):
                id_str_ir = '{:05d}'.format(id[0])+'_ir.jpg'
                id_str_vis = '{:05d}'.format(id[0])+'_vis.jpg'
                ir_image_savepath =f'{folder}/{id_str_ir}'
                vis_image_savepath =f'{folder}/{id_str_vis}'
                vis_image=vis_images[i]
                ir_image=ir_images[i]
                vis_image=draw_bbox_paired_vis(vis_image,bboxes_vis,i)
                ir_image=draw_bbox_paired_ir(ir_image,bboxes_ir,i) 
                #logger.info("Detection bbox results save in {}".format(ir_image_savepath+' '+vis_image_savepath))
                vis_image.save(vis_image_savepath, quality=100)
                ir_image.save(ir_image_savepath, quality=100)

    def paired_label_processing(self, data, teacher_bbox, teacher_class, teacher_score,iou_threshold=0.7):
        gt_rbox_ir=copy.deepcopy(data['gt_rbox_ir'])
        gt_class=copy.deepcopy(data['gt_class'])
        im_id=copy.deepcopy(data['im_id'])
        H=data['vis_image'].shape[2]
        W=data['vis_image'].shape[3]
        flipped=copy.deepcopy(data['flipped'])
        angle1=copy.deepcopy(data['angle1'])
        angle2=copy.deepcopy(data['angle2'])
        matrix1=copy.deepcopy(data['matrix1'])
        matrix2=copy.deepcopy(data['matrix2'])
        ratio1=copy.deepcopy(data['ratio1'])
        ratio2=copy.deepcopy(data['ratio2'])          
        def transform_bbox_rotate_filp(bboxes, img_w, img_h,angle1,angle2,ratio1,ratio2,flipped_mm,ratio3=1.0):
            cx, cy = img_w / 2.0, img_h / 2.0
            theta1 = -math.pi * angle1 /180   # in radians
            theta2 = -math.pi * angle2 /180
            x = bboxes[:, 0]
            y = bboxes[:, 1]
            w = bboxes[:, 2]
            h = bboxes[:, 3]
            a = bboxes[:, 4]
            # Step 1: rotate around image center
            x=cx-(cx-x)/ratio2
            y=cy-(cy-y)/ratio2
            w=w/ratio2
            h=h/ratio2
            a=a
            x_offset = x - cx
            y_offset = y - cy
            x_rot = x_offset * math.cos(theta2) - y_offset * math.sin(theta2)
            y_rot = x_offset * math.sin(theta2) + y_offset * math.cos(theta2)
            x1 = x_rot + cx
            y1 = y_rot + cy
            a1 = a + theta2

            y1 = y1
            x1=cx-(cx-x1)/ratio1
            y1=cy-(cy-y1)/ratio1
            w=w/ratio1
            h=h/ratio1
            a1=a1
            x_offset = x1 - cx
            y_offset = y1 - cy
            x_rot = x_offset * math.cos(theta1) - y_offset * math.sin(theta1)
            y_rot = x_offset * math.sin(theta1) + y_offset * math.cos(theta1)
            x1 = x_rot + cx
            y1 = y_rot + cy
            a1 = a1 + theta1
            if flipped_mm==1:
                x2 = img_w - x1
                a2 = -a1
            elif flipped_mm==0:
                x2 = x1
                a2 = a1
#######################640*544 --> 840*712#####################
            x2= x2/ratio3
            y1= y1/ratio3
            w = w/ratio3
            h = h/ratio3
            return paddle.stack([x2, y1, w, h, a2], axis=1)
        def transform_bbox_filp_rotate(bboxes, img_w, img_h,angle1,angle2,ratio1,ratio2,flipped_mm):
            cx, cy = img_w / 2.0, img_h / 2.0
            theta1 = math.pi * angle1 /180
            theta2 = math.pi * angle2 /180
            # Step 1: horizontal flip
            if flipped_mm == 1:
                x_flip = img_w - bboxes[:, 0]
                y_flip = bboxes[:, 1]
                w = bboxes[:, 2]
                h = bboxes[:, 3]
                a_flip = -bboxes[:, 4]
            if flipped_mm == 0:
                x_flip = bboxes[:, 0]
                y_flip = bboxes[:, 1]
                w = bboxes[:, 2]
                h = bboxes[:, 3]
                a_flip = bboxes[:, 4]
            # Step 2: rotate around image center (顺时针)
            x_offset = x_flip - cx
            y_offset = y_flip - cy
            x_rot = x_offset * math.cos(theta1) - y_offset * math.sin(theta1)
            y_rot = x_offset * math.sin(theta1) + y_offset * math.cos(theta1)
            x_new = x_rot + cx
            y_new = y_rot + cy
            a_new = a_flip + theta1
            x_new=cx-ratio1*(cx-x_new)
            y_new=cy-ratio1*(cy-y_new)
            w=w*ratio1
            h=h*ratio1
            x_offset = x_new - cx
            y_offset = y_new - cy
            x_rot = x_offset * math.cos(theta2) - y_offset * math.sin(theta2)
            y_rot = x_offset * math.sin(theta2) + y_offset * math.cos(theta2)
            x_new = x_rot + cx
            y_new = y_rot + cy
            a_new = a_new + theta2
            x_new=cx-ratio2*(cx-x_new)
            y_new=cy-ratio2*(cy-y_new)
            w=w*ratio2
            h=h*ratio2
            return paddle.stack([x_new, y_new, w, h, a_new], axis=1)

        def points_in_rotated_box(points, box):
            """
            判断哪些点在旋转矩形内

            参数:
                points: ndarray of shape (N, 2)，每行是一个 (x, y) 坐标
                box: tuple (cx, cy, w, h, theta)，矩形中心、宽、高、角度（弧度）

            返回:
                mask: ndarray of shape (N,)，bool 数组，表示哪些点在区域内
            """
            cx, cy, w, h, theta_rad = box

            # 平移坐标，使矩形中心变为原点
            translated = points - np.array([cx, cy])

            # 构建逆旋转矩阵，将点逆时针旋转 -theta 使矩形变为轴对齐
            rotation_matrix = np.array([
                [np.cos(-theta_rad), -np.sin(-theta_rad)],
                [np.sin(-theta_rad),  np.cos(-theta_rad)]
            ])

            # 对点进行旋转
            rotated = translated @ rotation_matrix.T

            # 判断是否在轴对齐的矩形内
            half_w, half_h = w / 2, h / 2
            inside_x = np.logical_and(rotated[:, 0] >= -half_w, rotated[:, 0] <= half_w)
            inside_y = np.logical_and(rotated[:, 1] >= -half_h, rotated[:, 1] <= half_h)
            mask = np.logical_and(inside_x, inside_y)

            return mask
        def compute_iou(rect1_np, rect2_np):
            # 使用OpenCV的函数计算旋转矩形的IoU
            intersection = cv2.rotatedRectangleIntersection(
                ((rect1_np[0], rect1_np[1]), (rect1_np[2], rect1_np[3]), rect1_np[4] * 180 / np.pi),
                ((rect2_np[0], rect2_np[1]), (rect2_np[2], rect2_np[3]), rect2_np[4] * 180 / np.pi)
            )
            # 如果相交区域存在
            if intersection[0] != 0:
                intersect_area = cv2.contourArea(intersection[1])  # 计算交集的面积
            else:
                return 0  
            # 计算两个框的面积
            area1 = rect1_np[2]*rect1_np[3]
            area2 = rect2_np[2]*rect2_np[3]
            # 计算IoU
            union_area = area1 + area2 - intersect_area
            iou = intersect_area / union_area if union_area != 0 else 0
            return iou
        def box2corners(pred_bbox):
            x, y, w, h, angle = pred_bbox
            cos_a_half = math.cos(angle) * 0.5
            sin_a_half = math.sin(angle) * 0.5
            w_x = cos_a_half * w
            w_y = sin_a_half * w
            h_x = -sin_a_half * h
            h_y = cos_a_half * h
            return np.array([
                x + w_x + h_x, y + w_y + h_y,
                x - w_x + h_x, y - w_y + h_y,
                x - w_x - h_x, y - w_y - h_y,
                x + w_x - h_x, y + w_y - h_y
            ])
        def draw_dashed_line(draw, p1, p2, dash_length=5, gap_length=5, **kwargs):
            """
            在 draw 上绘制从 p1 到 p2 的虚线段。
            """
            x1, y1 = p1
            x2, y2 = p2
            total_len = np.hypot(x2 - x1, y2 - y1)
            dash_gap = dash_length + gap_length
            num_dashes = int(total_len // dash_gap)

            for i in range(num_dashes + 1):
                start_frac = i * dash_gap / total_len
                end_frac = min((i * dash_gap + dash_length) / total_len, 1.0)

                sx = x1 + (x2 - x1) * start_frac
                sy = y1 + (y2 - y1) * start_frac
                ex = x1 + (x2 - x1) * end_frac
                ey = y1 + (y2 - y1) * end_frac

                draw.line([(sx, sy), (ex, ey)], **kwargs)
        def draw_bbox_vis(image, bboxes_vis,bboxes_ir):
            """
            只对单张图像的 bboxes_vis 和 bboxes_ir 做可视化
            bboxes_vis 是 list，bboxes_ir 是 dict 包含 'bbox'
            """
            bboxes_vis = copy.deepcopy(bboxes_vis)
            draw = ImageDraw.Draw(image)
            for bbox in bboxes_vis:
                x1, y1, x2, y2, x3, y3, x4, y4 = box2corners(bbox)
                draw.line([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                        width=1, fill=(0, 255, 0))
            for bbox in bboxes_ir:
                x1, y1, x2, y2, x3, y3, x4, y4 = box2corners(bbox)
                draw.line([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                        width=1, fill=(255, 0, 0))
            return image
        def draw_bbox_ir(image, bboxes_ir):
            """
            只对单张图像的 bboxes_vis 和 bboxes_ir 做可视化
            bboxes_vis 是 list，bboxes_ir 是 dict 包含 'bbox'
            """
            bboxes_ir = copy.deepcopy(bboxes_ir)
            draw = ImageDraw.Draw(image)
            for bbox in bboxes_ir:
                x1, y1, x2, y2, x3, y3, x4, y4 = box2corners(bbox)
                draw.line([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                        width=1, fill=(0, 255, 0))
            return image
        def draw_bbox_scale(image, bboxes, bbox_ir, bbox_scale):
            """
            只对单张图像的 bboxes_vis 和 bboxes_ir 做可视化
            bboxes_vis 是 list，bboxes_ir 是 dict 包含 'bbox'
            """
            bboxes = copy.deepcopy(bboxes)
            draw = ImageDraw.Draw(image)
            for bbox in bboxes:
                x1, y1, x2, y2, x3, y3, x4, y4 = box2corners(bbox)
                draw.line([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                        width=1, fill=(0, 255, 0))
            x1, y1, x2, y2, x3, y3, x4, y4 = box2corners(bbox_ir)
            draw.line([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                    width=1, fill=(255, 0, 0))
            x1, y1, x2, y2, x3, y3, x4, y4 = box2corners(bbox_scale)
            points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]
            for i in range(len(points) - 1):
                draw_dashed_line(draw, points[i], points[i + 1], dash_length=5, gap_length=5,
                                fill=(255, 255, 0), width=1)
            return image
        def tensor_to_pil_image(tensor):
            """
            只对单张tensor做反标准化和转换成PIL图像
            输入：tensor shape [3, H, W]
            输出：PIL.Image
            """
            tensor = copy.deepcopy(tensor)
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            
            tensor_np = tensor.numpy()
            img_data = tensor_np
            img_data = img_data * std + mean
            img_data = np.clip(img_data, 0, 1)
            img_data = np.transpose(img_data, (1, 2, 0))
            img_data = (img_data * 255).astype(np.uint8)
            return Image.fromarray(img_data)
        new_gt_rbox_ir_unmatch = []
        new_gt_rbox_ir_match = []
        new_gt_class_ir = []
        new_teacher_bbox_unmatch = []
        new_teacher_bbox_match = []

        for mm in range(len(gt_rbox_ir)):
            coordinates_ir = gt_rbox_ir[mm].numpy()
            coordinates_vis = teacher_bbox[mm].numpy()
            categories_ir = gt_class[mm].numpy()
            categories_vis = teacher_class[mm].numpy()
            im_id_mm = im_id[mm].numpy()[0]
            flipped_mm = flipped[mm].numpy()[0]
            angle1_mm = angle1[mm].numpy()[0]
            angle2_mm = angle2[mm].numpy()[0]
            matrix1_mm = matrix1[mm].numpy()[0]
            matrix2_mm = matrix2[mm].numpy()[0]
            ratio1_mm = ratio1[mm].numpy()[0]
            ratio2_mm = ratio2[mm].numpy()[0]
            coordinates_vis_match = []
            coordinates_ir_match = []
            coordinates_vis_unmatch = []
            coordinates_ir_unmatch = []
            categories_ir_new=[]
            categories_vis_match=[]
            # 为vis_label 匹配 ir_label 找到都有的目标标签,
            record_pop_idx_vis = []
            record_pop_idx_ir = []
            index = np.where(self.vis_gt_data['im_id'].clone().numpy() == im_id_mm)[0][0]
            self.vis_gt_data['gt_rbox_vis'][index] = transform_bbox_filp_rotate(self.vis_gt_data['gt_rbox_vis'][index],W,H,angle1_mm,angle2_mm,ratio1_mm,ratio2_mm,flipped_mm)
            for i in range(len(coordinates_ir)):
                center_points_vis=coordinates_vis[:,0:2]
                ir_search_scale=(coordinates_ir[i][0],coordinates_ir[i][1],coordinates_ir[i][2]*1.60,coordinates_ir[i][3]*1.60,coordinates_ir[i][4])
                mask = points_in_rotated_box(center_points_vis,ir_search_scale)
                index_in_search=np.where(mask)[0]
                max_iou=0.0
                best_k=-1
                # coordinates_vis_filtered=coordinates_vis[mask]
                # bboxes=copy.deepcopy(coordinates_vis_filtered)
                # image=tensor_to_pil_image(data['vis_image'][mm])
                # image=draw_bbox_scale(image,bboxes,coordinates_ir[i],np.array(ir_search_scale))
                # image.save(f'/data1/jinhang/DA-DPDETR-Prominent_Position_Shift/DA-DPDETR/dataset/rbox_Drone_Vehicle/train/label_trainimg/{(im_id_mm+1):05d}_epoch{self.epoch}_{i}.jpg', quality=100)   
                if np.any(mask):
                    for k in index_in_search:
                        if k in record_pop_idx_vis:
                            continue
                        coordinates_vis_k_transform2sameorigin=np.array([coordinates_ir[i][0],coordinates_ir[i][1],coordinates_vis[k][2],coordinates_vis[k][3],coordinates_vis[k][4]])
                        iou = compute_iou(coordinates_ir[i], coordinates_vis_k_transform2sameorigin)
                        if iou > max_iou:
                            max_iou = iou
                            if max_iou >= iou_threshold:
                                best_k = k
                    if best_k!=-1:  # 匹配上了
                        coordinates_vis_whcopy=copy.deepcopy(coordinates_vis[best_k])
                        coordinates_vis_whcopy[2]=coordinates_ir[i][2]
                        coordinates_vis_whcopy[3]=coordinates_ir[i][3]
                        coordinates_vis_whcopy[4]=coordinates_ir[i][4]
                        if self.epoch <= self.semi_start_epochs+4:
                            self.vis_gt_data['gt_rbox_vis'][index][i,:]=paddle.to_tensor(coordinates_vis_whcopy)
                            self.vis_gt_data['gt_class'][index][i,:]=paddle.to_tensor(categories_ir[i])
                        elif self.epoch>self.semi_start_epochs+4:
                            # deta_x=coordinates_vis_whcopy[0]-coordinates_ir[i][0]
                            # deta_y=coordinates_vis_whcopy[1]-coordinates_ir[i][1]
                            deta_x=coordinates_vis_whcopy[0]-self.vis_gt_data['gt_rbox_vis'][index][i][0]
                            deta_y=coordinates_vis_whcopy[1]-self.vis_gt_data['gt_rbox_vis'][index][i][1]
                            if self.epoch<self.stage3_start_epochs:
                                lam=(min(self.epoch,(self.stage2_start_epochs-3))-self.semi_start_epochs)/(self.stage2_start_epochs-self.semi_start_epochs-3)
                            elif self.epoch>=self.stage3_start_epochs:
                                lam=(min(self.epoch,(self.stage3_start_epochs+4))-self.stage3_start_epochs+1)/5
                            self.vis_gt_data['gt_rbox_vis'][index][i,:]=paddle.to_tensor(np.array([self.vis_gt_data['gt_rbox_vis'][index][i][0]+lam*deta_x,self.vis_gt_data['gt_rbox_vis'][index][i][1]+lam*deta_y,coordinates_ir[i][2],coordinates_ir[i][3],coordinates_ir[i][4]], dtype='float32'))
                        coordinates_ir_match.append(coordinates_ir[i])
                        coordinates_vis_match.append(self.vis_gt_data['gt_rbox_vis'][index][i,:])
                        categories_ir_new.append(categories_ir[i])
                        categories_vis_match.append(self.vis_gt_data['gt_class'][index][i,:])
                        record_pop_idx_ir.append(i)  # 记录匹配上的idx
                        record_pop_idx_vis.append(best_k)
                    else:
                        if self.epoch<self.semi_start_epochs+4:
                            self.vis_gt_data['gt_rbox_vis'][index][i,:]=paddle.to_tensor(np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), dtype='float32')
                            self.vis_gt_data['gt_class'][index][i,:]=paddle.to_tensor([-1], dtype='int32')

                        elif self.epoch==self.semi_start_epochs+4:
                            self.vis_gt_data['gt_rbox_vis'][index][i,:]=paddle.to_tensor(coordinates_ir[i])
                            self.vis_gt_data['gt_class'][index][i,:]=paddle.to_tensor(categories_ir[i])
                            coordinates_ir_match.append(coordinates_ir[i])
                            coordinates_vis_match.append(self.vis_gt_data['gt_rbox_vis'][index][i])
                            categories_ir_new.append(categories_ir[i])
                            categories_vis_match.append(self.vis_gt_data['gt_class'][index][i])  
                        elif self.epoch>self.semi_start_epochs+4:
                            coordinates_ir_match.append(coordinates_ir[i])
                            coordinates_vis_match.append(self.vis_gt_data['gt_rbox_vis'][index][i])
                            categories_ir_new.append(categories_ir[i])
                            categories_vis_match.append(self.vis_gt_data['gt_class'][index][i])                                
                else:
                    if self.epoch<self.semi_start_epochs+4:
                        self.vis_gt_data['gt_rbox_vis'][index][i,:]=paddle.to_tensor(np.array([np.nan, np.nan, np.nan, np.nan, np.nan]), dtype='float32')
                        self.vis_gt_data['gt_class'][index][i,:]=paddle.to_tensor([-1], dtype='int32')

                    elif self.epoch==self.semi_start_epochs+4:
                        self.vis_gt_data['gt_rbox_vis'][index][i,:]=paddle.to_tensor(coordinates_ir[i])
                        self.vis_gt_data['gt_class'][index][i,:]=paddle.to_tensor(categories_ir[i])
                        coordinates_ir_match.append(coordinates_ir[i])
                        coordinates_vis_match.append(self.vis_gt_data['gt_rbox_vis'][index][i])
                        categories_ir_new.append(categories_ir[i])
                        categories_vis_match.append(self.vis_gt_data['gt_class'][index][i])
                    elif self.epoch>self.semi_start_epochs+4:
                        coordinates_ir_match.append(coordinates_ir[i])
                        coordinates_vis_match.append(self.vis_gt_data['gt_rbox_vis'][index][i])
                        categories_ir_new.append(categories_ir[i])
                        categories_vis_match.append(self.vis_gt_data['gt_class'][index][i])              
            # bboxes=copy.deepcopy(self.vis_gt_data['gt_rbox_vis'][index]) 
            # image=tensor_to_pil_image(data['vis_image'][mm])
            # image=draw_bbox_vis(image,bboxes)
            # image.save(f'/data1/jinhang/DA-DPDETR-Prominent_Position_Shift/DA-DPDETR/dataset/rbox_Drone_Vehicle/train/label_trainimg/{(im_id_mm+1):05d}_before.jpg', quality=99)   
            origin_vis_rboxes_mm = transform_bbox_rotate_filp(copy.deepcopy(self.vis_gt_data['gt_rbox_vis'][index]),W,H,angle1_mm,angle2_mm,ratio1_mm,ratio2_mm,flipped_mm,ratio3=640/840)          
            origin_ir_rboxes_mm = transform_bbox_rotate_filp(paddle.to_tensor(copy.deepcopy(coordinates_ir),dtype='float32'),W,H,angle1_mm,angle2_mm,ratio1_mm,ratio2_mm,flipped_mm,ratio3=640/840)        
            folder = f'/data1/jinhang/Experiment/shiftcorrect_vision2/{self.epoch}/train'
            # 创建文件夹（如果不存在，且支持递归创建多级目录）
            os.makedirs(folder, exist_ok=True)
            if os.path.exists(f'{self.cfg.TrainDataset.dataset_dir}/train_1000/trainimg/{(im_id_mm+1):05d}.jpg'):
                image_vis=Image.open(f'{self.cfg.TrainDataset.dataset_dir}/train_1000/trainimg/{(im_id_mm+1):05d}.jpg').convert('RGB')
                bboxes_vis=copy.deepcopy(origin_vis_rboxes_mm)
                bboxes_ir=copy.deepcopy(origin_ir_rboxes_mm)
                image=draw_bbox_vis(image_vis,bboxes_vis,bboxes_ir)
                image.save(f'{folder}/{(im_id_mm+1):05d}_vis_shiftcorrect.jpg', quality=100)
            if os.path.exists(f'{self.cfg.TrainDataset.dataset_dir}/train_1000/trainimgr/{(im_id_mm+1):05d}.jpg'):
                image_ir=Image.open(f'{self.cfg.TrainDataset.dataset_dir}/train_1000/trainimgr/{(im_id_mm+1):05d}.jpg').convert('RGB')
                bboxes=copy.deepcopy(origin_ir_rboxes_mm)
                image=draw_bbox_ir(image_ir,bboxes)
                image.save(f'{folder}/{(im_id_mm+1):05d}_ir_gt.jpg', quality=100)
            self.vis_gt_data['gt_rbox_vis'][index] = transform_bbox_rotate_filp(self.vis_gt_data['gt_rbox_vis'][index],W,H,angle1_mm,angle2_mm,ratio1_mm,ratio2_mm,flipped_mm)
            # bboxes=copy.deepcopy(self.vis_gt_data['gt_rbox_vis'][index]) 
            # image=tensor_to_pil_image(data['vis_image'][mm])
            # image=draw_bbox_vis(image,bboxes)
            # image.save(f'/data1/jinhang/DA-DPDETR-Prominent_Position_Shift/DA-DPDETR/dataset/rbox_Drone_Vehicle/train/label_trainimg/{(im_id_mm+1):05d}_latter.jpg', quality=99)      
            if self.epoch<self.semi_start_epochs+4:
                coordinates_vis = [coordinates_vis[i] for i in range(len(coordinates_vis)) if i not in record_pop_idx_vis]
                categories_vis = [categories_vis[i] for i in range(len(categories_vis)) if i not in record_pop_idx_vis]
                coordinates_ir = [coordinates_ir[i] for i in range(len(coordinates_ir)) if i not in record_pop_idx_ir]
                categories_ir = [categories_ir[i] for i in range(len(categories_ir)) if i not in record_pop_idx_ir]

                # 接下来把剩余的label复制到两个模态上
                # 先复制可见光的
                for box, cls in zip(coordinates_ir, categories_ir):
                    coordinates_vis_unmatch.append(box)
                    #categories_vis_match.append(cls)
                    coordinates_ir_unmatch.append(box)
                    categories_ir_new.append(cls)

            new_gt_rbox_ir_match.append(paddle.to_tensor(coordinates_ir_match))
            new_teacher_bbox_match.append(paddle.to_tensor(coordinates_vis_match))
            new_gt_class_ir.append(paddle.to_tensor(categories_ir_new))

            if self.epoch<self.semi_start_epochs+4:
                new_gt_rbox_ir_unmatch.append(paddle.to_tensor([]))
            else:
                new_gt_rbox_ir_unmatch.append(paddle.to_tensor(coordinates_ir_unmatch))            
            
            if self.epoch<self.semi_start_epochs+4:
                new_teacher_bbox_unmatch.append(paddle.to_tensor([]))
            else:
                new_teacher_bbox_unmatch.append(paddle.to_tensor(coordinates_vis_unmatch))
            
        return new_gt_rbox_ir_match, new_gt_rbox_ir_unmatch, new_gt_class_ir, new_teacher_bbox_match, new_teacher_bbox_unmatch
        #match红外框、unmatch红外框、[match+unmatch]的红外类别、match可见光框、unmatch可见光框、match的可见光类别
    def _eval_with_loader(self, loader, eval_flag=0, my_eval_flag = 'eval_stage3'):
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'

        test_cfg = self.cfg['test_cfg']
        if test_cfg['inference_on'] == 'teacher' and eval_flag == 1:
            logger.info("***** teacher model evaluating *****")
            eval_model = self.ema.model
        else:
            logger.info("***** student model evaluating *****")
            eval_model = self.model

        eval_model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg.worker_num, self._eval_batch_sampler)
            self._flops(flops_loader)
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            data['flag'] = my_eval_flag
            # forward
            if self.use_amp:
                with paddle.amp.auto_cast(
                        enable=self.cfg.use_gpu or self.cfg.use_mlu,
                        custom_white_list=self.custom_white_list,
                        custom_black_list=self.custom_black_list,
                        level=self.amp_level):
                    outs = eval_model(data)
            else:
                outs = eval_model(data)

            # update metrics
            for metric in self._metrics:
                metric.update(data, outs)

            # multi-scale inputs: all inputs have same im_id
            if isinstance(data, typing.Sequence):
                sample_num += data[0]['im_id'].numpy().shape[0]
            else:
                sample_num += data['im_id'].numpy().shape[0]
            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        self._compose_callback.on_epoch_end(self.status)
        self._reset_metrics()

    def evaluate(self):
        # get distributed model
        if self.cfg.get('fleet', False):
            self.model = fleet.distributed_model(self.model)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
        elif self._nranks > 1:
            find_unused_parameters = self.cfg[
                'find_unused_parameters'] if 'find_unused_parameters' in self.cfg else False
            self.model = paddle.DataParallel(
                self.model, find_unused_parameters=find_unused_parameters)
        with paddle.no_grad():
            self._eval_with_loader(self.loader, 0)

    def _eval_with_loader_slice(self,
                                loader,
                                slice_size=[640, 640],
                                overlap_ratio=[0.25, 0.25],
                                combine_method='nms',
                                match_threshold=0.6,
                                match_metric='iou'):
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg.worker_num, self._eval_batch_sampler)
            self._flops(flops_loader)

        merged_bboxs = []
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            # forward
            if self.use_amp:
                with paddle.amp.auto_cast(
                        enable=self.cfg.use_gpu or self.cfg.use_npu or
                        self.cfg.use_mlu,
                        custom_white_list=self.custom_white_list,
                        custom_black_list=self.custom_black_list,
                        level=self.amp_level):
                    outs = self.model(data)
            else:
                outs = self.model(data)

            shift_amount = data['st_pix']
            outs['bbox'][:, 2:4] = outs['bbox'][:, 2:4] + shift_amount
            outs['bbox'][:, 4:6] = outs['bbox'][:, 4:6] + shift_amount
            merged_bboxs.append(outs['bbox'])

            if data['is_last'] > 0:
                # merge matching predictions
                merged_results = {'bbox': []}
                if combine_method == 'nms':
                    final_boxes = multiclass_nms(
                        np.concatenate(merged_bboxs), self.cfg.num_classes,
                        match_threshold, match_metric)
                    merged_results['bbox'] = np.concatenate(final_boxes)
                elif combine_method == 'concat':
                    merged_results['bbox'] = np.concatenate(merged_bboxs)
                else:
                    raise ValueError(
                        "Now only support 'nms' or 'concat' to fuse detection results."
                    )
                merged_results['im_id'] = np.array([[0]])
                merged_results['bbox_num'] = np.array(
                    [len(merged_results['bbox'])])

                merged_bboxs = []
                data['im_id'] = data['ori_im_id']
                # update metrics
                for metric in self._metrics:
                    metric.update(data, merged_results)

                # multi-scale inputs: all inputs have same im_id
                if isinstance(data, typing.Sequence):
                    sample_num += data[0]['im_id'].numpy().shape[0]
                else:
                    sample_num += data['im_id'].numpy().shape[0]

            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        self._compose_callback.on_epoch_end(self.status)
        # reset metric states for metric may performed multiple times
        self._reset_metrics()

    def evaluate_slice(self,
                       slice_size=[640, 640],
                       overlap_ratio=[0.25, 0.25],
                       combine_method='nms',
                       match_threshold=0.6,
                       match_metric='iou'):
        with paddle.no_grad():
            self._eval_with_loader_slice(self.loader, slice_size, overlap_ratio,
                                         combine_method, match_threshold,
                                         match_metric)

    def slice_predict(self,
                      images,
                      slice_size=[640, 640],
                      overlap_ratio=[0.25, 0.25],
                      combine_method='nms',
                      match_threshold=0.6,
                      match_metric='iou',
                      draw_threshold=0.5,
                      output_dir='output',
                      save_results=False,
                      visualize=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_slice_images(images, slice_size, overlap_ratio)
        loader = create('TestReader')(self.dataset, 0)
        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            # mem
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg[
                'save_prediction_only'] if 'save_prediction_only' in self.cfg else None
            output_eval = self.cfg[
                'output_eval'] if 'output_eval' in self.cfg else None

            # modify
            self.mode = '_test'
            self.cfg['save_prediction_only'] = True
            self.cfg['output_eval'] = output_dir
            self.cfg['imid2path'] = imid2path
            self._init_metrics()

            # restore
            self.mode = mode
            self.cfg.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg['save_prediction_only'] = save_prediction_only

            self.cfg.pop('output_eval')
            if output_eval is not None:
                self.cfg['output_eval'] = output_eval

            self.cfg.pop('imid2path')

            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file)

        # Run Infer 
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)

        results = []  # all images
        merged_bboxs = []  # single image
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)

            outs['bbox'] = outs['bbox'].numpy()  # only in test mode
            shift_amount = data['st_pix']
            outs['bbox'][:, 2:4] = outs['bbox'][:, 2:4] + shift_amount.numpy()
            outs['bbox'][:, 4:6] = outs['bbox'][:, 4:6] + shift_amount.numpy()
            merged_bboxs.append(outs['bbox'])

            if data['is_last'] > 0:
                # merge matching predictions
                merged_results = {'bbox': []}
                if combine_method == 'nms':
                    final_boxes = multiclass_nms(
                        np.concatenate(merged_bboxs), self.cfg.num_classes,
                        match_threshold, match_metric)
                    merged_results['bbox'] = np.concatenate(final_boxes)
                elif combine_method == 'concat':
                    merged_results['bbox'] = np.concatenate(merged_bboxs)
                else:
                    raise ValueError(
                        "Now only support 'nms' or 'concat' to fuse detection results."
                    )
                merged_results['im_id'] = np.array([[0]])
                merged_results['bbox_num'] = np.array(
                    [len(merged_results['bbox'])])

                merged_bboxs = []
                data['im_id'] = data['ori_im_id']

                for _m in metrics:
                    _m.update(data, merged_results)

                for key in ['im_shape', 'scale_factor', 'im_id']:
                    if isinstance(data, typing.Sequence):
                        merged_results[key] = data[0][key]
                    else:
                        merged_results[key] = data[key]
                for key, value in merged_results.items():
                    if hasattr(value, 'numpy'):
                        merged_results[key] = value.numpy()
                results.append(merged_results)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        if visualize:
            for outs in results:
                batch_res = get_infer_results(outs, clsid2catid)
                bbox_num = outs['bbox_num']

                start = 0
                for i, im_id in enumerate(outs['im_id']):
                    image_path = imid2path[int(im_id)]
                    image = Image.open(image_path).convert('RGB')
                    image = ImageOps.exif_transpose(image)
                    self.status['original_image'] = np.array(image.copy())

                    end = start + bbox_num[i]
                    bbox_res = batch_res['bbox'][start:end] \
                            if 'bbox' in batch_res else None
                    mask_res = batch_res['mask'][start:end] \
                            if 'mask' in batch_res else None
                    segm_res = batch_res['segm'][start:end] \
                            if 'segm' in batch_res else None
                    keypoint_res = batch_res['keypoint'][start:end] \
                            if 'keypoint' in batch_res else None
                    pose3d_res = batch_res['pose3d'][start:end] \
                            if 'pose3d' in batch_res else None
                    image = visualize_results(
                        image, bbox_res, mask_res, segm_res, keypoint_res,
                        pose3d_res, int(im_id), catid2name, draw_threshold)
                    self.status['result_image'] = np.array(image.copy())
                    if self._compose_callback:
                        self._compose_callback.on_step_end(self.status)
                    # save image with detection
                    save_name = self._get_save_image_name(output_dir,
                                                          image_path)
                    logger.info("Detection bbox results save in {}".format(
                        save_name))
                    image.save(save_name, quality=95)

                    start = end

    def predict(self,
                images,
                draw_threshold=0.5,
                output_dir='output',
                save_results=False,
                visualize=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            # mem
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg[
                'save_prediction_only'] if 'save_prediction_only' in self.cfg else None
            output_eval = self.cfg[
                'output_eval'] if 'output_eval' in self.cfg else None

            # modify
            self.mode = '_test'
            self.cfg['save_prediction_only'] = True
            self.cfg['output_eval'] = output_dir
            self.cfg['imid2path'] = imid2path
            self._init_metrics()

            # restore
            self.mode = mode
            self.cfg.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg['save_prediction_only'] = save_prediction_only

            self.cfg.pop('output_eval')
            if output_eval is not None:
                self.cfg['output_eval'] = output_eval

            self.cfg.pop('imid2path')

            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file)

        # Run Infer 
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)
        results = []
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            if hasattr(self.model, 'modelTeacher'):
                outs = self.model.modelTeacher(data)
            else:
                outs = self.model(data)
            for _m in metrics:
                _m.update(data, outs)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)

        # sniper
        if type(self.dataset) == SniperCOCODataSet:
            results = self.dataset.anno_cropper.aggregate_chips_detections(
                results)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        if visualize:
            for outs in results:
                batch_res = get_infer_results(outs, clsid2catid)
                bbox_num = outs['bbox_num']

                start = 0
                for i, im_id in enumerate(outs['im_id']):
                    image_path = imid2path[int(im_id)]
                    image = Image.open(image_path).convert('RGB')
                    image = ImageOps.exif_transpose(image)
                    self.status['original_image'] = np.array(image.copy())

                    end = start + bbox_num[i]
                    bbox_res = batch_res['bbox'][start:end] \
                            if 'bbox' in batch_res else None
                    mask_res = batch_res['mask'][start:end] \
                            if 'mask' in batch_res else None
                    segm_res = batch_res['segm'][start:end] \
                            if 'segm' in batch_res else None
                    keypoint_res = batch_res['keypoint'][start:end] \
                            if 'keypoint' in batch_res else None
                    pose3d_res = batch_res['pose3d'][start:end] \
                            if 'pose3d' in batch_res else None
                    image = visualize_results(
                        image, bbox_res, mask_res, segm_res, keypoint_res,
                        pose3d_res, int(im_id), catid2name, draw_threshold)
                    self.status['result_image'] = np.array(image.copy())
                    if self._compose_callback:
                        self._compose_callback.on_step_end(self.status)
                    # save image with detection
                    save_name = self._get_save_image_name(output_dir,
                                                          image_path)
                    logger.info("Detection bbox results save in {}".format(
                        save_name))
                    image.save(save_name, quality=95)

                    start = end
        return results

    def multi_predict_paired(self,
                vis_images,
                ir_images,
                draw_threshold=0.5,
                output_dir='output',
                save_results=False,
                visualize=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_images(vis_images,ir_images)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            # mem
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg[
                'save_prediction_only'] if 'save_prediction_only' in self.cfg else None
            output_eval = self.cfg[
                'output_eval'] if 'output_eval' in self.cfg else None

            # modify
            self.mode = '_test'
            self.cfg['save_prediction_only'] = True
            self.cfg['output_eval'] = output_dir
            self.cfg['imid2path'] = imid2path
            self._init_metrics()

            # restore
            self.mode = mode
            self.cfg.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg['save_prediction_only'] = save_prediction_only

            self.cfg.pop('output_eval')
            if output_eval is not None:
                self.cfg['output_eval'] = output_eval

            self.cfg.pop('imid2path')

            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file)

        # Run Infer
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)
        results = []
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            data['flag'] = 'infer_paired'
            # forward
            if hasattr(self.model, 'modelTeacher'):
                outs = self.model.modelTeacher(data)
            else:
                outs = self.model(data)
            for _m in metrics:
                _m.update(data, outs)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)

        # sniper
        if type(self.dataset) == SniperCOCODataSet:
            results = self.dataset.anno_cropper.aggregate_chips_detections(
                results)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        if visualize:
            for outs in results:
                batch_res = get_infer_results(outs, clsid2catid)
                bbox_num = outs['bbox_num']

                start = 0
                for i, im_id in enumerate(outs['im_id']):
                    vis_image_path = imid2path[int(im_id)][0]
                    ir_image_path = imid2path[int(im_id)][1]
                    vis_image = Image.open(vis_image_path).convert('RGB')
                    ir_image = Image.open(ir_image_path).convert('RGB')
                    vis_image = ImageOps.exif_transpose(vis_image)
                    ir_image = ImageOps.exif_transpose(ir_image)
                    self.status['original_vis_image'] = np.array(vis_image.copy())
                    self.status['original_ir_image'] = np.array(ir_image.copy())
                    end = start + bbox_num[i]
                    bbox_res_vis = batch_res['bbox_vis'][start:end] \
                            if 'bbox_vis' in batch_res else None
                    bbox_res_ir = batch_res['bbox_ir'][start:end] \
                        if 'bbox_ir' in batch_res else None
                    mask_res = batch_res['mask'][start:end] \
                            if 'mask' in batch_res else None
                    segm_res = batch_res['segm'][start:end] \
                            if 'segm' in batch_res else None
                    keypoint_res = batch_res['keypoint'][start:end] \
                            if 'keypoint' in batch_res else None
                    pose3d_res = batch_res['pose3d'][start:end] \
                            if 'pose3d' in batch_res else None
                    vis_image = visualize_results_paired(
                        vis_image, bbox_res_vis, mask_res, segm_res, keypoint_res,
                        pose3d_res, int(im_id), catid2name, draw_threshold)

                    ir_image = visualize_results_paired(
                        ir_image, bbox_res_ir, mask_res, segm_res, keypoint_res,
                        pose3d_res, int(im_id), catid2name, draw_threshold)

                    self.status['result_vis_image'] = np.array(vis_image.copy())
                    self.status['result_ir_image'] = np.array(ir_image.copy())
                    if self._compose_callback:
                        self._compose_callback.on_step_end(self.status)
                    # save image with detection
                    save_vis_name = self._get_save_vis_image_name(output_dir,
                                                          vis_image_path)
                    save_ir_name = self._get_save_ir_image_name(output_dir,
                                                                ir_image_path)
                    logger.info("Detection bbox results save in {}".format(
                        save_vis_name+' '+save_ir_name))
                    vis_image.save(save_vis_name, quality=95)
                    ir_image.save(save_ir_name, quality=95)
                    start = end
        return results

    def multi_predict(self,
                vis_images,
                ir_images,
                draw_threshold=0.5,
                output_dir='output',
                save_results=False,
                visualize=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_images(vis_images,ir_images)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            # mem
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg[
                'save_prediction_only'] if 'save_prediction_only' in self.cfg else None
            output_eval = self.cfg[
                'output_eval'] if 'output_eval' in self.cfg else None

            # modify
            self.mode = '_test'
            self.cfg['save_prediction_only'] = True
            self.cfg['output_eval'] = output_dir
            self.cfg['imid2path'] = imid2path
            self._init_metrics()

            # restore
            self.mode = mode
            self.cfg.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg['save_prediction_only'] = save_prediction_only

            self.cfg.pop('output_eval')
            if output_eval is not None:
                self.cfg['output_eval'] = output_eval

            self.cfg.pop('imid2path')

            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file)

        # Run Infer
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)
        results = []
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            if hasattr(self.model, 'modelTeacher'):
                outs = self.model.modelTeacher(data)
            else:
                outs = self.model(data)
            for _m in metrics:
                _m.update(data, outs)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)

        # sniper
        if type(self.dataset) == SniperCOCODataSet:
            results = self.dataset.anno_cropper.aggregate_chips_detections(
                results)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        if visualize:
            for outs in results:
                batch_res = get_infer_results(outs, clsid2catid)
                bbox_num = outs['bbox_num']

                start = 0
                for i, im_id in enumerate(outs['im_id']):
                    vis_image_path = imid2path[int(im_id)][0]
                    ir_image_path = imid2path[int(im_id)][1]
                    vis_image = Image.open(vis_image_path).convert('RGB')
                    ir_image = Image.open(ir_image_path).convert('RGB')
                    vis_image = ImageOps.exif_transpose(vis_image)
                    ir_image = ImageOps.exif_transpose(ir_image)
                    self.status['original_vis_image'] = np.array(vis_image.copy())
                    self.status['original_ir_image'] = np.array(ir_image.copy())
                    end = start + bbox_num[i]
                    bbox_res = batch_res['bbox'][start:end] \
                            if 'bbox' in batch_res else None
                    mask_res = batch_res['mask'][start:end] \
                            if 'mask' in batch_res else None
                    segm_res = batch_res['segm'][start:end] \
                            if 'segm' in batch_res else None
                    keypoint_res = batch_res['keypoint'][start:end] \
                            if 'keypoint' in batch_res else None
                    pose3d_res = batch_res['pose3d'][start:end] \
                            if 'pose3d' in batch_res else None
                    vis_image = visualize_results(
                        vis_image, bbox_res, mask_res, segm_res, keypoint_res,
                        pose3d_res, int(im_id), catid2name, draw_threshold)

                    ir_image = visualize_results(
                        ir_image, bbox_res, mask_res, segm_res, keypoint_res,
                        pose3d_res, int(im_id), catid2name, draw_threshold)

                    self.status['result_vis_image'] = np.array(vis_image.copy())
                    self.status['result_ir_image'] = np.array(ir_image.copy())
                    if self._compose_callback:
                        self._compose_callback.on_step_end(self.status)
                    # save image with detection
                    save_vis_name = self._get_save_vis_image_name(output_dir,
                                                          vis_image_path)
                    save_ir_name = self._get_save_ir_image_name(output_dir,
                                                                ir_image_path)
                    logger.info("Detection bbox results save in {}".format(
                        save_vis_name+' '+save_ir_name))
                    vis_image.save(save_vis_name, quality=95)
                    ir_image.save(save_ir_name, quality=95)
                    start = end
        return results


    def _get_save_image_name(self, output_dir, image_path):
        """
        Get save image name from source image path.
        """
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        return os.path.join(output_dir, "{}".format(name)) + ext

    def _get_save_vis_image_name(self, output_dir, image_path):
        """
        Get save image name from source image path.
        """
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        return os.path.join(output_dir, "{}".format(name)+'_vis') + ext

    def _get_save_ir_image_name(self, output_dir, image_path):
        """
        Get save image name from source image path.
        """
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        return os.path.join(output_dir, "{}".format(name)+'_ir') + ext

    def _get_infer_cfg_and_input_spec(self,
                                      save_dir,
                                      prune_input=True,
                                      kl_quant=False):
        image_shape = None
        im_shape = [None, 2]
        scale_factor = [None, 2]
        if self.cfg.architecture in MOT_ARCH:
            test_reader_name = 'TestMOTReader'
        else:
            test_reader_name = 'TestReader'
        if 'inputs_def' in self.cfg[test_reader_name]:
            inputs_def = self.cfg[test_reader_name]['inputs_def']
            image_shape = inputs_def.get('image_shape', None)
        # set image_shape=[None, 3, -1, -1] as default
        if image_shape is None:
            image_shape = [None, 3, -1, -1]

        if len(image_shape) == 3:
            image_shape = [None] + image_shape
        else:
            im_shape = [image_shape[0], 2]
            scale_factor = [image_shape[0], 2]

        if hasattr(self.model, 'deploy'):
            self.model.deploy = True

        if 'slim' not in self.cfg:
            for layer in self.model.sublayers():
                if hasattr(layer, 'convert_to_deploy'):
                    layer.convert_to_deploy()

        if hasattr(self.cfg, 'export') and 'fuse_conv_bn' in self.cfg[
                'export'] and self.cfg['export']['fuse_conv_bn']:
            self.model = fuse_conv_bn(self.model)

        export_post_process = self.cfg['export'].get(
            'post_process', False) if hasattr(self.cfg, 'export') else True
        export_nms = self.cfg['export'].get('nms', False) if hasattr(
            self.cfg, 'export') else True
        export_benchmark = self.cfg['export'].get(
            'benchmark', False) if hasattr(self.cfg, 'export') else False
        if hasattr(self.model, 'fuse_norm'):
            self.model.fuse_norm = self.cfg['TestReader'].get('fuse_normalize',
                                                              False)
        if hasattr(self.model, 'export_post_process'):
            self.model.export_post_process = export_post_process if not export_benchmark else False
        if hasattr(self.model, 'export_nms'):
            self.model.export_nms = export_nms if not export_benchmark else False
        if export_post_process and not export_benchmark:
            image_shape = [None] + image_shape[1:]

        # Save infer cfg
        _dump_infer_config(self.cfg,
                           os.path.join(save_dir, 'infer_cfg.yml'), image_shape,
                           self.model)

        input_spec = [{
            "image": InputSpec(
                shape=image_shape, name='image'),
            "im_shape": InputSpec(
                shape=im_shape, name='im_shape'),
            "scale_factor": InputSpec(
                shape=scale_factor, name='scale_factor')
        }]
        if self.cfg.architecture == 'DeepSORT':
            input_spec[0].update({
                "crops": InputSpec(
                    shape=[None, 3, 192, 64], name='crops')
            })
        if prune_input:
            static_model = paddle.jit.to_static(
                self.model, input_spec=input_spec)
            # NOTE: dy2st do not pruned program, but jit.save will prune program
            # input spec, prune input spec here and save with pruned input spec
            pruned_input_spec = _prune_input_spec(
                input_spec, static_model.forward.main_program,
                static_model.forward.outputs)
        else:
            static_model = None
            pruned_input_spec = input_spec

        # TODO: Hard code, delete it when support prune input_spec.
        if self.cfg.architecture == 'PicoDet' and not export_post_process:
            pruned_input_spec = [{
                "image": InputSpec(
                    shape=image_shape, name='image')
            }]
        if kl_quant:
            if self.cfg.architecture == 'PicoDet' or 'ppyoloe' in self.cfg.weights:
                pruned_input_spec = [{
                    "image": InputSpec(
                        shape=image_shape, name='image'),
                    "scale_factor": InputSpec(
                        shape=scale_factor, name='scale_factor')
                }]
            elif 'tinypose' in self.cfg.weights:
                pruned_input_spec = [{
                    "image": InputSpec(
                        shape=image_shape, name='image')
                }]

        return static_model, pruned_input_spec

    def export(self, output_dir='output_inference'):
        if hasattr(self.model, 'aux_neck'):
            self.model.__delattr__('aux_neck')
        if hasattr(self.model, 'aux_head'):
            self.model.__delattr__('aux_head')
        self.model.eval()

        model_name = os.path.splitext(os.path.split(self.cfg.filename)[-1])[0]
        save_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        static_model, pruned_input_spec = self._get_infer_cfg_and_input_spec(
            save_dir)

        # dy2st and save model
        if 'slim' not in self.cfg or 'QAT' not in self.cfg['slim_type']:
            paddle.jit.save(
                static_model,
                os.path.join(save_dir, 'model'),
                input_spec=pruned_input_spec)
        else:
            self.cfg.slim.save_quantized_model(
                self.model,
                os.path.join(save_dir, 'model'),
                input_spec=pruned_input_spec)
        logger.info("Export model and saved in {}".format(save_dir))

    def post_quant(self, output_dir='output_inference'):
        model_name = os.path.splitext(os.path.split(self.cfg.filename)[-1])[0]
        save_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, data in enumerate(self.loader):
            self.model(data)
            if idx == int(self.cfg.get('quant_batch_num', 10)):
                break

        # TODO: support prune input_spec
        kl_quant = True if hasattr(self.cfg.slim, 'ptq') else False
        _, pruned_input_spec = self._get_infer_cfg_and_input_spec(
            save_dir, prune_input=False, kl_quant=kl_quant)

        self.cfg.slim.save_quantized_model(
            self.model,
            os.path.join(save_dir, 'model'),
            input_spec=pruned_input_spec)
        logger.info("Export Post-Quant model and saved in {}".format(save_dir))

    def _get_base_model(self):
        if hasattr(self.model, '_layers'):
            return self.model._layers
        return self.model

    def _count_total_params(self, model):
        total = 0
        for name, param in model.named_parameters():
            if any(x in name for x in ['_mean', '_variance', 'aux_']):
                continue
            total += int(np.prod(param.shape))
        return total

    def _count_params_by_prefix(self, model, prefixes):
        total = 0
        matched = False
        for name, param in model.named_parameters():
            if any(x in name for x in ['_mean', '_variance', 'aux_']):
                continue
            for prefix in prefixes:
                if name.startswith(prefix):
                    matched = True
                    total += int(np.prod(param.shape))
                    break
        return total, matched

    def _build_flops_inputs(self, data):
        if isinstance(data, typing.Sequence):
            data = data[0]
        inputs = {}
        for key in ['vis_image', 'ir_image', 'im_shape', 'scale_factor', 'pad_mask']:
            if key not in data:
                continue
            value = data[key]
            if isinstance(value, paddle.Tensor):
                if len(value.shape) > 0:
                    value = value[0].unsqueeze(0)
            inputs[key] = value
        return inputs

    def _log_stage_flops_and_params(self, data, stage_name, stage_flag, param_prefixes):
        model = self._get_base_model()
        if param_prefixes:
            params, matched = self._count_params_by_prefix(model, param_prefixes)
        else:
            params, matched = self._count_total_params(model), True
        if not matched:
            params = self._count_total_params(model)
            logger.warning(
                "Stage {} param prefixes not found, fallback to total params.".
                format(stage_name))
        logger.info("Stage {} Params : {:.3f} M.".format(stage_name,
                                                         params / 1e6))
        try:
            import paddleslim
        except Exception:
            logger.warning(
                "Unable to calculate FLOPs for stage {}, please install paddleslim, for example: `pip install paddleslim`"
                .format(stage_name))
            return params, None
        from paddleslim.analysis import dygraph_flops as flops
        inputs = self._build_flops_inputs(data)
        required = ['ir_image']
        if stage_flag in ['eval_stage2', 'eval_stage3']:
            required.append('vis_image')
        missing = [k for k in required if k not in inputs]
        if missing:
            logger.warning(
                "Stage {} FLOPs skipped, missing inputs: {}".format(
                    stage_name, ', '.join(missing)))
            return params, None

        class _StageFlopsWrapper(nn.Layer):
            def __init__(self, model, flag):
                super(_StageFlopsWrapper, self).__init__()
                self.model = model
                self.flag = flag

            def forward(self, inputs):
                input_dict = dict(inputs)
                input_dict['flag'] = self.flag
                return self.model(input_dict)

        prev_training = model.training
        prev_exclude = getattr(model, 'exclude_post_process', None)
        model.eval()
        if prev_exclude is not None:
            model.exclude_post_process = True
        wrapper = _StageFlopsWrapper(model, stage_flag)
        try:
            with paddle.no_grad():
                stage_flops = flops(wrapper, [inputs]) / (1000**3)
        except Exception as e:
            if prev_exclude is not None:
                model.exclude_post_process = prev_exclude
            if prev_training:
                model.train()
            else:
                model.eval()
            logger.warning(
                "Stage {} FLOPs failed: {}".format(stage_name, e))
            return params, None
        if prev_exclude is not None:
            model.exclude_post_process = prev_exclude
        if prev_training:
            model.train()
        else:
            model.eval()
        shape_info = []
        if 'vis_image' in inputs:
            shape_info.append('vis {}'.format(list(inputs['vis_image'].shape)))
        if 'ir_image' in inputs:
            shape_info.append('ir {}'.format(list(inputs['ir_image'].shape)))
        shape_str = ', '.join(shape_info) if shape_info else 'unknown'
        logger.info("Stage {} FLOPs : {:.6f} G. (input {})".format(
            stage_name, stage_flops, shape_str))
        return params, stage_flops

    def _get_stage_name(self, curr_iter, stage2_iter, stage3_iter):
        if curr_iter < stage2_iter:
            return 'stage1'
        if curr_iter < stage3_iter:
            return 'stage2'
        return 'stage3'

    def _get_batch_size(self, data):
        if isinstance(data, typing.Sequence):
            data = data[0]
        if isinstance(data, dict):
            if 'im_id' in data and isinstance(data['im_id'], paddle.Tensor):
                return int(data['im_id'].shape[0])
            for key in ['ir_image', 'vis_image']:
                if key in data and isinstance(data[key], paddle.Tensor):
                    return int(data[key].shape[0])
        return 0

    def _calc_stage_images(self, stage_name, bs, curr_iter, st_iter):
        return bs

    def _update_stage_fps(self, curr_iter, stage2_iter, stage3_iter, st_iter,
                          data, iter_cost):
        stage_name = self._get_stage_name(curr_iter, stage2_iter, stage3_iter)
        if self._stage_fps_stats['name'] != stage_name:
            self._log_stage_fps(self._stage_fps_stats)
            self._stage_fps_stats = {
                'name': stage_name,
                'time': 0.0,
                'images': 0
            }
        bs = self._get_batch_size(data)
        if bs <= 0:
            return
        images = self._calc_stage_images(stage_name, bs, curr_iter, st_iter)
        self._stage_fps_stats['time'] += float(iter_cost)
        self._stage_fps_stats['images'] += int(images)

    def _log_stage_fps(self, stats):
        if not stats or stats.get('name') is None:
            return
        if stats.get('time', 0) <= 0:
            return
        fps = stats['images'] / stats['time']
        logger.info(
            "Stage {} FPS (images/s) : {:.3f}. ({} images / {:.2f}s)".format(
                stats['name'], fps, stats['images'], stats['time']))

    def _flops(self, loader):
        if hasattr(self.model, 'aux_neck'):
            self.model.__delattr__('aux_neck')
        if hasattr(self.model, 'aux_head'):
            self.model.__delattr__('aux_head')
        self.model.eval()
        try:
            import paddleslim
        except Exception as e:
            logger.warning(
                'Unable to calculate flops, please install paddleslim, for example: `pip install paddleslim`'
            )
            return

        from paddleslim.analysis import dygraph_flops as flops
        input_data = None
        for data in loader:
            input_data = data
            break

        input_spec = [{
            "vis_image": input_data['vis_image'][0].unsqueeze(0),
            "ir_image": input_data['ir_image'][0].unsqueeze(0),
            "im_shape": input_data['im_shape'][0].unsqueeze(0),
            "scale_factor": input_data['scale_factor'][0].unsqueeze(0)
        }]

        # input_spec = [{
        #     "image": input_data['image'][0].unsqueeze(0),
        #     "im_shape": input_data['im_shape'][0].unsqueeze(0),
        #     "scale_factor": input_data['scale_factor'][0].unsqueeze(0)
        # }]
        # flops = flops(self.model, input_spec) / (1000**3)
        # logger.info(" Model FLOPs : {:.6f}G. (image shape is {})".format(
        #     flops, input_data['image'][0].unsqueeze(0).shape))

    def parse_mot_images(self, cfg):
        import glob
        # for quant
        dataset_dir = cfg['EvalMOTDataset'].dataset_dir
        data_root = cfg['EvalMOTDataset'].data_root
        data_root = '{}/{}'.format(dataset_dir, data_root)
        seqs = os.listdir(data_root)
        seqs.sort()
        all_images = []
        for seq in seqs:
            infer_dir = os.path.join(data_root, seq)
            assert infer_dir is None or os.path.isdir(infer_dir), \
                "{} is not a directory".format(infer_dir)
            images = set()
            exts = ['jpg', 'jpeg', 'png', 'bmp']
            exts += [ext.upper() for ext in exts]
            for ext in exts:
                images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
            images = list(images)
            images.sort()
            assert len(images) > 0, "no image found in {}".format(infer_dir)
            all_images.extend(images)
            logger.info("Found {} inference images in total.".format(
                len(images)))
        return all_images
    
