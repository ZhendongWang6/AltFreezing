import sys
import os

config_text = """
TRAIN:
  ENABLE: True
  DATASET: kinetics
  BATCH_SIZE: 64
  EVAL_PERIOD: 10
  CHECKPOINT_PERIOD: 1
  AUTO_RESUME: True
DATA:
  NUM_FRAMES: 8
  SAMPLING_RATE: 8
  TRAIN_JITTER_SCALES: [256, 320]
  TRAIN_CROP_SIZE: 224
  TEST_CROP_SIZE: 256
  INPUT_CHANNEL_NUM: [3]
RESNET:
  ZERO_INIT_FINAL_BN: True
  WIDTH_PER_GROUP: 64
  NUM_GROUPS: 1
  DEPTH: 50
  TRANS_FUNC: bottleneck_transform
  STRIDE_1X1: False
  NUM_BLOCK_TEMP_KERNEL: [[3], [4], [6], [3]]
NONLOCAL:
  LOCATION: [[[]], [[]], [[]], [[]]]
  GROUP: [[1], [1], [1], [1]]
  INSTANTIATION: softmax
BN:
  USE_PRECISE_STATS: True
  NUM_BATCHES_PRECISE: 200
SOLVER:
  BASE_LR: 0.1
  LR_POLICY: cosine
  MAX_EPOCH: 196
  MOMENTUM: 0.9
  WEIGHT_DECAY: 1e-4
  WARMUP_EPOCHS: 34.0
  WARMUP_START_LR: 0.01
  OPTIMIZING_METHOD: sgd
  ALTER_FREQ: 10
MODEL:
  NUM_CLASSES: 1
  ARCH: i3d
  MODEL_NAME: ResNet
  LOSS_FUNC: cross_entropy
  DROPOUT_RATE: 0.5
  HEAD_ACT: sigmoid
TEST:
  ENABLE: True
  DATASET: kinetics
  BATCH_SIZE: 64
DATA_LOADER:
  NUM_WORKERS: 8
  PIN_MEMORY: True
NUM_GPUS: 8
NUM_SHARDS: 1
RNG_SEED: 0
OUTPUT_DIR: .
"""

from slowfast.models.video_model_builder import ResNet as ResNetOri
from slowfast.config.defaults import get_cfg
import torch
from torch import nn
from config import config as my_cfg
from utils import logger


class I3D8x8(nn.Module):
    def __init__(self) -> None:
        super(I3D8x8, self).__init__()
        cfg = get_cfg()
        cfg.merge_from_str(config_text)
        cfg.NUM_GPUS = 1
        cfg.TEST.BATCH_SIZE = 1
        cfg.TRAIN.BATCH_SIZE = 1
        cfg.DATA.NUM_FRAMES = my_cfg.clip_size
        SOLVER = my_cfg.model.inco.SOLVER
        # logger.info(str(SOLVER))
        if SOLVER is not None:
            for key, val in SOLVER.to_dict().items():
                old_val = getattr(cfg.SOLVER, key)
                val = type(old_val)(val)
                setattr(cfg.SOLVER, key, val)
        if my_cfg.model.inco.i3d_routine:
            self.cfg = cfg
        self.resnet = ResNetOri(cfg)

    def forward(
        self,
        images,
        noise=None,
        has_mask=None,
        freeze_backbone=False,
        return_feature_maps=False,
    ):
        assert not freeze_backbone
        inputs = [images]
        pred = self.resnet(inputs)
        output = {"final_output": pred}
        return output


from torch import nn
from typing import Callable, Type
from ._classifier_base import ClassifierBase


class Classifier(ClassifierBase):
    @property
    def module_to_build(self) -> Type[nn.Module]:
        return I3D8x8
