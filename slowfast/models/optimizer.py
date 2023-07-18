#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import torch
import torch.nn as nn

import slowfast.utils.lr_policy as lr_policy
from utils import logger

def temporal_spatial_sep(module):
    """
    Separate temporal and spatial parameters.
    """
    model_p = {}
    for name, p in module.named_parameters():
        logger.info(name)
        model_p[name] = p
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    temporal_p_bn = []
    temporal_p_bn_non = []
    spatial_p_bn = []
    spatial_p_bn_non = []
    for name,m in module.named_modules():
        if isinstance(m,nn.Conv3d):
            kernel_size = list(m.kernel_size)
            logger.info("%s [%d,%d,%d]"%(name,kernel_size[0],kernel_size[1],kernel_size[2]))
            assert kernel_size[1] == kernel_size[2]
            if kernel_size[0] ==1 and kernel_size[1] >1:
                # spatial conv
                if name+'.bias' in model_p.keys():
                    spatial_p_bn_non+=[model_p[name+'.weight'],model_p[name+'.bias']]
                else:
                    spatial_p_bn_non+=[model_p[name+'.weight']]
                # print("spatial_p",m.kernel_size)
            elif kernel_size[0] >1 and kernel_size[1]==1:
                # temporal conv
                if name+'.bias' in model_p.keys():
                    temporal_p_bn_non+=[model_p[name+'.weight'],model_p[name+'.bias']]
                else:
                    temporal_p_bn_non+=[model_p[name+'.weight']]
                # print("temporal_p",m.kernel_size)
            else:
                # [1,1,1] conv or [5,7,7] conv
                if name+'.bias' in model_p.keys():
                    spatial_p_bn_non+=[model_p[name+'.weight'],model_p[name+'.bias']]
                    temporal_p_bn_non+=[model_p[name+'.weight'],model_p[name+'.bias']]
                else:
                    spatial_p_bn_non+=[model_p[name+'.weight']]
                    temporal_p_bn_non+=[model_p[name+'.weight']]
        elif isinstance(m,nn.Linear) or isinstance(m,nn.BatchNorm3d):
            if 'head' in name:
                continue
            elif 'norm' in name:
                temporal_p_bn+=[model_p[name+'.weight'],model_p[name+'.bias']]
                spatial_p_bn+=[model_p[name+'.weight'],model_p[name+'.bias']]
            else:
                if name+'.bias' in model_p.keys():
                    temporal_p_bn_non+=[model_p[name+'.weight'],model_p[name+'.bias']] 
                    spatial_p_bn_non+=[model_p[name+'.weight'],model_p[name+'.bias']]
                else:
                    temporal_p_bn_non+=[model_p[name+'.weight']] 
                    spatial_p_bn_non+=[model_p[name+'.weight']]
            logger.info('%s added to spatial&temporal_p'%name)
    for name, p in module.named_parameters():
        if 'space_transformer' in name or 'space_token' in name:
            # if 'weight' in name and 'norm' not in name:
            #     model_p[name].data.zero_()
            # if 'bias' in name and 'norm' not in name: 
            #     model_p[name].data.zero_()
            spatial_p_bn_non+=[model_p[name]]
            logger.info('%s added to spatial_p'%name)
        elif 'temporal_transformer' in name or 'temporal_token' in name or 'time_T' in name: 
            # if 'weight' in name and 'norm' not in name:
            #     model_p[name].data.zero_()
            # if 'bias' in name and 'norm' not in name: 
            #     model_p[name].data.zero_()
            temporal_p_bn_non+=[model_p[name]]
            logger.info('%s added to temporal_p'%name)
        # elif 'mlp_head' in name or 'pos_embedding' in name: 
        elif 'head' in name or 'pos_embedding' in name: 
            spatial_p_bn_non+=[model_p[name]]
            temporal_p_bn_non+=[model_p[name]]
            logger.info('%s added to spatial&temporal_p'%name)
    return temporal_p_bn, temporal_p_bn_non, spatial_p_bn, spatial_p_bn_non
    


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    # Batchnorm parameters.
    bn_params = []
    # Non-batchnorm parameters.
    non_bn_parameters = []
    for name, p in model.named_parameters():
        if "bn" in name:
            bn_params.append(p)
        else:
            non_bn_parameters.append(p)
    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
    # Having a different weight decay on batchnorm might cause a performance
    # drop.
    # logger.info(str(cfg))
    optim_params = [
        {"params": bn_params, "weight_decay": cfg.BN.WEIGHT_DECAY},
        {"params": non_bn_parameters, "weight_decay": cfg.model.inco.SOLVER.WEIGHT_DECAY},
    ]
    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(
        bn_params
    ), "parameter size does not match: {} + {} != {}".format(
        len(non_bn_parameters), len(bn_params), len(list(model.parameters()))
    )

    if cfg.model.inco.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.model.inco.SOLVER.BASE_LR,
            momentum=cfg.model.inco.SOLVER.MOMENTUM,
            weight_decay=cfg.model.inco.SOLVER.WEIGHT_DECAY,
            dampening=cfg.model.inco.SOLVER.DAMPENING,
            nesterov=cfg.model.inco.SOLVER.NESTEROV,
        )
    elif cfg.model.inco.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.model.inco.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.model.inco.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.model.inco.SOLVER.OPTIMIZING_METHOD)
        )

def construct_optimizer_altertraining(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum for alternating training of
    Temporal and Spatial kernels.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    temporal_p_bn, temporal_p_bn_non, spatial_p_bn, spatial_p_bn_non = temporal_spatial_sep(model)
    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
    # Having a different weight decay on batchnorm might cause a performance
    # drop.
    # logger.info(str(cfg))
    optim_params_temporal = [
        {"params": temporal_p_bn, "weight_decay": cfg.BN.WEIGHT_DECAY},
        {"params": temporal_p_bn_non, "weight_decay": cfg.model.inco.SOLVER.WEIGHT_DECAY},
    ]
    optim_params_spatial = [
        {"params": spatial_p_bn, "weight_decay": cfg.BN.WEIGHT_DECAY},
        {"params": spatial_p_bn_non, "weight_decay": cfg.model.inco.SOLVER.WEIGHT_DECAY},
    ]

    if cfg.model.inco.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params_temporal,
            lr=cfg.model.inco.SOLVER.BASE_LR,
            momentum=cfg.model.inco.SOLVER.MOMENTUM,
            weight_decay=cfg.model.inco.SOLVER.WEIGHT_DECAY,
            dampening=cfg.model.inco.SOLVER.DAMPENING,
            nesterov=cfg.model.inco.SOLVER.NESTEROV,
        ),torch.optim.SGD(
            optim_params_spatial,
            lr=cfg.model.inco.SOLVER.BASE_LR,
            momentum=cfg.model.inco.SOLVER.MOMENTUM,
            weight_decay=cfg.model.inco.SOLVER.WEIGHT_DECAY,
            dampening=cfg.model.inco.SOLVER.DAMPENING,
            nesterov=cfg.model.inco.SOLVER.NESTEROV,
        )
    elif cfg.model.inco.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params_temporal,
            lr=cfg.model.inco.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.model.inco.SOLVER.WEIGHT_DECAY,
        ),torch.optim.Adam(
            optim_params_spatial,
            lr=cfg.model.inco.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.model.inco.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.model.inco.SOLVER.OPTIMIZING_METHOD)
        )

def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decays.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)

def get_iter_lr(cur_iter, cfg):
    """
    Retrieves the lr for the given iter (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decays.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr=lr_policy.get_lr_at_iter(cfg, cur_iter)
    
    return lr


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
