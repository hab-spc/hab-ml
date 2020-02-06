# Standard dist imports
import argparse
import logging
import os

import sys
sys.path.insert(0,os.path.realpath(__file__)[:-7])

from pprint import pformat
import time
from datetime import datetime

# Third party imports
from tensorboardX import SummaryWriter
import numpy as np
import torch
import matplotlib.pyplot as plt

# Project level imports
from model import resnet
from model.model import HABClassifier
from trainer import Trainer
from data.dataloader import get_dataloader, to_cuda
from utils.constants import Constants as CONST
from utils.config import opt, set_config
from utils.eval_utils import accuracy, get_meter, EvalMetrics, vis_training
from utils.logger import Logger
from utils.model_sql import model_sql

# Module level constants

def coral_train_and_evaluate(opt, logger=None, tb_logger=None):
    logger = logger if logger else logging.getLogger('train-and-evaluate')
    logger.setLevel(opt.logging_level)

    # Read in dataset
    # check the path for the data loader to make sure it is loading the right data set
    src_data_loader = {mode: get_coral_dataloader(data_dir=opt.data_dir, camera='pier',
                                        batch_size=opt.batch_size,
                                        mode=mode) for mode in [CONST.TRAIN, CONST.VAL]}
    target_data_loader = {mode: get_coral_dataloader(data_dir=opt.data_dir, camera='lab',
                                        batch_size=opt.batch_size,
                                        mode=mode) for mode in [CONST.TRAIN, CONST.VAL]}
    # Create model
    model = HABCORAL(arch=opt.arch, pretrained=opt.pretrained, num_classes=opt.class_num)

    # Initialize Trainer for initializing losses, optimizers, loading weights, etc
    trainer = Trainer(model=model, model_dir=opt.model_dir, mode=opt.mode,
                      resume=opt.resume, lr=opt.lr, momentum=opt.momentum,
                      class_count=data_loader[CONST.TRAIN].dataset.data[
                          CONST.LBL].value_counts())

    # ==== BEGIN OPTION 1: TRAINING ====#
    # Train and validate model if set to TRAINING
    # When training, we do both training and validation within the loop.
    # When set to the validation mode, this will run a full evaluation
    # and produce more summarized evaluation results. This is the default condition
    # if the mode is not training.
    if opt.mode == CONST.TRAIN:
        best_err = trainer.best_err
        # Evaluate on validation set
        Logger.section_break('Valid SOURCE (Epoch {})'.format(trainer.start_epoch))
        src_err, src_acc, _, src_metrics_test = evaluate(trainer.model, trainer,
                                                         src_data_loader[CONST.VAL],
                                                         trainer.start_epoch, opt.batch_size, logger,
                                                         tb_logger,
                                                         max_iters=None)
        Logger.section_break('Valid TARGET (Epoch {})'.format(trainer.start_epoch))
        target_err, target_acc, _, target_metrics_test = evaluate(trainer.model,
                                                                  trainer,
                                                                  target_data_loader[
                                                                      CONST.VAL],
                                                                  trainer.start_epoch, opt.batch_size,
                                                                  logger,
                                                                  tb_logger,
                                                                  max_iters=None)
        metrics_best = target_metrics_test

        eps_meter = get_meter(meters=['class_train_loss', 'class_val_loss',
                                      'coral_train_loss', 'coral_val_loss',
                                      'total_loss',
                                      'train_acc', 'val_acc'])

        for ii, epoch in enumerate(range(trainer.start_epoch,
                                         trainer.start_epoch + opt.epochs)):

            # Train for one epoch
            Logger.section_break('Train (Epoch {})'.format(epoch))
            class_train_loss, \
            coral_train_loss, \
            total_loss, train_acc = train(trainer.model,
                                          trainer,
                                          src_data_loader[CONST.TRAIN],
                                          target_data_loader[CONST.TRAIN],
                                          epoch, logger, tb_logger,
                                          opt.batch_size, opt.print_freq)

            eps_meter['class_train_loss'].update(class_train_loss)
            eps_meter['coral_train_loss'].update(coral_train_loss)
            eps_meter['total_loss'].update(total_loss)
            eps_meter['train_acc'].update(train_acc)

            # Evaluate on validation set
            Logger.section_break('Valid SOURCE (Epoch {})'.format(epoch))
            src_err, src_acc, _, src_metrics_test = evaluate(trainer.model, trainer,
                                                 src_data_loader[CONST.VAL],
                                                 epoch, opt.batch_size, logger,
                                                 tb_logger,
                                                 max_iters=None)
            Logger.section_break('Valid TARGET (Epoch {})'.format(epoch))
            target_err, target_acc, _, target_metrics_test = evaluate(trainer.model,
                                                                   trainer,
                                                 target_data_loader[CONST.VAL],
                                                 epoch, opt.batch_size, logger,
                                                 tb_logger,
                                                 max_iters=None)

            eps_meter['class_src_loss'].update(src_err)
            eps_meter['class_target_loss'].update(target_err)
            eps_meter['class_src_acc'].update(src_acc)
            eps_meter['class_target_acc'].update(target_acc)

            # Remember best error and save checkpoint
            err = target_err
            is_best = err < best_err
            best_err = min(err, best_err)
            state = trainer.generate_state_dict(epoch=epoch, best_err=best_err)

            if epoch % opt.save_freq == 0:
                trainer.save_checkpoint(state, is_best=False,
                                        filename='checkpoint-{}_{:0.4f}.pth.tar'.format(
                                            epoch, target_acc))
            if is_best:
                metrics_best = target_metrics_test
                trainer.save_checkpoint(state, is_best=is_best,
                                        filename='model_best.pth.tar')

        # ==== END OPTION 1: TRAINING LOOP ====#
        # Generate evaluation plots
        opt.train_acc = max(eps_meter['train_acc'].data)
        opt.test_acc = max(eps_meter['val_acc'].data)
        # plot loss over eps
        vis_training(eps_meter['train_loss'].data, eps_meter['val_loss'].data, loss=True)
        # plot acc over eps
        vis_training(eps_meter['train_acc'].data, eps_meter['val_acc'].data, loss=False)

        # plot best confusion matrix
        plt.figure()
        metrics_best.compute_cm(plot=True)

    # ==== BEGIN OPTION 2: EVALUATION ====#
    # EVALUATE the model if set to evaluation mode
    # Below you'll receive a more comprehensive report of the evaluation in the eval.log
    elif opt.mode == CONST.VAL:
        err, acc, run_time, metrics = evaluate(
            model=trainer.model, trainer=trainer, data_loader=data_loader[
                CONST.VAL], logger=logger, tb_logger=tb_logger)

        Logger.section_break('EVAL COMPLETED')
        model_parameters = filter(lambda p: p.requires_grad,
                                  trainer.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        metrics.print_eval(params, run_time, err, acc, metrics.results_dir)

        cm, mca = metrics.compute_cm(plot=True)
    # ==== END OPTION 2: EVALUATION ====#

