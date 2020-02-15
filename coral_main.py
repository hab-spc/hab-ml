# Standard dist imports
import argparse
from collections import namedtuple
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
from model.coral import HABCORAL
from trainer import Trainer, seed_torch
from data.coral_dataloader import get_coral_dataloader
from data.dataloader import get_dataloader
from data.d_utils import to_cuda
from utils.constants import Constants as CONST
from utils.config import opt, set_config
from utils.eval_utils import accuracy, get_meter, EvalMetrics, vis_training, update_meter
from utils.logger import Logger
from utils.model_sql import model_sql

# Module level constants
seed_torch(opt.SEED)
LossTuple = namedtuple('LossTuple',
                       ['class_train_loss',
                        'coral_train_loss',
                        'total_loss',
                        'train_acc'
                        ])

def coral_train_and_evaluate(opt, logger=None, tb_logger=None):
    logger = logger if logger else logging.getLogger('train-and-evaluate')
    logger.setLevel(opt.logging_level)

    # Read in dataset
    # check the path for the data loader to make sure it is loading the right data set
    train_data_loader = get_coral_dataloader(data_dir=opt.data_dir,
                                             batch_size=opt.batch_size,
                                             mode=CONST.TRAIN)

    src_data_loader = get_coral_dataloader(data_dir=opt.data_dir, camera='pier',
                                        batch_size=opt.batch_size,
                                        mode=CONST.VAL)
    target_data_loader = get_coral_dataloader(data_dir=opt.data_dir, camera='lab',
                                        batch_size=opt.batch_size,
                                        mode=CONST.VAL)
    # Create model
    Logger.section_break('MODEL ARCHITECTURE')
    model = HABCORAL(arch=opt.arch, num_classes=opt.class_num)
    logger.debug(model)

    # Initialize Trainer for initializing losses, optimizers, loading weights, etc
    Logger.section_break('MODEL TRAINER')
    trainer = Trainer(model=model, model_dir=opt.model_dir, mode=opt.mode,
                      resume=opt.resume, lr=opt.lr, momentum=opt.momentum,
                      class_count=train_data_loader.dataset.datasets[0].get_class_counts())

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
                                                         src_data_loader,
                                                         trainer.start_epoch,
                                                         opt.batch_size, logger,
                                                         tb_logger, max_iters=None)
        Logger.section_break('Valid TARGET (Epoch {})'.format(trainer.start_epoch))
        target_err, target_acc, _, target_metrics_test = evaluate(trainer.model, trainer,
                                                                  target_data_loader,
                                                                  trainer.start_epoch,
                                                                  opt.batch_size, logger,
                                                                  tb_logger,
                                                                  max_iters=None,
                                                                  hab_eval=True)
        metrics_best = src_metrics_test

        eps_meter = get_meter(meters=['class_train_loss', 'class_src_loss',
                                      'coral_train_loss', 'class_target_loss',
                                      'total_loss',
                                      'train_acc', 'class_src_acc', 'class_target_acc'])

        for ii, epoch in enumerate(range(trainer.start_epoch,
                                         trainer.start_epoch + opt.epochs)):
            # Train for one epoch
            Logger.section_break('Train (Epoch {})'.format(epoch))
            scores = train(trainer.model, trainer, train_data_loader,
                          epoch, logger, tb_logger, opt.batch_size, opt.print_freq)

            update_meter(eps_meter, scores)

            # Evaluate on validation set
            Logger.section_break('Valid SOURCE (Epoch {})'.format(epoch))
            err, acc, _, metrics_test = evaluate(trainer.model, trainer, src_data_loader,
                                                 epoch, opt.batch_size, logger,
                                                 tb_logger, max_iters=None)
            eps_meter['class_src_loss'].update(err)
            eps_meter['class_src_acc'].update(acc)

            Logger.section_break('Valid TARGET (Epoch {})'.format(epoch))
            err, acc, _, metrics_test = evaluate(trainer.model, trainer,
                                                 target_data_loader, epoch,
                                                 opt.batch_size, logger, tb_logger,
                                                 max_iters=None, hab_eval=True)
            eps_meter['class_target_loss'].update(err)
            eps_meter['class_target_acc'].update(acc)

            # Remember best error and save checkpoint
            is_best = err < best_err
            best_err = min(err, best_err)
            state = trainer.generate_state_dict(epoch=epoch, best_err=best_err)

            if epoch % opt.save_freq == 0:
                trainer.save_checkpoint(state, is_best=False,
                                        filename='checkpoint-{}_{:0.4f}.pth.tar'.format(
                                            epoch, acc))
            if is_best:
                metrics_best = metrics_test
                trainer.save_checkpoint(state, is_best=is_best,
                                        filename='model_best.pth.tar')

        Logger.section_break('TRAINING COMPLETED')
        # ==== END OPTION 1: TRAINING LOOP ====#
        # Generate evaluation plots
        opt.train_acc = max(eps_meter['train_acc'].data)
        opt.src_acc = max(eps_meter['class_src_acc'].data)
        opt.target_acc = max(eps_meter['class_target_acc'].data)
        # plot loss over eps
        vis_training(eps_meter['class_train_loss'].data,
                     eps_meter['class_src_loss'].data,
                     loss=True)
        # plot acc over eps
        vis_training(eps_meter['train_acc'].data,
                     eps_meter['class_src_acc'].data,
                     loss=False)

        # plot best confusion matrix
        metrics_best.compute_cm(plot=True, save=True)

    # ==== BEGIN OPTION 2: EVALUATION ====#
    # EVALUATE the model if set to evaluation mode
    # Below you'll receive a more comprehensive report of the evaluation in the eval.log
    elif opt.mode == CONST.VAL:
        src_err, src_acc, src_run_time, src_metrics = evaluate(model=trainer.model,
                                                               trainer=trainer,
                                                               data_loader=src_data_loader,
                                                               logger=logger,
                                                               tb_logger=tb_logger)

        target_err, target_acc, target_run_time, target_metrics = evaluate(
            model=trainer.model, trainer=trainer, data_loader=src_data_loader,
            logger=logger, tb_logger=tb_logger, hab_eval=True)

        Logger.section_break('EVAL COMPLETED')
        model_parameters = filter(lambda p: p.requires_grad,
                                  trainer.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        Logger.section_break('SOURCE EVAL')
        src_metrics.print_eval(params, src_run_time, src_err, src_acc, src_metrics.results_dir)
        Logger.section_break('TARGET EVAL')
        target_metrics.print_eval(params, target_run_time, target_err, target_acc, target_metrics.results_dir)
    # ==== END OPTION 2: EVALUATION ====#


def train(model, trainer, train_loader,
          epoch, logger, tb_logger, batch_size=opt.batch_size,
          print_freq=opt.print_freq):
    """ Train the model

    Outside of the typical training loops, `train()` incorporates other
    useful bookkeeping features and wrapper functions. This includes things
    like keeping track of accuracy, loss, batch time to wrapping optimizers
    and loss functions in the `trainer`. Be sure to reference `trainer.py`
    or `utils/eval_utils.py` if extra detail is needed.

    Args:
        model: Classification model
        trainer (Trainer): Training wrapper
        train_loader (torch.data.Dataloader): Generator data loading instance
        epoch (int): Current epoch
        logger (Logger): Logger. Used to display/log metrics
        tb_logger (SummaryWriter): Tensorboard Logger
        batch_size (int): Batch size
        print_freq (int): Print frequency

    Returns:
        None

    """
    criterion = trainer.criterion
    coral_criterion = trainer.coral_criterion
    optimizer = trainer.optimizer

    # Initialize meter to bookkeep the following parameters
    meter = get_meter(meters=['batch_time', 'data_time', 'class_loss',
                              'coral_loss', 'total_loss', 'acc'])

    # Switch to training mode
    model.train(True)

    end = time.time()
    num_iterations = len(train_loader)
    for i, batch in enumerate(train_loader):

        src_batch, target_batch = batch[0], batch[1]

        # process batch items: images, labels
        src_img = to_cuda(src_batch[CONST.IMG], trainer.computing_device)
        src_label = to_cuda(src_batch[CONST.LBL], trainer.computing_device, label=True)

        target_img = to_cuda(target_batch[CONST.IMG], trainer.computing_device)

        # measure data loading time
        meter['data_time'].update(time.time() - end)

        # compute output
        end = time.time()
        src_logits, target_logits = model(src_img, target_img)

        # compute losses
        class_loss = criterion(src_logits, src_label)
        coral_loss = coral_criterion(src_logits, target_logits)

        total_loss = class_loss + opt.lambda_coral * coral_loss

        # compute evaluation metric
        acc = accuracy(src_logits, src_label)

        # update metrics
        meter['acc'].update(acc, batch_size)
        meter['class_loss'].update(class_loss, batch_size)
        meter['coral_loss'].update(coral_loss, batch_size)
        meter['total_loss'].update(total_loss, batch_size)

        # compute gradient and do sgd step
        optimizer.zero_grad()
        total_loss.backward()

        if i % print_freq == 0:
            log = 'TRAIN [{:02d}][{:2d}/{:2d}] TIME {:10} DATA {:10} ACC {:10} ' \
                  'CLASS LOSS {:10} CORAL LOSS {:10} TOTAL LOSS {:10}'. \
                format(epoch, i, num_iterations,
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['batch_time']),
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['data_time']),
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['acc']),
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['class_loss']),
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['coral_loss']),
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['total_loss'])
                       )
            logger.info(log)

            tb_logger.add_scalar('train/class_loss', meter['class_loss'].val,
                                 epoch * num_iterations + i)
            tb_logger.add_scalar('train/coral_loss', meter['coral_loss'].val,
                                 epoch * num_iterations + i)
            tb_logger.add_scalar('train/total_loss', meter['total_loss'].val,
                                 epoch * num_iterations + i)
            tb_logger.add_scalar('train/accuracy', meter['acc'].val,
                                 epoch * num_iterations + i)
            tb_logger.add_scalar('data_time', meter['data_time'].val,
                                 epoch * num_iterations + i)
            tb_logger.add_scalar('compute_time',
                                 meter['batch_time'].val - meter['data_time'].val,
                                 epoch * num_iterations + i)

        optimizer.step()

        # measure elapsed time
        meter['batch_time'].update(time.time() - end)
        end = time.time()

    tb_logger.add_scalar('train-epoch/total_loss', meter['total_loss'].avg, epoch)
    tb_logger.add_scalar('train-epoch/accuracy', meter['acc'].avg, epoch)

    meter_values = [meter['class_loss'].avg, meter['coral_loss'].avg,\
           meter['total_loss'].avg, meter['acc'].avg]

    return LossTuple(*meter_values)


def evaluate(model, trainer, data_loader, epoch=0, batch_size=opt.batch_size,
             logger=None, tb_logger=None, max_iters=None, hab_eval=False):
    """ Evaluate model

    Similar to `train()` structure, where the function includes bookkeeping
    features and wrapper items. The only difference is that evaluation will
    only occur until the `max_iter` if it is specified and includes an
    `EvalMetrics` intiailization.

    The latter is currrently used to save predictions and ground truths to
    compute the confusion matrix.

    Args:
        hab_eval:
        model: Classification model
        trainer (Trainer): Training wrapper
        data_loader (torch.data.Dataloader): Generator data loading instance
        epoch (int): Current epoch
        logger (Logger): Logger. Used to display/log metrics
        tb_logger (SummaryWriter): Tensorboard Logger
        batch_size (int): Batch size
        max_iters (int): Max iterations

    Returns:
        float: Loss average
        float: Accuracy average
        float: Run time average
        EvalMetrics: Evaluation wrapper to compute CMs

    """
    criterion = trainer.criterion

    # Initialize meter and metrics
    meter = get_meter(meters=['batch_time', 'loss', 'acc'])
    predictions, gtruth, ids = [], [], []
    classes = data_loader.dataset.classes
    metrics = EvalMetrics(classes, predictions, gtruth, ids, trainer.model_dir)

    # Switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # process batch items: images, labels
            img = to_cuda(batch[CONST.IMG], trainer.computing_device)
            label = to_cuda(batch[CONST.LBL], trainer.computing_device, label=True)
            id = batch[CONST.ID]

            # compute output
            end = time.time()
            logits, _ = model(img, img)
            loss = criterion(logits, label)
            acc = metrics.accuracy(logits, label, hab_eval=hab_eval)
            batch_size = list(batch[CONST.LBL].shape)[0]

            # update metrics
            meter['acc'].update(acc, batch_size)
            meter['loss'].update(loss, batch_size)

            # update metrics2
            metrics.update(logits, label, id)

            # measure elapsed time
            meter['batch_time'].update(time.time() - end, batch_size)

            if i % opt.print_freq == 0:
                log = 'EVAL [{:02d}][{:2d}/{:2d}] TIME {:10} ACC {:10} LOSS {' \
                      ':10}'.format(epoch, i, len(data_loader),
                                    "{t.val:.3f} ({t.avg:.3f})".format(
                                        t=meter['batch_time']),
                                    "{t.val:.3f} ({t.avg:.3f})".format(t=meter['acc']),
                                    "{t.val:.3f} ({t.avg:.3f})".format(t=meter['loss'])
                                    )
                logger.info(log)

                if tb_logger is not None:
                    tb_logger.add_scalar('test/loss', meter['loss'].val, epoch)
                    tb_logger.add_scalar('test/accuracy', meter['acc'].val, epoch)

            if max_iters is not None and i >= max_iters:
                break

        # Print last eval
        log = 'EVAL [{:02d}][{:2d}/{:2d}] TIME {:10} ACC {:10} LOSS {' \
              ':10}'.format(epoch, i, len(data_loader),
                            "{t.val:.3f} ({t.avg:.3f})".format(t=meter['batch_time']),
                            "{t.val:.3f} ({t.avg:.3f})".format(t=meter['acc']),
                            "{t.val:.3f} ({t.avg:.3f})".format(t=meter['loss'])
                            )
        logger.info(log)

        if tb_logger is not None:
            tb_logger.add_scalar('test-epoch/loss', meter['loss'].avg, epoch)
            tb_logger.add_scalar('test-epoch/accuracy', meter['acc'].avg, epoch)

    return meter['loss'].avg, meter['acc'].avg, meter['batch_time'], metrics

def deploy(opt, logger=None):
    """ Deploy a model in production mode, assumes unseen/unlabeled data as input

    Function call is intended to run model on unseen/unlabeled data, hence deployment.

    Args:
        opt (Config): A state dictionary holding preset parameters
        logger (Logger): Logging instance

    Returns:

    """
    logger = logger if logger else logging.getLogger('deploy')
    logger.setLevel(opt.logging_level)
    start_datetime = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')

    # read data
    data_loader = get_dataloader(mode=CONST.DEPLOY, csv_file=opt.deploy_data,
                                 batch_size=opt.batch_size,
                                 input_size=opt.input_size)

    # load model
    model = HABCORAL(arch=opt.arch, num_classes=opt.class_num)

    # Initialize Trainer for initializing losses, optimizers, loading weights, etc
    trainer = Trainer(model=model, model_dir=opt.model_dir, mode=opt.mode,
                      resume=opt.resume)

    # Run Predictions
    Logger.section_break('Deploy')
    logger.info('Starting deployment...')
    err, acc, run_time, metrics = evaluate(model=trainer.model, trainer=trainer,
                                           data_loader=data_loader, logger=logger,
                                           tb_logger=tb_logger, hab_eval=opt.hab_eval)

    dest_dir = opt.deploy_data + '_static_html' if opt.lab_config else opt.deploy_data
    metrics.save_predictions(start_datetime, run_time.avg, opt.model_dir,
                             dest_dir)

    # compute hab accuracy
    hab_acc = metrics.compute_hab_acc() if opt.hab_eval else 'NOT EVALUATED'

    # plot confusion matrix and get mca
    _, mca_acc = metrics.compute_cm(plot=True)

    # plot roc curve
    _, _, auc_score = metrics.compute_roc_auc_score(plot=True)

    # plot precision recall curve
    _, _, average_precision = metrics.compute_precision_recall_ap_score(plot=True)

    Logger.section_break('DEPLOY COMPLETED')
    model_parameters = filter(lambda p: p.requires_grad,
                              trainer.model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    metrics.print_eval(params, run_time, err, acc, metrics.results_dir,
                       hab_accuracy=hab_acc,
                       mean_class_accuracy=mca_acc,
                       auc_score=auc_score['micro'],
                       average_precision=average_precision['micro'])


if __name__ == '__main__':
    # Initialize Logger
    base = '' if opt.mode != CONST.DEPLOY else '-'+ os.path.basename(
        opt.deploy_data).replace('.csv','')
    log_fname = '{}{}.log'.format(opt.mode, base)
    Logger(log_filename=os.path.join(opt.model_dir, log_fname),
                    level=opt.logging_level, log2file=opt.log2file)
    logger = logging.getLogger('go-train')
    Logger.section_break(f'*** MODE SELECTED: {opt.mode} ***')
    Logger.section_break('User Config')
    logger.info(pformat(opt._state_dict()))

    # Initialize Tensorboard Logger
    if opt.mode == CONST.TRAIN:
        tb_logger = SummaryWriter(opt.model_dir)
    else:
        tb_logger = None

    # Train and evaluate
    if opt.mode == CONST.TRAIN or opt.mode == CONST.VAL:
        coral_train_and_evaluate(opt, logger, tb_logger)
    else:
        deploy(opt, logger)
