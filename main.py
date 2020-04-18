"""Train and Evaluate Model

Script operates by requesting arguments from the user and feeds it into
`train_and_evaluate()`, which acts as the main() of the script.

Execution of the script is largely dependent upon the `--mode` of the model.
`train` will train the model and validate on a subset while `val` will go
through a full evaluation.

If the mode is set to `deploy`, then it will run the script assuming that
the model will be running on a test set (e.g. unseen/unlabelled data)

Logging is heavily incorporated into the script to track and log event
occurrences. If you are unfamiliar with `logging` please look into it.

"""
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

def train_and_evaluate(opt, logger=None, tb_logger=None):
    """ Train and evaluate a model

    The basic understanding of `train_and_evaluate()` can be broken down
    into two parts. Part 1 focuses on getting the dataloaders, model,
    and trainer to conduct the training/evaluation. Part 2.A and 2.B is about
    training or evaluating, respectively.

    Given the mode, train_and_evaluate can take two actions:

    1) mode == TRAIN ---> action: train_and_validate
    2) mode == VAL   ---> action: evaluate the model on the full validation/test set


    Args:
        opt (Config): A state dictionary holding preset parameters
        logger (Logger): Logging instance
        tb_logger (SummaryWriter): Tensorboard logging instance

    Returns:
        None

    """

    #TODO implement Early Stopping
    #TODO implement test code
    
    logger = logger if logger else logging.getLogger('train-and-evaluate')
    logger.setLevel(opt.logging_level)

    # Read in dataset
    # check the path for the data loader to make sure it is loading the right data set
    data_loader = {mode: get_dataloader(data_dir=opt.data_dir,
                                        batch_size=opt.batch_size,
                                        mode=mode) for mode in [CONST.TRAIN, CONST.VAL]}
    # Create model
    Logger.section_break('MODEL ARCHITECTURE')
    model = HABClassifier(arch=opt.arch, pretrained=opt.pretrained, num_classes=opt.class_num)
    logger.debug(model)

    # Initialize Trainer for initializing losses, optimizers, loading weights, etc
    Logger.section_break('MODEL TRAINER')
    trainer = Trainer(model=model, model_dir=opt.model_dir, mode=opt.mode,
                      resume=opt.resume, lr=opt.lr,
                      class_count=data_loader[CONST.TRAIN].dataset.data[CONST.LBL].value_counts())

    #==== BEGIN OPTION 1: TRAINING ====#
    # Train and validate model if set to TRAINING
    # When training, we do both training and validation within the loop.
    # When set to the validation mode, this will run a full evaluation
    # and produce more summarized evaluation results. This is the default condition
    # if the mode is not training.
    if opt.mode == CONST.TRAIN:
        best_err = trainer.best_err
        Logger.section_break('Valid (Epoch {})'.format(trainer.start_epoch))
        err, acc, _, metrics_test = evaluate(trainer.model, trainer, data_loader[CONST.VAL],
                                             0, opt.batch_size, logger, tb_logger, max_iters=None)
        metrics_best = metrics_test

        eps_meter = get_meter(meters=['train_loss', 'val_loss', 'train_acc', 'val_acc'])
        
        for ii, epoch in enumerate(range(trainer.start_epoch,
                                         trainer.start_epoch+opt.epochs)):

            # Train for one epoch
            Logger.section_break('Train (Epoch {})'.format(epoch))
            train_loss, train_acc = train(trainer.model, trainer, data_loader[CONST.TRAIN], epoch, logger,
                                          tb_logger, opt.batch_size, opt.print_freq)
            eps_meter['train_loss'].update(train_loss)
            eps_meter['train_acc'].update(train_acc)
            
            # Evaluate on validation set
            Logger.section_break('Valid (Epoch {})'.format(epoch))
            err, acc, _, metrics_test = evaluate(trainer.model, trainer, data_loader[CONST.VAL],
                                                 epoch, opt.batch_size, logger, tb_logger,
                                                 max_iters=None)
            eps_meter['val_loss'].update(err)
            eps_meter['val_acc'].update(acc)
                
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

        # ==== END OPTION 1: TRAINING LOOP ====#
        # Generate evaluation plots
        opt.train_acc = max(eps_meter['train_acc'].data)
        opt.test_acc = max(eps_meter['val_acc'].data)
        #plot loss over eps
        vis_training(eps_meter['train_loss'].data, eps_meter['val_loss'].data, loss=True)
        #plot acc over eps
        vis_training(eps_meter['train_acc'].data, eps_meter['val_acc'].data, loss=False)
        
        #plot best confusion matrix
        plt.figure()
        metrics_best.compute_cm(plot=True)

    #==== BEGIN OPTION 2: EVALUATION ====#
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
    #==== END OPTION 2: EVALUATION ====#

def train(model, trainer, train_loader, epoch, logger, tb_logger,
    batch_size=opt.batch_size, print_freq=opt.print_freq):
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
    optimizer = trainer.optimizer

    # Initialize meter to bookkeep the following parameters
    meter = get_meter(meters=['batch_time', 'data_time', 'loss', 'acc'])

    # Switch to training mode
    model.train(True)

    end = time.time()
    for i, batch in enumerate(train_loader):
        # process batch items: images, labels
        img = to_cuda(batch[CONST.IMG], trainer.computing_device)
        target = to_cuda(batch[CONST.LBL], trainer.computing_device, label=True)
        id = batch[CONST.ID]

        # measure data loading time
        meter['data_time'].update(time.time() - end)

        # compute output
        end = time.time()
        logits = model(img)
        loss = criterion(logits, target)
        acc = accuracy(logits, target)

        # update metrics
        meter['acc'].update(acc, batch_size)
        meter['loss'].update(loss, batch_size)
        
        # compute gradient and do sgd step
        optimizer.zero_grad()
        loss.backward()

        if i % print_freq == 0:
            log = 'TRAIN [{:02d}][{:2d}/{:2d}] TIME {:10} DATA {:10} ACC {:10} LOSS {:10}'.\
                format(epoch, i, len(train_loader),
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['batch_time']),
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['data_time']),
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['acc']),
                       "{t.val:.3f} ({t.avg:.3f})".format(t=meter['loss'])
                                )
            logger.info(log)

            tb_logger.add_scalar('train/loss', meter['loss'].val, epoch*len(train_loader)+i)
            tb_logger.add_scalar('train/accuracy', meter['acc'].val, epoch*len(train_loader)+i)
            tb_logger.add_scalar('data_time', meter['data_time'].val, epoch*len(train_loader)+i)
            tb_logger.add_scalar('compute_time', meter['batch_time'].val - meter['data_time'].val, epoch*len(train_loader)+i)

        optimizer.step()

        # measure elapsed time
        meter['batch_time'].update(time.time() - end)
        end = time.time()

    tb_logger.add_scalar('train-epoch/loss', meter['loss'].avg, epoch)
    tb_logger.add_scalar('train-epoch/accuracy', meter['acc'].avg, epoch)
    
    return meter['loss'].avg, meter['acc'].avg

    

def evaluate(model, trainer, data_loader, epoch=0,
             batch_size=opt.batch_size, logger=None, tb_logger=None,
             max_iters=None):
    """ Evaluate model

    Similar to `train()` structure, where the function includes bookkeeping
    features and wrapper items. The only difference is that evaluation will
    only occur until the `max_iter` if it is specified and includes an
    `EvalMetrics` intiailization.

    The latter is currrently used to save predictions and ground truths to
    compute the confusion matrix.

    Args:
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
            target = to_cuda(batch[CONST.LBL], trainer.computing_device, label=True)
            id = batch[CONST.ID]

            # compute output
            end = time.time()
            logits = model(img)
            loss = criterion(logits, target)
            acc = accuracy(logits, target)
            batch_size = list(batch[CONST.LBL].shape)[0]
            
            # update metrics
            meter['acc'].update(acc, batch_size)
            meter['loss'].update(loss, batch_size)

            # update metrics2
            metrics.update(logits, target, id)

            # measure elapsed time
            meter['batch_time'].update(time.time() - end, batch_size)

            if i % opt.print_freq == 0:
                log = 'EVAL [{:02d}][{:2d}/{:2d}] TIME {:10} ACC {:10} LOSS {' \
                      ':10}'.format(epoch, i, len(data_loader),
                    "{t.val:.3f} ({t.avg:.3f})".format(t=meter['batch_time']),
                    "{t.val:.3f} ({t.avg:.3f})".format(t=meter['acc']),
                    "{t.val:.3f} ({t.avg:.3f})".format(t=meter['loss'])
                                    )
                logger.info(log)

                if tb_logger is not None:
                    tb_logger.add_scalar('test/loss', meter['loss'].val, epoch)
                    tb_logger.add_scalar('test/accuracy',  meter['acc'].val, epoch)

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
                                 batch_size=opt.batch_size)

    # load model
    model = HABClassifier(arch=opt.arch, pretrained=opt.pretrained, num_classes=opt.class_num)

    # Initialize Trainer for initializing losses, optimizers, loading weights, etc
    trainer = Trainer(model=model, model_dir=opt.model_dir, mode=opt.mode,
                      resume=opt.resume)

    # Run Predictions
    Logger.section_break('Deploy')
    logger.info('Starting deployment...')
    err, acc, run_time, metrics = evaluate(
        model=trainer.model, trainer=trainer, data_loader=data_loader,
        logger=logger, tb_logger=tb_logger)

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
    
    """Argument Parsing"""
    parser = argparse.ArgumentParser("Harmful algae bloom CNN detection model")
    parser.add_argument('--mode', type=str, default=opt.mode)
    parser.add_argument('--interactive', action='store_true', dest=CONST.INTERACTIVE)
    parser.add_argument('--arch', type=str, default=opt.arch)
    parser.add_argument('--model_dir', type=str, default=opt.model_dir)
    parser.add_argument('--data_dir', type=str, default=opt.data_dir)

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=opt.lr)
    parser.add_argument('--epochs', type=int, default=opt.epochs)
    parser.add_argument('--batch_size', '-b', type=int, default=opt.batch_size)
    parser.add_argument('--weighted_loss', dest=CONST.WEIGHTED_LOSS,
                        action='store_true', default=opt.weighted_loss)
    parser.add_argument('--freezed_layers', type=int, default=opt.freezed_layers)
    parser.add_argument('--pretrained', dest=CONST.PRETRAINED, action='store_true',
                        default=opt.pretrained)

    # Model hyperparameters
    parser.add_argument('--input_size', type=int, default=opt.input_size)

    # Training flags
    parser.add_argument('--gpu', '-g', type=str, default=opt.gpu)
    parser.add_argument('--resume', type=str, default=opt.resume)
    parser.add_argument('--print_freq', type=int, default=opt.print_freq)
    parser.add_argument('--save_freq', type=int, default=opt.save_freq)
    parser.add_argument('--log2file', dest=CONST.LOG2FILE, action='store_true',
                        default=opt.log2file)
    parser.add_argument('--logging_level', type=int, default=opt.logging_level)
    parser.add_argument('--save_model_db', dest=CONST.SAVE_MODEL_DB,
                        action='store_true', default=opt.save_model_db)

    # Deploy hyperparameters
    parser.add_argument('--deploy_data', type=str, default=opt.deploy_data)
    parser.add_argument('--lab_config', dest=CONST.LAB_CONFIG, action='store_true',
                        default=opt.lab_config)
    parser.add_argument('--hab_eval', dest=CONST.HAB_EVAL, action='store_true',
                        default=opt.hab_eval)

    # # Parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__
    opt = set_config(**args.__dict__)

    #TODO save json file of config file here ????

    ## Usr Input
    if opt.mode != CONST.DEPLOY or opt.data_dir:
        if opt.interactive:
            date = input('Enter training set name (Baseline: no_green_80_random_spp_combined) : \n')
            opt.data_dir = '/data6/plankton_test_db_new/data/' + date

    if opt.mode != CONST.DEPLOY or opt.resume:
        if opt.interactive:
            resume = input('Do you want to load an existed checkpoint ? (y/n)\n')
            if resume == 'y':
                opt.resume = True
            else:
                opt.resume = False

    sql_yn = opt.sql_yn
    if opt.mode != CONST.DEPLOY or opt.save_model_db:
        if opt.interactive:
            sql_yn = input('Do you want to save model to sql database? (y/n)\n')

            if sql_yn == 'y':
                date = input('Enter today date as model directory name (ex.20190708) : \n')
        else:
            date = datetime.now().strftime("%Y%m%d")

        if sql_yn == 'y' or opt.save_model_db:
            temp_path = os.path.join(opt.model_sql_db, date)
            if not os.path.isdir(temp_path):
                os.mkdir(temp_path)
            opt.model_dir = os.path.join(opt.model_sql_db, date, datetime.now().strftime("%H:%M:%S"))
            os.mkdir(opt.model_dir)
            os.mkdir(os.path.join(opt.model_dir, 'figs'))
    
    # Initialize Logger
    base = '' if opt.mode != CONST.DEPLOY else '-'+ os.path.basename(
        opt.deploy_data).replace('.csv','')
    log_fname = '{}{}.log'.format(opt.mode, base)
    Logger(log_filename=os.path.join(opt.model_dir, log_fname),
                    level=opt.logging_level, log2file=opt.log2file)
    logger = logging.getLogger('go-train')
    Logger.section_break('User Config')
    logger.info(pformat(opt._state_dict()))

    # Initialize Tensorboard Logger
    if opt.mode == CONST.TRAIN:
        tb_logger = SummaryWriter(opt.model_dir)
    else:
        tb_logger = None

    # Train and evaluate
    if opt.mode == CONST.TRAIN or opt.mode == CONST.VAL:
        train_and_evaluate(opt, logger, tb_logger)
    else:
        deploy(opt, logger)

    if opt.save_model_db or sql_yn=='y':
        logger.info('Inserting current model to sql database')
        sql = model_sql()
        sql.save_model()
        sql.close()
