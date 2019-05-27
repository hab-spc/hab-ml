"""Train and Evaluate Model

Script operates by requesting arguments from the user and feeds it into
`train_and_evaluate()`, which acts as the main() of the script.

Execcution of the script is largely dependent upon the `--mode` of the model.
`train` will train the model and validate on a subset while `val` will go
through a full evaluation.

Logging is heavily incorporated into the script to track and log event
occurrences. If you are unfamiliar with `logging` please look into it.

Author: Kevin Le (ktl014)

"""
# Standard dist imports
import argparse
import logging
import os
import sys
#TODO change this to be dynamic
sys.path.insert(0, '/data6/lekevin/hab-master/hab-spc/')
#TODO figure out why module imports aren't working
from pprint import pformat
import time
from datetime import datetime

# Third party imports
from tensorboardX import SummaryWriter
import numpy as np
import torch

# Project level imports
from model import resnet
from trainer import Trainer
from data.dataloader import get_dataloader, to_cuda
from utils.constants import *
from utils.constants import SPCData as sd
from utils.config import opt, set_config
from utils.eval_utils import accuracy, get_meter, EvalMetrics
from utils.logger import Logger

# Module level constants

def train_and_evaluate(opt, logger=None, tb_logger=None):
    """ Train and evaluate a model

    The basic understanding of `train_and_evaluate()` can be broken down
    into two parts. Part 1 focuses on getting the dataloaders, model,
    and trainer to conduct the training/evaluation. Part 2.A and 2.B is about
    training or evaluating, respectively.

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

    # Read in dataset
    # check the path for the data loader to make sure it is loading the right data set
    data_loader = {mode: get_dataloader(opt.data_dir,
                                        batch_size=opt.batch_size,
                                        mode=mode) for mode in [TRAIN, VAL]}

    # Initialize model
    model = resnet.create_model(arch='resnet50', num_classes=2)
    Logger.section_break('Model')
    logger.debug(model)
    fn = 'model/' + opt.arch.split('_')[0] + '.pth'
    print("Name of the pretrained model filename is {}".format(fn))
    model.load_pretrained(fn)
    logger.debug("Loaded imagenet pretrained checkpoint")

    # Initialize Trainer for initializing losses, optimizers, loading weights, etc
    trainer = Trainer(model=model, model_dir=opt.model_dir, mode=opt.mode,
                      resume=opt.resume, lr=opt.lr)

    if opt.mode == TRAIN:
        best_err = trainer.best_err
        Logger.section_break('Valid (Epoch {})'.format(trainer.start_epoch))
        err, acc, _, _ = evaluate(trainer.model, trainer, data_loader[VAL],
                       0, opt.batch_size, logger, tb_logger, max_iters=200)

        for ii, epoch in enumerate(range(trainer.start_epoch,
                                         trainer.start_epoch+opt.epochs)):

            # Train for one epoch
            Logger.section_break('Train (Epoch {})'.format(epoch))
            train(trainer.model, trainer, data_loader[TRAIN], epoch, logger,
                  tb_logger, opt.batch_size, opt.print_freq)

            # Evaluate on validation set
            Logger.section_break('Valid (Epoch {})'.format(epoch))
            err, acc, _, _ = evaluate(trainer.model, trainer, data_loader[VAL],
                                   0, opt.batch_size, logger, tb_logger,
                                   max_iters=200)

            # Remember best error and save checkpoint
            is_best = err < best_err
            best_err = min(err, best_err)
            state = trainer.generate_state_dict(epoch=epoch, best_err=best_err)

            if epoch % opt.save_freq == 0:
                trainer.save_checkpoint(state, is_best=False,
                                        filename='checkpoint-{}_{:0.4f}.pth.tar'.format(
                                            epoch, acc))
            else:
                trainer.save_checkpoint(state, is_best=is_best,
                                        filename='model_best.pth.tar')
    elif opt.mode == VAL:
        err, acc, run_time, metrics = evaluate(
            model=trainer.model, trainer=trainer, data_loader=data_loader[
                opt.mode], logger=logger, tb_logger=tb_logger)

        #TODO write as print_eval()
        Logger.section_break('EVAL COMPLETED')
        model_parameters = filter(lambda p: p.requires_grad,
                                  trainer.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        log = '[PARAMETERS] {params}'.format(params=params)
        logger.info(log)
        log = '[RUN TIME] {time.avg:.3f} sec/sample'.format(time=run_time)
        logger.info(log)
        log = '[FINAL] {name:<30} {loss:.7f}'.format(
            name='{}/{}'.format(opt.mode.upper(), 'crossentropy'), loss=err)
        logger.info(log)
        log = '[FINAL] {name:<30} {acc:.7f}'.format(
            name='{}/{}'.format(opt.mode.upper(), 'accuracy'), acc=acc)
        logger.info(log)
        log = '[FIGS] {}'.format(metrics.results_dir)
        logger.info(log)

        cm = metrics.compute_cm(plot=True)

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
        img = to_cuda(batch[sd.IMG], trainer.computing_device)
        target = to_cuda(batch[sd.LBL], trainer.computing_device, label=True)
        id = batch[sd.ID]

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
            img = to_cuda(batch[sd.IMG], trainer.computing_device)
            target = to_cuda(batch[sd.LBL], trainer.computing_device, label=True)
            id = batch[sd.ID]

            # compute output
            end = time.time()
            logits = model(img)
            loss = criterion(logits, target)
            acc = accuracy(logits, target)

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
    """

    CURRENT: accepts data only from csv file generated through SPICI

    Args:
        opt:
        logger:

    Returns:

    """
    logger = logger if logger else logging.getLogger('deploy')
    start_datetime = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')

    # read data
    data_loader = get_dataloader(mode=DEPLOY, data_dir=opt.deploy_data,
                                batch_size=opt.batch_size,
                                input_size=opt.input_size)

    # load model
    model = resnet.create_model(arch='resnet50', num_classes=2)
    Logger.section_break('Model')
    logger.debug(model)
    fn = 'model/' + opt.arch.split('_')[0] + '.pth'
    print("Name of the pretrained model filename is {}".format(fn))
    model.load_pretrained(fn)
    logger.debug("Loaded imagenet pretrained checkpoint")

    # Initialize Trainer for initializing losses, optimizers, loading weights, etc
    trainer = Trainer(model=model, model_dir=opt.model_dir, mode=opt.mode,
                      resume=opt.resume)

    # return predictions back to image_ids
    _, _, run_time, metrics = evaluate(
        model=trainer.model, trainer=trainer, data_loader=data_loader,
        logger=logger, tb_logger=tb_logger)

    dest_dir = opt.deploy_data + '_static_html' if opt.lab_config else os.path.dirname(opt.deploy_data)
    metrics.save_predictions(start_datetime, run_time.avg, opt.model_dir,
                             dest_dir)

if __name__ == '__main__':
    #TODO write out help description
    """Argument Parsing"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=opt.mode)
    parser.add_argument('--arch', type=str, default=opt.arch)
    parser.add_argument('--model_dir', type=str, default=opt.model_dir)
    parser.add_argument('--data_dir', type=str, default=opt.data_dir)

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=opt.lr)
    parser.add_argument('--epochs', type=int, default=opt.epochs)
    parser.add_argument('--batch_size', '-b', type=int, default=opt.batch_size)

    # Model hyperparameters
    parser.add_argument('--input_size', type=int, default=opt.input_size)

    # Training flags
    parser.add_argument('--gpu', '-g', type=str, default=opt.gpu)
    parser.add_argument('--resume', type=str, default=opt.resume)
    parser.add_argument('--print_freq', type=int, default=opt.print_freq)
    parser.add_argument('--save_freq', type=int, default=opt.save_freq)
    parser.add_argument('--log2file', dest='log2file', action='store_true')

    # Deploy hyperparameters
    parser.add_argument('--deploy_data', type=str, default=opt.deploy_data)
    parser.add_argument('--lab_config', dest='lab_config', action='store_true')



    #TODO add more arguments here

    # # Parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # Example of passing in arguments as the new configurations
    #TODO find more efficient way to pass in arguments into configuration file
    mode = arguments.pop(MODE)
    arch = arguments.pop(ARCH)
    model_dir = arguments.pop(MODEL_DIR)
    data_dir = arguments.pop(DATA_DIR)
    lr = arguments.pop(LR)
    epochs = arguments.pop(EPOCHS)
    batch_size = arguments.pop(BATCH)
    input_size = arguments.pop(INPUT_SIZE)
    gpu = arguments.pop(GPU)
    resume = arguments.pop(RESUME)
    print_freq = arguments.pop(PRINT_FREQ)
    save_freq = arguments.pop(SAVE_FREQ)
    log2file = arguments.pop(LOG2FILE)
    deploy_data = arguments.pop(DEPLOY_DATA)
    lab_config = arguments.pop(LAB_CONFIG)
    opt = set_config(mode=mode, arch=arch, model_dir=model_dir, data_dir=data_dir,
                     lr=lr, epochs=epochs, batch_size=batch_size,
                     input_size=input_size, gpu=gpu, resume=resume,
                     print_freq=print_freq, save_freq=save_freq,
                     log2file=log2file, deploy_data=deploy_data,
                     lab_config=lab_config)

    # Initialize Logger
    Logger(log_filename=os.path.join(opt.model_dir, '{}.log'.format(opt.mode)),
                    level=logging.DEBUG, log2file=opt.log2file)
    logger = logging.getLogger('go-train')
    Logger.section_break('User Config')
    logger.info(pformat(opt._state_dict()))

    # Initialize Tensorboard Logger
    if opt.mode == TRAIN:
        tb_logger = SummaryWriter(opt.model_dir)
    else:
        tb_logger = None

    # Train and evaluate
    if opt.mode == TRAIN or opt.mode == VAL:
        train_and_evaluate(opt, logger, tb_logger)
    else:
        deploy(opt, logger)
