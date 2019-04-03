""" """
# Standard dist imports
import argparse

# Third party imports

# Project level imports
from utils.constants import *
from utils.config import opt, set_config

# Module level constants

def train_and_evaluate(opt):
    """
    #TODO @Sneha

    Args:
        opt:

    Returns:

    """
    """
    Pseudocode:
    
    # Initialize logger

    # Read in dataset using dataloader

    # Initialize model (refer to resnet.py for example)
    
    # Initialize Trainer for initializing losses, optimizers, loading 
    weights, etc.

    # Train model
    if mode == TRAIN:
        evaluate(val_loader...)
        for epoch in epochs:
            train(train_loader...)
            evaluate(val_loader)
            
            # remember best error and save checkpoint every 5 epochs
            best_err = ...
            if epoch % 5 == 0:
                trainer.save_checkpoint
            else:
                trainer.save_checkpoint(is_best=is_best)
    else:
        evaluate(test_loader...)
        # save pretty results to log
        
    """
    pass

def train():
    """
    #TODO @Sneha

    Returns:

    """
    """
    Pseudocode
    for batch in train_loader:
        # process batch items: images, labels
        img = batch[...]
        
        # compute output
        output = model(images)
        loss = ...
        acc = ...
        
        # compute gradient and do sgd step
        optimizer.-----()
        
        # print training updates
    """
    pass

def evaluate():
    """
    #TODO @Sneha

    Returns:

    """
    """
    Pseudocode
    for batch in train_loader:
        # process batch items: images, labels
        img = batch[...]
        
        # compute output
        output = model(images)
        loss = ...
        acc = ...
        
        # print results
        
        # save predictions
    """
    pass

def deploy():
    """
    #TODO @Sneha

    Returns:

    """
    """
    Pseudocode
    # process given input
    input = input...
    
    # compute output
    output = model(input...)
    
    # visualize prediction
    eval_utils.vis_prediction(...) #TODO write this function
    """
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=opt.mode)

    #TODO add more arguments here

    # Parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # Example of passing in arguments as the new configurations
    mode = arguments.pop(MODE)
    opt = set_config(mode=mode)

    # Train and evaluate
    train_and_evaluate(opt)
