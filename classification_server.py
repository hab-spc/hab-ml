from __future__ import print_function
import sys
import os
import argparse
import time
import io
import json
import threading
import re

import torch
from PIL import Image

from data.transforms import DATA_TRANSFORMS
from data.dataloader import to_cuda
from utils.logger import init_logger
from utils.config import set_config
from trainer import build_classifier

class Timer:
    def __init__(self, start=False):
        self.stime = -1
        self.prev = -1
        self.times = {}
        if start:
            self.start()

    def start(self):
        self.stime = time.time()
        self.prev = self.stime
        self.times = {}

    def tick(self, name=None, tot=False):
        t = time.time()
        if not tot:
            elapsed = t - self.prev
        else:
            elapsed = t - self.stime
        self.prev = t

        if name is not None:
            self.times[name] = elapsed
        return elapsed

class ServerModelError(Exception):
    pass

class ClassificationServer():
    def __init__(self):
        self.models = {}
        self.next_id = 0

    def start(self, config_file):
        """Read the config file and pre-/load the models"""
        self.config_file = config_file
        with open(self.config_file) as f:
            self.confs = json.load(f)

        self.models_root = self.confs.get('models_root', './available_models')
        for i, conf in enumerate(self.confs["models"]):
            if "models" not in conf:
                if "model" in conf:
                    # backwards compatibility for confs
                    conf["models"] = [conf["model"]]
                else:
                    raise ValueError("""Incorrect config file: missing 'models'
                                        parameter for model #%d""" % i)

            model_id = conf.get("id", None)
            opt = conf["opt"]
            opt["models"] = conf["models"]

            kwargs = {}

            self.preload_model(opt, model_id=model_id, **kwargs)

    def preload_model(self, opt, model_id=None, **model_kwargs):
        """Preloading the model: updating internal datastructure"""
        if model_id is not None:
            if model_id in self.models.keys():
                raise ValueError("Model ID %d already exists" % model_id)
        else:
            model_id = self.next_id
            while model_id in self.models.keys():
                model_id += 1
            self.next_id = model_id + 1
        print("Pre-loading model %d" % model_id)
        model = ServerModel(opt, model_id, **model_kwargs)
        self.models[model_id] = model

        return model_id

    def run(self, inputs):
        """Classify `inputs`
            #TODO specify inputs

        """
        #TODO handle multiple inputs for grabbing model id
        model_id = inputs[0].get("id", 0)
        if model_id in self.models and self.models[model_id] is not None:
            return self.models[model_id].run(inputs)
        else:
            print("Error No such model '%s'" % str(model_id))
            raise ServerModelError("No such model '%s'" % str(model_id))

    def list_models(self):
        """Return the list of available models
        """
        models = []
        for _, model in self.models.items():
            models += [model.to_dict()]
        return models


class ServerModel:
    def __init__(self, opt, model_id, tokenizer_opt=None, load=False,
                 timeout=-1, on_timeout="to_cpu", model_root="./"):
        """
            Args:
                opt: (dict) options for the Translator
                model_id: (int) model id
                tokenizer_opt: (dict) options for the tokenizer or None
                load: (bool) whether to load the model during __init__
                timeout: (int) seconds before running `do_timeout`
                         Negative values means no timeout
                on_timeout: (str) in ["to_cpu", "unload"] set what to do on
                            timeout (see function `do_timeout`)
                model_root: (str) path to the model directory
                            it must contain de model and tokenizer file

        """
        self.model_root = model_root
        self.opt = opt
        self.opt = self.parse_opt(opt)
        self.model_id = model_id
        self.tokenizer_opt = tokenizer_opt
        self.timeout = timeout
        self.on_timeout = on_timeout

        self.unload_timer = None
        self.user_opt = opt
        self.tokenizer = None
        self.logger = init_logger("")
        self.loading_lock = threading.Event()
        self.loading_lock.set()

        if load:
            self.load()

    def parse_opt(self, opt):
        """Parse the option set passed by the user using `onmt.opts`
           Args:
               opt: (dict) options passed by the user

           Returns:
               opt: (Namespace) full set of options for the Translator
        """

        models = opt.pop('models')
        opt = set_config(**opt)
        if not isinstance(models, (list, tuple)):
            models = [models]
        opt.models = [os.path.join(self.model_root, model)
                         for model in models]
        opt.src = "dummy_src"
        opt.cuda = torch.cuda.is_available()

        return opt

    @property
    def loaded(self):
        return hasattr(self, 'classifier')

    def load(self):
        self.loading_lock.clear()

        timer = Timer()
        self.logger.info("Loading model %d" % self.model_id)
        timer.start()

        try:
            self.classifier = build_classifier(self.opt,
                                               report_score=False,
                                               out_file=open(os.devnull, "w"))
        except RuntimeError as e:
            raise ServerModelError("Runtime Error: %s" % str(e))

        timer.tick("model_loading")

        self.load_time = timer.tick()
        self.reset_unload_timer()
        self.loading_lock.set()

    def run(self, inputs):
        """Classify `inputs` using this model

        #TODO Load data preprocessed fashion
        #TODO Get prediction
        #TODO Score the prediction

            Args:
                inputs: [{"src": "..."},{"src": ...}]

            Returns:
                result: (list) translations
                times: (dict) containing times
        """
        self.stop_unload_timer()

        timer = Timer()
        timer.start()
        self.logger.info("Running translation using %d" % self.model_id)

        if not self.loading_lock.is_set():
            self.logger.info(
                "Model #%d is being loaded by another thread, waiting"
                % self.model_id)
            if not self.loading_lock.wait(timeout=30):
                raise ServerModelError("Model %d loading timeout"
                                       % self.model_id)

        else:
            if not self.loaded:
                self.load()
                timer.tick(name="load")
            elif self.opt.cuda:
                self.to_gpu()
                timer.tick(name="to_gpu")

        """PREPROCESS TRANSFORM IMAGE HERE"""
        #TODO be able to handle batched images for preprocessing
        #TODO modularize preprocessing code
        image = Image.open(io.BytesIO(inputs[0]['input']))

        img = DATA_TRANSFORMS[self.opt.mode](image).unsqueeze(0)
        img = to_cuda(img, self.classifier.computing_device)

        imgs = [img]

        scores = []
        predictions = []
        try:
            self.classifier.model.eval()
            outputs = self.classifier.model.forward(img)
            _, y_hat = outputs.max(1)
            predicted_idx = str(y_hat.item())
            predictions.append(predicted_idx)
            scores.append(100)

        except RuntimeError as e:
            raise ServerModelError("Runtime Error: %s" % str(e))

        timer.tick(name="classification")
        self.logger.info("""Using model #%d\t%d inputs
               \tclassification time: %f""" % (self.model_id, len(imgs),
                                            timer.times['classification']))
        self.reset_unload_timer()

        # NOTE: translator returns lists of `n_best` list
        #       we can ignore that (i.e. flatten lists) only because
        #       we restrict `n_best=1`
        # def flatten_list(_list): return sum(_list, [])
        # results = flatten_list([predictions])
        results = predictions

        self.logger.info("Translation Results: %d", len(results))

        # return results, scores, self.opt.n_best, timer.times
        return results, scores, timer.times

    def stop_unload_timer(self):
        if self.unload_timer is not None:
            self.unload_timer.cancel()

    def reset_unload_timer(self):
        if self.timeout < 0:
            return

        self.stop_unload_timer()
        self.unload_timer = threading.Timer(self.timeout, self.do_timeout)
        self.unload_timer.start()

    def to_dict(self):
        hide_opt = ["models"]
        d = {"model_id": self.model_id,
             "opt": {k: self.user_opt[k] for k in self.user_opt.keys()
                     if k not in hide_opt},
             "models": self.user_opt["models"],
             "loaded": self.loaded,
             "timeout": self.timeout,
             }

        return d