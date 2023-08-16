"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict
import jax
import jax.numpy as jnp
import optax
from mingpt.utils import CfgNode as CN


class Trainer:
    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = "auto"
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(model)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            # The sampler might need to be adapted for your purposes
            shuffle=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        data_iter = iter(train_loader)

        @jax.jit
        def train_step(params, x, y, opt_state):
            def loss_fn(params):
                logits, loss = model.apply({"params": params}, x, y)
                return loss

            grads = jax.grad(loss_fn)(params)
            grads = jax.tree_map(
                lambda g: jnp.clip(g, -config.grad_norm_clip, config.grad_norm_clip),
                grads,
            )
            updates, opt_state = self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state

        self.iter_num = 0
        self.iter_time = time.time()

        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            x, y = batch

            model.params, self.opt_state = train_step(
                model.params, x, y, self.opt_state
            )

            # Trigger callbacks if you have them
            # self.trigger_callbacks('on_batch_end')

            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
