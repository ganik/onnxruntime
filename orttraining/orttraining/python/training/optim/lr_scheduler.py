from . import config

class LRScheduler(object):
    r"""Base class for implementing custom learning rate schedulers

    Once the scheduler is configured, no user code is needed to update learning rate.

    NOTE: This class should never be instantiated, but used as an abstract class.

    Args:
        optimizer_config (optim._OptimizerConfig): optimizer config.
            One of :py:class:`.optim.Adam`, :py:class:`.optim.Lamb` or :py:class:`.optim.SGD`.
    """

    def __init__(self, optimizer_config):
        pass

    def get_lr(self, train_step_info):
        r"""Returns the current learning rate

        Args:
            train_step_info (:py:class:`.TrainStepInfo`): runtime info for current training step

        Returns:
            ordered :py:obj:`list` of learning rates.
                The first entry is the default learning rate and
                    the following, if any, refer to each parameter group.
            NOTE: Currently, only default learning rate is supporte,
                which implies returning a list with size 1.
        """
        raise NotImplementedError


class LinearWarmupLRScheduler(LRScheduler):
    r"""Linear warmup strategy for learning rate update

    Args:
        optimizer_config (optim._OptimizerConfig): optimizer config.
            One of :py:class:`.optim.Adam`, :py:class:`.optim.Lamb` or :py:class:`.optim.SGD`.
        total_steps (int): total training steps for learning.
        warmup (float): portion of total steps for warmup.

    Example:
        .. code-block:: python

            # Initialize optimizer config
            optimizer_config = optim.SGD(lr=0.001)

            # Initialize lr scheduler
            lr_scheduler = LinearWarmupLRScheduler(optimizer_config, total_steps=512, warmup=0.002)

            # Initialize ORTTrainer with lr scheduler
            opts = ORTTrainerOptions({
                lr_scheduler: lr_scheduler
            })
            ort_trainer = ORTTrainer(..., options=opts)

            # Call step() in every batch update
            for inputs in batch_inputs:
                outputs = ort_trainer.train_step(**inputs)
    """

    def __init__(self, optimizer_config, total_steps, warmup):
        assert isinstance(optimizer_config, config._OptimizerConfig),\
            "optimizer_config must be :py:class:`.optim._OptimizerConfig"
        assert isinstance(total_steps, int) and total_steps > 0,\
            "total_steps must be a strict positive number"
        assert isinstance(warmup, int) and warmup >= 0,\
            "warmup must be a positive number"
        assert total_steps > warmup,\
            "total_steps must be greater than warmup"

        self.total_steps = total_steps
        self.warmup = warmup
        self.optimizer_config = optimizer_config

    def _warmup_linear(self, train_step_info):
        x = train_step_info.global_step / self.total_steps
        if x < self.warmup:
            return x / self.warmup
        return max((x - 1.) / (self.warmup - 1.), 0.)

    def get_lr(self, train_step_info):
        lrs_this_step = [group['lr'] * self._warmup_linear(
            train_step_info) for group in self.optimizer_config.param_groups]
        return lrs_this_step
