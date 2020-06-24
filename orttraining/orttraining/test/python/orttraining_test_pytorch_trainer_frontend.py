import pytest
from numpy.testing import assert_allclose

from onnxruntime.capi.training import optim


def testLRSchedulerWithSGD():
    rtol = 1e-03
    initial_lr = 0.5
    optimizer_config = optim.config.SGD(lr=initial_lr)
    total_steps = 10
    warmup = 5
    lr_scheduler = optim.lr_scheduler.LinearWarmupLRScheduler(optimizer_config,
                                                              total_steps,
                                                              warmup)

    # Initial state
    assert lr_scheduler.optimizer_config == optimizer_config
    assert lr_scheduler.total_steps == total_steps
    assert lr_scheduler.warmup == warmup
    assert_allclose(lr_scheduler.optimizer_config.hyper_parameters['lr'],
                    initial_lr, rtol=rtol, err_msg="lr mismatch")

