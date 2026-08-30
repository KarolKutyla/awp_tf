import tensorflow as tf

from awp_tf.trainers import trainer
from awp_tf.api import awp, layer_scales_selector
from awp_tf.api.awp_params import AWPParams
from awp_tf.attacks.attack import EvasionAttack
from awp_tf.losses.loss import AdversarialLoss

class Trainer(trainer.Trainer):

    def __init__(
            self,
            classifier: tf.keras.Model,
            attack: EvasionAttack,
            adversarial_loss: AdversarialLoss,
            layer_scales: tuple[float, ...] | None = None,
            awp_params=AWPParams(),
    ):
        super().__init__(classifier, attack, adversarial_loss)
        ls = layer_scales
        if ls is None:
            ls = layer_scales_selector.select_evenly(self._classifier)
        self._batch_processor = awp.AWP(self._classifier, self._robust_loss, self._attack, ls, awp_params)

    def _train_batches(self, dataset):
        train_iter = iter(dataset)
        for step, (x_batch, y_batch) in enumerate(train_iter):
            self._run_batch(x_batch, y_batch, step + 1)

    def _run_batch(self, x_batch: tf.Tensor, y_batch: tf.Tensor, step: int):
        self._callback_list.on_batch_begin(step)

        batch_results = self._batch_processor.batch_process(x_batch, y_batch)
        self._update_metrics(y_batch, batch_results)

        self._callback_list.on_batch_end(step, self._collect_train_logs())
