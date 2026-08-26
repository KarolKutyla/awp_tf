import tensorflow as tf

from awp_tf.trainers import trainer

class Trainer(trainer.Trainer):

    def _train_batches(self, dataset):
        for step, (x_batch, y_batch) in enumerate(dataset):
            self._run_batch(x_batch, y_batch, step + 1)

    def _run_batch(self, x_batch: tf.Tensor, y_batch: tf.Tensor, step: int):
        self._callback_list.on_batch_begin(step)

        batch_results = self._train_step(x_batch, y_batch)
        self._update_metrics(y_batch, batch_results)

        self._callback_list.on_batch_end(step, self._collect_train_logs())

    def _train_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            logits = self._classifier(x_batch, training=True)
            loss = self._classifier.loss(y_batch, logits)
        gradient = tape.gradient(loss, self._classifier.trainable_variables)
        self._classifier.optimizer.apply_gradients(zip(gradient, self._classifier.trainable_variables))
        return loss, logits, loss, logits