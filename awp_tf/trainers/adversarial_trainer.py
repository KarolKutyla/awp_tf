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

    def _train_step(self, x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        x_adv = self._attack.generate(x, y)
        validation_metrics = self._validation_metrics(x, y, x_adv)
        with tf.GradientTape() as tape:
            loss = self._robust_loss.calculate(x, y, x_adv, self._classifier, training=True)
        gradient = tape.gradient(loss, self._classifier.trainable_variables)
        self._classifier.optimizer.apply_gradients(zip(gradient, self._classifier.trainable_variables))
        return validation_metrics


    def _validation_metrics(self, x, y, x_adv) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        logits_clean = self._classifier(x, training=False)
        loss_on_clean_examples = self._classifier.loss(y_true=y, y_pred=logits_clean)
        logits_adv = self._classifier(x_adv, training=False)
        loss_on_adversarial_examples = self._classifier.loss(y_true=y, y_pred=logits_adv)
        return loss_on_clean_examples, logits_clean, loss_on_adversarial_examples, logits_adv