import tensorflow as tf

from awp_tf.trainers import trainer

class Trainer(trainer.Trainer):

    def _train_batches(self, dataset):
        train_iter = iter(dataset)
        alt_iter = iter(dataset)
        for step, ((x_batch, y_batch), (x_batch_alt, y_batch_alt)) in enumerate(zip(train_iter, alt_iter)):
            self._run_batch(x_batch, y_batch, x_batch_alt, y_batch_alt, step + 1)

    def _run_batch(self, x_batch: tf.Tensor, y_batch: tf.Tensor, x_batch_alt: tf.Tensor, y_batch_alt: tf.Tensor, step: int):
        self._callback_list.on_batch_begin(step)

        batch_results = self._trainer.awp_train_step(x_batch, y_batch, x_batch_alt, y_batch_alt)
        self._update_metrics(y_batch, batch_results)

        self._callback_list.on_batch_end(step, self._collect_train_logs())


    @tf.function(jit_compile=True)
    def awp_train_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor, x_batch_alt: tf.Tensor, y_batch_alt: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        x_batch_adv = self._attack.generate(x_batch, y_batch)
        x_batch_adv_alt = self._attack.generate(x_batch_alt, y_batch_alt)
        logits_clean = self._classifier(x_batch, training=False)
        clean_loss = self._clean_loss(y_true=y_batch, y_pred=logits_clean)
        logits_adv = self._classifier(x_batch_adv, training=False)
        adv_loss = self._clean_loss(y_true=y_batch, y_pred=logits_adv)

        self._weight_calculator.initiate_state_for_batch_process()
        self._weight_calculator.calculate_weight_perturbation(x_batch, y_batch, x_batch_adv, x_batch_alt, y_batch_alt, x_batch_adv_alt)
        self._weight_calculator.apply_weight_perturbations()

        with tf.GradientTape() as tape:
            robust_loss = self._robust_loss.calculate(x_batch, y_batch, x_batch_adv, self._classifier, training=True)
        gradient = tape.gradient(robust_loss, self._classifier.trainable_variables)
        self._weight_calculator.restore_model()
        self._classifier.optimizer.apply(gradient)
        return clean_loss, logits_clean, adv_loss, logits_adv