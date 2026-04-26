import tensorflow as tf
from tensorflow import keras


class AdversarialLoss:
    def __init__(self, model: keras.Model):
        self._model = model
        self._loss = model.loss
        self._trades_beta = 0.1
        self._eps = 8/255

    @tf.function
    def loss_clean(self, x: tf.Tensor, x_pert: tf.Tensor, y: tf.Tensor) -> float:
        y_pred = self._model(x_pert, training=True)
        return self._loss(y, y_pred)


    @tf.function
    def _loss_trades_activation(self, x: tf.Tensor, x_pert: tf.Tensor, y: tf.Tensor, eps=8/255):
        model_outputs = self._proxy_classifier(x, training=True)
        model_outputs_pert = self._proxy_classifier(x_pert, training=True)
        loss_clean = self._loss_obj(y, model_outputs)
        loss_pert = self._loss_obj(y, model_outputs_pert)
        loss_kl = keras.losses.KLDivergence(reduction="sum_over_batch_size")(
            tf.clip_by_value(model_outputs, axis=1, clip_value_min=eps,
                             clip_value_max=tf.dtypes.as_dtype(model_outputs.dtype).max),
            model_outputs_pert)
        loss = loss_clean + self._trades_beta * loss_kl
        return loss, model_outputs_pert, loss_pert


    @tf.function
    def _loss_trades_no_activation(self, x: tf.Tensor, x_pert: tf.Tensor, y: tf.Tensor, eps=8/255):
        model_outputs = self._proxy_classifier(x, training=True)
        model_outputs_pert = self._proxy_classifier(x_pert, training=True)
        loss_clean = self._loss_obj(y, model_outputs)
        loss_pert = self._loss_obj(y, model_outputs_pert)
        loss_kl = keras.losses.KLDivergence(reduction="sum_over_batch_size")(
            tf.clip_by_value(tf.nn.softmax(model_outputs, axis=1), clip_value_min=eps,
                             clip_value_max=tf.dtypes.as_dtype(model_outputs.dtype).max),
            tf.nn.softmax(model_outputs_pert, axis=1)
        )
        loss = loss_clean + self._trades_beta * loss_kl
        return loss, model_outputs_pert, loss_pert