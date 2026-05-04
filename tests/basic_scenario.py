import tensorflow
from art.attacks.evasion import ProjectedGradientDescentTensorFlowV2

import numpy as np
from tensorflow import keras
from tensorflow.python.profiler.profiler_v2 import warmup

from actions import models, datasets, attacks

from awp_protocol.attacks import pgd
from awp_protocol import awp

tensorflow.config.run_functions_eagerly(False)
print(tensorflow.executing_eagerly())

train_ds, x_test, y_test, _, _ = datasets.load_mnist_dataset()
model = models.load_tensorflow_resnet()

pgd_params = pgd.get_default_params()
pgd_attack = pgd.PGDAttack(model, pgd_params)

x_batch, y_batch = next(iter(train_ds))
x_adv = pgd_attack.generate(x_batch, y_batch)

tf_evaluation_clean = model.evaluate(x_batch, y_batch)
tf_evaluation_adv = model.evaluate(x_adv, y_batch)
print(tf_evaluation_clean)
print(tf_evaluation_adv)

attacks.show_adversarial_batch(x_batch, x_adv)

input_shape = model.inputs[0].shape[1:]
tfv2_classifier = awp.TensorFlowV2Classifier(model, 10, input_shape=input_shape, loss_object=model.loss)
proxy_model = awp.clone_classifier(model)
tfv2_classifier_proxy = awp.TensorFlowV2Classifier(proxy_model, 10, input_shape=input_shape, loss_object=model.loss)



# attack = ProjectedGradientDescentTensorFlowV2(
#     tfv2_classifier,
#     norm=np.inf,
#     eps=8/255,
#     eps_step=0.01,
#     max_iter=10,
#     targeted=False)

attack = pgd.PGDAttack(proxy_model)

trainer = awp.AdversarialTrainerAWPTensorflow(tfv2_classifier, tfv2_classifier_proxy, attack, warmup=2)
trainer.fit_dataset(train_ds)
