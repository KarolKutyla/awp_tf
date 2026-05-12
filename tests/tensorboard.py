import tensorflow as tf
from keras.src.optimizers import SGD
from keras.src.optimizers.schedules import learning_rate_schedule
from torch._inductor.template_heuristics import params

from actions import models, datasets, attacks

from awp_protocol.attacks import pgd
from awp_protocol import awp

from awp_protocol import awp_protocol_tf

tf.config.run_functions_eagerly(False)
print(f"tf executing eagerly: {tf.executing_eagerly()}")

# train_ds, tf_test_ds, _, _ = datasets.load_cifar_dataset()
model = models.load_tensorflow_resnet()

pgd_attack = pgd.PGDAttack(model)

# x_batch, y_batch = next(iter(train_ds))
# x_adv = pgd_attack.generate(x_batch, y_batch)

# tf_evaluation_clean = model.evaluate(x_batch, y_batch)
# tf_evaluation_adv = model.evaluate(x_adv, y_batch)
#
#
# labels = {0: "airplane",
# 1: "automobile",
# 2: "bird",
# 3: "cat",
# 4: "deer",
# 5: "dog",
# 6: "frog",
# 7: "horse",
# 8: "ship",
# 9: "truck" }

# plotter = attacks.AdversarialPlots(pgd_attack, labels)
# plotter.generate_and_show_adversarial_batch(x_batch, y_batch)


input_shape = model.inputs[0].shape[1:]


# attack = ProjectedGradientDescentTensorFlowV2(
#     tfv2_classifier,
#     norm=np.inf,
#     eps=8/255,
#     eps_step=0.01,
#     max_iter=10,
#     targeted=False)



proxy_model = awp.clone_classifier(model)

params = pgd.PGDParams(pgd_step=1)
attack = pgd.PGDAttack(proxy_model, params=params)

protocol_params = awp_protocol_tf.AWPProtocolParams(awp_steps=1)
params = awp.AWPParams(protocol_params=protocol_params)
trainer = awp.AdversarialTrainerAWPTensorflow(model, proxy_model, attack, warmup=0, params=params)

tensor_func = trainer._train_step