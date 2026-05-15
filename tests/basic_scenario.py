import tensorflow as tf
from keras.src.optimizers import SGD
from keras.src.optimizers.schedules import learning_rate_schedule

from actions import models, datasets, attacks

from awp_protocol.attacks import pgd
from awp_protocol import awp

from awp_protocol import batch_processor

tf.config.run_functions_eagerly(False)
print(f"tf executing eagerly: {tf.executing_eagerly()}")

train_ds, tf_test_ds = datasets.load_cifar_dataset()
model = models.load_tensorflow_resnet()

pgd_attack = pgd.PGDAttack(model)

x_batch, y_batch = next(iter(train_ds))
x_adv = pgd_attack.generate(x_batch, y_batch)

tf_evaluation_clean = model.evaluate(x_batch, y_batch)
tf_evaluation_adv = model.evaluate(x_adv, y_batch)


labels = {0: "airplane",
1: "automobile",
2: "bird",
3: "cat",
4: "deer",
5: "dog",
6: "frog",
7: "horse",
8: "ship",
9: "truck" }

plotter = attacks.AdversarialPlots(pgd_attack, labels)
plotter.generate_and_show_adversarial_batch(x_batch, y_batch)

# attack = ProjectedGradientDescentTensorFlowV2(
#     tfv2_classifier,
#     norm=np.inf,
#     eps=8/255,
#     eps_step=0.01,
#     max_iter=10,
#     targeted=False)

proxy_model = awp.clone_classifier(model)

params = pgd.PGDParams(perturbation_bound= 8/255, pgd_step=10, pgd_step_size= 2/255)
attack = pgd.PGDAttack(proxy_model, params=params)

protocol_params = batch_processor.AWPParams(alternate_iteration=1, awp_steps=10, weight_constraint=5.03-3)
awp_params = awp.Params(mode="trades", protocol_params=protocol_params)

params = awp.Params(protocol_params=protocol_params)
trainer = awp.AdversarialTrainerAWPTensorflow(model, proxy_model, attack, warmup=0, params=params)

trainer.fit_dataset(train_ds, nb_epochs=4)