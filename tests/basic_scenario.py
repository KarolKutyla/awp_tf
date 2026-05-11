import tensorflow

from actions import models, datasets, attacks

from awp_protocol.attacks import pgd
from awp_protocol import awp

tensorflow.config.run_functions_eagerly(False)
print(f"tf executing eagerly: {tensorflow.executing_eagerly()}")

train_ds, x_test, y_test, _, _ = datasets.load_cifar_dataset()
model = models.load_tensorflow_resnet()

pgd_params = pgd.get_default_params()
pgd_attack = pgd.PGDAttack(model, pgd_params)

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


input_shape = model.inputs[0].shape[1:]


# attack = ProjectedGradientDescentTensorFlowV2(
#     tfv2_classifier,
#     norm=np.inf,
#     eps=8/255,
#     eps_step=0.01,
#     max_iter=10,
#     targeted=False)



proxy_model = awp.clone_classifier(model)
attack = pgd.PGDAttack(proxy_model)

trainer = awp.AdversarialTrainerAWPTensorflow(model, proxy_model, attack, warmup=0)

trainer.fit_dataset(train_ds, nb_epochs=3)