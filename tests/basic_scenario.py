import tensorflow as tf

from actions import models, datasets, attacks

from awp_protocol.attacks.v1 import pgd
from awp_protocol import awp
from awp_protocol import batch_processor
from awp_protocol.callbacks import checkpoint_callback, epoch_logger

tf.config.run_functions_eagerly(False)

train_ds, tf_test_ds = datasets.load_cifar_dataset()
steps_per_epoch = train_ds.cardinality()
model = models.load_preact_resnet_18(steps_per_epoch)

attack_params = pgd.PGDParams(perturbation_bound=128 / 255, pgd_step=10, pgd_step_size=15 / 255, norm="l2")
pgd_attack = pgd.PGDAttack(model, attack_params)
x_batch, y_batch = next(iter(train_ds))
x_adv = pgd_attack.generate(x_batch, y_batch)
tf_evaluation_clean = model.evaluate(x_batch, y_batch)
tf_evaluation_adv = model.evaluate(x_adv, y_batch)

labels = datasets.load_cifar_labels()
plotter = attacks.AdversarialPlots(pgd_attack, labels)
plotter.generate_and_show_adversarial_batch(x_batch, y_batch)


proxy_model = awp.clone_classifier(model)

attack = pgd.PGDAttack(proxy_model, params=attack_params)

protocol_params = batch_processor.AWPParams(alternate_iteration=1, awp_steps=1, weight_constraint=1.0e-2)
awp_params = awp.Params(mode="trades", protocol_params=protocol_params)

params = awp.Params(protocol_params=protocol_params)
trainer = awp.AdversarialTrainerAWPTensorflow(model, proxy_model, attack, warmup=0, params=params)

save_callback = checkpoint_callback.EpochCheckpoint(f"checkpoints/{proxy_model.name}")
epoch_logger_callback = epoch_logger.EpochLogger(save_filepath=f"logs/{proxy_model.name}/logs.txt", attack_params=attack_params, training_params=awp_params)
callbacks = [save_callback, epoch_logger_callback]

trainer.fit_dataset(train_ds, validation_dataset=tf_test_ds, nb_epochs=200, callbacks=callbacks)