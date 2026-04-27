from tensorflow import keras
import keras_cv

import torch
from torch import nn
import torchvision

def load_tensorflow_resnet():
    backbone = keras_cv.models.ResNet18Backbone(
        include_rescaling=False,
        input_shape=(32, 32, 3)
    )

    x = backbone.outputs[0]
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(10)(x)

    keras_resnet = keras.Model(backbone.inputs, outputs)

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    keras_resnet.compile(loss=loss, optimizer=optimizer)
    return keras_resnet

def load_torch_resnet():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    torch_resnet = torchvision.models.resnet18(weights=None)
    torch_resnet.fc = nn.Linear(torch_resnet.fc.in_features, 10)
    torch_resnet = torch_resnet.to(dtype=torch.float32)
    torch_loss_fn = nn.CrossEntropyLoss()
    torch_optimizer = torch.optim.Adam(
        torch_resnet.parameters(),
        lr=1e-3
    )
    torch_resnet.to(device)
    torch_resnet.train()
    return torch_resnet