.. _layers_activation:

Supported Layers and Activation Functions
=======================================================================

The PEtab SciML NN model YAML format supports numerous
standard neural network layers and activation functions. Layer names and
associated keyword arguments follow the PyTorch naming scheme. PyTorch
is used because it is currently the most popular machine learning
framework, and its comprehensive documentation makes it easy to look up
details for any specific layer or activation function.

If support is lacking for a layer or activation function you would like
to see, please file an issue on
`GitHub <https://github.com/sebapersson/petab_sciml/issues>`__.

The table below lists the supported and tested neural network layers
along with links to their respective PyTorch documentation.
Additionally, the table indicates which tools support each layer.

+--------------------------------------------------------------+----+---+
| layer                                                        | PE | A |
|                                                              | ta | M |
|                                                              | b. | I |
|                                                              | jl | C |
|                                                              |    | I |
+==============================================================+====+===+
| `Linear <https://pytorch.org/do                              | ✔️ |   |
| cs/stable/generated/torch.nn.Linear.html#torch.nn.Linear>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Bilinear <https://pytorch.org/docs/s                        | ✔️ |   |
| table/generated/torch.nn.Bilinear.html#torch.nn.Bilinear>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Flatten <https://pytorch.org/docs                           | ✔️ |   |
| /stable/generated/torch.nn.Flatten.html#torch.nn.Flatten>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Dropout <https://pytorch.org/docs                           | ✔️ |   |
| /stable/generated/torch.nn.Dropout.html#torch.nn.Dropout>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Dropout1d <https://pytorch.org/docs/sta                     | ✔️ |   |
| ble/generated/torch.nn.Dropout1d.html#torch.nn.Dropout1d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Dropout2d <https://pytorch.org/docs/sta                     | ✔️ |   |
| ble/generated/torch.nn.Dropout2d.html#torch.nn.Dropout2d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Dropout3d <https://pytorch.org/docs/sta                     | ✔️ |   |
| ble/generated/torch.nn.Dropout3d.html#torch.nn.Dropout3d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `AlphaDropout <https://pytorch.org/docs/stable/ge            | ✔️ |   |
| nerated/torch.nn.AlphaDropout.html#torch.nn.AlphaDropout>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Conv1d <https://pytorch.org/do                              | ✔️ |   |
| cs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Conv2d <https://pytorch.org/do                              | ✔️ |   |
| cs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Conv3d <https://pytorch.org/do                              | ✔️ |   |
| cs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `ConvTranspose1d <https://pytorch.org/docs/stable/generate   | ✔️ |   |
| d/torch.nn.ConvTranspose1d.html#torch.nn.ConvTranspose1d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `ConvTranspose2d <https://pytorch.org/docs/stable/generate   | ✔️ |   |
| d/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `ConvTranspose3d <https://pytorch.org/docs/stable/generate   | ✔️ |   |
| d/torch.nn.ConvTranspose3d.html#torch.nn.ConvTranspose3d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `MaxPool1d <https://pytorch.org/docs/sta                     | ✔️ |   |
| ble/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `MaxPool2d <https://pytorch.org/docs/sta                     | ✔️ |   |
| ble/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `MaxPool3d <https://pytorch.org/docs/sta                     | ✔️ |   |
| ble/generated/torch.nn.MaxPool3d.html#torch.nn.MaxPool3d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `AvgPool1d <https://pytorch.org/docs/sta                     | ✔️ |   |
| ble/generated/torch.nn.AvgPool1d.html#torch.nn.AvgPool1d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `AvgPool2d <https://pytorch.org/docs/sta                     | ✔️ |   |
| ble/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `AvgPool3d <https://pytorch.org/docs/sta                     | ✔️ |   |
| ble/generated/torch.nn.AvgPool3d.html#torch.nn.AvgPool3d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `LPPool1 <https://pytorch.org/docs/s                         | ✔️ |   |
| table/generated/torch.nn.LPPool1d.html#torch.nn.LPPool1d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `LPPool2 <https://pytorch.org/docs/s                         | ✔️ |   |
| table/generated/torch.nn.LPPool2d.html#torch.nn.LPPool2d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `LPPool3 <https://pytorch.org/docs/s                         | ✔️ |   |
| table/generated/torch.nn.LPPool3d.html#torch.nn.LPPool3d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Ada                                                         | ✔️ |   |
| ptiveMaxPool1d <https://pytorch.org/docs/stable/generated/to |    |   |
| rch.nn.AdaptiveMaxPool1d.html#torch.nn.AdaptiveMaxPool1d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Ada                                                         | ✔️ |   |
| ptiveMaxPool2d <https://pytorch.org/docs/stable/generated/to |    |   |
| rch.nn.AdaptiveMaxPool2d.html#torch.nn.AdaptiveMaxPool2d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Ada                                                         | ✔️ |   |
| ptiveMaxPool3d <https://pytorch.org/docs/stable/generated/to |    |   |
| rch.nn.AdaptiveMaxPool3d.html#torch.nn.AdaptiveMaxPool3d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Ada                                                         | ✔️ |   |
| ptiveAvgPool1d <https://pytorch.org/docs/stable/generated/to |    |   |
| rch.nn.AdaptiveAvgPool1d.html#torch.nn.AdaptiveAvgPool1d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Ada                                                         | ✔️ |   |
| ptiveAvgPool2d <https://pytorch.org/docs/stable/generated/to |    |   |
| rch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `Ada                                                         | ✔️ |   |
| ptiveAvgPool3d <https://pytorch.org/docs/stable/generated/to |    |   |
| rch.nn.AdaptiveAvgPool3d.html#torch.nn.AdaptiveAvgPool3d>`__ |    |   |
+--------------------------------------------------------------+----+---+

Supported Activation Function
-----------------------------

The table below lists the supported and tested activation functions
along with links to their respective PyTorch documentation.
Additionally, the table indicates which tools support each layer.

+--------------------------------------------------------------+----+---+
| Function                                                     | PE | A |
|                                                              | ta | M |
|                                                              | b. | I |
|                                                              | jl | C |
|                                                              |    | I |
+==============================================================+====+===+
| `relu <https://pytorch.org/docs/stable/generate              | ✔️ |   |
| d/torch.nn.functional.relu.html#torch.nn.functional.relu>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `relu6 <https://pytorch.org/docs/stable/generated/           | ✔️ |   |
| torch.nn.functional.relu6.html#torch.nn.functional.relu6>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `hardtanh <https://pytorch.org/docs/stable/generated/torch.  | ✔️ |   |
| nn.functional.hardtanh.html#torch.nn.functional.hardtanh>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `h                                                           | ✔️ |   |
| ardswish <https://pytorch.org/docs/stable/generated/torch.nn |    |   |
| .functional.hardswish.html#torch.nn.functional.hardswish>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `selu <https://pytorch.org/docs/stable/generate              | ✔️ |   |
| d/torch.nn.functional.selu.html#torch.nn.functional.selu>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `leak                                                        | ✔️ |   |
| y_relu <https://pytorch.org/docs/stable/generated/torch.nn.f |    |   |
| unctional.leaky_relu.html#torch.nn.functional.leaky_relu>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `gelu <https://pytorch.org/docs/stable/generate              | ✔️ |   |
| d/torch.nn.functional.gelu.html#torch.nn.functional.gelu>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `tanh                                                        | ✔️ |   |
| shrink <https://pytorch.org/docs/stable/generated/torch.nn.f |    |   |
| unctional.tanhshrink.html#torch.nn.functional.tanhshrink>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `softsign <https://pytorch.org/docs/stable/generated/torch.  | ✔️ |   |
| nn.functional.softsign.html#torch.nn.functional.softsign>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `softplus <https://pytorch.org/docs/stable/generated/torch.  | ✔️ |   |
| nn.functional.softplus.html#torch.nn.functional.softplus>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `tanh <https://pytorch.org/docs/stable/generate              | ✔️ |   |
| d/torch.nn.functional.tanh.html#torch.nn.functional.tanh>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `sigmoid <https://pytorch.org/docs/stable/generated/torc     | ✔️ |   |
| h.nn.functional.sigmoid.html#torch.nn.functional.sigmoid>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `hardsig                                                     | ✔️ |   |
| moid <https://pytorch.org/docs/stable/generated/torch.nn.fun |    |   |
| ctional.hardsigmoid.html#torch.nn.functional.hardsigmoid>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `mish <https://pytorch.org/docs/stable/generate              | ✔️ |   |
| d/torch.nn.functional.mish.html#torch.nn.functional.mish>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `elu <https://pytorch.org/docs/stable/genera                 | ✔️ |   |
| ted/torch.nn.functional.elu.html#torch.nn.functional.elu>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `celu <https://pytorch.org/docs/stable/generate              | ✔️ |   |
| d/torch.nn.functional.celu.html#torch.nn.functional.celu>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `softmax <https://pytorch.org/docs/stable/generated/torc     | ✔️ |   |
| h.nn.functional.softmax.html#torch.nn.functional.softmax>`__ |    |   |
+--------------------------------------------------------------+----+---+
| `log_sof                                                     | ✔️ |   |
| tmax <https://pytorch.org/docs/stable/generated/torch.nn.fun |    |   |
| ctional.log_softmax.html#torch.nn.functional.log_softmax>`__ |    |   |
+--------------------------------------------------------------+----+---+
