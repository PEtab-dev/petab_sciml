In the first example (`example.py`) we:
1. create an NN model in pytorch
2. convert that pytorch model into a PEtab SciML `NNModel` and store it to disk (see `data/nn_model0.yaml`)
3. read the model from disk, reconstruct the pytorch model, then convert that reconstructed pytorch model back into an `NNModel`, and store it to disk once more (see `data/nn_model1.yaml`)

In total, this means we do:
```
pytorch model
-> petab sciml NN model
-> petab sciml NN model yaml
-> petab sciml NN model
-> pytorch model
-> petab sciml NN model
-> petab sciml NN model yaml
```
and then verify that the two YAML files match.


# TODO
- [ ] check that the original pytorch forward call provides that same output as the reconstructed pytorch forward call, for some different inputs.
- [ ] the following will have language-specific quirks that are currently not specified by pytorch as some attribute
  - python and julia differ in their flatten commands (one is row-major, the other column-major). For consistency, we only support tensors up to dimension 5, and their order is explicitly Width Height Depth Channel Batch. Larger tensors could be supported, but not with ops like flatten?
  - TODO: get input dimensions from first layer's input dimensions, and then annotate them with W,H,D,C,N up to the number of dimensions they have
