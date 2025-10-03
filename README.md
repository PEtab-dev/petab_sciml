# PEtab SciML
*A data format for scientific machine learning*

PEtab SciML is a table-based data format for creating training (parameter estimation)
problems for **scientific machine learning (SciML)** models that combine machine learning
and mechanistic ordinary differential equation (ODE) models.

**Beta Disclaimer**: this software is under active development and may contain bugs or instabilities. The PEtab SciML format is finalised and support for it has been implemented in PEtab importers, though not yet released.  Documentation and utility functions are currently being added. 

## Highlights

Extending the [PEtab format](https://petab.readthedocs.io) for mechanistic ODE models,
PEtab SciML provides a human readable, reproducible way to specify SciML training problems
across diverse scenarios, in a format directly importable by downstream tools. The main
aspects enabling this are:

- **Flexible hybridization.** Machine learning (ML) and ODE models can be combined in three
  ways: (1) ML within the ODE dynamics (includes **Neural ODEs**), (2) ML in the
  observable/measurement model linking simulations to data, and (3) ML upstream of the ODE,
  mapping high-dimensional inputs (e.g., images) to ODE model parameters.
- **Import across ecosystems.** PEtab SciML problems can be imported into state-of-the-art
  toolboxes for dynamic-model training in Julia
  ([PEtab.jl](https://github.com/sebapersson/PEtab.jl)) and Python/JAX
  ([AMICI](https://github.com/AMICI-dev/AMICI)).
- **Broad support for ML architectures.** A diverse set of ML architectures can be
  specified via an exchangeable PEtab SciML YAML format (supports export from PyTorch
  modules), or via importer-specific libraries (e.g., Lux.jl in PEtab.jl; Equinox in
  AMICI).
- **Diverse model types.** All model features of the
  [PEtab format](https://petab.readthedocs.io) are supported, like models with partial
  observability, multiple simulation conditions, diverse noise models, and/or events.
- **Efficient training strategies.** With minimal user input, PEtab SciML problems can be
  rewritten at the PEtab abstraction level to be compatible with training strategies such as
  multiple shooting, curriculum learning, and regularization (e.g., of ML outputs).
- **Thoroughly tested.** An extensive test suite ensures importers produce correct and
  consistent output.
- **Linting and helpers.** The PEtab SciML Python library provides a linter and utility
  functions for creating common problem types (e.g., Neural ODEs) and transformations (e.g.,
  rewriting a PEtab problem for multiple-shooting training).

## Installation

The PEtab SciML Python3 helper library can be installed with:

```bash
pip install petab-sciml
```

## Documentation

Information on features and tutorials can be found in the online [documentation](https://petab-sciml.readthedocs.io/latest/introduction.html).
