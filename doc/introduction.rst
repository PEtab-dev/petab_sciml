PEtab SciML — Scientific Machine Learning Format and Tooling
============================================================

PEtab SciML is a table-based data format for creating training (parameter estimation)
problems for **scientific machine learning (SciML)** models that combine machine learning
and mechanistic ordinary differential equation (ODE) models. Extending the
`PEtab format <https://petab.readthedocs.io/>`_ for mechanistic-model training,
PEtab SciML provides an accessible, reproducible data format for specifying the training
loss function across diverse scenarios.

Highlights
----------

Features that enable PEtab SciML to streamline SciML modelling across diverse use cases are:

- **Flexible hybridization.** Machine learning (ML) and ODE models can be combined in three
  ways: (1) ML within the ODE equations, (2) ML in the observable/measurement model
  linking simulations to data, and (3) ML upstream of the ODE, mapping high-dimensional
  inputs (e.g., images) to ODE model parameters.
- **Support across ecosystems.** PEtab SciML problems can be imported into state-of-the-art
  toolboxes for dynamic model training in Julia
  (`PEtab.jl <https://github.com/sebapersson/PEtab.jl>`_) and Python/JAX
  (`AMICI <https://github.com/AMICI-dev/AMICI>`_).
- **Diverse modeling scenarios.** All features of the
  `PEtab format <https://petab.readthedocs.io/>`_ standard are supported, such as
  partially observed systems, multiple simulation conditions, diverse noise models, and
  events/callbacks.
- **Efficient training strategies.** With minimal user input, PEtab SciML problems can be
  rewritten at the PEtab abstraction level to to be compatible with efficient training
  strategies such as multiple shooting, curriculum learning, and regularization
  (e.g., of parameters or ML outputs).
- **Thoroughly tested.** An extensive conformance test suite ensures importers produce
  correct, consistent output .
- **Linting and helpers.** The PEtab SciML Python library provides a linter and utilities
  for creating common problem types (e.g., Neural ODEs) and transformations (e.g.,
  rewriting a PEtab problem for multiple-shooting training).
- **Neural ODEs.** Neural ODEs are supported and benefit from all of the above.


Installation
------------

The PEtab SciML Python helper library can be installed with:

.. code-block:: bash

   pip install petab-sciml


How to read the documentation
-----------------------------

This documentation ranges from first steps to advanced usage. If you are new to PEtab SciML,
start with the Getting Started tutorial <tutorial>. It is required for the How-to guides,
which show different model types (e.g., Neural ODEs, ML components upstream of the ODE,
etc.). For a complete description of every option when creating a SciML problem, see the
Format specification.

Why a SciML data format?
------------------------

There are several technical challenges with training SciML models. First, coding a correct
and performant training objective/loss is non-trivial, especially for more complex models
like those  events or partial observability. Second, efficient training requires gradients
computed via automatic differentiation (AD). Yet writing fast, AD-compatible code in
frameworks like JAX, PyTorch, or the Julia SciML ecosystem can be challenging. Similarly,
porting models between frameworks to leverage framework-specific benefits is time-consuming
because syntax differ. As a result, training setups are often hard-coded to a single
framework. This undermines reusability, both for extending prior work and for creating
benchmark collections that run across toolchains.

All this motivated PEtab SciML. By providing a programming-language-independent,
table-based format, SciML training problems can be specified accessible. A standard format
also allows importers in suitable frameworks (e.g., JAX, Julia) to be engineered for
performance across the entire standard rather than for ad-hoc use cases. A standard format
further enables a conformance test suite to ensure correctness across importers. Together,
these points improve exchangeability, correctness, and overall reproducibility.

Getting Help with and Extending PEtab SciML
-------------------------------------------

If you run into problems:

- Check the Troubleshooting section of the docs.
- If you encounter unexpected behavior or a bug, please open an issue on GitHub.

If PEtab SciML is missing a feature you need, please open an issue on GitHub.
