PEtab SciML
===========

PEtab SciML is a table-based data format for creating training (parameter estimation)
problems for **scientific machine learning (SciML)** models that combine machine learning
and mechanistic ordinary differential equation (ODE) models.

Highlights
----------

Extending the `PEtab format <https://petab.readthedocs.io/>`_ for mechanistic models,
PEtab SciML provides an accessible, reproducible way to specify SciML training problems
across diverse scenarios, in a format directly importable by downstream tools. The main
aspects enabling this are:

- **Flexible hybridization.** Machine learning (ML) and ODE models can be combined in three
  ways: (1) ML within the ODE dynamics (includes **Neural ODEs**), (2) ML in the
  observable/measurement model linking simulations to data, and (3) ML upstream of the ODE,
  mapping high-dimensional inputs (e.g., images) to ODE model parameters.
- **Import across ecosystems.** PEtab SciML problems can be imported into state-of-the-art
  toolboxes for dynamic-model training in Julia
  (`PEtab.jl <https://github.com/sebapersson/PEtab.jl>`_) and Python/JAX
  (`AMICI <https://github.com/AMICI-dev/AMICI>`_).
- **Diverse model types.** All model features of the
  `PEtab format <https://petab.readthedocs.io/>`_ are supported, like models with partial
  observability, multiple simulation conditions, diverse noise models, and events/callbacks.
- **Efficient training strategies.** With minimal user input, PEtab SciML problems can be
  rewritten at the PEtab abstraction level to be compatible with training strategies such as
  multiple shooting, curriculum learning, and regularization (e.g., of ML outputs).
- **Thoroughly tested.** An extensive test suite ensures importers produce correct,
  consistent output.
- **Linting and helpers.** The PEtab SciML Python library provides a linter and utility
  functions for creating common problem types (e.g., Neural ODEs) and transformations (e.g.,
  rewriting a PEtab problem for multiple-shooting training).

Installation
------------

The PEtab SciML Python3 helper library can be installed with:

.. code-block:: bash

   pip install petab-sciml


How to read the documentation
-----------------------------

If you are new to PEtab SciML, start with the Getting Started tutorial. It is a
prerequisite for the How-to guides, which cover different model scenarios (e.g., Neural
ODEs, ML model upstream of the ODE). For a complete description of all options when
defining a SciML problem, see the Format specification.

Why a SciML data format?
------------------------

There are several technical challenges with training SciML models. First, coding a correct
and performant loss/objective function is non-trivial, especially for more complex models
like those events or partial observability. Second, efficient training requires gradients
computed via automatic differentiation (AD). Yet writing fast, AD-compatible code in
frameworks like JAX, PyTorch, or the Julia SciML ecosystem can be challenging. Similarly,
porting models between frameworks to leverage framework-specific benefits is time-consuming
because syntax and AD framework differ. As a result, training setups are often hard-coded
to a single framework. This undermines reusability, both for extending prior work and for
creating benchmark collections that can run across toolchains.

All this motivated PEtab SciML. By providing a programming-language-independent, accessible,
table-based format, specifying the training objective becomes simpler. A standard
specification also lets importers in suitable frameworks (e.g., JAX, Julia) be engineered
for performance and correctness across the entire standard rather than for ad-hoc use cases.
Together, these points improve exchangeability, correctness, and overall reproducibility.


Getting Help with and Extending PEtab SciML
-------------------------------------------

If you run into problems:

- Check the Troubleshooting section of the docs.
- If you encounter unexpected behavior or a bug, please open an issue on GitHub.

If PEtab SciML is missing a feature you need, please open an issue on GitHub.
