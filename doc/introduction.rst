PEtab SciML
===========

PEtab SciML is a table-based data format for creating training (parameter estimation)
problems for **scientific machine learning (SciML)** models that combine machine learning
and mechanistic ordinary differential equation (ODE) models.

**Beta Disclaimer**: this software is under active development and may contain bugs or 
instabilities. The PEtab SciML format is finalised and support for it has been implemented 
in PEtab importers, though not yet released.  Documentation and utility functions are
currently being added. 

Highlights
----------

Extending the `PEtab format <https://petab.readthedocs.io/>`_ for mechanistic ODE models,
PEtab SciML provides a human readable, reproducible way to specify SciML training problems
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
- **Broad support for ML architectures.** A diverse set of ML architectures can be
  specified via an exchangeable PEtab SciML YAML format (supports export from PyTorch
  modules), or via importer-specific libraries (e.g., Lux.jl in PEtab.jl; Equinox in
  AMICI).
- **Diverse model types.** All model features of the
  `PEtab format <https://petab.readthedocs.io/>`_ are supported, like models with partial
  observability, multiple simulation conditions, diverse noise models, and/or events.
- **Efficient training strategies.** With minimal user input, PEtab SciML problems can be
  rewritten at the PEtab abstraction level to be compatible with training strategies such as
  multiple shooting, curriculum learning, and regularization (e.g., of ML outputs).
- **Thoroughly tested.** An extensive test suite ensures importers produce correct and
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

If you are new to PEtab SciML, start with the :doc:`Getting Started tutorial <tutorial>`.
It is a prerequisite for the How-to guides, which cover different model scenarios (e.g.,
Neural ODEs, ML model upstream of the ODE). For a complete description of all options when
defining a SciML problem, see the :doc:`Format specification <format>`.

Why a SciML data format?
------------------------

There are several technical challenges with training SciML models. Firstly, coding a correct
and performant loss/objective function is non-trivial, especially for more complex scenarios
like models with events/callbacks or partial observability. Secondly, efficient training
requires gradients computed via automatic differentiation (AD). Although frameworks like
JAX, PyTorch and the Julia SciML ecosystem support AD-compatible code, it can still be
challenging to write performant code. Similarly, porting models between frameworks to leverage
framework-specific benefits is time-consuming because each AD backend uses different syntax.
As a result, training setups are often hard-coded to a single framework. This undermines
reusability, both for extending prior work and for creating benchmark collections that can
run across different toolboxes.

PEtab SciML solves these problems by providing a human-writable, tabular,
language-independent specification of the training objective. This further enables generic
importers in different ecosystems (e.g., JAX, Julia SciML) to generate correct, performant
training objectives without ad-hoc, per-project engineering. Together, this improves
exchangeability and overall reproducibility.


Getting Help with and Extending PEtab SciML
-------------------------------------------

If you run into problems:

- Check the :doc:`Troubleshooting <trouble>` section of the documentation.
- If you encounter unexpected behavior or a bug, please open an
  `issue <https://github.com/PEtab-dev/petab_sciml/issues/>`_ on GitHub

If PEtab SciML is missing a feature you need, please open an
`issue <https://github.com/PEtab-dev/petab_sciml/issues/>`_ on GitHub
