.. _test_suite_benchmarks:

Test suite and benchmark collection
===================================

Two companion resources support tools implementing PEtab SciML:
- a test suite for verifying correct implementation of the format
- a benchmark collection of real-data problems for evaluating methods

Test suite
----------

The `PEtab SciML test suite <https://github.com/PEtab-dev/petab_sciml_testsuite>`__
is a collection of test problems for verifying and quantifying the
capabilities of tools that support PEtab SciML. Each test case provides
reference values (e.g., model outputs, objective function values and
gradients) that an importer should reproduce.

The test cases in the suite are divided into three categories:

- **Machine learning model import**: tests import of
  neural networks defined in the
  :ref:`PEtab SciML neural network YAML format <layers_activation>`,
  covering different layers and activation functions.
- **SciML problem import**: tests import of problems that
  combine machine learning models with the mechanistic model in different
  ways, by comparing the objective function, its gradient and simulated
  outputs.
- **Initialization**: tests that the nominal machine learning
  model parameters of an imported problem are initialized correctly (e.g.,
  when a subset of layers is frozen).

Benchmark collection
--------------------

The `PEtab SciML benchmark collection <https://github.com/sebapersson/Benchmark-Models-PEtab-SciML>`__
is a collection of real-data SciML problems in the PEtab SciML format. The
problems combine mechanistic ODE models and machine learning models, and are
intended to support the development and evaluation of methodology for SciML
problems. Each problem includes expected loss or likelihood values for the
provided parameter values.

The benchmark collection currently contains 2 problems:

- ``Dandekar_Patterns2020``: a universal differential equation (UDE) model.
- ``Ko_NeurIPS2023``: a neural ODE model.

Contributions of new benchmark problems are welcome; see the
`contributing guide <https://github.com/sebapersson/Benchmark-Models-PEtab-SciML/blob/main/CONTRIBUTING.md>`__.
