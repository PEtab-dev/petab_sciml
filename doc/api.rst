PEtab SciML Python API
======================

Standard
--------

Read, write, and validate the PEtab SciML array data and neural network YAML
files.

.. autosummary::
   :toctree: generated

   petab_sciml.standard.ArrayData
   petab_sciml.standard.ArrayDataStandard
   petab_sciml.standard.NNModel
   petab_sciml.standard.NNModelStandard
   petab_sciml.standard.Input
   petab_sciml.standard.add_array_files_to_yaml
   petab_sciml.standard.extract_nn_yaml_parameters
   petab_sciml.standard.extract_torch_parameters

   petab_sciml.standard.nn_model
   petab_sciml.standard.array_data

Problem utilities
-----------------

Helpers for constructing common PEtab SciML problem types from their
components.

.. autosummary::
   :toctree: generated
   :recursive:

   petab_sciml.problem_utils.neural_ode
   petab_sciml.problem_utils
