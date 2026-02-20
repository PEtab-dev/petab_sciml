Format Specification
====================

A PEtab SciML problem extends the version 2 PEtab standard to accommodate hybrid models
(SciML problems) that combine machine learning (ML) and mechanistic components. In PEtab
SciML, the only supported ML models are neural networks (NNs). Three new file types are
introduced by the extension:

1. :ref:`Neural Network Files <nn_format>`: Files describing NN models.
2. :ref:`Hybridization Table <hybrid_table>`: Table for assigning NN inputs and outputs.
3. :ref:`Array Data Files <hdf5_array>`: HDF5 files for storing NN input data and/or
   parameter values.

PEtab SciML further extends the following standard PEtab files:

1. :ref:`Mapping Table <mapping_table>`: Extended to describe how NN
   inputs, outputs and parameters map to PEtab entities.
2. :ref:`Parameters Table <parameter_table>`: Extended to describe
   nominal values and priors for NN parameters.
3. :ref:`Problem YAML File <YAML_file>`: Extended to include a new
   SciML field for NN models and (optionally) array data.

All other PEtab files remain unchanged. This specification explains the
format for each file that is added or modified by the PEtab SciML
extension.

.. _hybrid_types:

High Level Overview
------------------------------------------

The PEtab SciML specification is designed to keep the mechanistic model,
ML model, and PEtab problem as independent as possible while linking
them through the hybridization and/or condition tables. In this context,
mechanistic models are typically defined using community standards like
SBML and are commonly simulated as systems of ordinary differential
equations (ODEs). In this specification, the terms mechanistic model and ODE are used
interchangeably. Essentially, the PEtab SciML approach takes a PEtab
problem involving a mechanistic model and supports the integration
of ML inputs and outputs.

PEtab SciML supports two classes of hybrid models:

1. **Pre-initialization hybridization**: The ML model is evaluated during the
   pre-initialization stage of each PEtab experiment (as defined in the
   `PEtab v2 specification <https://petab.readthedocs.io/en/latest/v2/documentation_data_format.html#initialization-and-parameter-application>`__
   ). This means ML model inputs are constant, and the ML model assigns
   parameter values and/or initial values in the ODE model prior to model initialization and
   simulation.
2. **Simulation hybridization**: ML inputs and outputs are computed dynamically over the
   course of a PEtab experiment (i.e., during simulation). This means the ML model appears
   in the ODE right-hand side (RHS) and/or in observable formulas.

A PEtab SciML problem can also include multiple ML models. Aside from ensuring
that models do not conflict (e.g., by sharing the same output), no special
considerations are required. Each additional ML model is included just as it
would be in the single ML model case.

.. _nn_format:

NN Model Format
------------------------------------------

The NN model format is flexible, meaning models can be provided in any format compatible
with the PEtab SciML specification. Additionally, the ``petab_sciml`` library provides a
NN model YAML format that can be imported by tools across various programming languages.

Regardless of format, a NN model must consist of two parts to be compatible with PEtab
SciML:

-  **layers**: Defines the NN layers, each with a unique identifier.
-  **forward**: A forward pass function that, given input arguments,
   specifies the order in which layers are called, applies any
   activation functions, and returns one or several arrays. The forward
   function can accept more than one input argument (``n > 1``), and in
   the :ref:`mapping table <mapping_table>`, the forward function’s
   ``n``\ th input argument (ignoring any potential class arguments such
   as ``self``) is referred to as ``inputArgumentIndex{n-1}``. Similar
   holds for the output. Aside from the NN output values, every
   component that should be visible to other parts of the PEtab SciML
   problem must be defined elsewhere (e.g. in **layers**). If input argument
   names can be extracted, they are considered valid PEtab identifiers
   provided they satisfy the PEtab identifier syntax.

.. _NN_YAML:

NN model YAML format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``petab_sciml`` library provides an NN model YAML format for model
exchange. This format follows PyTorch conventions for layer names and
arguments. The schema is provided as a :download:`JSON schema <standard/nn_model_schema.json>`,
which enables validation with various third-party tools, and also as a
:download:`YAML-formatted JSON Schema <standard/nn_model_schema.yaml>` for readability.

.. tip::

   **Use the NN model YAML format for interoperability**. The NN model specification format
   in PEtab SciML is flexible, to ensure all architectures can be used. However, where
   possible, the NN model YAML format should be used, to facilitate model exchange.

.. _hdf5_array:

Array data
---------------------------------

The standard PEtab format is unsuitable for incorporating large arrays
of values into an estimation problem. This includes the large datasets
used to train NNs, or the parameter values of wide or deep NNs. Hence,
PEtab SciML supports an HDF5-based file format to store and incorporate array data
efficiently.

Referencing array data
~~~~~~~~~~~~~~~~~~~~~~

To indicate that a PEtab variable (e.g., NN parameters or an NN input) takes its
values from an array data file, it must be explicitly assigned the reserved
keyword ``array`` in the relevant PEtab table entry.

Semantically, assigning ``array`` is interpreted as a global assignment to an array
variable whose potentially condition-specific values are provided in an array
data file. Therefore, specifying ``array`` is only valid in the
:ref:`hybridization table <_hybrid_table>` and the
:ref:`parameter Table <parameter_table>`, where assignments apply across all
PEtab experiments.

Array data file format
~~~~~~~~~~~~~~~~~~~~~~

Array data must be provided as HDF5. Input data and parameter values may be
stored in a single array data file or split across multiple array data files.
The general structure is:

.. code::

   arrays.hdf5                            # (arbitrary filename)
   ├── metadata                           # [GROUP]
   │   └── perm                           # [DATASET, STRING] reserved keyword. "row" for row-major, "column" for column-major
   ├── inputs                             # (optional) [GROUP] reserved keyword
   │   ├── inputId1                       # [GROUP] an input ID, must be a valid PEtab ID
   │   │   ├── conditionId1;conditionId2  # [DATASET, FLOAT ARRAY] the input data. The name is a semicolon-delimited list of relevant conditions, or "0" for all conditions.
   │   │   ├── conditionId3
   │   │   └── ...
   │   ├── inputId2
   │   │   └── 0                          # Unlike for inputId1, here the condition ID list is "0" to represent all conditions.
   │   └── ...
   └── parameters                    # (optional) [GROUP] reserved keyword
       ├── netId1                    # [GROUP] a NN ID
       │   ├── layerId1              # [GROUP] a layer ID
       │   │   ├── parameterId1      # [DATASET, FLOAT ARRAY] the parameter values
       │   │   └── ...
       │   └── ...
       └── ...

The schema is provided as a :download:`JSON schema <standard/array_data_schema.json>`.
Currently, validation is only provided via the PEtab SciML library and does not
check the validity of framework-specific IDs (e.g., input, parameter, and layer
IDs).

Inputs
^^^^^^

The optional ``inputs`` group stores NN input datasets. For a given input ID,
either a single global dataset (used for all PEtab conditions) or multiple
condition-specific datasets may be provided. In the global case, the dataset
name must be ``0`` (string). In the condition-specific case, the dataset name
must be a semicolon-delimited list of the relevant condition IDs. In either case,
a dataset  must be specified for **all initial PEtab conditions** (the first
condition per PEtab experiment).

The required dataset shape depends on the NN model format:

- If the model is provided in the PEtab SciML :ref:`NN model YAML format <NN_YAML>`,
  datasets must follow the PyTorch dimension ordering. For example, if the first
  layer is ``Conv2d``, the input should be in ``(C, W, H)`` format.
- For NN models in other framework-specific formats, input datasets must follow
  the dimension ordering of the respective framework.

.. tip::

   Multiple NNs may share the same input array data: Like PEtab parameters, NN
   inputs are global variables. Shared input data for multiple NNs can be
   specified by using the same input ID in each NN.

Parameters
^^^^^^^^^^

The optional ``parameters`` group stores NN parameter datasets in a hierarchical
structure: ``parameters/<netId>/<layerId>/<parameterId>``. ``parameterId`` and required
dataset shape depend on the NN model format:

- For NN models in the PEtab SciML :ref:`NN model YAML format <NN_YAML>`, ``parameterId``
  name and dataset dimension ordering follow PyTorch conventions. For example, in a PyTorch
  ``Linear`` layer, ``parameterId``s are ``weight`` and/or ``bias``.
- For NN models in other framework-specific formats, ``parameterId`` and datasets shape
  follow the conventions of the respective framework.

.. _mapping_table:

Mapping Table
---------------------------------------

Each NN is assigned an identifier in the PEtab problem :ref:`YAML file <YAML_file>`.
The NN identifier itself is not a valid PEtab identifier, to avoid ambiguity about
what it refers to (inputs, parameters, outputs). Consequently, every NN input,
parameter, and output referenced in the PEtab problem must be defined under
``modelEntityId`` and mapped to a PEtab identifier in ``petabEntityId``.

An exception applies if the NN model format supports extracting names for
inputs to the forward function. If such input names are valid PEtab
identifiers, they may be used directly as NN input IDs (e.g., for assigning
``array`` in the :ref:`hybridization table <hybrid_table>`). However,
the only way to assign the values of a subset of an input is to first
map the input subset to a new PEtab ID in the
:ref:`mapping table <mapping_table>`.

For ``petabEntityId``, the same rules as in PEtab v2 apply.

``modelEntityId`` syntax
~~~~~~~~~~~~~~~~~~~~~~~~

The valid ``modelEntityId`` syntax depends on whether it refers to NN parameters, inputs,
or outputs.

Parameters
^^^^^^^^^^

For a NN model with ID ``nnId``, a parameter reference has the form
``nnId.parameters[<layerId>].<arrayId>[<parameterIndex>]``:

- ``<layerId>``: Layer identifier (e.g., ``conv1``).
- ``<arrayId>``: Parameter array name (e.g., ``weight``).
- ``<parameterIndex>``: Index into the parameter array (:ref:`Indexing <mapping_table_indexing>`).

NN parameter PEtab identifiers may only be referenced in the parameter table.

Inputs
^^^^^^

For a NN model with ID ``nnId``, an input reference has the form
``nnId.inputs[<inputArgumentIndex>][<inputIndex>]``:

- ``<inputArgumentIndex>``: Input argument index in the NN forward function.
- ``<inputIndex>``: Index into the input argument (:ref:`Indexing <mapping_table_indexing>`).
  Should be omitted if ``[<inputIndex>]`` when the input is provided via an array file.

For restrictions on where NN inputs may be assigned values for different hybridization modes,
see :ref:`Hybridization types <hybrid_types>`.

Outputs
^^^^^^^

For a NN model with ID ``nnId``, an output reference has the form
``nnId.outputs[<outputArgumentIndex>][<outputIndex>]``:

- ``<outputArgumentIndex>``: Output argument index in the NN forward function (zero-based).
- ``<outputIndex>``: Index into the output argument (:ref:`Indexing <mapping_table_indexing>`).

For restrictions on where NN outputs may be assigned for different hybridization modes,
see :ref:`Hybridization types <hybrid_types>`.

.. _mapping_table_indexing:

Indexing
~~~~~~~~

For both NN inputs and outputs, indexing into arrays uses the format ``[i0, i1, ...]`` and
depends on the NN model format:

- Models in the PEtab SciML :ref:`NN model YAML format <NN_YAML>` follow PyTorch
  conventions and use zero-based indexing.
- Models in other formats follow the indexing and naming conventions of the
  respective framework.

.. _hybrid_table:

Hybridization Table
--------------------------------------------

The hybridization table assigns NN inputs and outputs across all PEtab experiments.
The hybridization file is expected to be in tab-separated values format and to have,
in any order, the following two columns:

======================= ===============
**targetId**            **targetValue**
======================= ===============
NON_ESTIMATED_ENTITY_ID MATH_EXPRESSION
nn1_input1              p1
nn1_input2              p1
…                       …
======================= ===============

Detailed Field Description
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``targetId`` [STRING, REQUIRED]: The identifier of
   the non-estimated entity that will be modified. Restrictions depend
   on hybridization type (see pre-initialization and simulation hybridization details below).
-  ``targetValue`` [STRING, REQUIRED]: The value or expression that will
   be used to change the target.

Pre-initialization hybridization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-initialization hybridization NN model inputs and outputs are constant targets.

.. _inputs-1:

Inputs
^^^^^^

Valid ``targetValue``\ s for a NN input are:

-  A parameter in the parameter table.
- ``array`` (values are read from an array data file; see :ref:`Array data <hdf5_array>`)

.. _outputs-1:

Outputs
^^^^^^^

Valid ``targetId``\ s for a NN output are:

-  A non-estimated model parameter.
-  A species’ initial value (referenced by the species’ ID). In this
   case, any other species initialization is overridden.

Condition-specific inputs
^^^^^^^^^^^^^^^^^^^^^^^^^

NN input variables are valid ``targetId``\ s for the condition table as
long as, following the PEtab standard, they are NON_PARAMETER_TABLE_ID.
Similarly, array inputs can be assigned condition-specific values using
the :ref:`Array data <hdf5_array>` format. In both cases, two restrictions
apply. Firstly, values can only be assigned for initial PEtab conditions (the
first condition per PEtab experiment) because, with pre-initialization hybridization,
the NN model is evaluated prior to model initialization and simulation. Assignments
to non-initial conditions are ignored. Secondly, since the hybridization table
defines assignments for all simulation conditions, any ``targetId`` value in
the condition table (or input ID in an array file) cannot appear in the hybridization
table, and vice versa.

NN output variables can also appear in the ``targetValue`` column of the
condition table.

Simulation hybridization
~~~~~~~~~~~~~~~~~~~~~~~~

Simulation hybridization NN models can depend on time-varying ODE model
quantities.

.. _inputs-2:

Inputs
^^^^^^

A valid ``targetValue`` for an NN input is:

- An expression depending on model species, time, and/or parameters. Species and
  parameter references are evaluated at the current simulation time.
- ``array`` (values are read from an array data file; see
  :ref:`Array data <hdf5_array>`). If PEtab condition-specific values are provided,
  the input is updated following the semantics of the PEtab standard, implying input
  values may change during a PEtab experiment.

.. _outputs-2:

Outputs
^^^^^^^

A valid ``targetId`` for a NN output is a constant model parameter. During
PEtab problem import, any assigned parameters are replaced by the NN
A valid ``targetId`` for a NN output is a model parameter. During
PEtab problem import, any assigned parameters are replaced by the NN
output in the ODE RHS.

.. _parameter_table:

Parameter Table
-------------------------------------------

The parameter table follows the same format as in PEtab v2, with
a subset of fields extended to accommodate NN parameters. This section
focuses on columns extended by the SciML extension.

.. note::

   **Specific Assignments Have Precedence**: More specific
   assignments (e.g., ``nnId.parameters[layerId]`` instead of
   ``nnId.parameters``) have precedence for nominal values, priors, and
   other settings. For example, if a nominal values is assigned to
   ``nnId.parameters`` and a different nominal value is assigned to
   ``nnId.parameters[layerId]``, the latter is used.

.. _detailed-field-description-1:

Detailed Field Description
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``parameterId`` [String, REQUIRED]: The NN or a specific
   layer/parameter array id. The target of the ``parameterId`` must be
   assigned via the :ref:`mapping table <mapping_table>`.

-  ``nominalValue`` [``array`` \| NUMERIC, REQUIRED]: Nominal values for NN parameters.
   If ``estimate = true``, this field can be empty. If ``estimate = false``, a
   nominal value must be provided. Valid values are:

   - ``array``, in which case values are taken from an existing :ref:`array file <hdf5_array>`.
   - A numeric value applied to all values under ``parameterId``. If values are
     also provided via an :ref:`array file <hdf5_array>`, the array file is ignored.

-  ``estimate`` [``false`` \| ``true``, REQUIRED]: Indicates whether the parameters are
   estimated (``true``) or fixed (``false``). Setting ``false`` for a NN identifier
   (e.g., ``nnId.parameters[layerId]``) freezes the parameters for the
   identifier.

Bounds for NN parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Parameter bounds can be specified for an entire NN or for nested NN identifiers.
For NN parameters, unbounded estimation is common. Therefore, for NN parameters
``lowerBound`` and ``upperBound`` can be set to ``-inf`` and ``inf`` respectively,
which following the PEtab standard is invalid for other PEtab parameters. The bounds
fields may also be left empty, in which case they default to ``-inf`` and ``inf``.

Priors for NN parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Priors following the standard PEtab syntax can be specified for an entire NN
or for nested NN identifiers. The prior is duplicated for each value under the
specified identifier, it does not specify a joint prior.

In PEtab v2, if any parameter is assigned a prior, all parameters with unassigned
priors are implicitly assigned a ``uniform(lowerBound, upperBound)`` prior. This
also applies to NN parameters. In this case, NN parameters must have finite
``lowerBound`` and ``upperBound`` so that the prior distribution is proper.

.. _YAML_file:

Problem YAML File
---------------------------------------

PEtab SciML files are defined within the ``extensions`` section of a
PEtab YAML file, with subsections for neural network models,
hybridization tables, and array files. The general structure is:

.. code::

   ...
   extensions:
     petab_sciml:
       version: 2.0.0        # see PEtab extensions spec.
       required: true        # see PEtab extensions spec.
       neural_networks:      # (required)
         netId1:
           location: ...     # location of NN model file (string).
           format: ...       # equinox | lux.jl | pytorch | yaml
           pre_initialization: ...       # the hybridization type (bool).
         ...
       hybridization_files:  # (required) list of location of hybridization table files
         - ...
         - ...
       array_files:          # list of location of array HDF5 files
         - ...
         - ...


The location fields (``location``, ``hybridization_files``, ``array_files``)
within this ``petab_sciml`` extension section are the same format as other
location fields in a PEtab v2 problem YAML file.

The ``neural_networks`` section is required and must define the following:

-  The keys (e.g. ``netId1`` in the example above) are the NN model IDs.
-  ``format`` [STRING, REQUIRED]: The format that the NN model is provided in.
   This should be a format supported by one of the frameworks that currently
   implement the PEtab SciML standard. Note that the ``equinox`` and ``lux.jl`` formats
   are not file formats; rather, they indicate that the NN model is specified in a
   programming language with the respective package.

   -  ``equinox``: the file contains an NN model specified in a Python file as
      a subclass of ``equinox.Module`` (see
      `Equinox documentation <https://docs.kidger.site/equinox/>`__).
      The subclass name must be the NN model ID.
   -  ``lux.jl``: the file contains an NN model specified in a Julia file as a
      Lux.jl function
      (see `Lux.jl documentation <https://lux.csail.mit.edu/stable/>`__).
      The function name must be the NN model ID.
   -  ``pytorch``: the file contains an NN model specified in a Python file as a
      subclass of ``torch.nn.Module`` (see
      `PyTorch documentation <https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html#define-the-class>`__).
      The subclass name must be the NN model ID.
   -  ``yaml``: the file contains an NN model specified in the PEtab SciML NN
      model YAML format (see :ref:`NN model YAML format <NN_YAML>`).

-  ``pre_initialization`` [BOOL, REQUIRED]: The hybridization type
   (see :ref:`hybridization types <hybrid_types>`). ``true`` indicates
   pre-initialization hybridization; ``false`` indicates simulation hybridization.

Notes for developers
--------------------

This section outlines recommendations and tips for developers interested in adding PEtab
SciML support to their packages.

Alternative model and neural-network formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both the ODE model and NN formats are flexible. Still, the most widely supported model
format is `SBML <https://sbml.org/>`_, the de facto standard for dynamical models in
computational biology (field of standard developers). We recommend supporting SBML whenever
possible to promote model exchange. Likewise, we recommend supporting the PEtab SciML
NN :ref:`YAML format <NN_YAML>`.

That said, alternative model formats (e.g., `BioNetGen <https://bionetgen.org/>`_) or
language-specific formulations, and alternative NN formats (e.g., architectures not yet
covered by the YAML format), may suit some tools, especially outside biology. The PEtab
SciML standard remains useful across formats by providing a high-level abstraction that
connects the dynamical model and NN components regardless of representation. For example,
leveraging this abstraction, `PEtab.jl <https://github.com/sebapersson/PEtab.jl/>`_
provides a Julia interface to create the PEtab tables and can accept a
`DifferentialEquations.jl <https://diffeq.sciml.ai/>`_ ``ODEProblem`` as the model together
with NNs defined in `Lux.jl <https://lux.csail.mit.edu/>`_. If adding support for other
formats, to **thoroughly test correctness**, the PEtab SciML
`test suite <https://github.com/sebapersson/petab_sciml_testsuite>`_ can be adapted by
replacing the NN and/or model files to match the formats any importer targets.

Dealing with arrays
~~~~~~~~~~~~~~~~~~~

For array handling, it is recommended to:

- **Respect memory layout and dimension ordering.**
  For computational efficiency, reorder input data and layer-parameter
  datasets to the target language’s native memory layout and dimension
  ordering when importing PEtab SciML problems. For example, PEtab.jl
  permutes image inputs to Julia’s ``(H, W, C)``convention instead of using
  the PyTorch ``(C, H, W)`` ordering.

- **Support exporting parameters to the PEtab SciML array format.**
  If a NN model is not provided in the PEtab SciML YAML format, HDF5 parameter
  datasets are generally not portable across tools, since they should follow
  the importer’s framework-native dimension ordering and memory layout. For
  example, highlighting differences in dimension ordering, a PyTorch tensor
  created as ``torch.zeros(2, 3, 3)`` would typically correspond to a Julia
  tensor created as ``zeros(3, 3, 2)``. To enable exchange, we therefore
  recommend that importers provide a utility to export NN parameters to the
  PEtab SciML array format (PyTorch conventions) and document the dimension
  ordering used when exporting arrays.
