Format Specification
====================

A PEtab SciML problem extends the version 2 PEtab standard to accommodate hybrid models
(SciML problems) that combine machine learning (ML) and mechanistic components. In PEtab
SciML, the only supported ML models are neural networks (NNs). Three new file types are
introduced by the extension:

1. :ref:`Neural Network Files <nn_format>`: Files
   describing NN models.
2. :ref:`Hybridization Table <hybrid_table>`: Table for assigning NN
   inputs and outputs across all model simulations.
3. :ref:`Array Data Files <hdf5_array>`: HDF5 files for storing NN
   input data or parameter values.

PEtab SciML further extends the following standard PEtab files:

1. :ref:`Mapping Table <mapping_table>`: Extended to describe how NN
   inputs, outputs and parameters map to PEtab entities.
2. :ref:`Parameters Table <parameter_table>`: Extended to describe
   nominal values for NN parameters.
3. :ref:`Problem YAML File <YAML_file>`: Extended to include a new
   SciML field for NN models and (optionally) array data.

All other PEtab files remain unchanged. This specification explains the
format for each file that is added or modified by the PEtab SciML
extension.

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

1. **Static hybridization**: For each simulation (PEtab experiment),
   inputs are constant and the ML model sets constant parameters and/or
   initial values in the ODE model prior to model simulation.
2. **Dynamic hybridization**: The ML model appears in the ODE
   right-hand-side (RHS) and/or observable formula.

A PEtab SciML problem can also include multiple ML models. Aside from ensuring
that models do not conflict (e.g., by sharing the same output), no special
considerations are required. Each additional ML model is included just as it
would be in the single ML model case.

.. _hybrid_types:

ML Model hybridization
------------------------------------------

PEtab SciML supports two ML model hybridization modes: static and dynamic. This
section explains each mode and its constraints.

Static hybridization
~~~~~~~~~~~~~~~~~~~~

For each simulation (PEtab experiment), inputs are constant and the ML model is
evaluated once before model simulation. Thus, inputs cannot be changed and
outputs cannot be re-evaluated during a simulation. Potential condition-specific
input assignments are therefore only valid for initial PEtab conditions (the first
condition per experiment in the experiment table).

Dynamic hybridization
~~~~~~~~~~~~~~~~~~~~~

The ML model appears in the ODE right-hand side (RHS) and/or observable formula,
for which inputs and outputs are evaluated  at the current time-point. Consequently,
inputs can only be assigned in the hybridization table. Assigning inputs in
the condition table is invalid, since it can change the model structure between
conditions, which is invalid in the PEtab standard. For example, if during a
simulation the ML model input is ``X`` in one condition but ``X + 1`` in another,
the ML model input is effectively embedded differently in the model.

Output variables may be used in observable formulas, or assigned in the
hybridization table to apply across all conditions.

.. _nn_format:

NN Model YAML Format
------------------------------------------

The NN model format is flexible, meaning models can be provided in any
format compatible with the PEtab SciML specification (see below).
Additionally, the ``petab_sciml`` library provides a NN model YAML format that
can be imported by tools across various programming languages.

A NN model must consist of two parts to be compatible with the PEtab
SciML specification:

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
   problem must be defined elsewhere (e.g. in **layers**).

.. tip::

   **Use the NN model YAML format for interoperability**. The NN model specification format
   in PEtab SciML is flexible, to ensure all architectures can be used. However, where
   possible, the NN model YAML format should be used, to facilitate model exchange.


.. _hdf5_array:

Array data
---------------------------------

The standard PEtab format is unsuitable for incorporating large arrays
of values into an estimation problem. This includes the large datasets
used to train NNs, or the parameter values of NNs.

Hence, we provide an HDF5-based file format to store and incorporate this
array data efficiently. Users can choose to provide input data and
parameter values in a single array data file, or to arbitrarily split
them across multiple array data files. The general structure is:

.. code::

   arrays.hdf5                            # (arbitrary filename)
   ├── metadata                           # [GROUP]
   │   └── perm                           # [DATASET, STRING] reserved keyword. "row" for row-major, "column" for column-major
   ├── inputs                             # (optional) [GROUP] reserved keyword
   │   ├── inputId1                       # [GROUP] an input ID
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

The parameters for a single NN model cannot be split across multiple array data files.

NN input data for :ref:`static hybridization <hybrid_types>` can be specified by a
single global array (used for all conditions) or by multiple condition-specific
arrays. In the global case, the dataset name must be ``0``. In the
condition-specific case, the dataset name must be a semicolon-delimited list of
the relevant condition IDs, and datasets may only be provided for initial PEtab
conditions (the first condition per experiment in the experiment table, see
explanation to why :ref:`here <hybrid_condition_table>`).

The schema is provided as a :download:`JSON schema <standard/array_data_schema.json>`.
Currently, validation is only provided via the PEtab SciML library, and does
not check the validity of framework-specific IDs (e.g. for inputs, parameters,
and layers).

The IDs of inputs or layer parameters are framework-specific or
user-specified. For inputs:

-  The PEtab SciML :ref:`NN model YAML format <NN_YAML>` follows
   PyTorch array dimension indexing. For example, if the first layer is
   ``Conv2d``, the input should be in ``(C, W, H)`` format.
-  NN models in other framework-specific formats follow the indexing and
   naming conventions of the respective framework.

For parameters:

-  The PEtab SciML :ref:`NN model YAML format <NN_YAML>` follows
   PyTorch indexing and naming conventions. For example, in a PyTorch
   ``Linear`` layer, the parameter array IDs are ``weight`` and/or
   ``bias``
-  NN models in other framework-specific formats follow the indexing and
   naming conventions of the respective framework.


.. tip::

   **Multiple NNs may share the same input array data**: Like PEtab
   parameters, NN inputs are global variables. Hence, shared input array
   data for multiple NNs can be specified by using the same input ID in each NN.
   Thus, be careful to only intentionally assign multiple inputs the same ID.


.. _NN_YAML:

NN model YAML format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``petab_sciml`` library provides a NN model YAML format for model
exchange. This format follows PyTorch conventions for layer names and
arguments. The schema is provided as a :download:`JSON schema <standard/nn_model_schema.json>`,
which enables validation with various third-party tools, and also as a
:download:`YAML-formatted JSON Schema <standard/nn_model_schema.yaml>` for readability.

.. tip::

   **For users: Define models in PyTorch**. The recommended approach
   to create a NN model YAML file is to first define a PyTorch model
   (``torch.nn.Module``) and use the Python ``petab_sciml`` library to
   export this to YAML. See the "Getting Started" and "How-to" guides for examples
   of this.

.. _mapping_table:

Mapping Table
---------------------------------------

All NNs are assigned an identifier in the PEtab problem
:ref:`YAML <YAML_file>` file. A NN identifier is not considered a
valid PEtab identifier in order to avoid confusion about what it refers to
(e.g. parameters, inputs, outputs). Consequently, every NN input,
parameter, and output referenced in the PEtab problem must be defined
under ``modelEntityId`` and mapped to a PEtab identifier. For the
``petabEntityId`` column the same rules as in PEtab v2 apply.

Detailed Field Description
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``modelEntityId`` [STRING, REQUIRED]: A modeling-language-independent syntax (see below)
   which refers to inputs, outputs, and parameters of NNs.
-  ``petabEntityId`` [STRING, REQUIRED]: Valid PEtab identifier that the
   ``modelEntityId`` maps to.

.. _nn_parameters:

``modelEntityId`` Syntax for Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model ID
``$nnId.parameters{[$layerId]}.{$arrayId}{[$parameterIndex]}}``
(e.g. ``$nnId.parameters[conv1].weight[0]``) refers to
the parameters of a NN identified by ``$nnId``.

-  ``$layerId``: The unique identifier of the layer (e.g., ``conv1``).
-  ``$arrayId``: The parameter array name specific to that layer (e.g.,
   ``weight``).
-  ``$parameterIndex``: The indexing into the parameter array
   (:ref:`syntax <mapping_table_indexing>`).

NN parameter PEtab identifiers can only be referenced in the parameters
table.

.. _nn_inputs:

Inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model ID ``$nnId.inputs{[$inputArgumentIndex]{[$inputIndex]}}``
refers to specific inputs of the NN identified by ``$nnId``.

-  ``$inputArgumentIndex``: The input argument number in the NN forward
   function (uses zero-based indexing).
-  ``$inputIndex`` Indexing into the input argument
   (:ref:`syntax <mapping_table_indexing>`). This should be omitted
   if the input is a file.

For static hybridization, NN input PEtab identifiers are considered
valid PEtab IDs without restrictions (e.g., they may be referenced
in the parameters table, condition table, hybridization table, etc.). For
dynamic hybridization inputs can only be assigned in the
:ref:`hybridization table <hybrid_table>`, as explained
:ref:`here <hybrid_types>`.

Outputs
^^^^^^^

The model ID ``$nnId.outputs{[outputArgumentIndex]{[$outputIndex]}}``
refers to specific outputs of a NN identified by ``$nnId``.

-  ``$outputId``: The output argument number in the NN forward function
   (uses zero-based indexing).
-  ``$outputIndex``: Indexing into the output argument
   (:ref:`syntax <mapping_table_indexing>`)

Nested Identifiers
^^^^^^^^^^^^^^^^^^

The PEtab SciML extension supports nested identifiers for mapping
structured or hierarchical elements. Identifiers are expressed in the
hierarchy indicated above using nested curly brackets. Valid examples
are:

-  ``nn1.parameters``
-  ``nn1.parameters[conv1]``
-  ``nn1.parameters[conv1].weight``

.. warning::

   **Do not break the hierarchy**: Identifiers that break the
   hierarchy (e.g., ``nn1.parameters.weight``) are not valid.

.. _mapping_table_indexing:

Indexing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Indexing into arrays follows the format ``[i0, i1, ...]``, and indexing
notation depends on the NN library:

-  NN models in the PEtab SciML :ref:`NN model YAML
   format <NN_YAML>` follow PyTorch indexing. Consequently,
   indexing is zero-based.
-  NN models in other formats follow the indexing and naming conventions
   of the respective package and programming language.

Assigning Values
^^^^^^^^^^^^^^^^

For assignments to nested PEtab identifiers (in the ``parameters``,
``hybridization``, or ``conditions`` tables), assigned values must
either:

-  Refer to another PEtab identifier with the same nested structure, or
-  Follow the corresponding hierarchical HDF5 structure for
   :ref:`inputs and parameters <hdf5_array>`.

.. _hybrid_table:

Hybridization Table
--------------------------------------------

Assignments of NN inputs and outputs in this table apply to all PEtab conditions.
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
   on hybridization type (see static and dynamic hybridization details below).
   The exact treatment of these entities by importers will also depend on whether
   the ML model is statically or dynamically hybridized.
-  ``targetValue`` [STRING, REQUIRED]: The value or expression that will
   be used to change the target.

Static hybridization
~~~~~~~~~~~~~~~~~~~~

Static hybridization NN inputs and outputs are constant targets, which
are evaluated once before model simulation and do not change over time.

For array inputs, the corresponding input row must be empty, and the input
must be assigned using the :ref:`HDF5 array file format <hdf5_array>`.
Otherwise, inputs and outputs must be specified as described below.

.. _inputs-1:

Inputs
^^^^^^

Valid ``targetValue``\ s are expressions that can be evaluated pre-simulation,
i.e., the expressions may contain parameters that are defined in the parameter
table, but not species or state variables in the ODE model, since even their
initial condition is not available pre-simulation.

.. _outputs-1:

Outputs
^^^^^^^

Valid ``targetId``\ s for an NN output are:

- A non-estimated model parameter.
- The initial value of a species (referenced by the species ID). In this case,
  any other species initialization is overridden.

.. _hybrid_condition_table:
Condition and Hybridization Tables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NN input variables are valid ``targetId``\ s in the condition table. For array
inputs, values must be assigned to specific conditions via the
:ref:`array input file <hdf5_array>`. Regardless of assignment mode, as explained
:ref:`here <hybrid_types>` inputs are only valid initial PEtab conditions
(the first condition in a PEtab experiment).

NN output variables may also appear in the ``targetValue`` column of the
condition table. Here, NN outputs are computed pre-simulation, and are
constant. NN outputs cannot appear in the ``targetValue`` expressions of NN
inputs.

Dynamic hybridization
~~~~~~~~~~~~~~~~~~~~~

Dynamic hybridization NN models depend on model simulated model quantities.

.. _inputs-2:

Inputs
^^^^^^

A valid ``targetValue`` for a NN input is an expression that depends on
model species, time, and/or parameters. Any model species or
parameters in the expression are expected to be evaluated at the given
time-value.

.. _outputs-2:

Outputs
^^^^^^^

A valid ``targetId`` for a NN output is a constant model parameter. During
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

-  ``nominalValue`` [String \| NUMERIC, REQUIRED]: Nominal values for NN parameters.
   If ``estimate = true``, this field can be empty. If ``estimate = false``, a
   nominal value must be provided or specified via an :ref:`array file <hdf5_array>`.
   Valid values are:

   - A numeric value applied to all values under ``parameterId``. If values are
     also provided via an :ref:`array file <hdf5_array>`, the array file is ignored.
   - Empty, in which case values are taken from an :ref:`array file <hdf5_array>`.


-  ``estimate`` [0 \| 1, REQUIRED]: Indicates whether the parameters are
   estimated (``1``) or fixed (``0``). Setting ``0`` for a NN identifier
   (e.g., ``nnId.parameters[layerId]``) freezes the parameters for the
   identifier.

Bounds for NN parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Bounds can be specified for an entire NN or its nested identifiers.
However, most optimization algorithms used for NNs, such as ADAM, do not
support parameter bounds in their standard implementations. Therefore,
NN bounds are optional and default to ``-inf`` for the lower bound and
``inf`` for the upper bound.

Priors for NN parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Priors following the standard PEtab syntax can be specified for an entire NN
or for nested NN identifiers. The prior applies to all values under the
specified identifier.

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
           static: ...       # the hybridization type (bool).
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

-  ``static`` [BOOL, REQUIRED]: The hybridization type
   (see :ref:`hybridization types <hybrid_types>`). ``true`` indicates
   static hybridization; ``false`` indicates dynamic hybridization.

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

- **Respect memory/axis order.**
   For computational efficiency, reorder input data and any layer-parameter arrays to the
   target language’s native indexing and memory layout. For example, PEtab.jl permutes
   image inputs to Julia’s ``(H, W, C)`` convention.

- **Allow export of parameters in PEtab SciML format**.
   If a NN model is not provided in the PEtab SciML YAML format for an importer, HDF5
   parameter files are generally not portable across tools. Because, In that case,
   parameters should follow the importer's framework native indexing and memory layout.
   To enable exchange, it it recommended for importers to implement a function that can
   export NN parameters PEtab SciML array format using PyTorch indexing.
