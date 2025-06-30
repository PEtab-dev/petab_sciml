Format Specification
====================

A PEtab SciML problem extends the PEtab standard version 2 to
accommodate hybrid models (SciML problems) that combine neural network
(NN) and mechanistic components. Three new file types are introduced by
the extension:

1. `Neural Network File(s) <@ref%20nn_format>`__: File(s)
   describing NN model(s).
2. `Hybridization table <@ref%20hybrid_table>`__: Table for assigning NN
   outputs and inputs.
3. `Array Data File(s) <@ref%20hdf5_array>`__: HDF5 files for storing NN
   input data or parameter values.

PEtab SciML further extends the following standard PEtab files:

1. `Mapping Table <@ref%20mapping_table>`__: Extended to describe how NN
   inputs, outputs and parameters map to PEtab entities.
2. `Parameters Table <@ref%20parameter_table>`__: Extended to describe
   nominal values for NN parameters.
3. `Problem YAML File <@ref%20YAML_file>`__: Extended to include a new
   SciML field for NN models and (optionally) array or tensor formatted
   data.

All other PEtab files remain unchanged. This specification explains the
format for each file that is added or modified by the PEtab SciML
extension.

`High Level Overview <@id%20hybrid_types>`__
--------------------------------------------

The PEtab SciML specification is designed to keep the mechanistic model,
NN model, and PEtab problem as independent as possible while linking
them through the hybridization and/or condition tables. In this context,
mechanistic models are typically defined using community standards like
SBML and are commonly simulated as systems of ordinary differential
equations (ODEs), and here the terms mechanistic model and ODE are used
interchangeably. Essentially, the PEtab SciML approach takes a PEtab
problem involving a mechanistic ODE model and supports the integration
of NN inputs and outputs.

PEtab SciML supports two classes of hybrid models:

1. **Static hybridization**: For each experimental/simulation condition,
   inputs are constant and the NN model sets constant parameters and/or
   initial values in the ODE model prior to model simulation.
2. **Dynamic hybridization**: The NN model appears in the ODE
   right-hand-side (RHS) and/or observable formula. Inputs and outputs
   are computed dynamically over the course of a simulation.

A PEtab SciML problem can also include multiple NNs. Aside from ensuring
that NNs do not conflict (e.g., by sharing the same output), no special
considerations are required. Each additional NN is included just as it
would be in the single NN case.

`NN Model YAML Format <@id%20nn_format>`__
------------------------------------------

The NN model format is flexible, meaning models can be provided in any
format compatible with the PEtab SciML specification (TODO see page
about supported formats/supporting frameworks). Additionally,
the ``petab_sciml`` library provides a NN model YAML format that can be
imported by tools across various programming languages.

!!! tip “For everyone: Use the NN model YAML format for
interoperability” The NN model specification format in PEtab SciML is
flexible, to ensure all architectures can be used. However, where
possible, the NN model YAML format should be used, to facilitate model
exchange.

A NN model must consist of two parts to be compatible with the PEtab
SciML specification:

-  **layers**: Defines the NN layers, each with a unique identifier.
-  **forward**: A forward pass function that, given input arguments,
   specifies the order in which layers are called, applies any
   activation functions, and returns one or several arrays. The forward
   function can accept more than one input argument (``n > 1``), and in
   the `mapping table <@ref%20mapping_table>`__, the forward function’s
   ``n``\ th input argument (ignoring any potential class arguments such
   as ``self``) is referred to as ``inputArgumentIndex{n-1}``. Similar
   holds for the output. Aside from the NN output values, every
   component that should be visible to other parts of the PEtab SciML
   problem must be defined elsewhere (e.g., in **layers**).

`Array data <@id%20hdf5_array>`__
---------------------------------

The standard PEtab format is unsuitable for incorporating large arrays
of values into an estimation problem. This includes the large datasets
used to train NNs, or the parameter values of wide or deep NNs.

Hence, we provide a HDF5-based file format to store and incorporate this
array data efficiently. Users can choose to provide input data and
parameter values in a single array data file, or to arbitrarily split
them across multiple array data files. The general structure is

.. code::

   arrays.hdf5                       # (arbitrary filename)
   ├── metadata
   │   └── perm                      # reserved keyword (string). "row" for row-major, "column" for column-major.
   ├── inputs                        # (optional)
   │   ├── inputId1
   │   │   ├─┬─ conditionIds         # (optional) an arbitrary number of PEtab condition IDs (list of string).
   │   │   │ │  ├── conditionId1 
   │   │   │ │  └── ... 
   │   │   │ └── data                # the input data (array).
   │   │   └── ...
   │   └── ...
   └── parameters                    # (optional)
       ├── netId1
       │   ├── layerId1
       │   │   ├── parameterId1      # the parameter values (array).
       │   │   └── ...
       │   └── ...
       └── ...

The schema is provided as `JSON
schema <standard/array_data_schema.json>`__. Currently, validation is only
provided via the PEtab SciML library, and does not check the validity of
framework-specific IDs (e.g. for inputs, parameters, and layers).

!!! tip “Multiple NNs may share the same input array data” Like PEtab
parameters, NN inputs are global variables. Hence, shared input array
data for multiple NNs can be specified by using the same input ID in
each NN. Tools and users should be careful to only intentionally assign
multiple inputs the same ID.

The IDs of inputs or layer parameters are framework-specific or
user-specified. For inputs:

-  The PEtab SciML `NN model YAML format <@ref%20NN_YAML>`__ follows
   PyTorch array dimension indexing. For example, if the first layer is
   ``Conv2d``, the input should be in ``(C, W, H)`` format.
-  NN models in other framework-specific formats follow the indexing and
   naming conventions of the respective framework.

For parameters:

-  The PEtab SciML `NN model YAML format <@ref%20NN_YAML>`__ follows
   PyTorch indexing and naming conventions. For example, in a PyTorch
   ``Linear`` layer, the parameter array IDs are ``weight`` and/or
   ``bias``
-  NN models in other framework-specific formats follow the indexing and
   naming conventions of the respective framework.

!!! tip “For developers: Respect memory order” Tools supporting the
SciML extension should, for computational efficiency, reorder input data
and potential layer parameter arrays to match the memory ordering of the
target language. For example, PEtab.jl converts input data to follow
Julia based indexing.

!!! tip “For developers: Allow export of parameters in PEtab SciML
format” If the NN is not provided in the YAML format, exchange of NN
parameters between software is not possible. To facilitate exchange, it
is recommended that tools supporting PEtab SciML implement a function
capable of exporting to the PEtab SciML format if all layers in the NN
correspond to layers supported by the PEtab SciML NN model YAML format.

.. _nn-model-yaml-format-1:

`NN model YAML format <@id%20NN_YAML>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``petab_sciml`` library provides a NN model YAML format for model
exchange. This format follows PyTorch conventions for layer names and
arguments. The schema is provided as `JSON
schema <standard/nn_model_schema.json>`__, which enables validation with
various third-party tools, and also as `YAML-formatted JSON
Schema <standard/nn_model_schema.yaml>`__ for readability.

!!! tip “For users: Define models in PyTorch” The recommended approach
to create a NN model YAML file is to first define a PyTorch model
(``torch.nn.Module``) and use the Python ``petab_sciml`` library to
export this to YAML. See the tutorials for examples of this.

`Mapping Table <@id%20mapping_table>`__
---------------------------------------

All NNs are assigned an identifier in the PEtab problem
`YAML <@ref%20YAML_file>`__ file. A NN identifier is not considered a
valid PEtab identifier, to avoid confusion about what it refers to
(e.g., parameters, inputs, outputs). Consequently, every NN input,
parameter, and output referenced in the PEtab problem must be defined
under ``modelEntityId`` and mapped to a PEtab identifier. For the
``PEtabEntityId`` column the same rules as in PEtab v2 apply.
Additionally array file IDs defined in the `YAML <@ref%20YAML_file>`__
file are considered valid PEtab entities.

``modelEntityId`` [STRING, REQUIRED]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A modeling-language-independent syntax which refers to inputs, outputs,
and parameters of NNs.

`Parameters <@id%20nn_parameters>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model ID
``$nnId.parameters[$layerId].{[$arrayId]{[$parameterIndex]}}`` refers to
the parameters of a NN identified by ``$nnId``.

-  ``$layerId``: The unique identifier of the layer (e.g., ``conv1``).
-  ``$arrayId``: The parameter array name specific to that layer (e.g.,
   ``weight``).
-  ``$parameterIndex``: The indexing into the parameter array
   (`syntax <@ref%20mapping_table_indexing>`__).

NN parameter PEtab identifiers can only be referenced in the parameters
table.

`Inputs <@id%20nn_inputs>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model ID ``$nnId.inputs{[$inputArgumentIndex]{[$inputIndex]}}``
refers to specific inputs of the NN identified by ``$nnId``.

-  ``$inputArgumentIndex``: The input argument number in the NN forward
   function. Starts from 0.
-  ``$inputIndex`` Indexing into the input argument
   (`syntax <@ref%20mapping_table_indexing>`__). Should not be specified
   if the input is a file.

For `static hybridization <@ref%20hybrid_types>`__ NN input PEtab
identifiers are considered valid PEtab IDs without restrictions (e.g.,
they may be referenced in the parameters table, condition table,
hybridization table, etc.). For `dynamic
hybridization <@ref%20hybrid_types>`__, input PEtab identifiers can only
be assigned an expression in the `hybridization
table <@ref%20hybrid_table>`__.

Outputs
^^^^^^^

The model ID ``$nnId.outputs{[outputArgumentIndex]{[$outputIndex]}}``
refers to specific outputs of a NN identified by ``$nnId``.

-  ``$outputId``: The output argument number in the NN forward function.
   Starts from 0.
-  ``$outputIndex``: Indexing into the output argument
   (`syntax <@ref%20mapping_table_indexing>`__)

Nested Identifiers
^^^^^^^^^^^^^^^^^^

The PEtab SciML extension supports nested identifiers for mapping
structured or hierarchical elements. Identifiers are expressed in the
hierarchical indicated above using nested curly brackets. Valid examples
are:

-  ``nn1.parameters``
-  ``nn1.parameters[conv1]``
-  ``nn1.parameters[conv1].weight``

!!! warn “Do not break the hierarchy” Identifiers that break the
hierarchy (e.g., ``nn1.parameters.weight``) are not valid.

`Indexing <@id%20mapping_table_indexing>`__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Indexing into arrays follows the format ``[i0, i1, ...]``, and indexing
notation depends on the NN library:

-  NN models in the PEtab SciML `NN model YAML
   format <@ref%20NN_YAML>`__ follow PyTorch indexing. Consequently,
   indexing is 0-based.
-  NN models in other formats follow the indexing and naming conventions
   of the respective package and programming language.

Assigning Values
^^^^^^^^^^^^^^^^

For assignments to nested PEtab identifiers (in the ``parameters``,
``hybridization``, or ``conditions`` tables), assigned values must
either:

-  Refer to another PEtab identifier with the same nested structure, or
-  Follow the corresponding hierarchical HDF5
   `input <@ref%20hdf5_input_structure>`__ or
   `parameter <@ref%20hdf5_ps_structure>`__ structure.

`Hybridization Table <@id%20hybrid_table>`__
--------------------------------------------

A tab-separated values file for assigning NN inputs and outputs.
Assignments in the table the table apply to all simulation conditions.
Expected to have, in any order, the following two columns:

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

-  ``targetId`` [NON_ESTIMATED_ENTITY_ID, REQUIRED]: The identifier of
   the non-estimated entity that will be modified. Restrictions depend
   on hybridization type (`static- or dynamic
   hybridization <@ref%20hybrid_types>`__). See below.
-  ``targetValue`` [STRING, REQUIRED]: The value or expression that will
   be used to change the target.

Static hybridization
~~~~~~~~~~~~~~~~~~~~

Static hybridization NN model inputs and outputs are constant targets
(case 1 `here <@ref%20hybrid_types>`__).

.. _inputs-1:

Inputs
^^^^^^

Valid ``targetValue``\ ’s for a NN input are:

-  A parameter in the parameter table.
-  An array input file (assigned an ID in the `YAML problem
   file <@ref%20YAML_file>`__).

.. _outputs-1:

Outputs
^^^^^^^

Valid ``targetId``\ ’s for a NN output are:

-  A non-estimated model parameter.
-  A species’ initial value (referenced by the species’ ID). In this
   case, any other species initialization is overridden.

Condition and Hybridization Tables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NN input variables are valid ``targetId``\ s for the condition table as
long as, following the PEtab standard, they are NON_PARAMETER_TABLE_ID.
**Importantly**, since the hybridization table defines assignments for
all simulation conditions, any ``targetId`` value in the condition table
cannot appear in the hybridization table, and vice versa.

NN output variables can also appear in the ``targetValue`` column of the
condition table.

Dynamic hybridization
~~~~~~~~~~~~~~~~~~~~~

Dynamic hybridization NN models depend on model simulated model
quantities (case 2 `here <@ref%20hybrid_types>`__).

.. _inputs-2:

Inputs
^^^^^^

Valid ``targetValue`` for a NN input is an expression that depend on
model species, time, and/or parameters. Any model species and/or
parameters in the expression are expected to be evaluated at the given
time-value.

.. _outputs-2:

Outputs
^^^^^^^

Valid ``targetId`` for a NN output is a constant model parameter. During
PEtab problem import, any assigned parameters is replaced by the NN
output in the ODE RHS.

`Parameter Table <@id%20parameter_table>`__
-------------------------------------------

The parameter table follows the same format as in PEtab version 2, with
a subset of fields extended to accommodate NN parameters. This section
focuses on columns extended by the SciML extension.

!!! note “Specific Assignments Have Precedence” More specific
assignments (e.g., ``nnId.parameters[layerId]`` instead of
``nnId.parameters``) have precedence for nominal values, priors, and
other setting. For example, if a nominal values is assigned to
``nnId.parameters`` and a different nominal value is assigned to
``nnId.parameters[layerId]``, the latter is used.

.. _detailed-field-description-1:

Detailed Field Description
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``parameterId`` [String, REQUIRED]: The NN or a specific
   layer/parameter array id. The target of the ``parameterId`` must be
   assigned via the `mapping table <@ref%20mapping_table>`__.
-  ``nominalValue`` [String \| NUMERIC, REQUIRED]: NN nominal values.
   This can be:

   -  A PEtab variable that via the problem `YAML
      file <@ref%20YAML_file>`__ corresponds to an HDF5 file with the
      required `structure <@ref%20hdf5_ps_structure>`__. If no file
      exists at the given path when the problem is imported and the
      parameters are set to be estimated, a file is created with
      randomly sampled values. Unless a numeric value is provided,
      referring to the same file is required for all assignments for a
      NN, since all NN parameters should be collected in a single HDF5
      file following the structure described
      `here <@ref%20hdf5_ps_structure>`__.
   -  A numeric value applied to all parameters under ``parameterId``.

-  ``estimate`` [0 \| 1, REQUIRED]: Indicates whether the parameters are
   estimated (``1``) or fixed (``0``).

Bounds for NN parameters
~~~~~~~~~~~~~~~~~~~~~~~~

Bounds can be specified for an entire NN or its nested identifiers.
However, most optimization algorithms used for NNs, such as ADAM, do not
support parameter bounds in their standard implementations. Therefore,
NN bounds are optional and default to ``-inf`` for the lower bound and
``inf`` for the upper bound.

`Problem YAML File <@id%20YAML_file>`__
---------------------------------------

PEtab SciML files are defined within the ``extensions`` section of a
PEtab YAML file, with subsections for neural network models,
hybridization tables, and array files. The general structure is

.. code::

   ...
   extensions:
     petab_sciml:
       version: 1.0.0        # see PEtab extensions spec.
       required: true        # see PEtab extensions spec.
       neural_networks:      # (required)
         netId1:
           location: ...     # location of NN model file (string).
           format: ...       # equinox | lux.jl | yaml
           dynamic: ...      # the hybridization type (bool).
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

``neural_networks`` [REQUIRED]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  The keys (e.g. ``netId1`` in the example above) are the NN model IDs.
-  ``format`` [STRING, REQUIRED]: The format that the NN model is provided in.
   This should be a format supported by one of the frameworks that currently
   implement the PEtab SciML standard (see TODO add page about PEtab.jl and
   AMICI/diffrax). Note that the ``equinox`` and ``lux.jl`` formats are not
   file formats; rather, they indicate that the NN model is specified in a
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
      model YAML format (see `NN model YAML format <@ref%20NN_YAML>`__).

-  ``dynamic`` [BOOL, REQUIRED]: The hybridization type
   (see `hybridization types <@ref%20hybrid_types>`__). ``true`` indicates
   dynamic hybridization; ``false`` indicates static hybridization.
