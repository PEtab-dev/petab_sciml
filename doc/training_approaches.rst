SciML Training strategies at the PEtab level
============================================

Training (parameter estimating) SciML models can be challenging, and often
standard ML training workflows (e.g., training with Adam for a fixed number of
epochs) fail to find a good minimum or require many training epochs.

Several training strategies have been developed to address this. These include
curriculum learning, multiple shooting, and combined curriculum multiple
shooting, all of which can be implemented at the PEtab abstraction level for
ODE models as well as hybrid PEtab SciML problems. This page describes these
PEtab-level abstractions for tool developers. The PEtab SciML library also
provides reference implementations.

Curriculum learning
-------------------

Curriculum learning is a training strategy where the training problem is
made progressively harder over successive curriculum stages. For PEtab
problems, a curriculum can be defined by gradually increasing the number of
measurement time points (and typically the simulation end time) over a fixed
number of stages. This can be implemented at the PEtab level as follows:

Inputs:

- A PEtab problem (PEtab v1 or v2).
- The number of curriculum stages, ``nStages``.
- A schedule ``n_i`` specifying how many measurements are included in stage
  ``i``.

1. Sort the measurement table in the input PEtab problem by the ``time``
   column.
2. Create ``nStages`` PEtab sub-problems by copying the input problem. For
   stage ``i``, filter the time-sorted measurement table to keep the first
   ``n_i`` measurements.
3. Optionally filter the condition, observable and experiment tables to only
   include entries required by the measurement table for each sub-problem.

A practical consideration for tools implementing and/or importing curriculum
problems is to keep parameter ordering consistent across stages, which
simplifies transferring parameters between stages.

.. _multiple_shooting:

Multiple shooting
-----------------

In multiple shooting, the simulation time span of each PEtab experiment is
split into windows that are fitted jointly. Each window has its own estimated
initial state values, and a continuity penalty is introduced to encourage a
continuous trajectory between adjacent windows. This can be implemented at the
PEtab level as follows:

Inputs:

- A PEtab problem (PEtab v2).
- The number of multiple-shooting windows, ``nWindows``.
- A window partition ``[t0_i, tf_i]`` for each window ``i = 1..nWindows`` such
  that the union of windows covers the full measurement time range, and
  ``t0_i != tf_i`` for all windows.
- A continuity penalty parameter ``lambda``.

1. Copy the input PEtab problem to create a multiple shooting (MS) PEtab
   problem.
2. In the MS PEtab problem, add the penalty weight parameter ``lambda`` to the
   parameter table as a non-estimated parameter and set an appropriate nominal
   value.
3. For each PEtab experiment with ID ``expId`` in the MS PEtab problem:
   1. Create ``nWindows`` new PEtab experiments with IDs ``WINDOW{i}_{expId}``
      and set the initial time to ``t0_i`` for window ``i = 1..nWindows``.
   2. In the experiment table, remove the original experiment IDs and keep
      only the windowed experiments. Assign each PEtab condition to the
      corresponding window experiment. If a PEtab condition occurs at a
      time point that lies in the overlap of windows ``i-1`` and ``i``, assign
      the condition to experiment ``WINDOW{i-1}_{expId}``.
   3. In the measurement table, assign all measurements in the time interval
      ``[t0_i, tf_i]`` for experiment ``expId`` to experiment
      ``WINDOW{i}_{expId}``. If MS windows overlap at time points that contain
      measurements, duplicate those measurements so they appear in each
      relevant window.
   4. For each window ``i > 1`` such that there exists at least one
      measurement for ``expId`` at time ``t >= t0_i`` in the original problem
      (i.e., at least one subsequent window contains measurements), assign
      initial window values and a continuity penalty:
      1. In the parameter table, create parameters
         ``WINDOW{i}_{expId}_init_stateId{j}`` for each model state
         ``stateId{j}``. Mark them as estimated and choose appropriate bounds.
      2. In the condition table, create a condition with ID
         ``WINDOW{i}_{expId}_condition0`` that assigns each ``stateId{j}`` to
         ``WINDOW{i}_{expId}_init_stateId{j}``.
      3. Assign condition ``WINDOW{i}_{expId}_condition0`` as the initial
         condition for experiment ``WINDOW{i}_{expId}`` at time ``t0_i``.
      4. In the observable table, create an observable with ID
         ``WINDOW{i}_{expId}_penalty_stateId{j}`` for each model state
         ``stateId{j}`` and set

         - ``observableFormula = sqrt(lambda) * (stateId{j} - WINDOW{i}_{expId}_init_stateId{j})``
         - ``noiseFormula = 1.0``
         - ``noiseDistribution = normal``

      5. In the measurement table, add a row for experiment
         ``WINDOW{i}_{expId}`` and observable
         ``WINDOW{i}_{expId}_penalty_stateId{j}`` at time ``t0_i`` with
         ``measurement = 0.0``. This yields an L2 (quadratic) penalty.

Naive multiple shooting can perform poorly when states have different scales,
since a single penalty weight may be impossible to tune. In this case, a
log-scale penalty such as

``sqrt(lambda) * (log(abs(stateId{j})) - log(WINDOW{i}_{expId}_init_stateId{j}))``

can be effective, where ``abs`` avoid potential problems with states going
below zero due to numerical errors.

From a runtime performance perspective, the number of initial-window
parameters scales with the number of windows, states, and PEtab experiments,
which can be impractical for larger problems. Moreover, since initial-window
parameters must be estimated, this approach typically performs poorly for
partially observed systems; this is addressed by the curriculum multiple
shooting approach.

Curriculum multiple shooting
----------------------------

Curriculum multiple shooting (CL+MS) combines multiple shooting with a
curriculum schedule. The idea is to start from a multiple-shooting formulation,
which is often easier to train, and then progressively reduce the number of
windows until the original (single-window) problem is recovered. This makes the
approach less sensitive to continuity-penalty tuning and ensures the final
parameters optimize the objective of the original PEtab problem.

Practically, CL+MS defines ``nStages`` curriculum stages. Stage 1 corresponds
to a multiple-shooting problem with ``nWindows = nStages`` windows. In each
subsequent stage, the first ``nWindows-1`` windows are expanded to cover the
union of two adjacent windows, and the last window is dropped. This reduces
the number of windows by one per stage while increasing the time span covered
by each remaining window. The final stage has a single window and corresponds
to the original problem. This can be implemented at the PEtab level as follows:

Inputs:

- A PEtab problem (PEtab v2).
- The number of curriculum stages, ``nStages``.
- An initial window partition ``[t0_i, tf_i]`` for stage 1 with
  ``i = 1..nStages``, such that the union of windows covers the full
  measurement time range and ``t0_i != tf_i`` for all windows.
- A continuity penalty parameter ``lambda`` (used in the multiple-shooting
  stages).

1. Construct stage 1 as a multiple-shooting (MS) PEtab problem with
   ``nWindows = nStages`` using the procedure in
   :ref:`Multiple shooting <multiple_shooting>`.
2. For curriculum stage ``k = 2..(nStages-1)``:
   1. Set the number of windows to ``nWindows = nStages - k + 1``.
   2. Define the MS window time spans for stage ``k`` by merging adjacent
      windows from the previous stage:
      - For ``i = 1..nWindows`` set ``t0_i^{(k)} = t0_i^{(k-1)}`` and
        ``tf_i^{(k)} = tf_{i+1}^{(k-1)}``.
      - Drop the last window of stage ``k-1``.
   3. Create the PEtab problem for stage ``k`` by applying the
      :ref:`Multiple shooting <multiple_shooting>` construction with the
      updated window partition. In particular:

      - Update the experiment table to contain only experiments
        ``WINDOW{i}_{expId}`` for ``i = 1..nWindows``.
      - Reassign and/or duplicate measurements to match
        ``[t0_i^{(k)}, tf_i^{(k)}]``.
        Measurements that in the original problem now appear in multiple
        windows must be duplicated so they appear in each window.
      - Include window-initial parameters and continuity-penalty observables
        for windows ``i > 1`` as in multiple shooting. Note that the penalty
        is applied at the initial time point of each window; in PEtab it is
        not possible to define a continuity penalty over the full overlap
        interval between two windows.

3. The final stage corresponds to the original PEtab problem. Use the parameter
   estimate from stage ``nStages-1`` to initialize optimization for the final
   stage.

A practical consideration for tools implementing and/or importing CL+MS is that
the number of window-initial parameters to estimate changes between stages. To
support transferring parameter values between stages, it can be beneficial to
provide a utility function for mapping parameters between stage problems.

Partitioning measurements and time windows
------------------------------------------

The above training approaches above require either splitting measurements into
curriculum stages (curriculum learning) or partitioning the simulation time
span into windows (multiple shooting and curriculum multiple shooting). We
recommend that tools supporting these methods provide the splitting schemes
outlined below.

For curriculum learning, the number of measurements per stage, ``n_i``, can be
chosen in two ways: (i) split by unique measurement time points and allocate
``n_i`` accordingly, or (ii) split by the total number of measurements, which
can be effective when there are few unique time points but many repeated
measurements. We recommend supporting both modes, as well as automatic
splitting (e.g., given ``nStages``, compute ``n_i`` for the user) and
user-defined schedules (e.g., explicit ``n_i`` per stage or a maximum time
point per stage).

For multiple shooting, window intervals ``[t0_i, tf_i]`` must be defined. We
recommend supporting automatic window construction (e.g., take ``nWindows`` as
input and allocate windows based on unique measurement time points) as well as
user-specified intervals. As a basic sanity check, tools should ensure that
each window contains at least one measurement.
