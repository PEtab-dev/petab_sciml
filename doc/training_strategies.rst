SciML Training strategies at the PEtab level
============================================

Training (parameter estimating) SciML models can be challenging, and often
standard ML training workflows such as training with Adam for a fixed number of
epochs fail to find a good minimum or require many training epochs.

Several training strategies have been developed to address this. These include
curriculum learning, multiple shooting, and combined curriculum multiple
shooting, all of which can be implemented at the PEtab abstraction level for
mechanistic ODE models as well as hybrid PEtab SciML problems. This page
describes these PEtab-level abstractions for tool developers. The PEtab SciML
library also provides reference implementations supporting both PEtab v2, and
PEtab SciML problems.

Curriculum learning
-------------------

Curriculum learning is a training strategy where the training problem is made
progressively harder over successive curriculum stages. For ODE-based models, a
curriculum can be defined by gradually increasing the measurement time horizon
over a fixed number of stages.

.. container:: figure align-center

   .. image:: assets/cl_illustration.png
      :alt: Hybrid model types
      :class: only-light

   .. image:: assets/cl_illustration_dark.png
      :alt: Hybrid model types
      :class: only-dark

   .. container:: caption text-start

      **Curriculum learning illustration**. Each stage extends the measurement
      time horizon, progressively adding more data points to train on.

Curriculum learning can be implemented at the PEtab level as follows:

Inputs:

- A PEtab problem (PEtab v1 or v2).
- A schedule of stage end times :math:`t_{\text{end},i}` for
  :math:`i = 1, \ldots, n_{\text{stages}}` such that
  :math:`t_{\text{end},1} < t_{\text{end},2} < \cdots < t_{\text{end},n_{\text{stages}}} = \max(\text{measurement times})`.

1. Create :math:`n_{\text{stages}}` PEtab sub-problems by copying the input
   problem. For stage :math:`i`, filter the measurement table to keep only
   measurements with time :math:`t \leq t_{\text{end},i}`.
2. Filter the experiment table to only include experiments referenced by the
   filtered measurement table. Within each remaining experiment, drop periods
   that start after :math:`t_{\text{end},i}`.
3. Filter the condition table to only include conditions referenced by the
   filtered experiment table.
4. Export each curriculum problem to disk, in directories named ``stage{i}``
   for :math:`i = 1, \ldots, n_{\text{stages}}`.

A practical consideration for tools implementing and/or importing curriculum
problems is to ensure that parameters are transferred consistently across
stages in training loops. Although the number of estimated parameters does not
change between stages, different PEtab importers may use different internal
parameter orderings when importing the stage-problems.

.. _multiple_shooting:

Multiple shooting
-----------------

In multiple shooting, the simulation time span of each PEtab experiment is
split into windows that are fitted jointly. Each window has its own estimated
initial state values, and a continuity penalty is introduced to encourage a
continuous trajectory between adjacent windows.

.. container:: figure align-center

   .. image:: assets/ms_illustration.png
      :alt: Hybrid model types
      :class: only-light
      :width: 75%

   .. image:: assets/ms_illustration_dark.png
      :alt: Hybrid model types
      :class: only-dark
      :width: 75%

   .. container:: caption text-start

      **Multiple shooting illustration.** Each window is fitted with its own
      initial state, and a continuity penalty encourages a continuous
      trajectory across window boundaries.


Multiple shooting can be implemented at the PEtab level as follows:

Inputs:

- A PEtab problem (PEtab v2). If the problem has no experiment table, add a
  default experiment to which all measurements are assigned. Problems with
  pre-equilibration are not supported.
- A window partition :math:`[t_{0,i}, t_{f,i}]` for each window such that the
  union of windows covers the full measurement time range and
  :math:`t_{0,i} \neq t_{f,i}` for all windows. Adjacent windows may overlap.
- A continuity penalty parameter :math:`\lambda`.
- An initial guess :math:`v_0` for each estimated window initial state.

1. Copy the input PEtab problem to create a multiple shooting (MS) PEtab
   problem. For this problem, create empty measurement, experiment and
   condition tables.
2. Add a non-estimated parameter ``MS_PENALTY_SQRT`` to the parameter table
   with nominal value :math:`\sqrt{\lambda}`.
3. For each PEtab experiment with ID ``expId`` in the MS PEtab problem and
   each window :math:`i = 1, \ldots, n_{\text{windows}}`:

   1. If the maximum measurement time of ``expId`` in the original problem is
      strictly less than :math:`t_{0,i}`, skip this window for this experiment
      (no PEtab experiment is created and no measurements, parameters,
      observables, or conditions are added). Otherwise, create a new PEtab
      experiment with ID ``WINDOW{i}_EXPERIMENT_{expId}``.
   2. Build the PEtab conditions of ``WINDOW{i}_EXPERIMENT_{expId}``:

      - For window :math:`i = 1`, keep original conditions of ``expId`` that
        fall in :math:`[t_{0,1}, t_{f,1}]`. If no condition is applied at
        :math:`t_{0,1}`, add a condition applied at :math:`t_{0,1}` with
        ``conditionId`` so the simulation starts with the original PEtab
        problem initialization.
      - For windows :math:`i > 1`, add a leading period at :math:`t_{0,i}`
        with the window's IC condition (defined below). Keep original
        conditions that are applied in :math:`[t_{0,i}, t_{f,i}]`.

   3. Assign all measurements of ``expId`` in :math:`[t_{0,i}, t_{f,i}]` to
      experiment ``WINDOW{i}_EXPERIMENT_{expId}``. Measurements at exactly
      the boundary between two adjacent windows are duplicated so they
      appear in both windows.
   4. If :math:`i > 1`, add per-experiment window initial values and
      continuity penalty:

      a. In the parameter table, create parameters
         ``WINDOW{i}_EXPERIMENT_{expId}_PARAMETER_{stateId}`` for each model
         state ``stateId``. Mark them as estimated, give them appropriate
         bounds, and use :math:`v_0` as the nominal value.
      b. In the condition table, create a condition with ID
         ``WINDOW{i}_EXPERIMENT_{expId}_IC`` that assigns each ``stateId`` to
         ``WINDOW{i}_EXPERIMENT_{expId}_PARAMETER_{stateId}``.
      c. In the observable table, create an observable with ID
         ``WINDOW{i}_EXPERIMENT_{expId}_PENALTY_{stateId}`` for each model
         state ``stateId`` and set

         - ``observableFormula = (stateId - WINDOW{i}_EXPERIMENT_{expId}_PARAMETER_{stateId}) * MS_PENALTY_SQRT``
         - ``noiseFormula = 1.0``
         - ``noiseDistribution = normal``

      d. In the measurement table, add a row for experiment
         ``WINDOW{i-1}_EXPERIMENT_{expId}`` and observable
         ``WINDOW{i}_EXPERIMENT_{expId}_PENALTY_{stateId}`` at time
         :math:`t_{0,i}` with ``measurement = 0.0``. This yields a quadratic
         (L2) penalty evaluated where the simulated trajectory of window
         :math:`i-1` meets the estimated initial state of window :math:`i`.

Note that all artifacts in step 3-4 are added per ``(window, experiment)``pair
rather than globally per window, since trajectories differ between experiments.
``MS_PENALTY_SQRT`` is added once and shared across all experiments and
windows.

Naive multiple shooting can perform poorly when states have different scales,
since a single penalty weight may be impossible to tune. In this case, a
log-scale penalty such as

``(log(abs(stateId)) - log(abs(WINDOW{i}_EXPERIMENT_{expId}_PARAMETER_{stateId}))) * MS_PENALTY_SQRT``

can be effective, where ``abs`` avoids potential problems with states going
below zero due to numerical errors.

Curriculum multiple shooting
----------------------------

Curriculum multiple shooting (CMS) combines multiple shooting with a
curriculum schedule. The idea is to start from a multiple-shooting formulation,
which is often easier to train, and then progressively reduce the number of
windows until the original (single-window) problem is recovered.

.. container:: figure align-center

   .. image:: assets/cms_illustration.png
      :alt: Hybrid model types
      :class: only-light

   .. image:: assets/cms_illustration_dark.png
      :alt: Hybrid model types
      :class: only-dark

   .. container:: caption text-start

      **Curriculum multiple shooting illustration.** Each stage is a multiple
      shooting problem with progressively fewer windows, until the final stage
      recovers the original problem.

CMS defines :math:`n_{\text{stages}}` curriculum stages. Stage 1 is a
multiple-shooting problem with :math:`n_{\text{stages}}` windows. At each
subsequent stage the last window is dropped and every remaining window's end
is shifted one position to the right; equivalently, window :math:`i` at stage
:math:`k` is :math:`[t_{0,i}, t_{f,i+k-1}]`. Each stage therefore has one fewer
window than the previous, with each remaining window covering more of the time
range. The final stage is a single window covering
:math:`[t_{0,1}, t_{f,n_{\text{stages}}}]` and corresponds to the original
problem. Stages 2 onwards have overlapping windows; the multiple-shooting
construction handles this naturally when the continuity penalty is placed at
:math:`t_{0,i+1}`, the first overlapping time point. The PEtab-level
implementation is then:

Inputs:

- A PEtab problem (PEtab v2).
- An initial window partition :math:`[t_{0,i}, t_{f,i}]` for stage 1 such that
  the union of windows covers the full measurement time range and
  :math:`t_{0,i} \neq t_{f,i}` for all windows. The number of curriculum
  stages equals the number of windows in this partition.
- A continuity penalty parameter :math:`\lambda`.
- An initial guess :math:`v_0` for each estimated window initial state.

1. Construct stage 1 as a multiple-shooting (MS) PEtab problem with
   :math:`n_{\text{windows}} = n_{\text{stages}}` using the procedure in
   :ref:`Multiple shooting <multiple_shooting>`.
2. For curriculum stage :math:`k = 2, \ldots, n_{\text{stages}} - 1`:

   1. Set the number of windows to
      :math:`n_{\text{windows}} = n_{\text{stages}} - k + 1`.
   2. Define the stage-:math:`k` windows by dropping the last window from
      stage :math:`k - 1` and extending the remaining windows. With the
      original window starts :math:`t_{0,1}, \ldots, t_{0,n_{\text{stages}}}`
      and ends :math:`t_{f,1}, \ldots, t_{f,n_{\text{stages}}}` from stage 1,
      the stage-:math:`k` windows are

      :math:`[t_{0,1}, t_{f,k}], [t_{0,2}, t_{f,k+1}], \ldots, [t_{0,n_{\text{stages}}-k+1}, t_{f,n_{\text{stages}}}]`.

      Note that windows now overlap pairwise.
   3. Create the PEtab problem for stage :math:`k` by applying the
      :ref:`Multiple shooting <multiple_shooting>` construction with the
      updated window partition. Measurements falling in the overlap between
      two windows are duplicated so they appear in each window. The continuity
      penalty between windows :math:`i` and :math:`i+1` is placed at
      :math:`t_{0,i+1}` (the first overlapping time point), evaluated in the
      experiment ``WINDOW{i}_EXPERIMENT_{expId}``.

3. The final stage (:math:`k = n_{\text{stages}}`) corresponds to the original
   PEtab problem. Use the parameter estimate from stage
   :math:`n_{\text{stages}} - 1` to initialize optimization for the final
   stage.

A practical consideration for tools implementing and/or importing CMS is that
the number of window-initial parameters to estimate changes between stages. To
support transferring parameter values between stages, it can be beneficial to
provide a utility function for mapping parameters between stage problems.

Partitioning time windows
-------------------------

The above training approaches above require either splitting measurements into
curriculum stages (curriculum learning) or partitioning the simulation time
span into windows (multiple shooting and curriculum multiple shooting). We
recommend that tools supporting these methods provide the splitting schemes
outlined below.

For curriculum learning, splitting is done by unique measurement time points:
stage boundaries are placed at time points from the measurement table, and a
stage includes all measurements up to its boundary. We recommend supporting
both automatic splitting (e.g., given :math:`n_{\text{stages}}`, compute stage
boundaries for the user) and user-defined schedules (e.g., explicit time
points per stage).

For multiple shooting, window intervals :math:`[t_{0,i}, t_{f,i}]` must be
defined. We recommend supporting automatic window construction (e.g., take
:math:`n_{\text{windows}}` as input and allocate windows based on unique
measurement time points) as well as user-specified intervals. As a basic
sanity check, tools should ensure that each window contains at least one
measurement.
