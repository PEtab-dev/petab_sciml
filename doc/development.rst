.. _development:

PEtab SciML development process
===============================

This page describes the development process for PEtab SciML, an extension to
the `PEtab format <https://petab.readthedocs.io/>`_ for parameter estimation
of dynamic models.

Format changes and proposals
----------------------------

PEtab SciML cannot accommodate every use case, but we are committed to
addressing current and future requirements in upcoming versions of the format.
We value backwards compatibility, but do not exclude breaking changes when
justified by clear benefits.

Anyone is welcome to propose changes or additions to PEtab SciML. Proposals
should be made via
`GitHub issues <https://github.com/PEtab-dev/petab_sciml/issues>`_, where
benefits, potential problems, and alternatives can be discussed.

A proposal is considered accepted for inclusion in the next PEtab SciML
version if it is endorsed by a majority of the maintainers and at least one
tool supporting PEtab SciML provides a prototype implementation. For accepted
changes, corresponding test cases should when relevant be added to the
`PEtab SciML test suite <https://github.com/sebapersson/petab_sciml_testsuite>`_
before release.

Release requirements
--------------------

For each new release, the following must be updated:

- Format specification
- Validator (linter)
- Test suite
- Changelog

For each release, the core maintainers ensure that:

- a new release is created in the GitHub repository
- the new version of the specification is deposited on Zenodo

Versioning of PEtab SciML
-------------------------

The PEtab SciML specification follows
`semantic versioning <https://semver.org/>`_. Changes to the specification
require a new release. Necessary clarifications or corrections may be collected
on an Errata page until the next release.

Release timing is at the discretion of the maintainers. However, accepted
changes should be released within two months of acceptance.

Maintainers
-----------

PEtab SciML, like PEtab, is intended as a community effort. Decisions should,
as far as possible, be made with input from the broader community.
Nevertheless, for efficiency, PEtab SciML has a set of core maintainers
responsible for maintaining the software and specification, and for approving
changes.

Since PEtab SciML is a PEtab extension, core maintainers are selected by the
`PEtab editorial board <https://petab.readthedocs.io/en/latest/editorial_board.html/>`_.
The PEtab editors ensure that the format has maintainers. New maintainers are
welcome, but a track record is expected (e.g., contributions to PEtab SciML or
related standards such as PEtab, SBML, etc.).

Values
------

We are committed to diversity, open communication, transparent processes, and
a welcoming environment. We aim for clear processes without unnecessary
over-formalization that could slow development.

Communication channels
----------------------

The primary communication channels are the GitHub
`issues <https://github.com/PEtab-dev/petab_sciml/issues>`_ and
`discussions <https://github.com/PEtab-dev/petab_sciml/discussions>`_ pages.
Additionally, since PEtab SciML is a PEtab extension, the PEtab
`mailing list <https://groups.google.com/g/petab-discuss>`_ can be used for
discussions that are not well suited for GitHub.
