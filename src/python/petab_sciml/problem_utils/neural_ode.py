from libsbml import parseL3Formula, SBMLDocument, UNIT_KIND_SECOND, writeSBML
import pandas as pd
from typing import Iterable
from yaml import safe_dump


def generate_neural_ode_problem(
    species_all: Iterable[str], 
    model_filename: str = "model.xml",
    measurements_filename: str = "measurements.tsv",
    network_name: str = "net1",
    network_filename: str = "net1.yaml",
    array_filenames: Iterable[str] = [],
) -> None:
    """
    Generate the PEtab files for a neural ODE problem with the given species.
    This function will generate files defining conditions, hybridization, mappings
    observables, parameters and a PEtab problem.yaml file. The measurements 
    need to be provided by the user but a filename for the measurements can be
    specified. 
    
    :param species_all:
        List of species names to include in the model.
    :param model_filename:
        Name of the SBML file to create.
    :param measurements_filename:
        Name of the measurements TSV file to be referenced in the PEtab problem.
    :param network_name:
        Name of the neural network to be used in the problem. This name will be 
        used by PEtab importers to identify the network in compiled models.
    :param network_filename:
        Name of the neural network YAML file defining the network architecture.
    :param array_filenames:
        List of names of array files to be referenced in the PEtab problem
        (e.g. network parameters, network inputs).
    """
    document = _write_sbml(species_all, model_filename)
    _write_petab(document, model_filename, measurements_filename, network_name, network_filename, array_filenames)


def _write_sbml(species_all: Iterable[str], model_filename: str) -> SBMLDocument:
    """
    Write the SBML model file with the given filename for a neural ODE PEtab 
    problem and return the created SBML document.
    """
    document = SBMLDocument(3, 1)

    model = document.createModel()

    per_second = model.createUnitDefinition()
    per_second.setId("per_second")
    unit = per_second.createUnit()
    unit.setKind(UNIT_KIND_SECOND)
    unit.setExponent(-1)
    unit.setScale(0)
    unit.setMultiplier(1)

    c1 = model.createCompartment()
    c1.setConstant(True)
    c1.setSize(1)

    for species in species_all:
        s = model.createSpecies()
        s.setId(species)
        s.setConstant(False)
        s.setInitialAmount(0.5)
        s.setBoundaryCondition(False)
        s.setHasOnlySubstanceUnits(True)

        param_name = species + "_param"
        k = model.createParameter()
        k.setId(param_name)
        k.setConstant(True)
        k.setValue(1.3)

        reaction_id = species + "_reaction"
        r = model.createReaction()
        r.setId(reaction_id)
        r.setReversible(False)

        species_ref1 = r.createProduct()
        species_ref1.setSpecies(species)
        math_ast = parseL3Formula(param_name)
        kinetic_law = r.createKineticLaw()
        kinetic_law.setMath(math_ast)

    writeSBML(document, model_filename)

    return document


def _write_petab(
        document: SBMLDocument, 
        model_filename: str, 
        measurements_filename: str,
        network_name: str,
        network_filename: str,
        array_filenames: Iterable[str],
) -> None:
    """
    Write the PEtab files for a neural ODE PEtab problem with the given SBML
    document and referencing the model, measurements, network and array filenames
    provided.
    """
    model = document.model
    # conditions
    conditions = {"conditionId": ["cond1"]}
    pd.DataFrame(conditions).to_csv("conditions.tsv", sep="\t", index=False)

    # hybridization
    species = [sp.id for sp in model.species]
    params = [param.id for param in model.parameters]
    target_ids = [f"{network_name}_input{s}" for s, _ in enumerate(species)]
    target_values = [f"{network_name}_output{s}" for s, _ in enumerate(params)]
    hybridization = {
        "targetId": target_ids + params,
        "target_value": species + target_values,
    }
    pd.DataFrame(hybridization).to_csv("hybridization.tsv", sep="\t", index=False)

    # mapping
    inputs = [f"{network_name}.inputs[0][{s}]" for s, _ in enumerate(target_ids)]
    outputs = [f"{network_name}.outputs[0][{s}]" for s, _ in enumerate(target_values)]
    mapping = {
        "petabEntityId": target_ids + target_values + [f"{network_name}_ps"],
        "modelEntityId": inputs + outputs + [f"{network_name}.parameters"],
    }
    pd.DataFrame(mapping).to_csv("mapping.tsv", sep="\t", index=False)

    # observables
    observable_ids = [f"{s}_o" for s in species]
    observables = {
        "observableId": observable_ids,
        "observableFormular": species,
        "noiseFormula": [0.05] * len(species),
        "observableTransformation": ["lin"] * len(species),
        "noiseDistribution": ["normal"] * len(species),
    }
    pd.DataFrame(observables).to_csv("observables.tsv", sep="\t", index=False)

    # parameters
    parameters = {
        "parameterId": [f"{network_name}_ps"],
        "parameterScale": ["lin"],
        "lowerBound": ["-inf"],
        "upperBound": ["inf"],
        "nominalValue": [None],
        "estimate": [1],
    }
    pd.DataFrame(parameters).to_csv("parameters.tsv", sep="\t", index=False)

    # problem.yaml
    problem = {
        "problems": [
            {
                "model_files": {
                    "model": {
                        "location": model_filename,
                        "language": "sbml",
                    },
                    "measurement_files": [measurements_filename],
                    "observable_files": ["observables.tsv"],
                    "condition_files": ["conditions.tsv"],
                    "mapping_files": ["mapping.tsv"],
                },
            }
        ],
        "format_version": "2.0.0",
        "extensions": {
            "sciml": {
                "array_files": array_filenames,
                "hybridization_files": ["hybridization.tsv"],
                "neural_nets": {
                    network_name: {
                        "location": network_filename,
                        "static": False,
                        "format": "YAML",
                    }
                },
            }
        },
        "parameter_file": "parameters.tsv",
    }
    with open("problem.yaml", "w") as file:
        safe_dump(problem, file, sort_keys=False)
