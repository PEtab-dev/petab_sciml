from libsbml import parseL3Formula, SBMLDocument, writeSBML
import os
import pandas as pd
from petab_sciml.standard.nn_model import NNModelStandard
from typing import Iterable
from ruamel.yaml import YAML

def create_neural_ode(
    species_all: Iterable[str] | dict, 
    model_filename: str = "model.xml",
    network_name: str = "net1",
    save_directory: str = ".",
) -> None:
    """Generate the PEtab files for a neural ODE problem with the given species.
    
    This function will generate the model, hybridization, mapping and parameter 
    files.

    Args:
        species_all:
            List of species names to include in the model, or a dictionary
            mapping species names to their initial values.
        model_filename:
            Name of the SBML file to create.
        network_name:
            Name of the neural network to reference in the PEtab files.
        save_directory:
            Directory where the generated files will be saved.

    Returns:
        None
    """
    model_path = os.path.join(save_directory, model_filename)
    document = _write_sbml(species_all, model_path)
    _write_petab(document, network_name, save_directory)


def _write_sbml(species_all: Iterable[str] | dict, model_filename: str) -> SBMLDocument:
    """
    Write the SBML model file with the given filename for a neural ODE PEtab 
    problem and return the created SBML document.
    """
    document = SBMLDocument(3, 1)

    model = document.createModel()

    c1 = model.createCompartment()
    c1.setConstant(True)
    c1.setSize(1)

    if isinstance(species_all, dict):
        species_items = species_all.items()
    else:
        species_items = ((species, 0.0) for species in species_all)

    for species, initial_amount in species_items:
        s = model.createSpecies()
        s.setId(species)
        s.setConstant(False)
        s.setInitialAmount(initial_amount)
        s.setBoundaryCondition(False)
        s.setHasOnlySubstanceUnits(True)

        param_name = species + "_param"
        k = model.createParameter()
        k.setId(param_name)
        k.setConstant(False)
        k.setValue(0.0)

        rule_id = species + "_reaction"
        r = model.createRateRule()
        r.setId(rule_id)
        r.setVariable(species)
        math_ast = parseL3Formula(param_name)
        r.setMath(math_ast)

    writeSBML(document, model_filename)

    return document


def _write_petab(
    document: SBMLDocument,
    network_name: str,
    save_directory: str,
) -> None:
    """
    Write the PEtab files for a neural ODE PEtab problem with the given SBML
    document and referencing the network provided.
    """
    model = document.getModel()
    
    # hybridization
    species = [sp.id for sp in model.species]
    params = [param.id for param in model.parameters]
    target_ids = [f"{network_name}_input{s}" for s, _ in enumerate(species)]
    target_values = [f"{network_name}_output{s}" for s, _ in enumerate(params)]
    hybridization = {
        "targetId": target_ids + params,
        "targetValue": species + target_values,
    }
    pd.DataFrame(hybridization).to_csv(
        os.path.join(save_directory, "hybridization.tsv"), 
        sep="\t", 
        index=False
    )

    # mapping
    inputs = [f"{network_name}.inputs[0][{s}]" for s, _ in enumerate(target_ids)]
    outputs = [f"{network_name}.outputs[0][{s}]" for s, _ in enumerate(target_values)]
    mapping = {
        "petabEntityId": target_ids + target_values + [f"{network_name}_ps"],
        "modelEntityId": inputs + outputs + [f"{network_name}.parameters"],
    }
    pd.DataFrame(mapping).to_csv(
        os.path.join(save_directory, "mapping.tsv"), 
        sep="\t", 
        index=False)

    # parameters
    parameters = {
        "parameterId": [f"{network_name}_ps"],
        "parameterScale": ["lin"],
        "lowerBound": ["-inf"],
        "upperBound": ["inf"],
        "nominalValue": [None],
        "estimate": [1],
    }
    pd.DataFrame(parameters).to_csv(
        os.path.join(save_directory, "parameters.tsv"), 
        sep="\t", 
        index=False
    )

def create_neural_ode_problem(
    model_filename: str,
    measurements_filename: str,
    observables_filename: str,
    network_filename: str,
    array_filenames: Iterable[str],
    mapping_filenames: Iterable[str] = ["mapping.tsv"],
    parameters_filename: str = "parameters.tsv",
    hybridization_filenames: Iterable[str] = ["hybridization.tsv"],
    save_directory: str = ".",
) -> None:
    """Write the PEtab files needed for the neural ODE PEtab problem.

    The mappings, parameters and hybridization files can be created using the 
    create_neural_ode function. This function will create the conditions and 
    problem.yaml files. The measurements and observables files need to be 
    provided by the user.

    Args:
        model_filename:
            Name of the SBML file to be referenced in the PEtab problem.
        measurements_filename:
            Name of the measurements TSV file to be referenced in the PEtab problem.
            This file should be located in the same directory where the remaining PEtab
            files are to be generated.
        observables_filename:
            Name of the observables TSV file to be created. This file should be 
            located in the same directory where the remaining PEtab files are to be 
            generated.
        network_filename:
            Name of the neural network YAML file defining the network architecture.
        array_filenames:
            List of names of array files to be referenced in the PEtab problem.
        mapping_filenames:
            List of names of mapping files to be referenced in the PEtab problem.
        parameters_filename:
            Name of the parameters TSV file to be referenced in the PEtab problem.
        hybridization_filenames:
            List of names of hybridization files to be referenced in the PEtab problem.
        save_directory:
            Directory where the generated files will be saved.

    Returns:
        None
    """
    measurements = pd.read_csv(measurements_filename, sep="\t")
    condition_ids = measurements["simulationConditionId"].unique()
    
    # conditions
    conditions = {"conditionId": condition_ids}
    pd.DataFrame(conditions).to_csv(
        os.path.join(save_directory, "conditions.tsv"), 
        sep="\t", 
        index=False
    )

    # get network name from network yaml file
    network_name = NNModelStandard.load_data(network_filename).nn_model_id

    # problem.yaml
    problem = {
        "problems": [
            {
                "model_files": {
                    "model": {
                        "location": model_filename,
                        "language": "sbml",
                    },    
                },
                "measurement_files": [measurements_filename],
                "observable_files": [observables_filename],
                "condition_files": ["conditions.tsv"],
                "mapping_files": mapping_filenames,
            }
        ],
        "format_version": "2.0.0",
        "extensions": {
            "sciml": {
                "array_files": array_filenames,
                "hybridization_files": hybridization_filenames,
                "neural_nets": {
                    network_name: {
                        "location": network_filename,
                        "static": False,
                        "format": "YAML",
                    }
                },
            }
        },
        "parameter_file": parameters_filename,
    }
    with open(os.path.join(save_directory, "problem.yaml"), "w") as file:
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.dump(problem, file)
