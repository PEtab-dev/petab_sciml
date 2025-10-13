from libsbml import Model, parseL3Formula, SBMLDocument, UNIT_KIND_SECOND, writeSBML
import pandas as pd
from typing import Iterable
from yaml import safe_dump

def generate_neural_ode_problem(species_all: Iterable[str], filename: str="model.xml") -> None:
    document = write_sbml(species_all, filename)
    breakpoint()
    write_petab(document, filename)

def write_sbml(species_all: Iterable[str], filename: str) -> SBMLDocument:
    document = SBMLDocument(3, 1)

    model = document.createModel()

    per_second = model.createUnitDefinition()
    per_second.setId('per_second')     
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


    writeSBML(document, filename)

    return document

def write_petab(document: SBMLDocument, filename: str) -> None:
    model = document.model
    # conditions
    conditions = {"conditionId": ["cond1"]}
    pd.DataFrame(conditions).to_csv("conditions.tsv", sep="\t", index=False)

    breakpoint()
    # hybridization
    species = [sp.id for sp in model.species]
    params = [param.id for param in model.parameters]
    target_ids = [f"net1_input{s}" for s, _ in enumerate(species)]
    target_values = [f"net1_output{s}" for s, _ in enumerate(params)]
    hybridization = {
        "targetId": target_ids + params,
        "target_value": species + target_values,
    }
    pd.DataFrame(hybridization).to_csv("hybridization.tsv", sep="\t", index=False)

    # mapping
    inputs = [f"net1.inputs[0][{s}]" for s, _ in enumerate(target_ids)]
    outputs = [f"net1.outputs[0][{s}]" for s, _ in enumerate(target_values)]
    mapping = {
        "petabEntityId": target_ids + target_values + ["net1_ps"],
        "modelEntityId": inputs + outputs + ["net1.parameters"]
    }
    pd.DataFrame(mapping).to_csv("mapping.tsv", sep="\t", index=False)
    
    # measurements - hmm skip?
    # network params hdf5 - different function
    # network yaml - different function
    
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
        "parameterId": ["net1_ps"],
        "parameterScale": ["lin"],
        "lowerBound": ["-inf"],
        "upperBound": ["inf"],
        "nominalValue": [None],
        "estimate": [1]
    }
    pd.DataFrame(parameters).to_csv("parameters.tsv", sep="\t", index=False)

    # problem.yaml 
    problem = {
        "problems": [{
            "model_files": {
                "model": {
                    "location": filename,
                    "language": "sbml",
                },
                "measurement_files": ["measurements.tsv"],
                "observable_files": ["observables.tsv"],
                "condition_files": ["conditions.tsv"],
                "mapping_files": ["mapping.tsv"],
            },
        }],
        "format_version": "2.0.0",
        "extensions": {
            "sciml": {
                "array_files": [],
                "hybridization_files": ["hybridization.tsv"],
                "neural_nets": {
                    "net1": {
                        "location": None,
                        "static": False,
                        "format": "YAML",
                    }
                }
            }
        },
        "parameter_file": "parameters.tsv"
    }
    with open("problem.yaml", "w") as file:
        safe_dump(problem, file, sort_keys=False)
