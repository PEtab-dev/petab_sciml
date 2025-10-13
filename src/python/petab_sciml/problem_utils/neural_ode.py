from libsbml import parseL3Formula, SBMLDocument, UNIT_KIND_SECOND, writeSBML
from typing import Iterable

def generate_neural_ode_problem(species_all: Iterable[str], filename: str="model.xml") -> None:
    write_sbml(species_all, filename)


def write_sbml(species_all: Iterable[str], filename: str) -> None:
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

def write_petab(species_all: Iterable[str]) -> None:
    # generate petab files and save them 
    return