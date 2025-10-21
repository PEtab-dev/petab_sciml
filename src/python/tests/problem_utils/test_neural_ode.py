from contextlib import chdir
from petab_sciml.problem_utils.neural_ode import generate_neural_ode_problem
import os
from yaml import safe_load


def test_generate_neural_ode_problem(tmp_path):
    """generate_neural_ode_problem"""

    test_dir = tmp_path / "petab"
    test_dir.mkdir()

    with chdir(test_dir):
        species = ["prey", "predator"]
        generate_neural_ode_problem(species)

        expected_files = [
            "model.xml",
            "conditions.tsv",
            "hybridization.tsv",
            "mapping.tsv",
            "observables.tsv",
            "parameters.tsv",
            "problem.yaml",
        ]
        for fname in expected_files:
            assert os.path.isfile(fname)

        with open("hybridization.tsv", "r") as f:
            # number of lines is header plus inputs and outputs for each species
            assert len(f.readlines()) == 1 + len(species) * 2

        with open("mapping.tsv", "r") as f:
            # number of lines is header plus network params plus inputs
            # and outputs for each species
            assert len(f.readlines()) == 2 + len(species) * 2

        with open("parameters.tsv", "r") as f:
            # number of lines is header plus network parameters
            assert len(f.readlines()) == 2


def test_generate_neural_ode_problem_with_options(tmp_path):
    """generate_neural_ode_problem_with_options"""

    test_dir = tmp_path / "petab"
    test_dir.mkdir()

    with chdir(test_dir):
        species = ["prey", "predator", "symbiote"]
        network_arrays = ["supernet_params.h5", "supernet_inputs.h5"]
        generate_neural_ode_problem(
            species, 
            model_filename="three_species_model.xml",
            measurements_filename="measurements_exp_test.tsv",
            network_name="supernet",
            network_filename="supernet.yaml",
            array_filenames=network_arrays,
        )

        expected_files = [
            "three_species_model.xml",
            "conditions.tsv",
            "hybridization.tsv",
            "mapping.tsv",
            "observables.tsv",
            "parameters.tsv",
            "problem.yaml",
        ]
        for fname in expected_files:
            assert os.path.isfile(fname)

        with open("hybridization.tsv", "r") as f:
            # number of lines is header plus inputs and outputs for each species
            assert len(f.readlines()) == 1 + len(species) * 2

        with open("mapping.tsv", "r") as f:
            # number of lines is header plus network params plus inputs
            # and outputs for each species
            assert len(f.readlines()) == 2 + len(species) * 2

        with open("parameters.tsv", "r") as f:
            # number of lines is header plus network parameters
            assert len(f.readlines()) == 2

        with open("problem.yaml", "r") as f:
            problem = safe_load(f)
            assert (
                problem["problems"][0]["model_files"]["model"]["location"]
                == "three_species_model.xml"
            )
            assert (
                problem["problems"][0]["measurement_files"][0]
                == "measurements_exp_test.tsv"
            )
            assert (
                "supernet" in problem["extensions"]["sciml"]["neural_nets"].keys()
            )
            assert (
                problem["extensions"]["sciml"]["neural_nets"]["supernet"]["location"]
                == "supernet.yaml"
            )
            assert problem["extensions"]["sciml"]["array_files"] == network_arrays

