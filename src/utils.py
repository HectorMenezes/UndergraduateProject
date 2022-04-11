from enum import Enum
import os
from typing import List, Optional

from argparse import ArgumentParser
from pydantic import BaseModel, Field, ValidationError, root_validator, validator

argp = ArgumentParser()

argp.add_argument(
    "-f", required=True, dest="configuration", help="Configuration File, a json"
)

argp.add_argument("-cv", required=False, dest="cross_validation", help="number of cross validation folds", type=int, default=0)

argp.add_argument("-max", required=False, dest="max_data", help="max number of data_points", type=int, default=0)


class Tech(Enum):
    QISKIT = "qiskit"
    PENNYLANTE = "pennylane"
    SKLEARN = "sklearn"


class ModelNature(Enum):
    CLASSICAL = "classical"
    QUANTUM = "quantum"


class Case(BaseModel):
    nature: ModelNature
    algorithm: str
    seed: Optional[int] = Field(42)
    supervised: bool
    data: str
    tech: Tech
    local: bool


class Cases(BaseModel):
    cases: List[Case]


def load_configuration(configuration_file: str) -> Cases:
    """
    Function to validate and load each case
    """
    if not os.path.exists(configuration_file):
        print(f"Configuration file {configuration_file} not foud")
        quit(100)

    try:
        return Cases.parse_file(configuration_file)
    except ValidationError as error:
        print(
            f"Failed to load configuration file {configuration_file}.\n"
            f"Details: {error.json()}"
        )
        quit(101)
