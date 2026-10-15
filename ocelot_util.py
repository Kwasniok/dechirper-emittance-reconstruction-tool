from enum import Enum
import copy
import numpy as np
from ocelot import MagneticLattice, Element, ParticleArray, track

# lattice


def has_lattice_element(lattice: MagneticLattice, eid: str) -> bool:
    """Returns True if the lattice contains an element with the given id."""
    # HACK __getitem__ is buggy
    return any(e.id == eid for e in lattice.sequence)


def get_lattice_element(lattice: MagneticLattice, eid: str) -> Element:
    """
    Returns the element with the given id from the lattice.

    Raises:
        KeyError: If no element with the given id is found in the lattice.
    """
    # HACK __getitem__ is buggy
    return {e.id: e for e in lattice.sequence}[eid]


def get_lattice_param(
    lattice: MagneticLattice, eid: str, param: str, default=None
) -> float:
    """Returns the value of the specified parameter for the element with the given id."""
    # HACK __getitem__ is buggy
    try:
        element = get_lattice_element(lattice, eid)
        return getattr(element, param)
    except KeyError:
        return default


def set_lattice_param(
    lattice: MagneticLattice, eid: str, param: str, value: float
) -> None:
    """Sets the value of the specified parameter for the element with the given id."""
    # HACK __getitem__ is buggy
    element = get_lattice_element(lattice, eid)
    setattr(element, param, value)


def make_sublattice(
    lattice: MagneticLattice,
    start: str | None = None,
    stop: str | None = None,
    inclusive: bool = False,
) -> MagneticLattice:
    """
    Returns a new lattice containing the elements from start to stop identified by their ids (inclusive if specified).
    """
    # NOTE: Logic is analogous to list slicing.

    # HACK __getitem__ is buggy
    inclusive |= stop is None  # implicitly include end if stop is not specified
    sequence = lattice.sequence
    start_index = (
        next(i for i, e in enumerate(sequence) if e.id == start)
        if start is not None
        else 0
    )
    stop_index = (
        next(i for i, e in enumerate(sequence) if e.id == stop)
        if stop is not None
        else len(sequence) - 1
    )
    stop_index += 1 if inclusive else 0
    return MagneticLattice(copy.deepcopy(sequence[start_index:stop_index]))


# dimensions
class Plane(str, Enum):
    X = "x"
    Y = "y"
