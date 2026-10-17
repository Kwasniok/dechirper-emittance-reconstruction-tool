"""Build T3 optics and visualize design or user-supplied initial Twiss states."""

# %%
from typing import Any
from dataclasses import dataclass, fields, asdict
import copy
import json
from pathlib import Path
import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.backend_bases import MouseEvent
from tqdm import tqdm
from ocelot import (
    m_e_GeV,
    MagneticLattice,
    Navigator,
    Twiss,
    twiss,
    Wake,
    WakeTableDechirperOffAxis,
    track,
)
import ocelot.gui as og
from ocelot.cpbd.matcher import MatchProblem, MatchState
from ocelot_util import (
    get_lattice_element,
    get_lattice_param,
    set_lattice_param,
    make_sublattice,
    Plane,
)
import t3

from table_aux import pretty_print_matrix

# %%
DECHIRPER_START = "VCCHIRP_S.2487.T3"
DECHIRPER_MID = "VCCHIRP.2490.T3"
DECHIRPER_STOP = "VCCHIRP_S.2492.T3"
LATTICE_SCREEN = "OTRC.2560.T3"
DEFAULT_LATTICE_START = DECHIRPER_START
DEFAULT_LATTICE_STOP = LATTICE_SCREEN

# conservative estimates from BKR panels, with some safety margin
LATTICE_LIMITS = {
    # ("QE.2480.T3", "k1"): sorted((t3.qe_2480_t3.k1 * 0.5, t3.qe_2480_t3.k1 * 2.5)),
    ("QE.2495.T3", "k1"): sorted((t3.qe_2495_t3.k1 * 0.5, t3.qe_2495_t3.k1 * 2.5)),
    ("QE.2506.T3", "k1"): sorted((t3.qe_2506_t3.k1 * 0.5, t3.qe_2506_t3.k1 * 2.5)),
    ("QE.2518.T3", "k1"): sorted((t3.qe_2518_t3.k1 * 0.5, t3.qe_2518_t3.k1 * 2.5)),
    ("QH.2529.T3", "k1"): sorted((t3.qh_2529_t3.k1 * 0.5, t3.qh_2529_t3.k1 * 2.5)),
    ("QH.2532.T3", "k1"): sorted((t3.qh_2532_t3.k1 * 0.5, t3.qh_2532_t3.k1 * 2.5)),
    ("QH.2544.T3", "k1"): sorted((t3.qh_2544_t3.k1 * 0.5, t3.qh_2544_t3.k1 * 1.5)),
    ("QM.2549.T3", "k1"): sorted((t3.qm_2549_t3.k1 * 0.5, t3.qm_2549_t3.k1 * 1.25)),
    ("QM.2554.T3", "k1"): sorted((t3.qm_2554_t3.k1 * 0.5, t3.qm_2554_t3.k1 * 1.25)),
    ("QM.2559.T3", "k1"): sorted((t3.qm_2559_t3.k1 * 0.5, t3.qm_2559_t3.k1 * 1.25)),
}


@dataclass
class LatticeHighLevelParams:
    """High-level parameters for the lattice to be targeted for a desired configuration of the lattice."""

    beam_energy: float = 14.0
    start: str = DEFAULT_LATTICE_START
    stop: str = DEFAULT_LATTICE_STOP
    mu_x_start: str = DEFAULT_LATTICE_START
    dispersion_x: float = 0.0
    mu_x: complex = 0.0
    global_max_beta: float = 300
    final_beta_min: float | None = 10.0
    r_34: float | None = None
    minimize_second_order_t116: bool = False
    dechirper_enabled: bool = False
    dechirper_dist_y: float = 500e-6  # distance from the plate in [m]
    dechirper_sigma_z: float = (
        10e-6  # characteristic (rms) longitudinal beam size in [m]
    )


@dataclass
class LatticeLowLevelParams:
    """Low-level parameters for the lattice (actual state)."""

    start: str = None
    stop: str = None
    mu_x_start: str = None
    # qe_2480_t3_k1: float = None # before dechirper
    qe_2495_t3_k1: float = None
    qe_2506_t3_k1: float = None
    qe_2518_t3_k1: float = None
    qh_2529_t3_k1: float = None
    qh_2532_t3_k1: float = None
    qh_2544_t3_k1: float = None
    qm_2549_t3_k1: float = None
    qm_2554_t3_k1: float = None
    qm_2559_t3_k1: float = None
    dechirper_enabled: bool = LatticeHighLevelParams.dechirper_enabled
    dechirper_dist_y: float = LatticeHighLevelParams.dechirper_dist_y
    dechirper_sigma_z: float = LatticeHighLevelParams.dechirper_sigma_z

    @classmethod
    def from_high_level(
        cls,
        params: LatticeHighLevelParams,
        tol: float = 1e-3,
        initial_lattice: MagneticLattice | None = None,
    ) -> "LatticeLowLevelParams":
        """
        Create low-level lattice parameters from high-level parameters by making the corresponding lattice and extracting the actual low-level parameters.

        Arguments:
            initial_lattice: Optional lattice to start the optimization from. If None, a lattice based on the default design parameters is used. This is usefull for finding a configuration of the lattice close to a previously given one.
        """

        lattice = make_lattice(params, tol=tol, initial_lattice=initial_lattice)

        return LatticeLowLevelParams(
            start=params.start,
            stop=params.stop,
            # qe_2480_t3_k1=get_lattice_param(lattice, "QE.2480.T3", "k1"),
            qe_2495_t3_k1=get_lattice_param(lattice, "QE.2495.T3", "k1"),
            qe_2506_t3_k1=get_lattice_param(lattice, "QE.2506.T3", "k1"),
            qe_2518_t3_k1=get_lattice_param(lattice, "QE.2518.T3", "k1"),
            qh_2529_t3_k1=get_lattice_param(lattice, "QH.2529.T3", "k1"),
            qh_2532_t3_k1=get_lattice_param(lattice, "QH.2532.T3", "k1"),
            qh_2544_t3_k1=get_lattice_param(lattice, "QH.2544.T3", "k1"),
            qm_2549_t3_k1=get_lattice_param(lattice, "QM.2549.T3", "k1"),
            qm_2554_t3_k1=get_lattice_param(lattice, "QM.2554.T3", "k1"),
            qm_2559_t3_k1=get_lattice_param(lattice, "QM.2559.T3", "k1"),
            dechirper_enabled=params.dechirper_enabled,
            dechirper_dist_y=params.dechirper_dist_y,
            dechirper_sigma_z=params.dechirper_sigma_z,
        )

    @classmethod
    def doocs_params(cls) -> set[str]:
        # magnets only
        return {
            f.name
            for f in fields(cls)
            if f.name[:3]
            in [
                "qe_",
                "qh_",
                "qm_",
            ]
        }

    def to_doocs_params_dict(self) -> dict[str, Any]:
        lattice = MagneticLattice(t3.cell)

        res = {}
        for param in LatticeLowLevelParams.doocs_params():

            # magnets only
            if param[:3] not in [
                "qe_",
                "qh_",
                "qm_",
            ]:
                continue

            if getattr(self, param) is None:
                raise ValueError(f"Field {param} is None. All fields must be set.")

            eid = param.split("_")
            if len(eid) != 4:
                raise ValueError(
                    f"Unexpected field name format: {param}. Expected format: <type>_<id>_<location>_<param>"
                )
            if eid[-1] != "k1":
                raise ValueError(
                    f"Unexpected field name format: {param}. Expected last part to be 'k1'"
                )
            eid = f"{eid[0].upper()}.{eid[1]}.{eid[2].upper()}"
            el = get_lattice_param(lattice, eid, "l")
            # convert k1 in SI to generalized gradient in mrad/m
            res["XFEL.MAGNETS/MAGNET.ML/" + eid + "/KICK_MRAD.SP"] = (
                getattr(self, param) * 1e3 * el
            )

        return res


def abs_phase_advance_from_matrix(R: np.ndarray, plane: Plane) -> complex:
    match plane:
        case Plane.X:
            return np.arccos((R[0, 0] + R[1, 1]) / 2 + 0j)
        case Plane.Y:
            return np.arccos((R[2, 2] + R[3, 3]) / 2 + 0j)


def abs_phase_advance_from_lattice(
    lattice: MagneticLattice, energy: float, plane: Plane
) -> complex:
    R = lattice.transfer_maps(energy=energy)[1]
    return abs_phase_advance_from_matrix(R, plane)


# %%
def make_twiss0(start: str = DEFAULT_LATTICE_START) -> Twiss:
    # HACK: empty lattice not allowed -> handle cases separately
    if start == t3.cell[0].id:
        # empty lattice = identity transfer map
        return copy.deepcopy(t3.twiss0)
    else:
        # propagate twiss
        lattice = make_sublattice(
            MagneticLattice(t3.cell),
            stop=start,
            inclusive=False,
        )
        return twiss(lattice, t3.twiss0)[-1]


_prepared_lattices = {}


def _make_prepared_lattice(
    dispersion_x: float, sextupole_enabled: bool
) -> MagneticLattice:
    """
    prepared lattice = lattice with all properties fixed during experiment already preset.
    Prevents unnecessary noise in fixable parameters.
    """

    global _prepared_lattices

    key = (dispersion_x, sextupole_enabled)
    if key in _prepared_lattices:
        return copy.deepcopy(_prepared_lattices[key])

    lattice = copy.deepcopy(MagneticLattice(t3.cell))

    def add_vary(
        problem: MatchProblem, lattice: MagneticLattice, eid: str, param: str
    ) -> None:
        problem.vary_element(
            get_lattice_element(lattice, eid),
            param,
            limits=LATTICE_LIMITS[(eid, param)],
        )

    # disable sextupole if not enabled
    if not sextupole_enabled:
        set_lattice_param(lattice, "SAOX.2555.T3", "k2", 0.0)

    # resolve dispersion in x at screen (preliminary to further optimizations)
    # adjust last quad in lattice
    twiss0 = make_twiss0(start=lattice.sequence[0].id)
    problem = MatchProblem(lattice, twiss0, periodic=False)
    add_vary(problem, lattice, "QM.2554.T3", "k1")

    problem.target_twiss(
        get_lattice_element(lattice, LATTICE_SCREEN),
        "Dx",
        dispersion_x,
        weight=1e6,
    )

    result = problem.solve(solver="ls_trf", max_iter=3_000)

    if not result.success:
        raise RuntimeError(
            f"Dispersion matching failed (with message=`{result.message}`)!"
        )

    _prepared_lattices[key] = lattice
    return copy.deepcopy(lattice)


def make_lattice(
    params: LatticeHighLevelParams | LatticeLowLevelParams,
    *,
    tol: float = 1e-3,
    initial_lattice: MagneticLattice | None = None,
) -> MagneticLattice:
    """
    Make lattice based on the given parameters.
    If low-level parameters are given, the lattice is directly constructed.
    If high-level parameters are given, an optimization problem is solved to find a lattice that matches the desired high-level configuration.

    Arguments:
        tol: Tolerance for checking the results of the optimization (only used for high-level parameters)
        initial_lattice: Optional lattice to start the optimization from (only used for high-level parameters). If None, a lattice based on the default design parameters is used.
    """

    def make_lattice_from_low_level_params(
        params: LatticeLowLevelParams,
    ) -> MagneticLattice:
        lattice = make_sublattice(
            MagneticLattice(t3.cell),
            start=params.start,
            stop=params.stop,
            inclusive=True,
        )

        # disable sextupole
        set_lattice_param(lattice, "SAOX.2555.T3", "k2", 0.0)

        # set_lattice_param(lattice, "QE.2480.T3", "k1", params.qe_2480_t3_k1)
        set_lattice_param(lattice, "QE.2495.T3", "k1", params.qe_2495_t3_k1)
        set_lattice_param(lattice, "QE.2506.T3", "k1", params.qe_2506_t3_k1)
        set_lattice_param(lattice, "QE.2518.T3", "k1", params.qe_2518_t3_k1)
        set_lattice_param(lattice, "QH.2529.T3", "k1", params.qh_2529_t3_k1)
        set_lattice_param(lattice, "QH.2532.T3", "k1", params.qh_2532_t3_k1)
        set_lattice_param(lattice, "QH.2544.T3", "k1", params.qh_2544_t3_k1)
        set_lattice_param(lattice, "QM.2549.T3", "k1", params.qm_2549_t3_k1)
        set_lattice_param(lattice, "QM.2554.T3", "k1", params.qm_2554_t3_k1)
        set_lattice_param(lattice, "QM.2559.T3", "k1", params.qm_2559_t3_k1)

        return lattice

    def make_lattice_from_high_level_params(
        params: LatticeHighLevelParams,
    ) -> MagneticLattice:

        lattice = make_sublattice(
            initial_lattice  # asserts that initial lattice is None or has matching dispersion and sextupole settings
            or _make_prepared_lattice(
                dispersion_x=params.dispersion_x,
                sextupole_enabled=False,
            ),
            start=params.start,
            stop=params.stop,
            inclusive=True,
        )

        # assert: abs_mu_x in [0, pi) + iR
        assert (
            # re only
            (
                np.real(params.mu_x) != 0
                and np.isclose(np.imag(params.mu_x), 0, tol)
                and np.real(params.mu_x) >= 0
                and np.real(params.mu_x) < np.pi
            )
            # im only
            or (np.isclose(np.real(params.mu_x), 0, tol) and np.imag(params.mu_x) != 0)
            # zero
            or np.isclose(params.mu_x, 0, tol)
        ), f"Only positive absolute phase advances up to but excluding pi or imaginary values are supported. Current mu_x={params.mu_x} not in [0, pi) + iR."

        twiss0 = make_twiss0(start=params.start)

        def add_vary(
            problem: MatchProblem, lattice: MagneticLattice, eid: str, param: str
        ) -> None:
            problem.vary_element(
                get_lattice_element(lattice, eid),
                param,
                limits=LATTICE_LIMITS[(eid, param)],
            )

        # fix phase advance
        # note: if mu_x real, then the lattice is automatically stable

        # adjust quads in lattice (after dechirper, before bend)
        # note: solution does not need to be stable in both x AND y, just x is good enough
        problem = MatchProblem(lattice, twiss0, periodic=False)
        # add_vary(problem, lattice, "QE.2480.T3", "k1") # before dechirper
        add_vary(problem, lattice, "QE.2495.T3", "k1")
        add_vary(problem, lattice, "QE.2506.T3", "k1")
        add_vary(problem, lattice, "QE.2518.T3", "k1")
        add_vary(problem, lattice, "QH.2529.T3", "k1")
        add_vary(problem, lattice, "QH.2532.T3", "k1")
        add_vary(problem, lattice, "QH.2544.T3", "k1")

        # target phase advance in x (post dechirper to screen)
        # WARNING: phase advance for unmachted Twiss and unmatchable lattice seems to be broken
        #          use explicit cost function instead (see below)
        # problem.target_twiss_delta(
        #     start=get_lattice_element(lattice, DECHIRPER_STOP),
        #     end=get_lattice_element(lattice, LATTICE_SCREEN),
        #     quantity="mux",
        #     value=params.mu_x,
        #     relation="==",
        #     wrap_phase=True,  # ensure mu in [-pi, pi]
        #     weight=1e6,
        #     tol=1e-3,
        #     name="phase_advance_x",
        # )

        # alternative with cost function
        def phase_advance_x_cost_function(state: MatchState) -> float:
            R = state.r_matrix(
                get_lattice_element(state.lat, params.mu_x_start),
                state.lat.sequence[-1],
            )
            abs_mu_x = np.arccos((R[0, 0] + R[1, 1]) / 2 + 0j)
            diff = abs_mu_x - params.mu_x
            re = np.real(diff)
            im = np.imag(diff)
            # cost: approx abs(diff)**2 for small diff, and 2*pi-periodic + flat only near 0 in re
            # return np.sqrt(2 * (1 - np.cos(re))), im # rejected: flat near 0 but also +/- pi, which can cause issues with convergence
            return 2 * np.tan(re / 2), im  # improved version

        problem.objective_function(
            phase_advance_x_cost_function,
            mode="residual",
            name="phase_advance_x_func",
            weight=1e3,  # no need to be very accurate (conflicts too much with beta constraints if weighted too high)
        )

        # optional: fix y streaking
        if params.r_34 is not None:
            problem.target_rmatrix(
                get_lattice_element(lattice, params.start),
                get_lattice_element(lattice, LATTICE_SCREEN),
                i=2,
                j=3,
                value=params.r_34,
                relation="==",
                weight=1e6,
                name="R34_const",
            )

        # restrict beam widths
        problem.target_global(
            quantity="beta_x",
            value=params.global_max_beta,
            relation="<=",
            weight=1e6,
            name="beta_x_max",
        )
        problem.target_global(
            quantity="beta_y",
            value=params.global_max_beta,
            relation="<=",
            weight=1e6,
            name="beta_y_max",
        )

        # restrict beta at screen to ensure smaller relative errors in beam width
        if params.final_beta_min is not None:
            problem.target_twiss(
                get_lattice_element(lattice, LATTICE_SCREEN),
                "beta_x",
                value=params.final_beta_min,
                relation=">=",
                weight=1e6,
                name="screen_beta_x_min",
            )

        result = problem.solve(solver="ls_trf", max_iter=10_000)

        if not result.success:
            raise RuntimeError(
                f"Lattice optimization failed (with message=`{result.message}`)!"
            )

        # check results with tolerance

        # post dechirper to screen transfer map
        mu_x_lattice = make_sublattice(
            lattice,
            start=params.mu_x_start,
            stop=LATTICE_SCREEN,
            inclusive=True,
        )
        R = lattice.transfer_maps(energy=params.beam_energy)[1]
        R_mu_x = mu_x_lattice.transfer_maps(energy=params.beam_energy)[1]
        # dispersion
        if not np.isclose(R[0, 5], params.dispersion_x, atol=tol):
            raise RuntimeError(
                f"Dispersion in x does not match target within tolerance={tol:e}! actual: {R[0,5]}, target: {params.dispersion_x}"
            )
        # phase advance x post dechirper to screen
        abs_mu_x = np.arccos((R_mu_x[0, 0] + R_mu_x[1, 1]) / 2 + 0j)
        if not np.isclose(
            np.real(abs_mu_x),
            np.real(params.mu_x),
            atol=np.pi / 10,  # no need to be very accurate
        ) or not np.isclose(np.imag(abs_mu_x), np.imag(params.mu_x), atol=tol):
            raise RuntimeError(
                f"Phase advance in x does not match target within tolerance={tol:e}! actual: {abs_mu_x}, target: {params.mu_x}"
            )
        # y streaking
        if params.r_34 is not None and not np.isclose(R[2, 3], params.r_34, atol=tol):
            raise RuntimeError(
                f"R34 does not match target within tolerance={tol:e}! actual: {R[2,3]}, target: {params.r_34}"
            )
        # global beta max
        twisses = twiss(lattice, twiss0)
        beta_x_max = max(t.beta_x for t in twisses)
        beta_y_max = max(t.beta_y for t in twisses)
        if beta_x_max > params.global_max_beta * (
            1 + tol
        ) or beta_y_max > params.global_max_beta * (1 + tol):
            raise RuntimeError(
                f"At least one global beta_max exceeds target! actual: beta_x_max={beta_x_max}, beta_y_max={beta_y_max}, target: {params.global_max_beta}"
            )
        # final beta min at screen
        if params.final_beta_min is not None:
            twiss1 = twisses[-1]
            if twiss1.beta_x < params.final_beta_min * (1 - tol):
                raise RuntimeError(
                    f"Final beta_x at screen is below target! actual: {twiss1.beta_x}, target: {params.final_beta_min}"
                )

        return lattice

    if isinstance(params, LatticeLowLevelParams):
        lattice = make_lattice_from_low_level_params(params)
    if isinstance(params, LatticeHighLevelParams):
        lattice = make_lattice_from_high_level_params(params)

    return lattice


if __name__ == "__main__":
    lattice = make_lattice(LatticeHighLevelParams(beam_energy=14, mu_x=np.pi / 4))


# %%
def make_lattice_and_navigator(
    params: LatticeHighLevelParams | LatticeLowLevelParams,
    *,
    lattice_tol: float = 1e-3,
    initial_lattice: MagneticLattice | None = None,
) -> tuple[MagneticLattice, Navigator]:

    lattice = make_lattice(params, tol=lattice_tol, initial_lattice=initial_lattice)
    navigator = Navigator(lattice)

    if params.dechirper_enabled:
        # add wake field phys process for dechirper

        assert (
            params.dechirper_dist_y is not None
        ), "dechirper_dist_y must be set if dechirper_enabled is True"
        assert (
            params.dechirper_sigma_z is not None
        ), "dechirper_sigma_z must be set if dechirper_enabled is True"

        wk_tv_kick = WakeTableDechirperOffAxis(
            b=params.dechirper_dist_y,  # distance from the plate in [m]
            a=0.01,  # half gap between plates in [m]
            width=0.02,  # width of the corrugated structure in [m], PhysRevAccelBeams.27.050702 uses 12cm; exact value does not matter due to internal infinite width approximation
            t=0.25 * 1e-3,  # longitudinal gap in [m]
            p=0.5 * 1e-3,  # period of corrugation in [m]
            length=5.0,  # length of the corrugated structure in [m]
            sigma=params.dechirper_sigma_z,  # characteristic (rms) longitudinal beam size in [m]
            orient="horz",  # "horz" or "vert" plate orientation
        )

        wake = Wake(
            step=1,  # step in Navigator.unit_step, dz = Navigator.unit_step * wake.step [m]
            wake_table=wk_tv_kick,
            w_sampling=500,  # defines the number of the equidistant sampling points for the one-dimensional wake coefficients in the Taylor expansion of the 3D wake function.
        )
        try:
            dechirper_start = get_lattice_element(lattice, DECHIRPER_START)
            dechirper_stop = get_lattice_element(lattice, DECHIRPER_STOP)
            navigator.add_physics_proc(wake, dechirper_start, dechirper_stop)
        except KeyError:
            raise ValueError(
                f"Cannot attach wake field. Lattice does not contain expected elements for dechirper placement ({DECHIRPER_START} and {DECHIRPER_STOP})."
            )

    return lattice, navigator


if __name__ == "__main__":
    from beam import (
        BeamParameters,
        make_beam_with_random_gaussian_mixture_current_profile,
    )

    lattice, navigator = make_lattice_and_navigator(
        LatticeHighLevelParams(
            beam_energy=14,
            mu_x=np.pi / 4,
            dechirper_dist_y=500e-6,
            dechirper_sigma_z=60e-6,
        )
    )

    twiss0 = make_twiss0(start=lattice.sequence[0].id)

    bea_params0 = BeamParameters(
        energy=14,
        emit_x=1e-8,
        emit_y=1e-8,
        alpha_x=twiss0.alpha_x,
        beta_x=twiss0.beta_x,
        alpha_y=twiss0.alpha_y,
        beta_y=twiss0.beta_y,
    )

    parts0 = make_beam_with_random_gaussian_mixture_current_profile(
        bea_params0, num_particles=20_000
    )

    twisses, partss = track(lattice, parts0, navigator)


# %%
def show_lattice(
    lattice: MagneticLattice,
    twiss0: Twiss | None = None,
    *,
    figure: Figure | SubFigure | None = None,
    twisses: list[Twiss] | None = None,
    show: bool = True,
    diagnostics: bool = True,
) -> Figure | SubFigure:
    """Show optics, or draw into an embedded figure without opening a window.

    An explicit initial state overrides the design optics at the actual lattice
    entrance. Embedded callers can reuse a propagated profile and suppress the
    interactive window and printed transfer-map diagnostics.
    """
    twiss0 = twiss0 if twiss0 is not None else make_twiss0(lattice.sequence[0].id)
    twisses = twisses if twisses is not None else twiss(lattice, twiss0)
    if figure is None:
        figure = plt.figure()
    # Ocelot's plot_opt_func uses pyplot's current axes even with plt_figure.
    # Use its axes-based helpers so embedded figures need no pyplot window.
    top = figure.add_axes((0.12, 0.66, 0.84, 0.25))
    betas = figure.add_axes((0.12, 0.23, 0.84, 0.43), sharex=top)
    elements = figure.add_axes((0.12, 0.12, 0.84, 0.11), sharex=top)
    positions = [state.s for state in twisses]
    og.plot_disp(top, twisses, ["Dx", "Dxp"], 10)
    og.plot_betas(
        betas,
        positions,
        [state.beta_x for state in twisses],
        [state.beta_y for state in twisses],
        10,
    )
    previous_callbacks = set(
        figure.canvas.callbacks.callbacks.get("motion_notify_event", {})
    )
    og.plot_elems(
        figure,
        elements,
        lattice,
        s_point=positions[0],
        legend=False,
        y_scale=0.8,
        font_size=10,
    )
    # Ocelot's hover handler also uses pyplot. Redraw the owning canvas instead.
    new_callbacks = (
        set(figure.canvas.callbacks.callbacks.get("motion_notify_event", {}))
        - previous_callbacks
    )
    for callback in new_callbacks:
        figure.canvas.mpl_disconnect(callback)

    def hover(event: MouseEvent) -> None:
        """Show element names on the embedded canvas without activating pyplot."""
        changed = False
        for patch, annotation in zip(elements.patches, elements.texts):
            visible = event.inaxes is elements and patch.contains(event)[0]
            if visible != annotation.get_visible():
                annotation.set_visible(visible)
                changed = True
        if changed:
            figure.canvas.draw_idle()

    figure.canvas.mpl_connect("motion_notify_event", hover)
    for axis in (top, betas, elements):
        axis.grid(True, alpha=0.25)
        axis.set_xlim(positions[0], positions[-1])
    top.tick_params(labelbottom=False)
    betas.tick_params(labelbottom=False)
    elements.set_yticks([])
    if show:
        plt.show()
    if not diagnostics:
        return figure

    lat_vec, lat_matrix = lattice.transfer_maps(energy=twiss0.E)[:2]

    print("R matrix:")
    pretty_print_matrix(lat_matrix)
    print("B vector:")
    pretty_print_matrix(lat_vec)
    print(f"Dx: {lat_matrix[0, 5]}")

    print(
        f"phase advance x: {abs_phase_advance_from_lattice(lattice, twiss0.E, Plane.X)}"
    )
    print(
        f"phase advance y: {abs_phase_advance_from_lattice(lattice, twiss0.E, Plane.Y)}"
    )
    return figure


if __name__ == "__main__":

    show_lattice(
        make_lattice(LatticeHighLevelParams(beam_energy=14, mu_x=np.pi / 4)),
    )

    paramss: list[LatticeHighLevelParams] = [
        LatticeHighLevelParams(beam_energy=14, mu_x=mu, dispersion_x=dx)
        for mu in [
            0,
            np.pi / 7,
            np.pi / 4,
        ]
        for dx in [0.0]
    ]
    tol = 1e-3
    for params in paramss:
        lat = make_lattice(params, tol=tol)


# %%
def find_lattice_low_level_params(
    params_list: list[LatticeHighLevelParams],
    *,
    tol: float = 1e-3,
    keep_consecutive_configs_close: bool = False,
) -> list[LatticeLowLevelParams]:
    """
    Find low-level lattice parameters for a list of high-level lattice parameters by making the corresponding lattices and extracting the actual low-level parameters.

    Arguments:
        tol: Tolerance for checking the results of the optimization (only used for high-level parameters)
        keep_consecutive_configs_close: If True, the optimization for high-level parameters is performed sequentially, starting from the lattice of the previous configuration.
            This can help to keep consecutive configurations close to each other in terms of low-level parameters.
            However, it also means that the order of the configurations in the list can affect the resulting low-level parameters.
    """

    lattice_low_level_params_list = []
    previous_lattice = None
    for ps in tqdm(params_list, desc="Finding low level lattice parameters"):
        lattice_low_level_params_list.append(
            LatticeLowLevelParams.from_high_level(
                ps,
                tol=tol,
                initial_lattice=previous_lattice,
            )
        )
        if keep_consecutive_configs_close:
            previous_lattice = lattice

    return lattice_low_level_params_list


# %%
def save_lattice_config_list(
    lattice_high_level_params_list: list[LatticeHighLevelParams],
    lattice_low_level_params_list: list[LatticeLowLevelParams],
    *,
    experiment_name: str,
    data_path: str = "data",
):
    """
    Save a list of lattice configurations to a JSON file.

    Warning: Low and high level parameters must be consistent with each other, i.e. the low-level parameters must correspond to the high-level parameters. This is not checked by this function and can lead to inconsistencies if not ensured by the user.
    Note: DOOCS parameters for the lattice are also saved, which can be used for setting the lattice in the control system. These are derived from the low-level parameters, so consistency between low-level parameters and DOOCS parameters is ensured by construction.
    """

    if len(lattice_low_level_params_list) == 0:
        raise ValueError("Lattice parameter list is empty. Cannot save.")

    if len(lattice_high_level_params_list) != len(lattice_low_level_params_list):
        raise ValueError(
            "High-level and low-level parameter lists must have the same length."
        )

    lattice_doocs_params_list = [
        ps.to_doocs_params_dict() for ps in lattice_low_level_params_list
    ]

    # write to json
    hl_dict_list = (
        [asdict(params) for params in lattice_high_level_params_list]
        if lattice_high_level_params_list is not None
        else None
    )
    ll_dict_list = [asdict(params) for params in lattice_low_level_params_list]

    json_dict = {
        "high_level_params": hl_dict_list,
        "low_level_params": ll_dict_list,
        "target_doocs_params": lattice_doocs_params_list,
    }

    path = Path(data_path, experiment_name, "lattice_configs.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_dict, f, indent=4)


def load_lattice_config_list(
    *,
    experiment_name: str,
    data_path: str = "data",
) -> tuple[
    list[LatticeLowLevelParams],
    list[LatticeHighLevelParams] | None,
    list[dict[str, float]] | None,
]:
    path = Path(data_path, experiment_name, "lattice_configs.json")
    with open(path, "r") as f:
        json_dict = json.load(f)

    ll_dict_list = json_dict["low_level_params"]
    hl_dict_list = json_dict.get("high_level_params", None)

    lattice_low_level_params_list = [LatticeLowLevelParams(**d) for d in ll_dict_list]
    lattice_high_level_params_list = (
        [LatticeHighLevelParams(**d) for d in hl_dict_list]
        if hl_dict_list is not None
        else None
    )

    lattice_doocs_params_list = json_dict.get("target_doocs_params", None)

    return (
        lattice_low_level_params_list,
        lattice_high_level_params_list,
        lattice_doocs_params_list,
    )


def _test():
    lattice_param_num_samples = 10
    lattice_high_level_params_list = list(
        LatticeHighLevelParams(beam_energy=14, dispersion_x=0, mu_x=mu)
        for mu in np.linspace(
            0,
            np.pi / 3,
            num=lattice_param_num_samples,
            endpoint=False,
        )
    )

    lattice_low_level_params_list = find_lattice_low_level_params(
        lattice_high_level_params_list,
        tol=1e-3,
        keep_consecutive_configs_close=True,
    )

    save_lattice_config_list(
        lattice_high_level_params_list,
        lattice_low_level_params_list,
        experiment_name="TEST",
    )
    print(load_lattice_config_list(experiment_name="TEST"))


if __name__ == "__main__":
    _test()


# %%
def save_lattice_configs_plots(
    *,
    experiment_name: str,
    data_path: str = "data",
    show: bool = False,
):

    lattice_low_level_params_list, lattice_high_level_params_list, _ = (
        load_lattice_config_list(data_path=data_path, experiment_name=experiment_name)
    )
    lattices = [make_lattice(params) for params in lattice_low_level_params_list]
    for i, (lattice, hl_params) in enumerate(
        zip(lattices, lattice_high_level_params_list)
    ):
        twiss0 = make_twiss0(start=lattice.sequence[0].id)

        og.plot_opt_func(lattice, twiss(lattice, twiss0), legend=False)
        plt.title(f"mu_x={hl_params.mu_x:.2f}/rad, Dx={hl_params.dispersion_x:.2g}/m")
        plt.legend(loc="upper left", bbox_to_anchor=(0, 7))
        path = Path(data_path, experiment_name, "lattice_plots", f"{i:02d}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        if show:
            plt.show()
        plt.close()


# test
if __name__ == "__main__":
    save_lattice_configs_plots(experiment_name="TEST")

# %%
