# %%
from dataclasses import dataclass, astuple, fields, asdict
import scipy.constants
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from ocelot import (
    m_e_GeV,
    Twiss,
    ParticleArray,
    generate_parray,
)
import ocelot.gui as og


# %%
@dataclass
class BeamParameters:
    """Physical macro parameters of a beam."""

    energy: float  # [GeV]
    emit_x: float  # horizontal geometric emittance [m rad]
    emit_y: float  # vertical geometric emittance [m rad]
    beta_x: float  # horizontal beta function [m]
    beta_y: float  # vertical beta function [m]
    alpha_x: float = 0.0  # horizontal alpha function [1]
    alpha_y: float = 0.0  # vertical alpha function [1]
    sigma_tau: float = 10e-6  # bunch length [m]
    sigma_p: float = 0  # relative momentum spread
    chirp: float = 0.0  # linear energy chirp [1]
    charge: float = 1e-9  # bunch charge [C]

    __UNITS__ = {
        "energy": "GeV",
        "emit_x": "m rad",
        "emit_y": "m rad",
        "beta_x": "m",
        "beta_y": "m",
        "alpha_x": "",
        "alpha_y": "",
        "sigma_tau": "m",
        "sigma_p": "",
        "chirp": "",
        "charge": "C",
    }

    def to_series(self) -> pd.Series:
        return pd.Series(asdict(self))

    @classmethod
    def from_series(cls, series: pd.Series) -> "BeamParameters":
        return cls(**series.to_dict())

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, dict_: dict[str, float]) -> "BeamParameters":
        return cls(**dict_)

    def __str__(self) -> str:
        s = ""
        for k, v in asdict(self).items():
            s += f"{k}: {v} {self.__UNITS__.get(k, '')}\n"
        return s


if __name__ == "__main__":

    def _test():
        energy = 14.0
        gamma_rel = energy / m_e_GeV
        params = BeamParameters(
            energy=14.0,
            emit_x=0.8e-6 / gamma_rel,
            emit_y=0.8e-6 / gamma_rel,
            beta_x=10.0,
            beta_y=11.0,
            alpha_x=1.0,
            alpha_y=2.0,
            sigma_tau=10e-6,
            sigma_p=1e-4,
            chirp=0.7,
        )
        print(params)
        print(params.to_series())
        print(asdict(params))

        assert params.to_series().equals(pd.Series(asdict(params)))
        assert params.to_dict() == asdict(params)
        assert BeamParameters.from_series(params.to_series()) == params
        assert BeamParameters.from_dict(params.to_dict()) == params

    _test()


# %%
def make_beam_with_random_gaussian_mixture_current_profile(
    params: BeamParameters,
    # technical parameters:
    num_current_components: int = 1,
    num_particles: int = 200_000,
    seed: int | None = None,
) -> ParticleArray:
    """
    Returns (randomized) relativistic Gaussian initial beam state with multi-Gaussian current profile as particle array.

    Note: For one current mode, the current profile is a single deterministic Gaussian.

    Arguments:
        - num_current_components: number of Gaussian modes to sum for the current profile

    Asserts:
        - beta_rel approx. 1
    """

    gamma_rel = params.energy / m_e_GeV
    beta_rel = np.sqrt(1 - 1 / gamma_rel**2)
    assert beta_rel > 0.99, "This function is intended for relativistic beams only."

    gamma = lambda a, b: (1 + a**2) / b
    gamma_x = gamma(params.alpha_x, params.beta_x)
    gamma_y = gamma(params.alpha_y, params.beta_y)
    emit_xn = params.emit_x * gamma_rel
    emit_yn = params.emit_y * gamma_rel

    rng = np.random.default_rng(seed=seed)

    # randomized Gaussian mixture current/ longitudinal density
    amp = (
        rng.random(num_current_components)
        if num_current_components > 1
        else np.array([1.0])
    )
    mu = (
        (rng.random(num_current_components) - 0.5) * 2 * params.sigma_tau
        if num_current_components > 1
        else np.array([0.0])
    )
    sigma = (
        rng.random(num_current_components) * params.sigma_tau
        if num_current_components > 1
        else np.array([params.sigma_tau])
    )
    s = np.linspace(-5 * params.sigma_tau, 5 * params.sigma_tau, num=200)
    density = np.sum(
        amp * np.exp(-((s[:, np.newaxis] - mu) ** 2) / (2.0 * sigma**2)), axis=1
    )

    particles = generate_parray(
        sigma_tau=params.sigma_tau,
        sigma_p=params.sigma_p,
        chirp=params.chirp,
        charge=params.charge,
        nparticles=num_particles,
        energy=params.energy,
        tws=Twiss(
            emit_xn=emit_xn,
            emit_yn=emit_yn,
            beta_x=params.beta_x,
            beta_y=params.beta_y,
            alpha_x=params.alpha_x,
            alpha_y=params.alpha_y,
            gamma_x=gamma_x,
            gamma_y=gamma_y,
            E=params.energy,
        ),
        shape=(s, density),
    )
    particles.tau()[:] -= np.mean(particles.tau())

    return particles


def make_beam_plot(
    beam: ParticleArray,
    *,
    num_bins: int = 100,
    num_slices: int = 100,
    smoothing_window_size: int = 10,
    cmap: matplotlib.colors.Colormap | None = None,
    norm: matplotlib.colors.Normalize | None = None,
):

    cmap = cmap or plt.cm.inferno_r.with_extremes(under="white")
    norm = norm or matplotlib.colors.PowerNorm(gamma=0.5, vmin=1e-12)

    num_coords = 6
    fig = plt.figure(figsize=(10, 10))

    # 2d phase space projections in in lower left triange of grid
    def append_2d_phase_space_projections():

        coords = {
            "x": (beam.x() * 1e3, r"$x$", "mm"),
            "px": (beam.px() * 1e3, r"$x^\prime$", "mrad"),
            "y": (beam.y() * 1e3, r"$y$", "mm"),
            "py": (beam.py() * 1e3, r"$y^\prime$", "mrad"),
            "tau": (beam.tau() * 1e3, r"$\tau$", "mm"),
            "p": (beam.p() * 1e2, r"$p$", r"$\%$"),
        }
        assert num_coords == len(coords)

        for i, (key2, (data2, label2, unit2)) in enumerate(coords.items()):
            for j, (key1, (data1, label1, unit1)) in enumerate(coords.items()):
                if i > j:
                    ax = fig.add_subplot(num_coords, num_coords, i * num_coords + j + 1)
                    # normalize data to 0-1 range for better visualization
                    ax.hist2d(
                        data1,
                        data2,
                        weights=beam.q_array * 1e12,  # -> pC
                        bins=num_bins,
                        cmap=cmap,
                        norm=norm,
                    )
                    if i == num_coords - 1:
                        ax.set_xlabel(
                            f"{label1} [{unit1}]" if unit1 is not None else f"{label1}"
                        )
                        # rotate x ticks
                        ax.tick_params(axis="x", rotation=45)
                    else:
                        ax.set_xticklabels([])
                    if j == 0:
                        ax.set_ylabel(
                            f"{label2} [{unit2}]" if unit2 is not None else f"{label2}"
                        )
                    else:
                        ax.set_yticklabels([])

    append_2d_phase_space_projections()

    # upper right triange of grid:

    # helper to align tau axes
    tau_limits = (None, None)

    # 1D current density
    def append_current(label_x_axis: bool = False):
        nonlocal tau_limits

        # note: equal tau slicing is used for current and with same binning resolution as 2D phase space projections
        equal_tau_bin_charge, equal_tau_bin_edges = np.histogram(
            beam.tau(),
            weights=beam.q_array,
            bins=num_bins,  # use the same number of bins as for the 2D histograms
        )
        equal_tau_bin_delta = equal_tau_bin_edges[1] - equal_tau_bin_edges[0]
        equal_tau_bin_positions = (
            equal_tau_bin_edges[:-1] + equal_tau_bin_edges[1:]
        ) / 2

        ax = fig.add_subplot(num_coords, num_coords, num_coords * 2 - 1)
        ax.plot(
            equal_tau_bin_positions * 1e3,  # bin centers [mm]
            equal_tau_bin_charge
            * scipy.constants.c
            / equal_tau_bin_delta
            * 1e-3,  # I [kA]
            color="blue",
        )
        if label_x_axis:
            ax.set_xlabel(r"$\tau$ [mm]")
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(r"$I$ [kA]")
        ax.tick_params(axis="x", rotation=45)

        tau_limits = np.array(ax.get_xlim()) * 1e-3

    append_current()

    def append_sliced_quantities(label_x_axis: bool = False):

        # tau bins by equal charge
        # note: this will create tau bins such that each bin contains approximately the same total charge
        tau_sorted_indices = np.argsort(beam.tau())
        tau_sorted_in = beam.tau()[tau_sorted_indices]
        q_sorted = beam.q_array[tau_sorted_indices]
        total_charge = np.sum(beam.q_array)
        tau_bin_edges = np.interp(
            np.linspace(0, total_charge, num_slices + 1),
            np.cumsum(q_sorted),
            tau_sorted_in,
        )
        tau_bin_indices = np.digitize(beam.tau(), tau_bin_edges) - 1
        tau_bin_positions = (tau_bin_edges[:-1] + tau_bin_edges[1:]) / 2
        # for weighted averages
        weights = beam.q_array / np.sum(beam.q_array)
        weight_slices = [
            weights[tau_bin_indices == i] for i in range(len(tau_bin_edges) - 1)
        ]
        x_slices = [
            beam.x()[tau_bin_indices == i] for i in range(len(tau_bin_edges) - 1)
        ]
        px_slices = [
            beam.px()[tau_bin_indices == i] for i in range(len(tau_bin_edges) - 1)
        ]
        y_slices = [
            beam.y()[tau_bin_indices == i] for i in range(len(tau_bin_edges) - 1)
        ]
        py_slices = [
            beam.py()[tau_bin_indices == i] for i in range(len(tau_bin_edges) - 1)
        ]
        p_slices = [
            beam.p()[tau_bin_indices == i] for i in range(len(tau_bin_edges) - 1)
        ]

        def moving_average(data, window_size=smoothing_window_size):
            # padding with first and last values to minimize edge effects
            padded_data = np.pad(
                data,
                (window_size // 2, window_size - 1 - window_size // 2),
                mode="edge",
            )
            return np.convolve(
                padded_data, np.ones(window_size) / window_size, mode="valid"
            )

        # x,y sliced normalized emittances
        def append_slice_normalized_emittances(label_x_axis: bool = False):

            def geometric_emittance(x_slices, px_slices, weight_slices):

                # subtract centriod
                centered_x_slices = [
                    x - np.average(x, weights=w)
                    for x, w in zip(x_slices, weight_slices)
                ]
                centered_px_slices = [
                    px - np.average(px, weights=w)
                    for px, w in zip(px_slices, weight_slices)
                ]

                emit_x_slices = [
                    np.sqrt(
                        np.average(x**2, weights=w) * np.average(px**2, weights=w)
                        - np.average(x * px, weights=w) ** 2
                    )
                    for x, px, w in zip(
                        centered_x_slices, centered_px_slices, weight_slices
                    )
                ]
                return np.array(emit_x_slices)

            geo_emit_x = geometric_emittance(x_slices, px_slices, weight_slices)
            geo_emit_y = geometric_emittance(y_slices, py_slices, weight_slices)

            proj_geo_emit_x = geometric_emittance(
                [beam.x()], [beam.px()], [beam.q_array / beam.q_array.sum()]
            )[0]
            proj_geo_emit_y = geometric_emittance(
                [beam.y()], [beam.py()], [beam.q_array / beam.q_array.sum()]
            )[0]

            # calculate relativistic factors for each tau slice
            def normalization_factor(p_slices, weight_slices):
                p_mean = np.array(
                    [np.average(p, weights=w) for p, w in zip(p_slices, weight_slices)]
                )
                gamma_rel = beam.E * (1 + p_mean * beam.p0c) / m_e_GeV
                beta_rel = np.sqrt(1 - 1 / gamma_rel**2)
                return gamma_rel * beta_rel

            norm_emit_x = geo_emit_x * normalization_factor(p_slices, weight_slices)
            norm_emit_y = geo_emit_y * normalization_factor(p_slices, weight_slices)

            proj_norm_emit_x = (
                proj_geo_emit_x
                * normalization_factor([beam.p()], [beam.q_array / beam.q_array.sum()])[
                    0
                ]
            )
            proj_norm_emit_y = (
                proj_geo_emit_y
                * normalization_factor([beam.p()], [beam.q_array / beam.q_array.sum()])[
                    0
                ]
            )

            ax = fig.add_subplot(num_coords, num_coords, num_coords * 3 - 1)
            # raw:
            # ax.plot(tau_bin_positions * 1e3, norm_emit_x * 1e6, label=r"$x$")
            # ax.plot(tau_bin_positions * 1e3, norm_emit_y * 1e6, label=r"$y$")

            # smoothed:
            # plot mean and std as error band with moving window
            norm_emit_x_mean = moving_average(norm_emit_x)
            norm_emit_y_mean = moving_average(norm_emit_y)
            norm_emit_x_std = np.sqrt(
                moving_average((norm_emit_x - norm_emit_x_mean) ** 2)
            )
            norm_emit_y_std = np.sqrt(
                moving_average((norm_emit_y - norm_emit_y_mean) ** 2)
            )

            ax.plot(
                tau_bin_positions * 1e3,
                norm_emit_x_mean * 1e6,
                label=r"$x$ (smoothed)",
                color="blue",
            )
            ax.plot(
                tau_bin_positions * 1e3,
                norm_emit_y_mean * 1e6,
                label=r"$y$ (smoothed)",
                color="orange",
            )
            ax.fill_between(
                tau_bin_positions * 1e3,
                (norm_emit_x_mean - norm_emit_x_std) * 1e6,
                (norm_emit_x_mean + norm_emit_x_std) * 1e6,
                color="blue",
                alpha=0.2,
                label=r"$x$ std",
            )
            ax.fill_between(
                tau_bin_positions * 1e3,
                (norm_emit_y_mean - norm_emit_y_std) * 1e6,
                (norm_emit_y_mean + norm_emit_y_std) * 1e6,
                color="orange",
                alpha=0.2,
                label=r"$y$ std",
            )

            # projected geometric emittances as dashed lines
            # with annotated values
            ax.axhline(proj_norm_emit_x * 1e6, color="blue", linestyle="--")
            ax.axhline(proj_norm_emit_y * 1e6, color="orange", linestyle="--")
            ax.plot(
                [],
                [],
                linestyle="--",
                color="blue",
                label=rf"$\varepsilon^\text{{proj.}}_\text{{{{n}},x}}$ = {proj_norm_emit_x * 1e6:.2g} mm mrad",
            )
            ax.plot(
                [],
                [],
                linestyle="--",
                color="orange",
                label=rf"$\varepsilon^\text{{proj.}}_\text{{{{n}},y}}$ = {proj_norm_emit_y * 1e6:.2g} mm mrad",
            )

            ax.set_xlim(tau_limits * 1e3)
            if label_x_axis:
                ax.set_xlabel(r"$\tau$ [mm]")
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(r"$\epsilon_\text{n}$ [mm mrad]")
            ax.tick_params(axis="x", rotation=45)
            ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))

        append_slice_normalized_emittances(label_x_axis=label_x_axis)

        def append_slice_energy_spread(label_x_axis: bool = False):

            def rel_momentum_spread(p):
                p_slices = [
                    p[tau_bin_indices == i] for i in range(len(tau_bin_edges) - 1)
                ]
                p_mean = [
                    np.average(p, weights=w) for p, w in zip(p_slices, weight_slices)
                ]
                p_std = [
                    np.sqrt(np.average((p - p0) ** 2, weights=w))
                    for p, p0, w in zip(p_slices, p_mean, weight_slices)
                ]
                return np.array(p_std)

            sigma_E = rel_momentum_spread(beam.p()) * beam.E * 1e9  # [eV]

            ax = fig.add_subplot(num_coords, num_coords, num_coords * 4 - 1)
            # raw:
            # ax.plot(tau_bin_positions * 1e3, sigma_E * 1e-6, label=r"$\sigma_E$")

            # smoothed:
            # plot mean and std as error band with moving window
            sigma_E_smooth = moving_average(sigma_E * 1e-6)
            sigma_E_smooth_std = np.sqrt(
                moving_average((sigma_E * 1e-6 - sigma_E_smooth) ** 2)
            )
            ax.plot(
                tau_bin_positions * 1e3,
                sigma_E_smooth,
                label=r"$\sigma_E$ (smoothed)",
                color="blue",
            )
            ax.fill_between(
                tau_bin_positions * 1e3,
                sigma_E_smooth - sigma_E_smooth_std,
                sigma_E_smooth + sigma_E_smooth_std,
                color="blue",
                alpha=0.2,
                label=r"$\sigma_E$ (smoothed ± std)",
            )

            ax.set_xlim(tau_limits * 1e3)
            if label_x_axis:
                ax.set_xlabel(r"$\tau$ [mm]")
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(r"$\sigma_E$ [MeV]")
            ax.tick_params(axis="x", rotation=45)

        append_slice_energy_spread(label_x_axis=label_x_axis)

    append_sliced_quantities()

    # general annotations

    def append_head_direction_indicator():
        ax = fig.add_subplot(num_coords, num_coords, num_coords * 5 - 1)
        ax.axis("off")
        ax.grid(False)
        # tex box
        ax.annotate(
            r"$\Downarrow \substack{\Longleftarrow \\ \text{head}}$",
            xy=(0.5, 0.5),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize="large",
            ha="center",
        )

    append_head_direction_indicator()

    # add colorbar
    # note: has to be last to ensure proper alignment of sub plots in grid
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=fig.get_axes(),
        orientation="vertical",
        fraction=0.02,
        pad=0.04,
    )
    cbar.set_label(r"charge density [a.u.]")
    # rescale to a.u.
    vmin, vmax = 0, norm.vmax
    cbar.ax.yaxis.set_major_locator(
        matplotlib.ticker.FixedLocator(np.linspace(vmin, vmax, 6))
    )
    cbar.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda x, pos: f"{(x - vmin) / (vmax - vmin):.1f}"
        )
    )
    # adjust position to avoid overlap
    pos = cbar.ax.get_position()
    cbar.ax.set_position(
        [
            pos.x0,
            0,
            pos.width,
            pos.height,
        ]
    )

    return fig


if __name__ == "__main__":

    def _test():
        from pathlib import Path
        from ocelot import load_particle_array

        energy = 14.0
        test_gamma_rel = energy / m_e_GeV
        beam = make_beam_with_random_gaussian_mixture_current_profile(
            params=BeamParameters(
                energy=energy,
                emit_x=1.0e-6 / test_gamma_rel,
                emit_y=1.0e-6 / test_gamma_rel,
                beta_x=10.0,
                beta_y=10.0,
                alpha_x=0.0,
                alpha_y=0.0,
                chirp=0.0,
                sigma_p=1e-4,
            ),
            num_current_components=3,
            num_particles=1_000_000,
            seed=42,
        )

        fig = make_beam_plot(beam, num_bins=300)
        plt.show(fig)

        def attempt_showing_beam_file(path):
            if Path(path).exists():
                beam = load_particle_array(path)
                fig = make_beam_plot(beam)
                plt.show(fig)

        attempt_showing_beam_file("data/before_ws.npz")
        attempt_showing_beam_file("data/chirper_beam.npz")

    _test()

# %%
