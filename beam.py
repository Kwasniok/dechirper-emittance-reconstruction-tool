# %%
from dataclasses import dataclass, astuple, fields
import numpy as np
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
    """Physical macro parameters of the beam."""

    energy: float
    emit_x: float
    emit_y: float
    beta_x: float
    beta_y: float
    alpha_x: float = 0.0
    alpha_y: float = 0.0
    sigma_tau: float = 10e-6
    sigma_p: float = 0
    chirp: float = 0.0
    charge: float = 1e-9

    def to_twiss(self) -> Twiss:
        return Twiss(
            E=self.energy,
            emit_x=self.emit_x,
            emit_y=self.emit_y,
            beta_x=self.beta_x,
            beta_y=self.beta_y,
            alpha_x=self.alpha_x,
            alpha_y=self.alpha_y,
            sigma_tau=self.sigma_tau,
            sigma_p=self.sigma_p,
            chirp=self.chirp,
        )


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
    print(str(params.to_twiss()))


if __name__ == "__main__":
    _test()


# %%
def make_beam(
    params: BeamParameters,
    # technical parameters:
    num_current_modes: int = 1,
    num_particles: int = 200_000,
) -> ParticleArray:
    """
    Returns (randomized)relativistic Gaussian initial beam state with multi-Gaussian current profile as particle array.

    Note: For one current mode, the current profile is a single deterministic Gaussian.
    
    Arguments:
        - num_current_modes: number of Gaussian modes to sum for the current profile

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

    # randomized multi-Gaussian current/ longitudinal density
    amp = np.random.rand(num_current_modes) if num_current_modes > 1 else np.array([1.0])
    mu = (np.random.rand(num_current_modes) - 0.5) * 2 * params.sigma_tau if num_current_modes > 1 else np.array([0.0])
    sigma = np.random.rand(num_current_modes) * params.sigma_tau if num_current_modes > 1 else np.array([params.sigma_tau])
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


def show_beam(params: BeamParameters):

    bunch = make_beam(
        params,
        num_current_modes=3,
    )

    og.show_e_beam(bunch)
    plt.show()

    og.show_density(
        bunch.x(),
        bunch.px(),
        xlabel=r"$x$ [m]",
        ylabel=r"$x^\prime$",
        title="horizontal phase space",
    )
    plt.show()


if __name__ == "__main__":
    energy = 14.0
    test_gamma_rel = energy / m_e_GeV
    show_beam(
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
        )
    )


# %%
def beam_mean_rel_error(true: BeamParameters, pred: BeamParameters):
    x = np.array(astuple(true))
    y = np.array(astuple(pred))
    return np.mean(np.abs((y - x) / (x + 1e-12)))


def plot_beam_rel_errors(true: BeamParameters, pred: BeamParameters):
    x = np.array(astuple(true))
    y = np.array(astuple(pred))
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    rel_errors = (y - x) / (x + 1e-12)
    ax.bar([f.name for f in fields(pred)], rel_errors)
    plt.ylabel("relative error")
    plt.title("relative errors of fitted parameters")
    # angle labels
    plt.xticks(rotation=45, ha="right")
    plt.grid()
    plt.tight_layout()
    plt.show()


# %%
