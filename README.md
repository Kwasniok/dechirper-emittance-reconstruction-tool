## requirements
- python v3.12+

## install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

IMPORTANT: Ensure that the `t3` file is up to date!

## usage
Always activate the virtual environment before running any commands.
```bash
source .venv/bin/activate
```

### measurement
For your `<EXPERIMENT_NAME>` manually go through each `<LATTICE_CONFIG_INDEX>` values and run the setup command followed by the measurement command:
```bash
python setup.py --experiment_name <EXPERIMENT_NAME> --lattice_config_index <LATTICE_CONFIG_INDEX>
python measure.py --experiment_name <EXPERIMENT_NAME> --lattice_config_index <LATTICE_CONFIG_INDEX>
```

### data analysis
```bash
python tool.py gui data/<EXPERIMENT_NAME>
```

The **Lattice Preview** tab shows the selected target configuration's optics and
compares all configurations' horizontal and vertical phase advances on the initial
Twiss ellipses. Choose a configuration ID on the left; the red star highlights it
in both phase plots. Hover points to see IDs and phase advances, including
overlapping configurations.

Each initial Twiss dropdown accepts **Design** or a typed value (normalized
emittance in mm mrad, beta in m, alpha dimensionless). Design optics come from
`make_twiss0` at the configured lattice entrance. If the design specifies no
emittance, the preview uses 1 mm mrad, matching the simulation default. Choose
**Original** for physical mm/mrad coordinates or **Normalized** for a unit circle
in dimensionless Courant–Snyder coordinates. Preview settings apply only to these
plots; the tab also works before samples have been acquired.

JSON formats, field definitions, units, and replay requirements are documented
in [protocols.md](protocols.md).

## contributors

- @Kwasniok
