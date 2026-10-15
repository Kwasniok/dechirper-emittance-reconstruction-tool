## requirements
- python v3.13+

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
./setup --experiment_name <EXPERIMENT_NAME> --lattice_config_index <LATTICE_CONFIG_INDEX>
./measure --experiment_name <EXPERIMENT_NAME> --lattice_config_index <LATTICE_CONFIG_INDEX>
```

### data analysis
```bash
python tool gui data/<EXPERIMENT_NAME>
```

## contributors
- @Kwasniok