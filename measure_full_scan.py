#!/usr/bin/env python3

import argparse
from pathlib import Path
from time import sleep
from measure_crisp_signal import measure as measure_crisp
from measure_screen_background import measure_series as measure_background
from setup import setup as setup_configuration
from measure import measure_series as measure_per_config

def prompt_confirm(message: str):
    response = input(f"{message} Do you want to continue? (y/n): ").strip().lower()
    return response == "y"

def main():
    parser = argparse.ArgumentParser(description="Measure emittance at EuXFEL T3")
    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        required=True,
        help="Name of the experiment (used for data storage)",
    )
    parser.add_argument(
        "--lattice_config_indices",
        "-i",
        type=int,
        nargs="+",
        required=True,
        help="Indices of the lattice configurations to use for the measurement",
    )
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="data",
        help="Path to store the measurement data",
    )
    parser.add_argument(
        "--quads",
        "-q",
        type=str,
        nargs="*",
        default=None,
        # [
        #     "QE.2518.T3",
        #     "QH.2529.T3",
        #     "QH.2532.T3",
        #     "QH.2544.T3",
        # ],
        help="Quads to be set by the script. All others are ignored.",
    )
    parser.add_argument(
        "--cycle_quads",
        action="store_true",
        help="Whether to cycle quads during the setup",
    )
    parser.add_argument(
        "--close_valve_during_cycling",
        action="store_true",
        help="Whether to close the valve during quad cycling",
    )
    parser.add_argument(
        "--valve_wait_time_sec",
        type=int,
        default=10,
        help="Time to wait for the valve to open/close in seconds",
    )
    parser.add_argument(
        "--num_measurements_per_config",
        "-n",
        type=int,
        default=40,
        help="Number of measurements to take",
    )
    parser.add_argument(
        "--wait_between_shots",
        "-w",
        type=float,
        default=0.333,  # screen refresh rate
        help="Time to wait between shots in seconds",
    )
    args = parser.parse_args()

    # screen background
    if prompt_confirm("Take background image?"):
        if not prompt_confirm("Beam off?"):
            raise RuntimeError("Aborting.")    
        measure_background(
            args.data_path,
            args.experiment_name,
            num_measurements=args.num_measurements_per_config,
            wait_between_shots=args.wait_between_shots,
        )

    # crisp
    if prompt_confirm("Take crisp measurement?"):
        if not prompt_confirm("Beam on?"):
            raise RuntimeError("Aborting.")
        measure_crisp(
            args.data_path,
            args.experiment_name,
        )

    for lattice_config_index in args.lattice_config_indices:
        while not prompt_confirm(f"Beam off?"):
            pass
        setup_configuration(
            data_path=args.data_path,
            experiment_name=args.experiment_name,
            lattice_config_index=lattice_config_index,
            quads=args.quads,
            perform_cycle_quads=args.cycle_quads,
            close_valve_during_cycling=args.close_valve_during_cycling,
            valve_wait_time_sec=args.valve_wait_time_sec,
        )

        while not prompt_confirm(f"Beam on & cycling done?"):
            pass
        measure_per_config(
            args.data_path,
            args.experiment_name,
            lattice_config_index=lattice_config_index,
            num_measurements=args.num_measurements_per_config,
            wait_between_shots=args.wait_between_shots,
        )


if __name__ == "__main__":
    main()
