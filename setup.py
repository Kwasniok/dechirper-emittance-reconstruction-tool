#!/usr/bin/env python3
"""Apply T3 lattice configurations and coordinate operator-confirmed magnet cycling."""

import argparse
import numpy as np
import pandas as pd
from time import sleep
from pathlib import Path
from datetime import datetime
from uuid import uuid4 as uuid
import pydoocs


def prompt_confirm(message: str) -> bool:
    response = input(f"{message} (y/n): ").strip().lower()
    return response == "y"


def load_lattice_configs(data_path: str, experiment_name: str) -> pd.DataFrame:
    path = Path(data_path, experiment_name, "lattice_configs.json")
    return pd.read_json(path)


def set_valve(
    state: str,  # "open" or "close"
    valve_wait_time_sec: int,
) -> None:
    if state == "open":
        print("opening valve ...")
        pydoocs.write("XFEL/VAC/OSD.SV2/T1#.00000.01/CMD_Open", 1)
        sleep(valve_wait_time_sec)
        if not prompt_confirm("Valve opened?"):
            raise RuntimeError("Valve not opened, aborting.")
    elif state == "close":
        print("closing valve ...")
        pydoocs.write("XFEL/VAC/OSD.SV2/T1#.00000.01/CMD_Close", 1)
        print(f"waiting for {valve_wait_time_sec} seconds...")
        sleep(valve_wait_time_sec)
        if not prompt_confirm("Valve closed?"):
            raise RuntimeError("Valve not closed, aborting.")
    else:
        raise ValueError(f"Invalid valve state: {state}")


def cycle_quads(
    target_doocs_params: dict[str, float],
    quads: list[str] | None = None,
    close_valve_during_cycling: bool = False,
    valve_wait_time_sec: int = 10,
) -> None:

    if close_valve_during_cycling:
        set_valve("close", valve_wait_time_sec=valve_wait_time_sec)

    print("cycling quads...")
    for ch, value in target_doocs_params.items():
        if quads is not None and not any(q in ch for q in quads):
            print(f"WARNING: SKIPPING channel {ch}")
            continue
        ch_status = ch.replace("KICK_MRAD.SP", "COMBINED_STATUS")
        status_flag = pydoocs.read(ch_status)["data"]
        dirty = bool(status_flag & 0x10)
        if dirty:
            print(f"{ch} is dirty, cycling...")
            ch_cycle = ch.replace("KICK_MRAD.SP", "CYCLE")
            pydoocs.write(ch_cycle, 1)
        else:
            print(f"{ch} is not dirty, skipping cycle.")

    while not prompt_confirm("Cycling done for all quads?"):
        pass

    if close_valve_during_cycling:
        set_valve("open", valve_wait_time_sec=valve_wait_time_sec)


def _setup_core(
    target_doocs_params: dict[str, float],
    quads: list[str] | None = None,
) -> None:

    for ch, value in target_doocs_params.items():

        if quads is not None and not any(q in ch for q in quads):
            print(f"WARNING: SKIPPING channel {ch}")
            continue

        print(f"setting {ch} to {value}")
        pydoocs.write(ch, value)


def setup(
    data_path: str,
    experiment_name: str,
    lattice_config_index: int,
    quads: list[str] | None = None,
    perform_cycle_quads: bool = False,
    close_valve_during_cycling: bool = False,
    valve_wait_time_sec: int = 10,
):
    lattice_configs = load_lattice_configs(data_path, experiment_name)
    target_lattice_config = lattice_configs.iloc[lattice_config_index]
    target_doocs_params = target_lattice_config["target_doocs_params"]

    # TODO: confirm no beam in T1

    if perform_cycle_quads and not close_valve_during_cycling:
        if not prompt_confirm(
            "Are you sure that you want to cycle quads without closing the valve?"
        ):
            raise RuntimeError("Aborting.")

    _setup_core(target_doocs_params, quads=quads)
    if perform_cycle_quads:
        cycle_quads(
            target_doocs_params=target_doocs_params,
            quads=quads,
            close_valve_during_cycling=close_valve_during_cycling,
            valve_wait_time_sec=valve_wait_time_sec,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Setup lattice configuration at EuXFEL T3"
    )
    parser.add_argument(
        "--lattice_config_index",
        "-i",
        type=int,
        default=0,
        help="Index of the lattice configuration to use for the measurement",
    )
    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        required=True,
        help="Name of the experiment (used for data storage)",
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
        "-c",
        action="store_true",
        help="Enable autocycling of the quads",
    )
    parser.add_argument(
        "--close_valve_during_cycling",
        action="store_true",
        help="Close the valve during quad cycling",
    )
    parser.add_argument(
        "--valve_wait_time_sec",
        type=int,
        default=10,
        help="Minimum time to wait for the valve to open/close (in seconds)",
    )
    args = parser.parse_args()

    setup(
        data_path=args.data_path,
        experiment_name=args.experiment_name,
        lattice_config_index=args.lattice_config_index,
        quads=args.quads,
        perform_cycle_quads=args.cycle_quads,
        close_valve_during_cycling=args.close_valve_during_cycling,
        valve_wait_time_sec=args.valve_wait_time_sec,
    )


if __name__ == "__main__":
    main()
