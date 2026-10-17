#!/usr/bin/env python3
"""Acquire T3 screen images and machine-state snapshots for emittance reconstruction."""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from uuid import uuid4 as uuid
from time import sleep
import pydoocs


def measure_shot(
    data_path: str, experiment_name: str, lattice_config_index: int
) -> dict[str, float]:
    sample_id = (
        "MES_"
        + str(lattice_config_index)
        + "_"
        # time stamp incl milliseconds
        + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
        + "-"
        + uuid().hex[:8]
    )

    magnets = pydoocs.read("XFEL.MAGNETS/MAGNET.ML/*.T3/KICK_MRAD.SP")
    rf_i1_br1_sumvoltage = pydoocs.read(
        "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/BR1.SP.SUMVOLTAGE"
    )
    rf_i1_br1_chirp = pydoocs.read("XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/BR1.SP.CHIRP")
    rf_i1_br1_curvature = pydoocs.read(
        "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/BR1.SP.CURVATURE"
    )
    rf_i1_br1_thirdderivative = pydoocs.read(
        "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/BR1.SP.THIRDDERIVATIVE"
    )
    rf_i1_br2_sumvoltage = pydoocs.read(
        "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/BR2.SP.SUMVOLTAGE"
    )
    rf_i1_br2_chirp = pydoocs.read("XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/BR2.SP.CHIRP")
    rf_i1_br2_curvature = pydoocs.read(
        "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/BR2.SP.CURVATURE"
    )
    rf_i1_br2_thirdderivative = pydoocs.read(
        "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/I1/BR2.SP.THIRDDERIVATIVE"
    )
    rf_l1_br1_sumvoltage = pydoocs.read(
        "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/BR1.SP.SUMVOLTAGE"
    )
    rf_l1_br1_chirp = pydoocs.read("XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/BR1.SP.CHIRP")
    rf_l1_br2_sumvoltage = pydoocs.read(
        "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/BR2.SP.SUMVOLTAGE"
    )
    rf_l1_br2_chirp = pydoocs.read("XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L1/BR2.SP.CHIRP")
    rf_l2_br1_sumvoltage = pydoocs.read(
        "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/BR1.SP.SUMVOLTAGE"
    )
    rf_l2_br1_chirp = pydoocs.read("XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/BR1.SP.CHIRP")
    rf_l2_br2_sumvoltage = pydoocs.read(
        "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/BR2.SP.SUMVOLTAGE"
    )
    rf_l2_br2_chirp = pydoocs.read("XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/BR2.SP.CHIRP")
    bpms_t1 = pydoocs.read("XFEL.DIAG/BPM/*.T1/X.ALL")
    bpms_sa2 = pydoocs.read("XFEL.DIAG/BPM/*.SA2/X.ALL")
    bpms_t3 = pydoocs.read("XFEL.DIAG/BPM/*.T3/X.ALL")
    orbits_x_t1 = pydoocs.read("XFEL.DIAG/ORBIT/*.T1/X.ALL.HOLD")
    orbits_y_t1 = pydoocs.read("XFEL.DIAG/ORBIT/*.T1/Y.ALL.HOLD")
    orbits_x_sa2 = pydoocs.read("XFEL.DIAG/ORBIT/*.SA2/X.ALL.HOLD")
    orbits_y_sa2 = pydoocs.read("XFEL.DIAG/ORBIT/*.SA2/Y.ALL.HOLD")
    orbits_x_t3 = pydoocs.read("XFEL.DIAG/ORBIT/*.T3/X.ALL.HOLD")
    orbits_y_t3 = pydoocs.read("XFEL.DIAG/ORBIT/*.T3/Y.ALL.HOLD")
    screens = pydoocs.read("XFEL.DIAG/CAMERA/OTRC.2560.T3/IMAGE_EXT_ZMQ")

    path_raw = Path(data_path, experiment_name, "samples_raw", f"{sample_id}.npz")
    path_raw.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "magnets": magnets,
        "rf_i1_br1_sumvoltage": rf_i1_br1_sumvoltage,
        "rf_i1_br1_chirp": rf_i1_br1_chirp,
        "rf_i1_br1_curvature": rf_i1_br1_curvature,
        "rf_i1_br1_thirdderivative": rf_i1_br1_thirdderivative,
        "rf_i1_br2_sumvoltage": rf_i1_br2_sumvoltage,
        "rf_i1_br2_chirp": rf_i1_br2_chirp,
        "rf_i1_br2_curvature": rf_i1_br2_curvature,
        "rf_i1_br2_thirdderivative": rf_i1_br2_thirdderivative,
        "rf_l1_br1_sumvoltage": rf_l1_br1_sumvoltage,
        "rf_l1_br1_chirp": rf_l1_br1_chirp,
        "rf_l1_br2_sumvoltage": rf_l1_br2_sumvoltage,
        "rf_l1_br2_chirp": rf_l1_br2_chirp,
        "rf_l2_br1_sumvoltage": rf_l2_br1_sumvoltage,
        "rf_l2_br1_chirp": rf_l2_br1_chirp,
        "rf_l2_br2_sumvoltage": rf_l2_br2_sumvoltage,
        "rf_l2_br2_chirp": rf_l2_br2_chirp,
        "bpms_t1": bpms_t1,
        "bpms_t3": bpms_t3,
        "bpms_sa2": bpms_sa2,
        "orbits_x_t1": orbits_x_t1,
        "orbits_y_t1": orbits_y_t1,
        "orbits_x_sa2": orbits_x_sa2,
        "orbits_y_sa2": orbits_y_sa2,
        "orbits_x_t3": orbits_x_t3,
        "orbits_y_t3": orbits_y_t3,
        "screens": screens,
    }

    np.savez(path_raw, **data)

    return data


def measure_series(
    data_path: str,
    experiment_name: str,
    lattice_config_index: int,
    num_measurements: int,
    wait_between_shots: float,
) -> list[dict[str, float]]:
    all_data = []
    for i in range(num_measurements):
        print(f"Measurement {i+1}/{num_measurements}")
        data = measure_shot(data_path, experiment_name, lattice_config_index)
        all_data.append(data)
        sleep(wait_between_shots)
    return all_data


def main():
    parser = argparse.ArgumentParser(description="Measure emittance at EuXFEL T3")
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
        "--num_measurements",
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

    measure_series(
        data_path=args.data_path,
        experiment_name=args.experiment_name,
        lattice_config_index=args.lattice_config_index,
        num_measurements=args.num_measurements,
        wait_between_shots=args.wait_between_shots,
    )


if __name__ == "__main__":
    main()
