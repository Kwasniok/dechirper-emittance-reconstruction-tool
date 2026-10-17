#!/usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from uuid import uuid4 as uuid
from time import sleep
import pydoocs


def measure_shot(data_path: str, experiment_name: str) -> dict[str, float]:
    sample_id = (
        "BACKGROUND_"
        # time stamp incl milliseconds
        + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
        + "-"
        + uuid().hex[:8]
    )
    screens = pydoocs.read("XFEL.DIAG/CAMERA/OTRC.2560.T3/IMAGE_EXT_ZMQ")

    path_raw = Path(data_path, experiment_name, "data_raw", f"{sample_id}.npz")
    path_raw.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "screens": screens,
    }

    np.savez(path_raw, **data)

    return data


def measure_series(data_path: str, experiment_name: str, num_measurements: int, wait_between_shots: float) -> list[dict[str, float]]:
    all_data = []
    for i in range(num_measurements):
        print(f"Measurement {i+1}/{num_measurements}")
        data = measure_shot(data_path, experiment_name)
        all_data.append(data)
        sleep(wait_between_shots)

    return all_data


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
        args.data_path,
        args.experiment_name,
        num_measurements=args.num_measurements,
        wait_between_shots=args.wait_between_shots,
    )


if __name__ == "__main__":
    main()
