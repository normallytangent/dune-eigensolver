# !/usr/bin/python

import os
import enum
from typing import List

import numpy as np

from my_data_frames import Measure_Factory

ReadStates = enum.Enum('ReadStates', ['readingMeta', 'readingData'])

# can be used to find multiple files
def get_measurement_files() -> List[str]:
    files = []
    for root, dirs, files in os.walk("../plots", topdown=True):
        for name in files:
            files.append(os.path.join(root, name))
        break # scan only top lvl directory
    return files


if __name__ == "__main__":
    data_factory = Measure_Factory()

    with open("../measurements/barrier-333.txt","rt") as f:
        curr_state = ReadStates.readingMeta
        for line in f:  # iterate over lines
            # Line Pre Processing-------------------
            if line.isspace():  # continue if empty line
                continue
            if line.startswith("End-Meta-Info"):
                curr_state=ReadStates.readingData
            # ----------------------------------------

            if curr_state == ReadStates.readingMeta:
                data_factory.make_Meta_Measure(line)
            elif curr_state == ReadStates.readingData:
                data_factory.make_Data_Measure(line)
            else:
                raise NotImplementedError

    meta_info = data_factory.get_meta()
    data_points = data_factory.get_data()

    meta_info.loop_iters.val_parser

    times1 = map(lambda p : p.t_my_barrier.value, data_points)
    times2 = map(lambda p : p.t_mpi_barrier.value, data_points)

    t_my_barrier_vals = np.fromiter(times1, dtype=np.float64)
    t_mpi_barrier_vals = np.fromiter(times2, dtype=np.float64)

    n_iter = meta_info.loop_iters
    node_name = meta_info.NodeList

    debug = 1