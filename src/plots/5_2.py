from dataclasses import dataclass
from typing import List, Callable, TypeVar, Generic, Union
import enum
import os


T = TypeVar('T')

@dataclass
class Member(Generic[T]):

    val_parser: Callable[[str], T]
    """Function transforming data string to the desired type
    Could be for example just int("13") if T is int""" 
    
    value: T
    """Actual value of that member""" 

    id: str
    """Idenitifier string to look for in the file to find the member value""" 
    
# Adapt below -----------------------------------------------------------------
def int_or_None(string: str) -> int:
    """Tasks-per-Node is unfortunatelly not always filled"""
    if string == '':
        return None
    else:
        return int(string)

class Meta_Measure:
    """Class holding meta information for measurements"""
    def __init__(self) -> None:
        self.NodeList:       Member[str] = Member(str, "", "NodeList: ")
        self.Nodes:          Member[int] = Member(int, 0, "Nodes: ")
        self.Tasks_per_Node: Member[int] = Member(int_or_None, 0, "Tasks-per-Node: ")
        self.N_Tasks:        Member[int] = Member(int_or_None, 0, "Number-of-Tasks: ")
        # self.N:            Member[int] = Member(int, 0, "N: ")

def remove_ms(string: str) -> float:
    """Parser for time measurements removes trailing ms (milli seconds)
    from strings"""
    return float(string[:-3])

class Data_Measure:
    """Class holding meta information for measurements"""
    def __init__(self) -> None:
        self.N: Member[int] = Member(int, 0, "N: ")
        self.Iters: Member[int] = Member(int, 0, "Iters: ")
        self.t_iter: Member[float] = Member(remove_ms, 0.0, "Time per iter: ")
        self.flop_iter: Member[int] = Member(int, 0, "Flop per iter: ")
        self.flop_total: Member[int] = Member(int, 0, "Flops total: ")
        self.Gflops: Member[float] = Member(float, 0.0, "GFlops/s: ")

# Adapt above -----------------------------------------------------------------




class Measure_Factory:

    def __init__(self, file_path: str) -> None:
        self.file_path: str = file_path
        self.experiment_meta: Meta_Measure = Meta_Measure()
        self.experiment_data: List[Data_Measure] = [Data_Measure()]
        self.meta_progress: int = 0
        self.data_progress: int = 0
        self.n_meta = len(vars(Meta_Measure())) # gives # of fields
        self.n_data = len(vars(Data_Measure()))

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, f_path: str):
        if os.path.isfile(f_path):
            self._file_path = f_path  # makes private variable outside __init__
        else:
            raise ValueError("Passed path seems to be no valid path")

    def read(self):
        ReadStates = enum.Enum('ReadStates', ['readingMeta', 'readingData'])

        with open(self.file_path, "rt") as f:
            curr_state = ReadStates.readingMeta
            for line in f:  # iterate over lines
                # Line Pre Processing-------------------
                if line.isspace():  # continue if empty line
                    continue
                if line.startswith("End-Meta-Info"):
                    curr_state=ReadStates.readingData
                # ----------------------------------------

                if curr_state == ReadStates.readingMeta:
                    self.make_Meta_Measure(line)
                elif curr_state == ReadStates.readingData:
                    self.make_Data_Measure(line)
                else:
                    raise NotImplementedError

    def make_Meta_Measure(self, line: str):
        success = self.fill_data_cls(line, self.experiment_meta)
        if success:
            self.meta_progress += 1

    def make_Data_Measure(self, line: str):
        if self.data_progress == self.n_data:
            self.experiment_data.append(Data_Measure())
            self.data_progress = 0

        success = self.fill_data_cls(line, self.experiment_data[-1])
        if success:
            self.data_progress += 1

    def fill_data_cls(self, line: str, data_cls: Union[Meta_Measure, Data_Measure]):
        # check for every member of the data class if line starts
        # with corresponding identifier
        # Maybe not best performance but least constraints on file format
        for attr_name, member_obj in vars(data_cls).items():
            id = member_obj.id
            # get substring to search for

            if line.startswith(id):
                # if found cast remaining string to type and assign
                member_obj.value = member_obj.val_parser(line[len(id):-1])
                return True
            else:
                continue
        return False # no matching attribute found


    def get_meta(self) -> Meta_Measure:
        return self.experiment_meta

    def get_data(self) -> List[Data_Measure]:
        return self.experiment_data

#-----------------------------------------------------------------------------

def get_measurement_files() -> List[str]:
    m_files = []
    for root, dirs, files in os.walk("./measurements", topdown=True):
        for name in files:
            m_files.append(os.path.join(root, name))
        break # scan only top lvl directory
    return m_files


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # plt.rcParams.update({'font.size': 11})
    from collections import namedtuple

    raw_files = get_measurement_files()

    files = [f for f in raw_files] # option to filter raw_filies

    factories: List[Measure_Factory] = [None]*len(files)

    for i, file in enumerate(files) :
        factories[i] = Measure_Factory(file)
        factories[i].read()

    meta: List[Meta_Measure] = []
    data: List[List[Data_Measure]] = []

    for i in range(len(factories)):
        meta.append(factories[i].get_meta())
        data.append(factories[i].get_data())

    # we want to plot G_flop/s over tile_width
    Point = namedtuple('Point', ['x', 'y'])

    # https://stackoverflow.com/a/17496530/4960953
    # make list of dicts to create pd.Dataframe at the end
    pd_raw_rows = []

    key_resolver = lambda tpl : tpl[1][0].N.value
    for m, d in sorted(zip(meta, data), key=key_resolver):
        # m is Meta_Measure d is LIST of Data_Measure
        pd_raw_row = dict()
        N = d[0].N.value
        pd_raw_row["Grid size"] = str(N) + " x " + str(N)
        pd_raw_row["Time/iteration"] = d[0].t_iter.value
        pd_raw_row["Flops total"] = d[0].flop_iter.value
        pd_raw_row["GFLOP/s"] = d[0].Gflops.value
        pd_raw_row["N"] = N

        pd_raw_rows.append(pd_raw_row)

    df = pd.DataFrame(pd_raw_rows)


    df.plot(x="N", y="GFLOP/s", marker='o', legend=False)
    plt.ylabel("GFlop/s")
    plt.xlabel("Grid Width")
    plt.title("Sequential 5 point stencil")
    # plt.legend()

    df[["Grid size", "Time/iteration", "Flops total", "GFLOP/s"]].to_latex("./tex/tbl.tex")

    # plt.show()
    plt.savefig("plots/img/5_2.pdf", bbox_inches='tight')
