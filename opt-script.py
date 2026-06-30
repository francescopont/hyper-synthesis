import argparse
import re
import csv
#  from tabulate import tabulate
import os
from file_read_backwards import FileReadBackwards


optimum_pattern = re.compile(r"optimum: ([0-9]+\.[0-9]+)\n")
progress_pattern = re.compile(r"> progress(.*) opt = ([0-9]+\.[0-9]+)\n")

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--input', type=str, default='', help="Input Log file.")
    argp.add_argument('--dir', type=str, default='', help="Current experiment directory.")
    args = argp.parse_args()

    path = args.input
    name = args.dir
    if not os.path.isfile(path):
        raise ValueError(f"the log file {path} does not exist")
    with FileReadBackwards(path, encoding="utf-8") as file:
        line = file.readline()
        progress_match = progress_pattern.search(line)
        if progress_match is not None:
            opt = progress_match.group(2)
        else:
            line = file.readline()
            optimum_match = optimum_pattern.search(line)
            if optimum_match is not None:
                opt = optimum_match.group(1)
            else:
                raise "Cannot find the optimum."

    # propagate optimum
    memfolders = [f"{name}+mem", f"{name}XX"]
    for memfolder in memfolders:
        if os.path.isdir(memfolder):
            output_path = f"{memfolder}/opt-temp.txt"
            with open(output_path, 'w', newline='') as f:
                f.write(opt)
        else:
            pass
            # print(f"Do not propagate to experiment {memfolder}")


