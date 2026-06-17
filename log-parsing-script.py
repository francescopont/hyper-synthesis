#!/usr/bin/env python3

import argparse
import re
import csv
#  from tabulate import tabulate
import os

#
time_out_symbol = "dagger"  # default
name_pattern = re.compile(r"loading sketch from (.*)/(\S+)/sketch.templ")
d_size_pattern = re.compile(r"constructed explicit quotient having ([0-9]+) states")
time_pattern = re.compile(r"synthesis time: ([0-9]+\.[0-9]+) s\n")
optimum_pattern = re.compile(r"optimum: ([0-9]+\.[0-9]+)\n")
progress_pattern = re.compile(r"> progress(.*) opt = ([0-9]+\.[0-9]+)\n")
start_pattern = re.compile(r"cli.py - This is Paynt version 0.1.0")
nr_states_pattern = re.compile(r"initial states and ([0-9]+) states")
nr_observations_pattern = re.compile(r"Number of observations of the input model: ([0-9]+)")
bound_pattern = re.compile(r"Bound on the maximum achievable value function: ([0-9]+\.[0-9]+)")
design_space_pattern = re.compile(r"synthesis initiated, design space: ([0-9]+)")
nr_holes_pattern = re.compile(r"Current family has ([0-9]+) holes")


def collect_results(path):
    results = []
    if not os.path.isfile(path):
        raise ValueError(f"the log file {path} does not exist")
    with open(path) as file:
        name = None
        d_size = None
        time = time_out_symbol
        opt = -1
        nr_states = None
        nr_observations = None
        bound = None
        design_space = None
        nr_holes = None

        for line in (file.readlines())[1:]:
            name_match = name_pattern.search(line)
            if name_match is not None:
                name = name_match.group(2)

            d_size_match = d_size_pattern.search(line)
            if d_size_match is not None:
                d_size = d_size_match.group(1)

            time_match = time_pattern.search(line)
            if time_match is not None:
                time = time_match.group(1)

            progress_match = progress_pattern.search(line)
            if progress_match is not None:
                opt = progress_match.group(2)
            else:
                optimum_match = optimum_pattern.search(line)
                if optimum_match is not None:
                    opt = optimum_match.group(1)

            nr_states_match = nr_states_pattern.search(line)
            if nr_states_match is not None:
                nr_states = nr_states_match.group(1)

            nr_observations_match = nr_observations_pattern.search(line)
            if nr_observations_match is not None:
                nr_observations = nr_observations_match.group(1)

            bound_match = bound_pattern.search(line)
            if bound_match is not None:
                bound = bound_match.group(1)

            design_space_match = design_space_pattern.search(line)
            if design_space_match is not None:
                design_space = design_space_match.group(1)

            nr_holes_match = nr_holes_pattern.search(line)
            if nr_holes_match is not None:
                nr_holes = nr_holes_match.group(1)

            new_experiment = start_pattern.search(line)
            if new_experiment is not None:
                # fix these values, start a new experiment
                record = {
                    "name": name,
                    "nr_states": nr_states,
                    "nr_observations": nr_observations,
                    "nr_holes": nr_holes,
                    "design_space": design_space,
                    "bound": bound,
                    "d_size": d_size,
                    "time": time,
                    "opt": opt,
                }
                #print(f"New record collected: {record}")
                print(f"Name: {name}")
                results.append(record)
                name = None
                d_size = None
                time = time_out_symbol
                opt = -1
                nr_states = 0
                nr_observations = None
                bound = None
                design_space = None
                nr_holes = None

    # last experiment does not have a new one
    assert name is not None, f"name: {name}"
    print(f"Name: {name}")
    record = {
        "name": name,
        "nr_states": nr_states,
        "nr_observations": nr_observations,
        "nr_holes": nr_holes,
        "design_space": design_space,
        "bound": bound,
        "d_size": d_size,
        "time": time,
        "opt": opt,
    }
    results.append(record)
    return results


def sorting_criterion(record):
  return record["name"]


def to_list(results, key_map_list):
    results.sort(key=sorting_criterion)
    print(f"Sorted results: {results}")
    return [[mapf(r[key]) for key, mapf in key_map_list] for r in results]


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--input', type=str, help="Input Log file.")
    argp.add_argument('--output', type=str, help='Output .csv file')
    args = argp.parse_args()

    print(f'Collecting benchmark results...')
    results = collect_results(args.input)

    key_list = ['name', 'nr_states', 'nr_observations', 'nr_holes', 'design_space', 'bound', 'd_size', 'time', 'opt']
    results_matrix = to_list(results, list(map(lambda k: (k, lambda x: x), key_list)))
    header = ["Name", "|M|", "|O|", "H", "|DS|", "b", "|D|", "t", "opt"]

    # store results somewhere
    with open(args.output, 'w', newline='') as f:
            cw = csv.writer(f)
            cw.writerow(header)
            cw.writerows(results_matrix)
