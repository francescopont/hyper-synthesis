import argparse
import re
#  from tabulate import tabulate
import os


optimum_pattern = re.compile(r"optimum: ([0-9]+\.[0-9]+)\n")
progress_pattern = re.compile(r"> progress(.*) opt = ([0-9]+\.[0-9]+)\n")


def check_line(line, acc, memory_level):
    for (old, new) in [("up0",f"up{memory_level}"), ("ri0",f"ri{memory_level}"), ("do0",f"do{memory_level}"), ("le0",f"le{memory_level}")]:

        if old in line: # record this line, I will need to add it
            line = re.sub(old, new, line)

            if "memory'=0" in line:  # record this line, I will need to add it
                line = re.sub("memory'=0", f"memory'={memory_level}", line)

            acc.append(line)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--input', type=str, default='', help="Input file.")
    argp.add_argument('--memory', type=int, default='', help="Input file.")
    args = argp.parse_args()

    dir = args.input
    if not os.path.isdir(dir):
        raise ValueError(f"the input directory {dir} does not exist")

    memory_level = args.memory
    assert memory_level in [2,3]

    check = "+memXX" if memory_level == 3 else "+mem"
    output_dir = f"{dir}X" if memory_level == 3 else f"{dir}XX"

    if check in dir:

        os.mkdir(output_dir)

        props_input_file = f"{dir}/sketch.props"
        props_output_file = f"{output_dir}/sketch.props"



        # props
        with open(props_input_file) as input_file:
            with open(props_output_file, 'x') as output_file:
                for line in (input_file.readlines()):
                    output_file.write(line)

        # model
        templ_input_file = f"{dir}/sketch.templ"
        templ_output_file = f"{output_dir}/sketch.templ"

        with open(templ_input_file) as input_file:
            with open(templ_output_file, 'x') as output_file:
                # maze lines to add
                maze_mode = False
                maze_lines = []

                # discounting lines to add
                discounting_mode = False
                discounting_lines = []

                # memory lines to add
                memory_mode = False
                memory_lines = []

                for line in (input_file.readlines()):
                    # modes
                    if "module maze" in line:
                        maze_mode = True
                    if "module discounting" in line:
                        discounting_mode = True
                    if "module memory" in line:
                        memory_mode = True

                    if maze_mode:
                        check_line(line, maze_lines, memory_level)
                        if "endmodule" in line:
                            for additional_line in maze_lines:
                                output_file.write(additional_line)
                            maze_mode = False
                            maze_lines = []

                    if discounting_mode:
                        check_line(line, discounting_lines, memory_level)
                        if "endmodule" in line:
                            for additional_line in discounting_lines:
                                output_file.write(additional_line)
                            discounting_mode = False
                            discounting_lines = []

                    if memory_mode:
                        check_line(line, memory_lines, memory_level)
                        if "endmodule" in line:
                            for additional_line in memory_lines:
                                output_file.write(additional_line)
                            memory_mode = False
                            memory_lines = []

                    if f"memory : [0..{memory_level -1}];" in line:
                        line = f"   memory : [0..{memory_level}];"
                    # write current line
                    output_file.write(line)











