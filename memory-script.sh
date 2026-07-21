#!/bin/bash

# example: bash memory-script.sh hyper-mdps

projects_dir="./eval"
if [ -n "$1" ]; then
  projects_dir="$1"
fi

# workspace settings
memory_exe="./increase-memory.py"

cwd=$(pwd)
cd $projects_dir

# find subdirectories that do not contain subdirectories, sort them, strip leading ./
dirs=$(find . -type d -exec sh -c '(ls -p "{}"|grep />/dev/null)||echo "{}"' \; | sort -V | cut -c 3-)
echo "Start increasing memory."
cd $cwd
for d in $dirs; do
    opt_call="python3 ${memory_exe} --input ${projects_dir}/${d}"
    eval ${opt_call}
done

