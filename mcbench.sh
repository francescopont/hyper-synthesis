#!/bin/bash

# example: bash mcbench.sh ar hyper-models/meeting-experiments mylog.txt
method="ar"
if [ -n "$1" ]; then
  method=$1
fi

projects_dir="./eval"
if [ -n "$2" ]; then
  projects_dir="$2"
fi

log_dir="./logs"
file_name="log.txt"
if [ -n "$3" ]; then
  file_name="$3"
fi
log_file="${log_dir}/${file_name}"

param=''
if [ -n "$4" ]; then
  param="$4"
fi

# default parameters
timeout=3600

# workspace settings
SYNTHESIS=`pwd`
paynt_exe="./paynt.py"

# ------------------------------------------------------------------------------
# functions
function paynt() {
    # argument settings
    local project="--project $1"
    local method="--method $2"

    local paynt_call="python3 ${paynt_exe} ${project} ${method} ${param} --hyper"
    echo \$ ${paynt_call}
    eval timeout ${timeout} ${paynt_call} >> ${log_file}
}

# empty the current content of the log file
#rm -rf $log_dir
#mkdir $log_dir
touch $log_file

cwd=$(pwd)
cd $projects_dir
dirs=$(find . -type d -exec sh -c '(ls -p "{}"|grep />/dev/null)||echo "{}"' \;)
cd $cwd
for d in $dirs; do
  paynt "${projects_dir}/${d}" $method
done

