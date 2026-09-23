#!/bin/bash
# Execute run for a number of files
#
#machine=Dell
machine=JZ

case="FIRZ4"
ln -sf ../infos/info_run_${machine}_${case}.txt ../infos/info_run.txt
ln -sf ../infos/files_${case}.txt ../infos/files.txt

# Need to run the first file
file_sh="run_"$machine".sh"

listfiles="../infos/files.txt"
while IFS= read -r varname; do
    printf '%s\n' "$varname"
    sbatch $file_sh $varname
done < "$listfiles"
