#PBS -l select=1:ncpus=4:mem=40gb
#PBS -l walltime=5:59:59
#PBS -q short_cpuQ

module load mpich-3.2
module load gcc91

cd $PBS_O_WORKDIR

data_files=("2000data.txt")
process_counts=(1 2 4 8 16)

for data_file in "${data_files[@]}"; do
    executable_name="cure_auto_${data_file%.txt}"
    compile_output_file="compile_output_${data_file%.txt}.txt"
    mpicc -g -Wall -fopenmp CURE_parallel.c -std=c99 -o "$executable_name" -lm -DDATA_FILE=\"$data_file\" > "$compile_output_file" 2>&1

    if [ ! -f "./$executable_name" ]; then
        echo "Compilation failed or $executable_name not found. See $compile_output_file for details."
        exit 1
    fi

    base_name="${data_file%.txt}"

    for num_procs in "${process_counts[@]}"; do
        export DATA_FILE_NAME="$base_name"
        
        mpiexec -n "$num_procs" ./"$executable_name" > "output_${data_file%.txt}_procs${num_procs}.txt" 2>&1
    done
done
