#PBS -l select=1:ncpus=4:mem=40gb
#PBS -l walltime=5:59:59
#PBS -q short_cpuQ

# Load necessary modules
module load mpich-3.2
module load gcc91

# Change directory to the working directory of the job
cd $PBS_O_WORKDIR

# Define list of data files to be processed
data_files=("2000data.txt")
# Specify the numbers of processes to be used for each execution
process_counts=(1 2 4 8 16)

# Loop over each data file
for data_file in "${data_files[@]}"; do
    # Compile the C program for each data file and redirect compile output to a file
    executable_name="cure_auto_${data_file%.txt}"
    compile_output_file="compile_output_${data_file%.txt}.txt"
    mpicc -g -Wall -fopenmp cure_auto.c -std=c99 -o "$executable_name" -lm -DDATA_FILE=\"$data_file\" > "$compile_output_file" 2>&1

    # Check if compilation was successful
    if [ ! -f "./$executable_name" ]; then
        echo "Compilation failed or $executable_name not found. See $compile_output_file for details."
        exit 1
    fi

    # Loop over each process count
    for num_procs in "${process_counts[@]}"; do
        # Execute the program with the current number of processes for the current data file
        mpiexec -n "$num_procs" ./"$executable_name" > "output_${data_file%.txt}_procs${num_procs}.txt" 2>&1
    done
done
