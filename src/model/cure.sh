#PBS -l select=1:ncpus=4:mem=40gb
#PBS -l walltime=5:59:59
#PBS -q short_cpuQ

module load mpich-3.2
module load gcc91

cd $PBS_O_WORKDIR

mpicc -g -Wall -fopenmp curev1.c -std=c99 -o curev1 -lm > compile_output.txt 2>&1

if [ ! -f "./curev1" ]; then
    echo "Compilation failed or cure2_exec not found."
    exit 1
fi

# Run the executable and tail the output in real-time
mpiexec -n 8 ./curev1 full_data.txt > output.txt 2>&1 
