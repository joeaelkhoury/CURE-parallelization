#PBS -l select=1:ncpus=4:mem=40gb
#PBS -l walltime=1:59:59
#PBS -q short_cpuQ

# Load necessary modules
module load mpich-3.2
module load gcc91

# cd /home/joe.elkhoury/project
cd $PBS_O_WORKDIR

# Compile the code
mpicc -g -Wall -fopenmp CURE.c -std=c99 -o out4 -lm > compile_outputwith1.txt 2>&1
gdb ./out4


# Run the compiled code with MPI
mpiexec -n 1 ./out4 4 augmented_data4.txt > outwith1.txt 2>&1


