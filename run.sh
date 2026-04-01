# This is a SLURM batch script to run the Python program that generates fractal images using MPI for parallel processing. The script requests resources, sets up the environment, and runs the program with different complex numbers as input. I used this to generate the huge fractal images for my project of size 30,000x30,000 pixels which is almost 1 billion pixels!

#!/bin/bash
#SBATCH --nodes=2 # Run on two nodes
#SBATCH --ntasks-per-node=40 # 1 process per core (40 cores per node)
#SBATCH --time=12:00:00 # My job will take at most 12 hours to run, so find and reserve resources for 12 hours
#SBATCH --job-name=Fractal-Generation # Name of my job
#SBATCH --output=Fractal-Generation%j.txt # Write the output to Fractal-Generation<jobID>.txt
#SBATCH --error=Fractal-Generation%j.err # Write stderr to Fractal-Generation<jobID>.err

cd $SLURM_SUBMIT_DIR
module restore TP_MODULES

dimension_of_image=30000
processor_count=80

# Complex Numbers to use for the fractal generation
mpirun -np $processor_count python3 ./main.py -- -1 $dimension_of_image
mpirun -np $processor_count python3 ./main.py -- 0.3-0.4j $dimension_of_image
mpirun -np $processor_count python3 ./main.py -- 0.360284+0.100376j $dimension_of_image
mpirun -np $processor_count python3 ./main.py -- -0.1+0.8j $dimension_of_image