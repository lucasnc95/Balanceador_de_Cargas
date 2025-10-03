mpic++ -o teste *.cpp -O3  -lOpenCL -fpermissive
#cluster
--Compila:
/opt/ohpc/pub/mpi/mpich-ofi-gnu13-ohpc/3.4.3/bin/mpic++ *.cpp -o load_balancer -L/usr/lib64/  -L/usr/local/cuda-12.6/targets/x86_64-linux/lib/ -L/opt/intel/oneapi/2024.0/lib/ -I/usr/local/cuda-12.6/targets/x86_64-linux/include/ -I/opt/intel/oneapi/2024.0/opt/oclfpga/host/include/ -lOpenCL -g
--executa
export LD_LIBRARY_PATH=/share/apps/AMDAPPSDK-3.0/lib/x86_64/:$LD_LIBRARY_PATH
/opt/ohpc/pub/mpi/mpich-ofi-gnu13-ohpc/3.4.3/bin/mpiexec ./load_balancer


job cluster

#!/bin/bash
#----------------------------------------------------------
# Job name
#PBS -N OpenCLWrapper

# Name of stdout output file
#PBS -o job.out

# Total number of nodes and MPI tasks/node requested
#PBS -l nodes=compute-1-0:ppn=1+compute-1-1:ppn=1

# Run time (hh:mm:ss) - 1.5 hours
#PBS -l walltime=01:30:00
#----------------------------------------------------------

# Change to submission directory
cd $PBS_O_WORKDIR

cat $PBS_NODEFILE
sort $PBS_NODEFILE | uniq > machines2
cat machines2
export LD_LIBRARY_PATH=/share/apps/AMDAPPSDK-3.0/lib/x86_64/:$LD_LIBRARY_PATH
# Launch MPI-based executable
/opt/ohpc/pub/mpi/mpich-ofi-gnu13-ohpc/3.4.3/bin/mpirun -np 2 -machinefile machines2 ./a.out

# OpenCLWrapper
