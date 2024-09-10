mpic++ -o teste *.cpp -O3  -lOpenCL -fpermissive

#cluster
--Compila:
/opt/ohpc/pub/mpi/mpich-ofi-gnu13-ohpc/3.4.3/bin/mpic++ *.cpp -o load_balancer -L/usr/lib64/  -L/usr/local/cuda-12.6/targets/x86_64-linux/lib/ -L/opt/intel/oneapi/2024.0/lib/ -I/usr/local/cuda-12.6/targets/x86_64-linux/include/ -I/opt/intel/oneapi/2024.0/opt/oclfpga/host/include/ -lOpenCL -g
--executa
export LD_LIBRARY_PATH=/share/apps/AMDAPPSDK-3.0/lib/x86_64/:$LD_LIBRARY_PATH
/opt/ohpc/pub/mpi/mpich-ofi-gnu13-ohpc/3.4.3/bin/mpiexec ./load_balancer
