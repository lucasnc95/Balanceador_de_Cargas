locate libOpenCL export LD_LIBRARY_PATH=/opt/intel/oneapi/lib/:/usr/local/cuda-12.3/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH vi machines

mpic++ -I/usr/local/cuda-12.3/targets/x86_64-linux/include/ -L/opt/intel/oneapi/lib/ -D CL_TARGET_OPENCL_VERSION=120 -D MAX_NUMBER_OF_DEVICES_PER_PLATFORM=10 -D ALL_DEVICES  -O3 -o HIS_Dinamico_ALL Balanceador.cpp OpenCLWrapper.cpp main.cpp -lOpenCL -lrt

mpiexec -machinefile machines -n 1 ./HIS_Dinamico_ALL 50 50 320

mpic++ -I/usr/local/cuda-12.3/targets/x86_64-linux/include/ -L/opt/intel/oneapi/lib/ -D CL_TARGET_OPENCL_VERSION=120 -D MAX_NUMBER_OF_DEVICES_PER_PLATFORM=10 -D ALL_DEVICES -O3 -o HIS_Dinamico_ALL Balanceador.cpp OpenCLWrapper.cpp main.cpp -lOpenCL -lrt -g

mpiexec -machinefile machines -n 1 xterm -e gdb ./HIS_Dinamico_ALL 50 50 320
