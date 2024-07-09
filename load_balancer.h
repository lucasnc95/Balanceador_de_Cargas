#ifndef LOAD_BALANCER_H
#define LOAD_BALANCER_H

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <mpi.h>
#include <CL/cl.hpp>
//#include "device_info.h"
//#include "kernel.h"

class LoadBalancer {
public:
    LoadBalancer();
    ~LoadBalancer();

    void initializeMPI(int argc, char *argv[]);
    void finalizeMPI();
    void initializeOpenCLDevices();
    void createKernel(const std::string& source_file, const std::string& kernel_name, const std::vector<size_t>& arg_sizes, const std::vector<void*>& arg_values);
    void probing(int steps, bool use_default_kernel = false);
    void balanceLoad();
    void executeKernel();
    void gatherData();
    void setDataDivision(const std::vector<int>& custom_division);

private:
    void computeLoad(const long int *ticks, const float *old_loads, float *new_loads, int participants);
    bool computeIntersection(int offset1, int length1, int offset2, int length2, int *intersect_offset, int *intersect_length);
    int getHistogramPosition(const std::vector<int>& histogram, int index);
    float computeStdDevPercent(const long int *ticks, int participants);
    float computeNorm(const float *old_loads, const float *new_loads, int participants);
    void initDeviceLengthsOffsets(unsigned int offset, unsigned int length, int count);
    void precisionBalance();
    void Probing();

    int world_size, world_rank;
    int myDeviceOffset, myDeviceLength, totalDevices;
    std::vector<Device_Info> devices;
    std::vector<int> devicesWorld;
    std::vector<int> deviceOffsets;
    std::vector<unsigned long int> deviceLengths;

    cl::Kernel kernel;
    cl::Program program;

    long int *ticks;
    double *times;
    float *new_loads;
    float *old_loads;

    // Additional fields for load balancing
    double writeByte, banda, latencia, tempoComputacaoInterna, tempoBalanceamento;
    std::vector<int> DataToKernelDispositivo;
    std::vector<int> swapBufferDispositivo;
    std::vector<int> kernelDispositivo;
    std::vector<int> kernelEventoDispositivo;
    std::vector<int> dataEventoDispositivo;
    std::vector<unsigned int> offset;
    std::vector<unsigned long int> length;
    bool custom_type_set;
    MPI_Datatype mpi_data_type, mpi_custom_type;
    int CPU_WORK_GROUP_SIZE, GPU_WORK_GROUP_SIZE, PRECISAO_BALANCEAMENTO;
};

#endif // LOAD_BALANCER_H
