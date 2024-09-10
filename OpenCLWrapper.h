#ifndef OPENCLWRAPPER_H
#define OPENCLWRAPPER_H

#include <CL/cl.h>
#include <mpi.h>
#include <vector>
#include <climits>
#include <unordered_map>
#include <iostream>


class OpenCLWrapper {
public:
    OpenCLWrapper(int &argc, char** &argv);
    ~OpenCLWrapper();
    int InitDevices(const std::string &_device_types, const unsigned int _maxNumberOfDevices);
    void FinishParallelProcessor();

    int AllocateMemoryObject(size_t _size, cl_mem_flags _memoryType, void *_hostMemory);    
    int CreateKernel(int devicePosition, const char *source, const char *kernelName);
    void SetKernelAttribute(int devicePosition, int kernelID, int attribute, int memoryObjectID);
    void SetKernelArg(int kernelID, int argIndex, size_t argSize, const void *argValue);
    void ExecuteKernel();
    void GatherResults(int dataIndex,void *resultData);
    void setKernel(const std::string &sourceFile, const std::string &kernelName);
    int WriteToMemoryObject(int devicePosition, int memoryObjectID, const char *data, int offset, size_t size);
    int ReadFromMemoryObject(int devicePosition, int memoryObjectID, char *data, int offset, size_t size);
    void setLoadBalancer(size_t _elementSize, int N_Elements, int units_per_elements, int _divisionSize);
    void setSubdomainBoundary(size_t _sdSize, int _nArgs, int * _args);
    void setAttribute(int attribute, int globalMemoryObjectID);
    int WriteObject(int GlobalObjectID, const char *data, int offset, size_t size);
    void LoadBalancing();
    void setBalancingTargetID(int targetID);

    int getMaxNumberOfPlatforms() const;
    void setMaxNumberOfPlatforms(int value);
    int getMaxNumberOfDevices() const;
    void setMaxNumberOfDevices(int value);
    int getMaxNumberOfDevicesPerPlatform() const;
    void setMaxNumberOfDevicesPerPlatform(int value);
    int getMaxMemoryObjects() const;
    void setMaxMemoryObjects(int value);
    int getMaxKernels() const;
    void setMaxKernels(int value);
    int getMaxEvents() const;
    void setMaxEvents(int value);
    void Probing();
private:
    int InitParallelProcessor();
    void Initialize();
    
    void PrecisaoBalanceamento();
    void ComputarCargas(const long int *ticks, const float *cargasAntigas, float *cargasNovas, int participantes);
    int RecuperarPosicaoHistograma(int *histograma, int tamanho, int indice);
    bool ComputarIntersecao(int offset1, int length1, int offset2, int length2, int *intersecaoOffset, int *intersecaoLength);
    float ComputarDesvioPadraoPercentual(const long int *ticks, int participantes);
    float ComputarNorma(const float *cargasAntigas, const float *cargasNovas, int participantes);
    void initializeLengthOffset(int offset, int length, int deviceIndex);
    int CreateMemoryObject(int devicePosition, size_t size, cl_mem_flags memoryType, void *hostMemory);
    size_t elementSize;
    int divisionSize;
    int unitsPerElement;
    int balancingTargetID;
    int deviceIndex;
    long int itCounter = 0;
    int world_rank, world_size;
    bool kernelSet = false;
    bool loadBalancerSet = false;
    bool device_init = false;
    std::string kernelSourceFile;
    std::string kernelFunctionName;
    size_t sdSize;
    cl_platform_id *platformIDs;
    cl_uint numberOfPlatforms;
    cl_device_id *deviceList;
    cl_uint numberOfDevices;
    bool sdSet = false;
    struct Device {
        cl_device_id deviceID;
        cl_device_type deviceType;
        cl_context context;
        cl_command_queue kernelCommandQueue;
        cl_command_queue dataCommandQueue;
        cl_program program;
        cl_mem *memoryObjects;
        cl_kernel *kernels;
        int *memoryObjectID;
        int *kernelID;
        cl_event *events;
        size_t deviceMaxWorkItemsPerWorkGroup;
        cl_uint deviceComputeUnits;
        int numberOfMemoryObjects;
        int numberOfKernels;
        int numberOfEvents;
    };

    struct MemoryObject {
        size_t size; 
        cl_mem_flags flags; 
        void* host_ptr;
    };

    
    Device *devices;
   
    std::unordered_map<int, std::vector<int>> *memoryObjectIDs;
    int globalMemoryObjectIDCounter = 0;
    int automaticNumber = 0;

    // Variáveis do balanceador de carga
    
    long int offsetComputacao;
    long int lengthComputacao;
	int *dispositivosWorld;
    long int *ticks;
    double *tempos_por_carga;
    float *cargasNovas;
    float *cargasAntigas;
    int *DataToKernelDispositivo;
    int **swapBufferDispositivo;
    float *tempos;
    size_t *offset;
    size_t *length;
    float latencia, banda, tempoBalanceamento, fatorErro;
    int *kernelDispositivo;
    int *kernelEventoDispositivo;
    int *parametrosMalhaDispositivo;
    int **parametrosMalha;
    int DataToKernel_Size;
    int meusDispositivosOffset;
	int meusDispositivosLength;
    int todosDispositivos;
    int interv_balance;
    int nElements;
    int simulacao;
    int n_devices;
    int *offsetDispositivo; 
    int *lengthDispositivo;
    int *memObjects;
    int nArgs;
    int *args;
    // Parâmetros ajustáveis
    int maxNumberOfPlatforms = 10;
    int maxNumberOfDevices = 10;
    int maxNumberOfDevicesPerPlatform = 10;
    int maxMemoryObjects = 100;
    int maxKernels = 100;
    int maxEvents = 1000;
    int MAX_SOURCE_BUFFER_LENGTH = 1000000;
    std::string device_types; 
    double tempoCB;
    int Maximum(int a, int b);
    int GetMemoryObjectPosition(int devicePosition, int memoryObjectID);
    int GetKernelPosition(int devicePosition, int kernelID);

    int RunKernel(int devicePosition, int kernelID, int parallelDataOffset, size_t parallelData, int workGroupSize);
    void SynchronizeCommandQueue(int devicePosition);
    void SynchronizeEvent(int eventPosition);
    long int GetEventTaskOverheadTicks(int devicePosition, int eventPosition);
    long int GetEventTaskTicks(int devicePosition, int eventPosition);
    int GetDeviceMemoryObjectID(int globalMemObjID, int deviceIndex);
    cl_device_type GetDeviceType();
    size_t GetDeviceMaxWorkItemsPerWorkGroup();
    cl_uint GetDeviceComputeUnits();
    bool isDeviceCPU(int devicePosition);
    void Comms();

    bool RemoveKernel(int devicePosition, int kernelID);
    bool RemoveMemoryObject(int devicePosition, int memoryObjectID);
    
};

#endif
