#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <mpi.h>

// Constantes
const int MAX_SOURCE_BUFFER_LENGTH = 1000000;
const int MAX_NUMBER_OF_PLATFORMS = 10;
const int MAX_NUMBER_OF_DEVICES = 10;
const int MAX_NUMBER_OF_DEVICES_PER_PLATFORM = 10;
const int MAX_MEMORY_OBJECTS = 100;
const int MAX_KERNELS = 100;
const int MAX_EVENTS = 1000;

cl_platform_id platformIDs[MAX_NUMBER_OF_PLATFORMS];

struct Device {
    cl_device_id deviceID;
    cl_device_type deviceType;
    cl_context context;
    cl_command_queue kernelCommandQueue;
    cl_command_queue dataCommandQueue;
    cl_program program;

    cl_mem memoryObjects[MAX_MEMORY_OBJECTS];
    cl_kernel kernels[MAX_KERNELS];

    int memoryObjectID[MAX_MEMORY_OBJECTS];
    int kernelID[MAX_KERNELS];

    cl_event events[MAX_EVENTS];

    size_t deviceMaxWorkItemsPerWorkGroup;
    cl_uint deviceComputeUnits;

    int numberOfMemoryObjects;
    int numberOfKernels;
    int numberOfEvents;
};

cl_uint numberOfDevices;
Device devices[MAX_NUMBER_OF_DEVICES];

int automaticNumber = 0;

int Maximum(int a, int b) {
    return (a > b) ? a : b;
}

int GetMemoryObjectPosition(int devicePosition, int memoryObjectID) {
    for (int count = 0; count < devices[devicePosition].numberOfMemoryObjects; count++) {
        if (devices[devicePosition].memoryObjectID[count] == memoryObjectID) {
            return count;
        }
    }
    return -1;
}

int GetKernelPosition(int devicePosition, int kernelID) {
    for (int count = 0; count < devices[devicePosition].numberOfKernels; count++) {
        if (devices[devicePosition].kernelID[count] == kernelID) {
            return count;
        }
    }
    return -1;
}

int InitParallelProcessor() {
    cl_int state;

    // Get platforms.
    cl_uint numberOfPlatforms = 0;
    state = clGetPlatformIDs(MAX_NUMBER_OF_PLATFORMS, platformIDs, &numberOfPlatforms);
    if (state != CL_SUCCESS) {
        std::cerr << "OpenCL Error: Platform couldn't be found.\n";
    }
    std::cout << numberOfPlatforms << " platform(s) found.\n";

    cl_device_id deviceList[MAX_NUMBER_OF_DEVICES];
    numberOfDevices = 0;
    for (int platform = 0; platform < numberOfPlatforms; platform++) {
        // Get devices.
        cl_uint numberOfDevicesOfPlatform;
        state = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL, MAX_NUMBER_OF_DEVICES_PER_PLATFORM, deviceList + numberOfDevices, &numberOfDevicesOfPlatform);
        if (state != CL_SUCCESS) {
            std::cerr << "OpenCL Error: Devices couldn't be resolved.\n";
        } else {
            if (numberOfDevicesOfPlatform > MAX_NUMBER_OF_DEVICES_PER_PLATFORM) {
                numberOfDevicesOfPlatform = MAX_NUMBER_OF_DEVICES_PER_PLATFORM;
            }
            std::cout << numberOfDevicesOfPlatform << " device(s) found on platform " << platform << ".\n";
        }

        // Set devices.
        for (int count = numberOfDevices; count < numberOfDevices + numberOfDevicesOfPlatform; count++) {
            // Get ID.
            devices[count].deviceID = deviceList[count];

            // Get type.
            clGetDeviceInfo(devices[count].deviceID, CL_DEVICE_TYPE, sizeof(cl_device_type), &devices[count].deviceType, NULL);
            clGetDeviceInfo(devices[count].deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &devices[count].deviceMaxWorkItemsPerWorkGroup, NULL);
            clGetDeviceInfo(devices[count].deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &devices[count].deviceComputeUnits, NULL);

            if (devices[count].deviceType == CL_DEVICE_TYPE_GPU) {
                std::cout << "Device (" << count << ") type: GPU\n";
            } else if (devices[count].deviceType == CL_DEVICE_TYPE_CPU) {
                std::cout << "Device (" << count << ") type: CPU\n";
            }

            // Create context.
            devices[count].context = clCreateContext(NULL, 1, &devices[count].deviceID, NULL, NULL, &state);
            if (state != CL_SUCCESS) {
                std::cerr << "OpenCL Error: Context couldn't be created.\n";
            }

            // Create command queues.
            devices[count].kernelCommandQueue = clCreateCommandQueue(devices[count].context, devices[count].deviceID, CL_QUEUE_PROFILING_ENABLE, &state);
            if (state != CL_SUCCESS) {
                std::cerr << "OpenCL Error: Kernel command queue couldn't be created.\n";
            }
            devices[count].dataCommandQueue = clCreateCommandQueue(devices[count].context, devices[count].deviceID, CL_QUEUE_PROFILING_ENABLE, &state);
            if (state != CL_SUCCESS) {
                std::cerr << "OpenCL Error: Data command queue couldn't be created.\n";
            }

            // Initialize memory objects, kernel and events.
            devices[count].numberOfMemoryObjects = 0;
            devices[count].numberOfKernels = 0;
            devices[count].numberOfEvents = 0;

            std::memset(devices[count].memoryObjects, 0, sizeof(cl_mem) * MAX_MEMORY_OBJECTS);
            std::memset(devices[count].kernels, 0, sizeof(cl_kernel) * MAX_KERNELS);
            std::memset(devices[count].memoryObjectID, 0, sizeof(int) * MAX_MEMORY_OBJECTS);
            std::memset(devices[count].kernelID, 0, sizeof(int) * MAX_KERNELS);
            std::memset(devices[count].events, 0, sizeof(cl_event) * MAX_EVENTS);

            devices[count].program = 0;
        }
        numberOfDevices += numberOfDevicesOfPlatform;
    }

    return numberOfDevices;
}

void FinishParallelProcessor() {
    for (int count = 0; count < numberOfDevices; count++) {
        clFlush(devices[count].kernelCommandQueue);
        clFinish(devices[count].kernelCommandQueue);

        clFlush(devices[count].dataCommandQueue);
        clFinish(devices[count].dataCommandQueue);

        for (int count2 = 0; count2 < devices[count].numberOfKernels; count2++) {
            clReleaseKernel(devices[count].kernels[count2]);
        }
        clReleaseProgram(devices[count].program);
        for (int count2 = 0; count2 < devices[count].numberOfMemoryObjects; count2++) {
            clReleaseMemObject(devices[count].memoryObjects[count2]);
        }
        clReleaseCommandQueue(devices[count].kernelCommandQueue);
        clReleaseCommandQueue(devices[count].dataCommandQueue);
    }
}

int CreateKernel(int devicePosition, const char *source, const char *kernelName) {
    if (devices[devicePosition].program != 0) {
        clReleaseProgram(devices[devicePosition].program);
    }
    devices[devicePosition].program = 0;

    cl_int state;
    FILE *fileHandle;
    char *sourceBuffer = (char *)malloc(sizeof(char) * MAX_SOURCE_BUFFER_LENGTH);
    if ((fileHandle = fopen(source, "r")) == NULL) {
        std::cerr << "Error reading " << source << "\n";
        return -1;
    }
    size_t sourceBufferLength = fread(sourceBuffer, 1, sizeof(char) * MAX_SOURCE_BUFFER_LENGTH, fileHandle);
    devices[devicePosition].program = clCreateProgramWithSource(devices[devicePosition].context, 1, (const char **)&sourceBuffer, (const size_t *)&sourceBufferLength, &state);

    fclose(fileHandle);
    free(sourceBuffer);

    if (state != CL_SUCCESS) {
        std::cerr << "Error creating program!\n";
        return -1;
    }

    state = clBuildProgram(devices[devicePosition].program, 1, &devices[devicePosition].deviceID, NULL, NULL, NULL);
    if (state != CL_SUCCESS) {
        std::cerr << "Error compiling program!\n";
        return -1;
    }

    devices[devicePosition].kernels[devices[devicePosition].numberOfKernels] = clCreateKernel(devices[devicePosition].program, kernelName, &state);
    if (state != CL_SUCCESS) {
        std::cerr << "Error creating kernel!\n";
        return -1;
    }
    devices[devicePosition].kernelID[devices[devicePosition].numberOfKernels] = automaticNumber;
    devices[devicePosition].numberOfKernels += 1;
    automaticNumber += 1;
    return automaticNumber - 1;
}

template<typename T>
void SetKernelAttributeInternal(int devicePosition, int kernelID, int attribute, const T& value) {
    int kernelPosition = GetKernelPosition(devicePosition, kernelID);
    if (kernelPosition != -1) {
        cl_int state = clSetKernelArg(devices[devicePosition].kernels[kernelPosition], attribute, sizeof(T), &value);
        if (state != CL_SUCCESS) {
            std::cerr << "Error setting kernel argument!\n";
        }
    } else {
        std::cerr << "Error setting kernel argument: Kernel ID=" << kernelID << " doesn't exist!\n";
    }
}

template<>
void SetKernelAttributeInternal<cl_mem>(int devicePosition, int kernelID, int attribute, const cl_mem& value) {
    int kernelPosition = GetKernelPosition(devicePosition, kernelID);
    if (kernelPosition != -1) {
        cl_int state = clSetKernelArg(devices[devicePosition].kernels[kernelPosition], attribute, sizeof(cl_mem), &value);
        if (state != CL_SUCCESS) {
            std::cerr << "Error setting kernel argument!\n";
        }
    } else {
        std::cerr << "Error setting kernel argument: Kernel ID=" << kernelID << " doesn't exist!\n";
    }
}

template<typename... Args>
void SetKernelAttribute(int devicePosition, int kernelID, int attribute, const Args&... args) {
    (SetKernelAttributeInternal(devicePosition, kernelID, attribute, args), ...);
}



bool RemoveKernel(int devicePosition, int kernelID) {
    int kernelPosition = GetKernelPosition(devicePosition, kernelID);
    if (kernelPosition != -1) {
        clReleaseKernel(devices[devicePosition].kernels[kernelPosition]);
        std::memcpy(devices[devicePosition].kernels + kernelPosition, devices[devicePosition].kernels + kernelPosition + 1, sizeof(cl_kernel) * (devices[devicePosition].numberOfKernels - 1));
        std::memcpy(devices[devicePosition].kernelID + kernelPosition, devices[devicePosition].kernelID + kernelPosition + 1, sizeof(int) * (devices[devicePosition].numberOfKernels - 1));
        devices[devicePosition].numberOfKernels -= 1;
        return true;
    }
    return false;
}

int CreateMemoryObject(int devicePosition, int size, cl_mem_flags memoryType, void *hostMemory) {
    cl_int state;
    if (devices[devicePosition].numberOfMemoryObjects < MAX_MEMORY_OBJECTS) {
        devices[devicePosition].memoryObjects[devices[devicePosition].numberOfMemoryObjects] = clCreateBuffer(devices[devicePosition].context, memoryType, size, hostMemory, &state);
        if (state != CL_SUCCESS) {
            std::cerr << "Error creating memory object!\n";
            return -1;
        } else {
            devices[devicePosition].memoryObjectID[devices[devicePosition].numberOfMemoryObjects] = automaticNumber;
            devices[devicePosition].numberOfMemoryObjects += 1;
        }
        automaticNumber += 1;
        return automaticNumber - 1;
    }
    std::cerr << "Error creating memory object, limit exceeded!";
    return -1;
}

int WriteToMemoryObject(int devicePosition, int memoryObjectID, const char *data, int offset, int size) {
    cl_int state;
    int memoryObjectPosition = GetMemoryObjectPosition(devicePosition, memoryObjectID);
    if (memoryObjectPosition != -1 && devices[devicePosition].numberOfEvents < MAX_EVENTS) {
        state = clEnqueueWriteBuffer(devices[devicePosition].dataCommandQueue, devices[devicePosition].memoryObjects[memoryObjectPosition], CL_FALSE, offset, size, data, 0, NULL, &devices[devicePosition].events[devices[devicePosition].numberOfEvents]);
        if (state != CL_SUCCESS) {
            std::cerr << "Error writing to memory object " << state << ".\n";
        } else {
            clFlush(devices[devicePosition].dataCommandQueue);
            devices[devicePosition].numberOfEvents += 1;
            return devices[devicePosition].numberOfEvents - 1;
        }
    }

    std::cerr << "Error! Couldn't find memory object position " << memoryObjectPosition << " or number of events " << devices[devicePosition].numberOfEvents << " exceeded limit.\n";
    return -1;
}

int ReadFromMemoryObject(int devicePosition, int memoryObjectID, char *data, int offset, int size) {
    cl_int state;
    int memoryObjectPosition = GetMemoryObjectPosition(devicePosition, memoryObjectID);
    if (memoryObjectPosition != -1 && devices[devicePosition].numberOfEvents < MAX_EVENTS) {
        state = clEnqueueReadBuffer(devices[devicePosition].dataCommandQueue, devices[devicePosition].memoryObjects[memoryObjectPosition], CL_FALSE, offset, size, data, 0, NULL, &devices[devicePosition].events[devices[devicePosition].numberOfEvents]);
        if (state != CL_SUCCESS) {
            std::cerr << "Error reading from memory object " << state << ".\n";
            return -1;
        } else {
            clFlush(devices[devicePosition].dataCommandQueue);
            devices[devicePosition].numberOfEvents += 1;
            return devices[devicePosition].numberOfEvents - 1;
        }
    }

    std::cerr << "Error! Couldn't find memory object position " << memoryObjectPosition << " or number of events " << devices[devicePosition].numberOfEvents << " exceeded limit.\n";
    return -1;
}

bool RemoveMemoryObject(int devicePosition, int memoryObjectID) {
    int memoryObjectPosition = GetMemoryObjectPosition(devicePosition, memoryObjectID);
    if (memoryObjectPosition != -1) {
        clReleaseMemObject(devices[devicePosition].memoryObjects[memoryObjectPosition]);
        std::memcpy(devices[devicePosition].memoryObjects + memoryObjectPosition, devices[devicePosition].memoryObjects + memoryObjectPosition + 1, sizeof(cl_mem) * (devices[devicePosition].numberOfMemoryObjects - 1));
        std::memcpy(devices[devicePosition].memoryObjectID + memoryObjectPosition, devices[devicePosition].memoryObjectID + memoryObjectPosition + 1, sizeof(int) * (devices[devicePosition].numberOfMemoryObjects - 1));
        devices[devicePosition].numberOfMemoryObjects -= 1;
        return true;
    }
    return false;
}

int RunKernel(int devicePosition, int kernelID, int parallelDataOffset, int parallelData, int workGroupSize) {
    int kernelPosition = GetKernelPosition(devicePosition, kernelID);
    if (kernelPosition != -1 && devices[devicePosition].numberOfEvents < MAX_EVENTS) {
        size_t globalItemsOffset = Maximum(parallelDataOffset, 0);
        size_t globalItems = Maximum(workGroupSize, parallelData + workGroupSize - (parallelData % workGroupSize));

        cl_int state;
        size_t localItems = workGroupSize;

        state = clEnqueueNDRangeKernel(devices[devicePosition].kernelCommandQueue, devices[devicePosition].kernels[kernelPosition], 1, &globalItemsOffset, &globalItems, &localItems, 0, NULL, &devices[devicePosition].events[devices[devicePosition].numberOfEvents]);
        if (state != CL_SUCCESS) {
            std::cerr << "Error queueing task! " << state << "\n";
            return -1;
        } else {
            clFlush(devices[devicePosition].kernelCommandQueue);
            devices[devicePosition].numberOfEvents += 1;
            return devices[devicePosition].numberOfEvents - 1;
        }
    }

    std::cerr << "Error! Couldn't find kernel position " << kernelPosition << " or number of events " << devices[devicePosition].numberOfEvents << " exceeded limit.\n";
    return -1;
}

void SynchronizeCommandQueue(int devicePosition) {
    clFinish(devices[devicePosition].kernelCommandQueue);
    clFinish(devices[devicePosition].dataCommandQueue);
    devices[devicePosition].numberOfEvents = 0;
}

void SynchronizeEvent(int devicePosition, int eventPosition) {
    clWaitForEvents(1, &devices[devicePosition].events[eventPosition]);
}

long int GetEventTaskOverheadTicks(int devicePosition, int eventPosition) {
    long int ticksStart;
    long int ticksEnd;

    clGetEventProfilingInfo(devices[devicePosition].events[eventPosition], CL_PROFILING_COMMAND_QUEUED, sizeof(long int), &ticksStart, NULL);
    clGetEventProfilingInfo(devices[devicePosition].events[eventPosition], CL_PROFILING_COMMAND_START, sizeof(long int), &ticksEnd, NULL);
    return (ticksEnd - ticksStart);
}

long int GetEventTaskTicks(int devicePosition, int eventPosition) {
    long int ticksStart;
    long int ticksEnd;

    clGetEventProfilingInfo(devices[devicePosition].events[eventPosition], CL_PROFILING_COMMAND_START, sizeof(long int), &ticksStart, NULL);
    clGetEventProfilingInfo(devices[devicePosition].events[eventPosition], CL_PROFILING_COMMAND_END, sizeof(long int), &ticksEnd, NULL);
    return (ticksEnd - ticksStart);
}

cl_device_type GetDeviceType(int devicePosition) {
    return devices[devicePosition].deviceType;
}

size_t GetDeviceMaxWorkItemsPerWorkGroup(int devicePosition) {
    return devices[devicePosition].deviceMaxWorkItemsPerWorkGroup;
}

cl_uint GetDeviceComputeUnits(int devicePosition) {
    return devices[devicePosition].deviceComputeUnits;
}

bool isDeviceCPU(int devicePosition) {
    return devices[devicePosition].deviceType == CL_DEVICE_TYPE_CPU;
}




int main(int argc, char **argv) {
    // Inicializar MPI
   // MPI_Init(&argc, &argv);

    // Inicializar OpenCL e obter dispositivos
    int num_devices = InitParallelProcessor();

    // Criar buffers e dados de exemplo
    const int N = 1024;
    float *buffer_A = new float[N * N];
    float *buffer_B = new float[N * N];
    float *buffer_C = new float[N * N];

    // Inicializar buffers de dados
    for (int i = 0; i < N * N; ++i) {
        buffer_A[i] = static_cast<float>(i);
        buffer_B[i] = static_cast<float>(i);
        buffer_C[i] = 0.0f;
    }

    // Criar objetos de memória no OpenCL
    int mem_A = CreateMemoryObject(0, N * N * sizeof(float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_A);
    int mem_B = CreateMemoryObject(0, N * N * sizeof(float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_B);
    int mem_C = CreateMemoryObject(0, N * N * sizeof(float), CL_MEM_WRITE_ONLY, nullptr);

    // Criar e compilar o kernel
    int kernel_id = CreateKernel(0, "kernel.cl", "matrix_multiply");

    // Setar atributos do kernel
    SetKernelAttribute(0, kernel_id, 0, devices[0].memoryObjects[mem_A], devices[0].memoryObjects[mem_B], devices[0].memoryObjects[mem_C], N);

    // Executar o kernel
    RunKernel(0, kernel_id, 0, N * N, 16);

    // Sincronizar e obter os resultados
    SynchronizeCommandQueue(0);
    ReadFromMemoryObject(0, mem_C, reinterpret_cast<char*>(buffer_C), 0, N * N * sizeof(float));
    SynchronizeCommandQueue(0);

    // Liberar recursos
    FinishParallelProcessor();
    MPI_Finalize();

    // Imprimir resultado para verificação
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << buffer_C[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // Limpar buffers de dados
    delete[] buffer_A;
    delete[] buffer_B;
    delete[] buffer_C;

    return 0;
}
