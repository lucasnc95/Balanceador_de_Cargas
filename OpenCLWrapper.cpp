#include "OpenCLWrapper.h"
#include <cstring>
#include <cmath>

OpenCLWrapper::OpenCLWrapper(int &argc, char** &argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
}

OpenCLWrapper::~OpenCLWrapper() {
  
   for (int count = 0; count < numberOfDevices; count++) {
        if (devices[count].context != nullptr) {
            clReleaseContext(devices[count].context);
        }
        for (int i = 0; i < devices[count].numberOfKernels; i++) {
            clReleaseKernel(devices[count].kernels[i]);
        }
        for (int j = 0; j < devices[count].numberOfMemoryObjects; j++) {
            clReleaseMemObject(devices[count].memoryObjects[j]);
        }
        clReleaseCommandQueue(devices[count].kernelCommandQueue);
        clReleaseCommandQueue(devices[count].dataCommandQueue);
        
        delete[] devices[count].memoryObjects;
        delete[] devices[count].kernels;
        delete[] devices[count].memoryObjectID;
        delete[] devices[count].kernelID;
        delete[] devices[count].events;
    }
    delete[] devices;
    MPI_Finalize();
}



int OpenCLWrapper::InitDevices(const std::string &_device_types, const unsigned int _maxNumberOfDevices )
{   maxNumberOfDevices = _maxNumberOfDevices;
    device_types = _device_types;
	int dispositivos;
    dispositivos = InitParallelProcessor();
    int dispositivosLocal[world_size];
	dispositivosWorld = new int[world_size];
	
	memset(dispositivosLocal, 0, sizeof(int)*world_size);
	dispositivosLocal[world_rank] = dispositivos;
    MPI_Allreduce(dispositivosLocal, dispositivosWorld, world_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    todosDispositivos = 0;
	
	
	for(int count = 0; count < world_size; count++)
	{
		if(count == world_rank)
		{
			meusDispositivosOffset = todosDispositivos;
			meusDispositivosLength = dispositivosWorld[count];
		}
		todosDispositivos += dispositivosWorld[count];
	}
    
    device_init = true;
	memoryObjectIDs = new std::unordered_map<int, std::vector<int>>();
 return todosDispositivos;

}


int OpenCLWrapper::InitParallelProcessor()
{
    cl_int state;
    platformIDs = new cl_platform_id[maxNumberOfPlatforms];
    if (!platformIDs) {
        printf("Memory allocation failed for platformIDs.\n");
        return -1;
    }

    cl_uint numberOfPlatforms = 0;
    state = clGetPlatformIDs(maxNumberOfPlatforms, platformIDs, &numberOfPlatforms);
    if (state != CL_SUCCESS || numberOfPlatforms == 0) {
        printf("OpenCL Error: Platform couldn't be found.\n");
        return -1;
    }
    printf("%i platform(s) found.\n", numberOfPlatforms);

    cl_device_id deviceList[maxNumberOfDevices];
    devices = new Device[maxNumberOfDevices];
    if (!devices) {
        printf("Memory allocation failed for devices.\n");
        return -1;
    }

    numberOfDevices = 0;
    int minMajorVersion = INT_MAX, minMinorVersion = INT_MAX;

    for (int platform = 0; platform < numberOfPlatforms; platform++) {
        cl_uint numberOfDevicesOfPlatform = 0;

        if (device_types == "CPU_DEVICES") {
            state = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_CPU, maxNumberOfDevicesPerPlatform, deviceList + numberOfDevices, &numberOfDevicesOfPlatform);
        } else if (device_types == "GPU_DEVICES") {
            state = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_GPU, maxNumberOfDevicesPerPlatform, deviceList + numberOfDevices, &numberOfDevicesOfPlatform);
        } else {
            state = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL, maxNumberOfDevicesPerPlatform, deviceList + numberOfDevices, &numberOfDevicesOfPlatform);
        }

        if (state != CL_SUCCESS || numberOfDevicesOfPlatform == 0) {
            printf("OpenCL Error: Devices couldn't be resolved on platform %d.\n", platform);
            continue;
        }

        for (int count = numberOfDevices; count < numberOfDevices + numberOfDevicesOfPlatform; count++) {
            devices[count].deviceID = deviceList[count];

            clGetDeviceInfo(devices[count].deviceID, CL_DEVICE_TYPE, sizeof(cl_device_type), &devices[count].deviceType, NULL);
            char versionStr[128];
            clGetDeviceInfo(devices[count].deviceID, CL_DEVICE_VERSION, sizeof(versionStr), versionStr, NULL);

            int majorVersion = 0, minorVersion = 0;
            sscanf(versionStr, "OpenCL %d.%d", &majorVersion, &minorVersion);
            printf("Device (%i) supports OpenCL version: %d.%d\n", count, majorVersion, minorVersion);

            if (majorVersion < minMajorVersion || (majorVersion == minMajorVersion && minorVersion < minMinorVersion)) {
                minMajorVersion = majorVersion;
                minMinorVersion = minorVersion;
            }
        }

        numberOfDevices += numberOfDevicesOfPlatform;
    }

    printf("Minimum OpenCL version supported by all devices: %d.%d\n", minMajorVersion, minMinorVersion);

    for (int count = 0; count < numberOfDevices; count++) {
        cl_context_properties contextProperties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[count],
            0
        };

        devices[count].context = clCreateContext(contextProperties, 1, &devices[count].deviceID, NULL, NULL, &state);
        if (state != CL_SUCCESS) {
            printf("OpenCL Error: Context couldn't be created for device %d.\n", count);
            devices[count].context = NULL;
            continue;
        }

        cl_command_queue_properties properties[] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
            0
        };

        devices[count].kernelCommandQueue = clCreateCommandQueueWithProperties(devices[count].context, devices[count].deviceID, properties, &state);
        if (state != CL_SUCCESS) {
            printf("OpenCL Error: Kernel command queue couldn't be created for device %d.\n", count);
            clReleaseContext(devices[count].context);
            devices[count].context = NULL;
            continue;
        }

        devices[count].dataCommandQueue = clCreateCommandQueueWithProperties(devices[count].context, devices[count].deviceID, properties, &state);
        if (state != CL_SUCCESS) {
            printf("OpenCL Error: Data command queue couldn't be created for device %d.\n", count);
            clReleaseCommandQueue(devices[count].kernelCommandQueue);
            clReleaseContext(devices[count].context);
            devices[count].context = NULL;
            continue;
        }

        devices[count].numberOfMemoryObjects = 0;
        devices[count].numberOfKernels = 0;
        devices[count].numberOfEvents = 0;

        devices[count].memoryObjects = new cl_mem[maxMemoryObjects];
        devices[count].kernels = new cl_kernel[maxKernels];
        memset(devices[count].memoryObjects, 0, sizeof(cl_mem) * maxMemoryObjects);
        memset(devices[count].kernels, 0, sizeof(cl_kernel) * maxKernels);

        devices[count].memoryObjectID = new int[maxMemoryObjects];
        devices[count].kernelID = new int[maxKernels];
        memset(devices[count].memoryObjectID, 0, sizeof(int) * maxMemoryObjects);
        memset(devices[count].kernelID, 0, sizeof(int) * maxKernels);

        devices[count].events = new cl_event[maxEvents];
        memset(devices[count].events, 0, sizeof(cl_event) * maxEvents);

        devices[count].program = 0;
    }

    return numberOfDevices;
}



// int OpenCLWrapper::InitParallelProcessor() {
//     cl_int state;
//     platformIDs = new cl_platform_id[maxNumberOfPlatforms];
//     if (!platformIDs) {
//         printf("Memory allocation failed for platformIDs.\n");
//         return -1;
//     }

//     // Obter plataformas
//     cl_uint numberOfPlatforms = 0;
//     state = clGetPlatformIDs(maxNumberOfPlatforms, platformIDs, &numberOfPlatforms);
//     if (state != CL_SUCCESS || numberOfPlatforms == 0) {
//         printf("OpenCL Error: Platform couldn't be found.\n");
//         return -1;
//     }
//     printf("%i platform(s) found.\n", numberOfPlatforms);

//     cl_device_id deviceList[maxNumberOfDevices];
//     devices = new Device[maxNumberOfDevices];
//     if (!devices) {
//         printf("Memory allocation failed for devices.\n");
//         return -1;
//     }

//     numberOfDevices = 0;
//     int minMajorVersion = INT_MAX, minMinorVersion = INT_MAX;

//     // Identificar a versão mínima do OpenCL suportada entre todos os dispositivos
//     for (int platform = 0; platform < numberOfPlatforms; platform++) {
//         cl_uint numberOfDevicesOfPlatform = 0;

//         state = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL, maxNumberOfDevicesPerPlatform, deviceList + numberOfDevices, &numberOfDevicesOfPlatform);
//         if (state != CL_SUCCESS || numberOfDevicesOfPlatform == 0) {
//             printf("OpenCL Error: Devices couldn't be resolved on platform %d.\n", platform);
//             continue;
//         }

//         for (int count = numberOfDevices; count < numberOfDevices + numberOfDevicesOfPlatform; count++) {
//             devices[count].deviceID = deviceList[count];

//             // Obter a versão do OpenCL do dispositivo
//             char versionStr[128];
//             clGetDeviceInfo(devices[count].deviceID, CL_DEVICE_VERSION, sizeof(versionStr), versionStr, NULL);

//             int majorVersion = 0, minorVersion = 0;
//             sscanf(versionStr, "OpenCL %d.%d", &majorVersion, &minorVersion);
//             printf("Device (%i) supports OpenCL version: %d.%d\n", count, majorVersion, minorVersion);

//             // Atualizar a versão mínima do OpenCL
//             if (majorVersion < minMajorVersion || (majorVersion == minMajorVersion && minorVersion < minMinorVersion)) {
//                 minMajorVersion = majorVersion;
//                 minMinorVersion = minorVersion;
//             }
//         }

//         numberOfDevices += numberOfDevicesOfPlatform;
//     }

//     printf("Using OpenCL version: %d.%d as target for all devices.\n", minMajorVersion, minMinorVersion);

//     // Configurar contextos e filas de comando para cada dispositivo usando a versão mínima detectada
//     for (int platform = 0; platform < numberOfPlatforms; platform++) {
//         for (int count = 0; count < numberOfDevices; count++) {
//             cl_context_properties contextProperties[] = {
//                 CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[platform],
//                 0
//             };

//             devices[count].context = clCreateContext(contextProperties, 1, &devices[count].deviceID, NULL, NULL, &state);
//             if (state != CL_SUCCESS) {
//                 printf("OpenCL Error: Context couldn't be created for device %d.\n", count);
//                 devices[count].context = NULL;
//                 continue;
//             }

//             // Usar clCreateCommandQueueWithProperties com versão mais antiga
//             cl_command_queue_properties properties[] = {
//                 CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
//                 0
//             };

//             devices[count].kernelCommandQueue = clCreateCommandQueueWithProperties(devices[count].context, devices[count].deviceID, properties, &state);
//             if (state != CL_SUCCESS) {
//                 printf("OpenCL Error: Kernel command queue couldn't be created for device %d.\n", count);
//                 clReleaseContext(devices[count].context);
//                 devices[count].context = NULL;
//                 continue;
//             }

//             devices[count].dataCommandQueue = clCreateCommandQueueWithProperties(devices[count].context, devices[count].deviceID, properties, &state);
//             if (state != CL_SUCCESS) {
//                 printf("OpenCL Error: Data command queue couldn't be created for device %d.\n", count);
//                 clReleaseCommandQueue(devices[count].kernelCommandQueue);
//                 clReleaseContext(devices[count].context);
//                 devices[count].context = NULL;
//                 continue;
//             }

//             // Inicializar outros atributos do dispositivo
//             devices[count].numberOfMemoryObjects = 0;
//             devices[count].numberOfKernels = 0;
//             devices[count].numberOfEvents = 0;

//             devices[count].memoryObjects = new cl_mem[maxMemoryObjects];
//             devices[count].kernels = new cl_kernel[maxKernels];
//             memset(devices[count].memoryObjects, 0, sizeof(cl_mem) * maxMemoryObjects);
//             memset(devices[count].kernels, 0, sizeof(cl_kernel) * maxKernels);

//             devices[count].memoryObjectID = new int[maxMemoryObjects];
//             devices[count].kernelID = new int[maxKernels];
//             memset(devices[count].memoryObjectID, 0, sizeof(int) * maxMemoryObjects);
//             memset(devices[count].kernelID, 0, sizeof(int) * maxKernels);

//             devices[count].events = new cl_event[maxEvents];
//             memset(devices[count].events, 0, sizeof(cl_event) * maxEvents);

//             devices[count].program = 0;
//         }
//     }

//     return numberOfDevices;
// }

// int OpenCLWrapper::InitParallelProcessor()
// {
//     cl_int state;
//     platformIDs = new cl_platform_id[maxNumberOfPlatforms];
//     if (!platformIDs) {
//         printf("Memory allocation failed for platformIDs.\n");
//         return -1;
//     }

//     // Obter plataformas
//     cl_uint numberOfPlatforms = 0;
//     state = clGetPlatformIDs(maxNumberOfPlatforms, platformIDs, &numberOfPlatforms);
//     if (state != CL_SUCCESS || numberOfPlatforms == 0) {
//         printf("OpenCL Error: Platform couldn't be found.\n");
//         return -1;
//     }
//     printf("%i platform(s) found.\n", numberOfPlatforms);

//     cl_device_id deviceList[maxNumberOfDevices];
//     devices = new Device[maxNumberOfDevices];
//     if (!devices) {
//         printf("Memory allocation failed for devices.\n");
//         return -1;
//     }

//     numberOfDevices = 0;
//     int minMajorVersion = INT_MAX, minMinorVersion = INT_MAX;

//     for (int platform = 0; platform < numberOfPlatforms; platform++) {
//         cl_uint numberOfDevicesOfPlatform = 0;

//         // Obter dispositivos
//         if (device_types == "CPU_DEVICES") {
//             state = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_CPU, maxNumberOfDevicesPerPlatform, deviceList + numberOfDevices, &numberOfDevicesOfPlatform);
//         } else if (device_types == "GPU_DEVICES") {
//             state = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_GPU, maxNumberOfDevicesPerPlatform, deviceList + numberOfDevices, &numberOfDevicesOfPlatform);
//         } else {
//             state = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL, maxNumberOfDevicesPerPlatform, deviceList + numberOfDevices, &numberOfDevicesOfPlatform);
//         }

//         if (state != CL_SUCCESS || numberOfDevicesOfPlatform == 0) {
//             printf("OpenCL Error: Devices couldn't be resolved on platform %d.\n", platform);
//             continue;
//         }

//         for (int count = numberOfDevices; count < numberOfDevices + numberOfDevicesOfPlatform; count++) {
//             devices[count].deviceID = deviceList[count];

//             // Obter tipo e versão do OpenCL do dispositivo
//             clGetDeviceInfo(devices[count].deviceID, CL_DEVICE_TYPE, sizeof(cl_device_type), &devices[count].deviceType, NULL);
//             char versionStr[128];
//             clGetDeviceInfo(devices[count].deviceID, CL_DEVICE_VERSION, sizeof(versionStr), versionStr, NULL);

//             // Extrair a versão do OpenCL
//             int majorVersion = 0, minorVersion = 0;
//             sscanf(versionStr, "OpenCL %d.%d", &majorVersion, &minorVersion);
//             printf("Device (%i) supports OpenCL version: %d.%d\n", count, majorVersion, minorVersion);

//             // Atualizar a versão mínima do OpenCL
//             if (majorVersion < minMajorVersion || (majorVersion == minMajorVersion && minorVersion < minMinorVersion)) {
//                 minMajorVersion = majorVersion;
//                 minMinorVersion = minorVersion;
//             }

//             // Criar propriedades do contexto com a versão mínima do OpenCL
//             cl_context_properties contextProperties[] = {
//                 CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[platform],
//                 0 // Finaliza a lista de propriedades
//             };

//             devices[count].context = clCreateContext(contextProperties, 1, &devices[count].deviceID, NULL, NULL, &state);
//             if (state != CL_SUCCESS) {
//                 printf("OpenCL Error: Context couldn't be created for device %d.\n", count);
//                 devices[count].context = NULL;
//                 continue;
//             }

//             // Usando clCreateCommandQueueWithProperties no lugar de clCreateCommandQueue
//             cl_command_queue_properties properties[] = {
//                 CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
//                 0 // Finaliza a lista de propriedades
//             };

//             devices[count].kernelCommandQueue = clCreateCommandQueueWithProperties(devices[count].context, devices[count].deviceID, properties, &state);
//             if (state != CL_SUCCESS) {
//                 printf("OpenCL Error: Kernel command queue couldn't be created for device %d.\n", count);
//                 clReleaseContext(devices[count].context);
//                 devices[count].context = NULL;
//                 continue;
//             }

//             devices[count].dataCommandQueue = clCreateCommandQueueWithProperties(devices[count].context, devices[count].deviceID, properties, &state);
//             if (state != CL_SUCCESS) {
//                 printf("OpenCL Error: Data command queue couldn't be created for device %d.\n", count);
//                 clReleaseCommandQueue(devices[count].kernelCommandQueue);
//                 clReleaseContext(devices[count].context);
//                 devices[count].context = NULL;
//                 continue;
//             }

//             // Inicializar outros atributos do dispositivo
//             devices[count].numberOfMemoryObjects = 0;
//             devices[count].numberOfKernels = 0;
//             devices[count].numberOfEvents = 0;

//             devices[count].memoryObjects = new cl_mem[maxMemoryObjects];
//             devices[count].kernels = new cl_kernel[maxKernels];
//             memset(devices[count].memoryObjects, 0, sizeof(cl_mem) * maxMemoryObjects);
//             memset(devices[count].kernels, 0, sizeof(cl_kernel) * maxKernels);

//             devices[count].memoryObjectID = new int[maxMemoryObjects];
//             devices[count].kernelID = new int[maxKernels];
//             memset(devices[count].memoryObjectID, 0, sizeof(int) * maxMemoryObjects);
//             memset(devices[count].kernelID, 0, sizeof(int) * maxKernels);

//             devices[count].events = new cl_event[maxEvents];
//             memset(devices[count].events, 0, sizeof(cl_event) * maxEvents);

//             devices[count].program = 0;
//         }

//         numberOfDevices += numberOfDevicesOfPlatform;
//     }

//     return numberOfDevices;
// }
// int OpenCLWrapper::InitParallelProcessor()
// {
//     cl_int state;
// 	platformIDs = new cl_platform_id[maxNumberOfPlatforms];
// 	//Get platforms.
// 	cl_uint numberOfPlatforms = 0;
// 	state = clGetPlatformIDs(maxNumberOfPlatforms, platformIDs, &numberOfPlatforms);
// 	if(state != CL_SUCCESS)
// 	{
// 		printf("OpenCL Error: Platform couldn't be found.\n");
// 	}
// 	printf("%i platform(s) found.\n", numberOfPlatforms);
	
// 	cl_device_id deviceList[maxNumberOfDevices];
// 	devices = new Device[maxNumberOfDevices];
// 	numberOfDevices = 0;
// 	for(int platform = 0; platform < numberOfPlatforms; platform++)
// 	{
// 		//Get devices.
// 		cl_uint numberOfDevicesOfPlatform;
// 	 if (device_types == "CPU_DEVICES")
// 		state = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_CPU, maxNumberOfDevicesPerPlatform, deviceList+numberOfDevices, &numberOfDevicesOfPlatform);
// 	else if (device_types == "GPU_DEVICES")
// 		state = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_GPU, maxNumberOfDevicesPerPlatform, deviceList+numberOfDevices, &numberOfDevicesOfPlatform);
// 		else{
// 		state = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL, maxNumberOfDevicesPerPlatform, deviceList+numberOfDevices, &numberOfDevicesOfPlatform);}
// 		if(state != CL_SUCCESS)
// 		{
// 			printf("OpenCL Error: Devices couldn't be resolved.\n");
// 		}
// 		else
// 		{
// 			if(numberOfDevicesOfPlatform > maxNumberOfDevicesPerPlatform)
// 			{
// 				numberOfDevicesOfPlatform = maxNumberOfDevicesPerPlatform;
// 			}
// 			printf("%i device(s) found on platform %i.\n", numberOfDevicesOfPlatform, platform);
// 		}

// 		//Set devices.
// 		for(int count = numberOfDevices; count < numberOfDevices+numberOfDevicesOfPlatform; count++)
// 		{
// 			//Get ID.
// 			devices[count].deviceID = deviceList[count];

// 			//Get type.
// 			clGetDeviceInfo(devices[count].deviceID, CL_DEVICE_TYPE, sizeof(cl_device_type), &devices[count].deviceType, NULL);
// 			clGetDeviceInfo(devices[count].deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_device_type), &devices[count].deviceMaxWorkItemsPerWorkGroup, NULL);
// 			clGetDeviceInfo(devices[count].deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_device_type), &devices[count].deviceComputeUnits, NULL);

// 			if(devices[count].deviceType == CL_DEVICE_TYPE_GPU)
// 			{
// 				printf("Device (%i) type: GPU\n", count);
// 			}
// 			else if(devices[count].deviceType == CL_DEVICE_TYPE_CPU)
// 			{
// 				printf("Device (%i) type: CPU\n", count);
// 			}

// 			//Create context.
// 			devices[count].context = clCreateContext(NULL, 1, &devices[count].deviceID, NULL, NULL, &state);
// 			if(state != CL_SUCCESS)
// 			{
// 				printf("OpenCL Error: Context couldn't be created.\n");
// 			}

// 			//Create command queue.
// 			devices[count].kernelCommandQueue = clCreateCommandQueue(devices[count].context, devices[count].deviceID, CL_QUEUE_PROFILING_ENABLE, &state);
// 			if(state != CL_SUCCESS)
// 			{
// 				printf("OpenCL Error: Kernel message queue couldn't be created.\n");
// 			}

// 			//Create command queue.
// 			devices[count].dataCommandQueue = clCreateCommandQueue(devices[count].context, devices[count].deviceID, CL_QUEUE_PROFILING_ENABLE, &state);
// 			if(state != CL_SUCCESS)
// 			{
// 				printf("OpenCL Error: Data message queue couldn't be created.\n");
// 			}

// 			//Initialize memory objects, kernel and events.
// 			devices[count].numberOfMemoryObjects = 0;
// 			devices[count].numberOfKernels = 0;
// 			devices[count].numberOfEvents = 0;

// 			devices[count].memoryObjects = new cl_mem[maxMemoryObjects];
// 			devices[count].kernels = new cl_kernel[maxKernels];
// 			memset(devices[count].memoryObjects, 0, sizeof(cl_mem)*maxMemoryObjects);
// 			memset(devices[count].kernels, 0, sizeof(cl_kernel)*maxKernels);

// 			devices[count].memoryObjectID = new int[maxMemoryObjects];
// 			devices[count].kernelID = new int[maxKernels];
// 			memset(devices[count].memoryObjectID, 0, sizeof(int)*maxMemoryObjects);
// 			memset(devices[count].kernelID, 0, sizeof(int)*maxKernels);

// 			devices[count].events = new cl_event[maxEvents];
// 			memset(devices[count].events, 0, sizeof(cl_kernel)*maxEvents);

// 			devices[count].program = 0;
// 		}
// 		numberOfDevices += numberOfDevicesOfPlatform;
// 	}
// 	return numberOfDevices;
// }






void OpenCLWrapper::setKernel(const std::string &sourceFile, const std::string &kernelName) {
    kernelSourceFile = sourceFile;
    kernelFunctionName = kernelName;
    kernelDispositivo = new int[n_devices];
	
    for(int count = 0; count < todosDispositivos; count++)
	{
		if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
		{
			
			kernelDispositivo[count] = CreateKernel(count-meusDispositivosOffset, kernelSourceFile.c_str(), kernelFunctionName.c_str());
			}
		
	}
    
    
    kernelSet = true;
}
int OpenCLWrapper::CreateKernel(int devicePosition, const char *source, const char *kernelName)
{
	if (devices[devicePosition].program != 0)
	{
		clReleaseProgram(devices[devicePosition].program);
	}
	devices[devicePosition].program = 0;

	cl_int state;

	// Read kernel file.
	FILE *fileHandle;
	char *sourceBuffer = (char *)malloc(sizeof(char) * MAX_SOURCE_BUFFER_LENGTH);
	if ((fileHandle = fopen(source, "r")) == NULL)
	{
		printf("Error reading %s\n!", source);
		return -1;
	}
	size_t sourceBufferLength = fread(sourceBuffer, 1, sizeof(char) * MAX_SOURCE_BUFFER_LENGTH, fileHandle);

	// Create program.
	devices[devicePosition].program = clCreateProgramWithSource(devices[devicePosition].context, 1, (const char **)&sourceBuffer, (const size_t *)&sourceBufferLength, &state);

	// Close kernel file.
	fclose(fileHandle);
	fileHandle = NULL;
	free(sourceBuffer);
	sourceBuffer = NULL;

	// Program created?
	if (state != CL_SUCCESS)
	{
		printf("Error creating program!\n");
		return -1;
	}

	// Compile program.
	state = clBuildProgram(devices[devicePosition].program, 1, &devices[devicePosition].deviceID, NULL, NULL, NULL);
	if (state != CL_SUCCESS)
	{
		printf("Error compiling program!\n");
		return -1;
	}

	// Create kernel.
	devices[devicePosition].kernels[devices[devicePosition].numberOfKernels] = clCreateKernel(devices[devicePosition].program, kernelName, &state);
	if (state != CL_SUCCESS)
	{
		printf("Error creating kernel!\n");
		return -1;
	}
	devices[devicePosition].kernelID[devices[devicePosition].numberOfKernels] = automaticNumber;
	devices[devicePosition].numberOfKernels += 1;
	automaticNumber += 1;
	return automaticNumber - 1;
}


void OpenCLWrapper::SetKernelAttribute(int devicePosition, int kernelID, int attribute, int memoryObjectID)
{
    int kernelPosition = GetKernelPosition(devicePosition, kernelID);
    int memoryObjectPosition = GetMemoryObjectPosition(devicePosition, memoryObjectID);
    if (kernelPosition != -1 && memoryObjectPosition != -1) {
        cl_int state = clSetKernelArg(devices[devicePosition].kernels[kernelPosition], attribute, sizeof(cl_mem), (void *)&devices[devicePosition].memoryObjects[memoryObjectPosition]);
        if (state != CL_SUCCESS) {
            printf("Error setting kernel argument!\n");
        }
    } else {
        printf("Error setting kernel argument: Either kernel ID=%i or MemOBJ=%i don't exist!\n", kernelID, memoryObjectID);
    }


}


int OpenCLWrapper::CreateMemoryObject(int devicePosition, int size, cl_mem_flags memoryType, void *hostMemory) {
	cl_int state;
	if(devices[devicePosition].numberOfMemoryObjects < maxMemoryObjects)
	{
		devices[devicePosition].memoryObjects[devices[devicePosition].numberOfMemoryObjects] = clCreateBuffer(devices[devicePosition].context, memoryType, size, hostMemory, &state);
		if(state != CL_SUCCESS)
		{
			printf("Error creating memory object!\n");
			return -1;
		}
		else
		{
			devices[devicePosition].memoryObjectID[devices[devicePosition].numberOfMemoryObjects] = automaticNumber;
			devices[devicePosition].numberOfMemoryObjects += 1;
		}
		automaticNumber += 1;
		return automaticNumber-1;
	}
	printf("Error creating memory object, limit exceeded!");
	return -1;



}

void OpenCLWrapper::ExecuteKernel() {
  
 //for(int count2 = 0; count2 < world_size; count2++)
	//{
		//if(count2 == world_rank)
	//	{
    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {

            
            
            int deviceIndex = count - meusDispositivosOffset;
            if (deviceIndex >= 0 && deviceIndex < todosDispositivos) {
                
                kernelEventoDispositivo[count] = RunKernel(deviceIndex, kernelDispositivo[deviceIndex], offset[deviceIndex], length[deviceIndex], isDeviceCPU(deviceIndex)? 8 : 64);
            } else {
                std::cerr << "Invalid device index: " << deviceIndex << std::endl;
            }
        }
    }
      //  }
    //}


     //for(int count2 = 0; count2 < world_size; count2++)
	//{
	//	if(count2 == world_rank)
	//	{
    
    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            int deviceIndex = count - meusDispositivosOffset;
            if (deviceIndex >= 0 && deviceIndex < todosDispositivos) {
                SynchronizeCommandQueue(deviceIndex);
            } else {
                std::cerr << "Invalid device index: " << deviceIndex << std::endl;
            }
        }
    }
     }
// }
// }

int OpenCLWrapper::RunKernel(int devicePosition, int kernelID, int parallelDataOffset, int parallelData, int workGroupSize) {
    
   
    // Checa se o devicePosition é válido e se há eventos disponíveis
    // if (devicePosition < 0 || devicePosition >= maxNumberOfDevices ||
    //     devices[devicePosition].numberOfEvents >= maxEvents)
    // {
    //     printf("Invalid device position or events limit exceeded.\n");
    //     return -1;
    // }

    // int kernelPosition = GetKernelPosition(devicePosition, kernelID);
    // if (kernelPosition == -1)
    // {
    //     printf("Kernel position not found.\n");
    //     return -1;
    // }

    // size_t globalItemsOffset = Maximum(parallelDataOffset, 0);
    // size_t globalItems = Maximum(parallelData, workGroupSize);
    // size_t localItems = workGroupSize;

    // Ajusta globalItems para ser múltiplo de localItems
    // if (globalItems % localItems != 0)
    // {
    //     globalItems = (globalItems / localItems + 1) * localItems;
    // }

	//Make sure parallelData is a power of 2.
	// 	size_t globalItemsOffset = Maximum(parallelDataOffset, 0);
	// 	size_t globalItems = parallelData;
	// 	size_t mask = 0;
	// 	globalItems = Maximum(workGroupSize, parallelData + workGroupSize - (parallelData%workGroupSize));

		
	// 	size_t localItems = workGroupSize;




    // cl_int state;
    // cl_event event;

    // state = clEnqueueNDRangeKernel(devices[devicePosition].kernelCommandQueue, devices[devicePosition].kernels[kernelPosition], 1, &globalItemsOffset, &globalItems, &localItems, 0, NULL, &event);
    // if (state != CL_SUCCESS)
    // {
    //     printf("Error queueing task! %i\n", state);
    //     return -1;
    // }

    // // Aqui você pode querer liberar a fila de comandos se não for necessária mais
    // clFlush(devices[devicePosition].kernelCommandQueue);

    // // Atualiza o array de eventos
    // devices[devicePosition].events[devices[devicePosition].numberOfEvents] = event;
    // devices[devicePosition].numberOfEvents += 1;

    // return devices[devicePosition].numberOfEvents - 1;

	 int kernelPosition = GetKernelPosition(devicePosition, kernelID);
    if (kernelPosition != -1 && devices[devicePosition].numberOfEvents < maxEvents)
    {
        // Garantir que parallelData seja um múltiplo de workGroupSize.
        size_t globalItemsOffset = Maximum(parallelDataOffset, 0);
        size_t globalItems = parallelData;
        size_t localItems = workGroupSize;

        // Obter o tamanho máximo do grupo de trabalho suportado pelo dispositivo
        size_t maxWorkGroupSize;
        clGetDeviceInfo(devices[devicePosition].deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
		std::cout<<"MaxworkGroupSize: "<<maxWorkGroupSize<<std::endl;
        // Ajustar o tamanho do grupo de trabalho para não exceder o limite e ser um divisor de globalItems
        if (localItems > maxWorkGroupSize) {
            localItems = maxWorkGroupSize;
        }
        while (globalItems % localItems != 0) {
            localItems--;
        }
		std::cout<<"locaItens: "<<localItems<<std::endl;
        cl_int state;
        globalItems = Maximum(localItems, parallelData + localItems - (parallelData % localItems));

        // Enfileirar o kernel para execução
        state = clEnqueueNDRangeKernel(devices[devicePosition].kernelCommandQueue, devices[devicePosition].kernels[kernelPosition], 1, &globalItemsOffset, &globalItems, &localItems, 0, NULL, &devices[devicePosition].events[devices[devicePosition].numberOfEvents]);
        if (state != CL_SUCCESS)
        {
            printf("Error queueing task! %i\n", state);
            return -1;
        }
        else
        {
            clFlush(devices[devicePosition].kernelCommandQueue);
            devices[devicePosition].numberOfEvents += 1;
            return devices[devicePosition].numberOfEvents - 1;
        }
    }

    printf("Error! Couldn't find kernel position %i or number of events %i exceeded limit.\n", kernelPosition, devices[devicePosition].numberOfEvents);
    return -1;




}

   


void OpenCLWrapper:: SynchronizeCommandQueue(int devicePosition)
{
    
	clFinish(devices[devicePosition].kernelCommandQueue);
	clFinish(devices[devicePosition].dataCommandQueue);
	devices[devicePosition].numberOfEvents = 0;
}

void OpenCLWrapper::GatherResults(int dataIndex, void *resultData) {
   
	char *resultPtr = reinterpret_cast<char*>(resultData);
	
//   for(int count2 = 0; count2 < world_size; count2++)
// 	{
// 		if(count2 == world_rank)
// 		{
			
			for(int count = 0; count < todosDispositivos; count++)
			{
				if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
				{	int id = GetDeviceMemoryObjectID(dataIndex, count);
					std::cout<<"offset[count]"<<offset[count]<<std::endl;
                    std::cout<<"length[count]"<<length[count]<<std::endl;
					ReadFromMemoryObject(count-meusDispositivosOffset, id, resultPtr+offset[count], offset[count]*sizeof(float), length[count]*sizeof(float));
					SynchronizeCommandQueue(count-meusDispositivosOffset);
					std::cout<<"------------------------------------------------------------------------------------------"<<std::endl;
				}
			}
		//}
		//MPI_Barrier(MPI_COMM_WORLD);
	//}
   
}

void OpenCLWrapper::setLoadBalancer(void *data, int N_Elements, int units_per_elements) {
   ticks = new long int[todosDispositivos];
    tempos_por_carga = new double[todosDispositivos];
    cargasNovas = new float[todosDispositivos];
    cargasAntigas = new float[todosDispositivos];
    swapBufferDispositivo = new int*[todosDispositivos];
    memObjects = new int[todosDispositivos];
    tempos = new float[todosDispositivos];
    offset = new unsigned long int[todosDispositivos];
    length = new unsigned long int[todosDispositivos];
    offsetDispositivo = new int[todosDispositivos];
    lengthDispositivo = new int[todosDispositivos];
    kernelEventoDispositivo = new int[todosDispositivos];
    N_elements = N_Elements;
    memset(ticks, 0, sizeof(long int) * todosDispositivos);
    memset(tempos_por_carga, 0, sizeof(double) * todosDispositivos);
    memset(cargasNovas, 0, sizeof(float) * todosDispositivos);
    memset(cargasAntigas, 0, sizeof(float) * todosDispositivos);

    offsetComputacao = 0;
    lengthComputacao = N_Elements / todosDispositivos;
    
    if (kernelSet)  {
        for (int count = 0; count < todosDispositivos; count++) {
            if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
               
			    initializeLengthOffset(offsetComputacao, (count + 1 == todosDispositivos) ? (N_Elements - offsetComputacao) : lengthComputacao, count);
                SynchronizeCommandQueue(count - meusDispositivosOffset);
            }
            offsetComputacao += lengthComputacao;
        }
        loadBalancerSet = true;
    } else {
        std::cerr << "Error: Kernel is not initialized." << std::endl;
    }

    for (int count = 0; count < todosDispositivos; count++) {
        cargasNovas[count] = static_cast<float>(count + 1) * (1.0f / static_cast<float>(todosDispositivos));
        cargasAntigas[count] = cargasNovas[count];
        tempos[count] = 1;
    }
}

// void Balanceador::Probing(int simulacao)
// {
// 	// if (balanceamento && ((simulacao == 0) || (simulacao == 1) ))

// 	double tempoInicioProbing = MPI_Wtime();
// 	double localLatencia = 0, localBanda = 0;
// 	PrecisaoBalanceamento(simulacao);

// 	// Computar novas cargas.

// 	for (int count = 0; count < todosDispositivos; count++)
// 	{
// 		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
// 		{
// 			int overlapNovoOffset = ((int)(((count == 0) ? 0.0f : cargasNovas[count - 1]) * ((float)(xMalhaLength * yMalhaLength * zMalhaLength))));
// 			int overlapNovoLength = ((int)(((count == 0) ? cargasNovas[count] - 0.0f : cargasNovas[count] - cargasNovas[count - 1]) * ((float)(xMalhaLength * yMalhaLength * zMalhaLength))));
// 			for (int count2 = 0; count2 < todosDispositivos; count2++)
// 			{
// 				if (count > count2)
// 				{
// 					// Atender requisicoes de outros processos.
// 					if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
// 					{
// 						int overlap[2];
// 						int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
// 						float *malha = ((simulacao % 2) == 0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];
// 						int malhaDevice = ((simulacao % 2) == 0) ? malhaSwapBufferDispositivo[count][0] : malhaSwapBufferDispositivo[count][1];
// 						MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
// 						// Podem ocorrer requisicoes vazias.
// 						if (overlap[1] > 0)
// 						{
// 							ReadFromMemoryObject(count - meusDispositivosOffset, malhaDevice, (char *)(malha + (overlap[0] * MALHA_TOTAL_CELULAS)), overlap[0] * MALHA_TOTAL_CELULAS * sizeof(float), overlap[1] * MALHA_TOTAL_CELULAS * sizeof(float));
// 							SynchronizeCommandQueue(count - meusDispositivosOffset);

// 							sizeCarga = overlap[1] * MALHA_TOTAL_CELULAS;

// 							double tempoInicioBanda = MPI_Wtime();
// 							MPI_Ssend(malha + (overlap[0] * MALHA_TOTAL_CELULAS), sizeCarga, MPI_FLOAT, alvo, 0, MPI_COMM_WORLD);
// 							double aux = (MPI_Wtime() - tempoInicioBanda) / sizeCarga;
// 							localBanda = aux > localBanda ? aux : localBanda;
// 						}
// 					}
// 				}
// 				else if (count < count2)
// 				{
// 					// Fazer requisicoes a outros processos.
// 					int overlapAntigoOffset = ((int)(((count2 == 0) ? 0 : cargasAntigas[count2 - 1]) * (xMalhaLength * yMalhaLength * zMalhaLength)));
// 					int overlapAntigoLength = ((int)(((count2 == 0) ? cargasAntigas[count2] - 0.0f : cargasAntigas[count2] - cargasAntigas[count2 - 1]) * (xMalhaLength * yMalhaLength * zMalhaLength)));

// 					int intersecaoOffset;
// 					int intersecaoLength;

// 					if (((overlapAntigoOffset <= overlapNovoOffset - (xMalhaLength * yMalhaLength)) && ComputarIntersecao(overlapAntigoOffset, overlapAntigoLength, overlapNovoOffset - (xMalhaLength * yMalhaLength), overlapNovoLength + (xMalhaLength * yMalhaLength), &intersecaoOffset, &intersecaoLength)) ||
// 							((overlapAntigoOffset > overlapNovoOffset - (xMalhaLength * yMalhaLength)) && ComputarIntersecao(overlapNovoOffset - (xMalhaLength * yMalhaLength), overlapNovoLength + (xMalhaLength * yMalhaLength), overlapAntigoOffset, overlapAntigoLength, &intersecaoOffset, &intersecaoLength)))
// 					{
// 						if (count2 >= meusDispositivosOffset && count2 < meusDispositivosOffset + meusDispositivosLength)
// 						{
// 							float *malha = ((simulacao % 2) == 0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];

// 							int malhaDevice[2] = {((simulacao % 2) == 0) ? malhaSwapBufferDispositivo[count][0] : malhaSwapBufferDispositivo[count][1],
// 																		((simulacao % 2) == 0) ? malhaSwapBufferDispositivo[count2][0] : malhaSwapBufferDispositivo[count2][1]};

// 							ReadFromMemoryObject(count2 - meusDispositivosOffset, malhaDevice[1], (char *)(malha + (intersecaoOffset * MALHA_TOTAL_CELULAS)), intersecaoOffset * MALHA_TOTAL_CELULAS * sizeof(float), intersecaoLength * MALHA_TOTAL_CELULAS * sizeof(float));
// 							SynchronizeCommandQueue(count2 - meusDispositivosOffset);

// 							WriteToMemoryObject(count - meusDispositivosOffset, malhaDevice[0], (char *)(malha + (intersecaoOffset * MALHA_TOTAL_CELULAS)), intersecaoOffset * MALHA_TOTAL_CELULAS * sizeof(float), intersecaoLength * MALHA_TOTAL_CELULAS * sizeof(float));
// 							SynchronizeCommandQueue(count - meusDispositivosOffset);
// 						}
// 						else
// 						{
// 							// Fazer uma requisicao.
// 							if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
// 							{
// 								int overlap[2] = {intersecaoOffset, intersecaoLength};
// 								int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
// 								float *malha = ((simulacao % 2) == 0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];
// 								int malhaDevice = ((simulacao % 2) == 0) ? malhaSwapBufferDispositivo[count][0] : malhaSwapBufferDispositivo[count][1];
// 								SynchronizeCommandQueue(count - meusDispositivosOffset);
// 								double tempoInicioLatencia = MPI_Wtime();
// 								MPI_Ssend(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
// 								double aux = (MPI_Wtime() - tempoInicioLatencia) / 2;
// 								localLatencia = aux > localLatencia ? aux : localLatencia;

// 								MPI_Recv(malha + (overlap[0] * MALHA_TOTAL_CELULAS), overlap[1] * MALHA_TOTAL_CELULAS, MPI_FLOAT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// 								WriteToMemoryObject(count - meusDispositivosOffset, malhaDevice, (char *)(malha + (overlap[0] * MALHA_TOTAL_CELULAS)), overlap[0] * MALHA_TOTAL_CELULAS * sizeof(float), overlap[1] * MALHA_TOTAL_CELULAS * sizeof(float));
// 								SynchronizeCommandQueue(count - meusDispositivosOffset);
// 							}
// 						}
// 					}
// 					else
// 					{
// 						// Fazer uma requisicao vazia.
// 						if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
// 						{
// 							int overlap[2] = {0, 0};
// 							int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
// 							float *malha = ((simulacao % 2) == 0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];
// 							MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
// 						}
// 					}
// 				}
// 			}

// 			parametrosMalha[count][OFFSET_COMPUTACAO] = overlapNovoOffset;
// 			parametrosMalha[count][LENGTH_COMPUTACAO] = overlapNovoLength;

// 			WriteToMemoryObject(count - meusDispositivosOffset, parametrosMalhaDispositivo[count], (char *)parametrosMalha[count], 0, sizeof(int) * NUMERO_PARAMETROS_MALHA);
// 			SynchronizeCommandQueue(count - meusDispositivosOffset);
// 		}
// 	}
// 	memcpy(cargasAntigas, cargasNovas, sizeof(float) * todosDispositivos);

// 	MPI_Allreduce(&localLatencia, &latencia, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
// 	MPI_Allreduce(&localBanda, &banda, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

// 	MPI_Barrier(MPI_COMM_WORLD);
// 	double tempoFimProbing = MPI_Wtime();
// 	tempoBalanceamento += tempoFimProbing - tempoInicioProbing;
// 	fatorErro = tempoBalanceamento;
// }

void OpenCLWrapper::PrecisaoBalanceamento() {
  
  
  	memset(ticks, 0, sizeof(long int) * todosDispositivos);
	memset(tempos, 0, sizeof(float) * todosDispositivos);

	for (int precisao = 0; precisao < 10; precisao++)
	{
		
		// Computação.
		for (int count = 0; count < todosDispositivos; count++)
		{
			
			if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
			{
				
				kernelEventoDispositivo[count] = RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count], length[count], 8);
			}
		}
	

	
	// // Ticks.
	for (int count = 0; count < todosDispositivos; count++)
	{	
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{	
			SynchronizeCommandQueue(count - meusDispositivosOffset);
			
			ticks[count] += GetEventTaskTicks(count - meusDispositivosOffset, kernelEventoDispositivo[count]);
			
		}
	}
	
}	
	// Reduzir ticks.
	
	long int ticks_root[todosDispositivos];
	MPI_Allreduce(ticks, ticks_root, todosDispositivos, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	memcpy(ticks, ticks_root, sizeof(long int) * todosDispositivos);
	ComputarCargas(ticks, cargasAntigas, cargasNovas, todosDispositivos);
	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
	 	{
	 		SynchronizeCommandQueue(count - meusDispositivosOffset);
	 		tempos[count] = ((float)ticks[count]) / ((float)cargasNovas[count]);
	 	}
	}
	float tempos_root[todosDispositivos];
	MPI_Allreduce(tempos, tempos_root, todosDispositivos, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	memcpy(tempos, tempos_root, sizeof(float) * todosDispositivos);
  


}

// void OpenCLWrapper::LoadBalancing(){
//     double tempoInicioBalanceamento = MPI_Wtime();
//     double localTempoCB;

//     for (int count = 0; count < todosDispositivos; count++) {
//         if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
//             SynchronizeCommandQueue(count - meusDispositivosOffset);
//             localTempoCB = cargasNovas[count] * tempos[count];
//         }
//     }
//     MPI_Allreduce(&localTempoCB, &tempoCB, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//     tempoCB *= N_Elements;

//     if (latencia + ComputarNorma(cargasAntigas, cargasNovas, todosDispositivos) * (writeByte + banda) + tempoCB < tempoComputacaoInterna) {
//         for (int count = 0; count < todosDispositivos; count++) {
//             if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
//                 int overlapNovoOffset = ((count == 0 ? 0.0f : cargasNovas[count - 1]) * (N_Elements));
//                 int overlapNovoLength = ((count == 0 ? cargasNovas[count] - 0.0f : cargasNovas[count] - cargasNovas[count - 1]) * (N_Elements));
//                 for (int count2 = 0; count2 < todosDispositivos; count2++) {
//                     if (count > count2) {
//                         if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
//                             int overlap[2];
//                             int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
//                             T *data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
//                             int dataDevice = (simulacao % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
//                             MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                             if (overlap[1] > 0) {
//                                 ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice, (char *)(data + overlap[0] * Element_size), overlap[0] * Element_size * sizeof(float), overlap[1] * Element_size * sizeof(float));
//                                 SynchronizeCommandQueue(count - meusDispositivosOffset);
//                                 size_t sizeCarga = overlap[1] * Element_size;
//                                 MPI_Send(data + overlap[0] * Element_size, sizeCarga, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD);
//                             }
//                         }
//                     } else if (count < count2) {
//                         int overlapAntigoOffset = ((count2 == 0 ? 0 : cargasAntigas[count2 - 1]) * N_Elements);
//                         int overlapAntigoLength = ((count2 == 0 ? cargasAntigas[count2] - 0.0f : cargasAntigas[count2] - cargasAntigas[count2 - 1]) * N_Elements);

//                         int intersecaoOffset;
//                         int intersecaoLength;

//                         if ((overlapAntigoOffset <= overlapNovoOffset - interv_balance && ComputarIntersecao(overlapAntigoOffset, overlapAntigoLength, overlapNovoOffset - interv_balance, overlapNovoLength + interv_balance, &intersecaoOffset, &intersecaoLength)) ||
//                             (overlapAntigoOffset > overlapNovoOffset - interv_balance && ComputarIntersecao(overlapNovoOffset - interv_balance, overlapNovoLength + interv_balance, overlapAntigoOffset, overlapAntigoLength, &intersecaoOffset, &intersecaoLength))) {
//                             if (count2 >= meusDispositivosOffset && count2 < meusDispositivosOffset + meusDispositivosLength) {
//                                 T *data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
//                                 int dataDevice[2] = {(simulacao % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1],
//                                                      (simulacao % 2) == 0 ? swapBufferDispositivo[count2][0] : swapBufferDispositivo[count2][1]};

//                                 ReadFromMemoryObject(count2 - meusDispositivosOffset, dataDevice[1], (char *)(data + intersecaoOffset * Element_size), intersecaoOffset * Element_size * sizeof(float), intersecaoLength * Element_size * sizeof(float));
//                                 SynchronizeCommandQueue(count2 - meusDispositivosOffset);
//                                 WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(data + intersecaoOffset * Element_size), intersecaoOffset * Element_size * sizeof(float), intersecaoLength * Element_size * sizeof(float));
//                                 SynchronizeCommandQueue(count - meusDispositivosOffset);
//                             } else {
//                                 if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
//                                     int overlap[2] = {intersecaoOffset, intersecaoLength};
//                                     int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
//                                     T *data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
//                                     int dataDevice = (simulacao % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
//                                     MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
//                                     MPI_Recv(data + overlap[0] * Element_size, overlap[1] * Element_size, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                                     WriteToMemoryObject(count - meusDispositivosOffset, dataDevice, (char *)(data + overlap[0] * Element_size), overlap[0] * Element_size * sizeof(float), overlap[1] * Element_size * sizeof(float));
//                                     SynchronizeCommandQueue(count - meusDispositivosOffset);
//                                 }
//                             }
//                         } else {
//                             if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
//                                 int overlap[2] = {0, 0};
//                                 int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
//                                 T *data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
//                                 MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
//                             }
//                         }
//                     }
//                 }
//                 offset[count] = overlapNovoOffset;
//                 length[count] = overlapNovoLength;
//                 WriteToMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)DataToKernel, 0, sizeof(int) * 8);
//                 SynchronizeCommandQueue(count - meusDispositivosOffset);
//             }
//         }
//         memcpy(cargasAntigas, cargasNovas, sizeof(float) * todosDispositivos);

//         MPI_Barrier(MPI_COMM_WORLD);
//         double tempoFimBalanceamento = MPI_Wtime();
//         tempoBalanceamento += tempoFimBalanceamento - tempoInicioBalanceamento;
//     }
// }








void OpenCLWrapper::ComputarCargas(const long int *ticks, const float *cargasAntigas, float *cargasNovas, int participantes) {
    if (participantes == 1) {
        cargasNovas[0] = 1.0f;
        return;
    }

    float cargaTotal = 0.0f;
    for (int count = 0; count < participantes; count++) {
        cargaTotal += ((count == 0) ? (cargasAntigas[count] - 0.0f) : (cargasAntigas[count] - cargasAntigas[count - 1])) * ((count == 0) ? 1.0f : static_cast<float>(ticks[0]) / static_cast<float>(ticks[count]));
    }

    for (int count = 0; count < participantes; count++) {
        float cargaNova = (((count == 0) ? (cargasAntigas[count] - 0.0f) : (cargasAntigas[count] - cargasAntigas[count - 1])) * ((count == 0) ? 1.0f : static_cast<float>(ticks[0]) / static_cast<float>(ticks[count]))) / cargaTotal;
        cargasNovas[count] = ((count == 0) ? cargaNova : cargasNovas[count - 1] + cargaNova);
    }
}

int OpenCLWrapper::RecuperarPosicaoHistograma(int *histograma, int tamanho, int indice) {
    int offset = 0;
    for (int count = 0; count < tamanho; count++) {
        if (indice >= offset && indice < offset + histograma[count]) {
            return count;
        }
        offset += histograma[count];
    }
    return -1;
}

bool OpenCLWrapper::ComputarIntersecao(int offset1, int length1, int offset2, int length2, int *intersecaoOffset, int *intersecaoLength) {
    if (offset1 + length1 <= offset2) {
        return false;
    }

    if (offset1 + length1 > offset2 + length2) {
        *intersecaoOffset = offset2;
        *intersecaoLength = length2;
    } else {
        *intersecaoOffset = offset2;
        *intersecaoLength = (offset1 + length1) - offset2;
    }
    return true;
}

float OpenCLWrapper::ComputarDesvioPadraoPercentual(const long int *ticks, int participantes) {
    double media = 0.0;
    for (int count = 0; count < participantes; count++) {
        media += static_cast<double>(ticks[count]);
    }
    media /= static_cast<double>(participantes);

    double variancia = 0.0;
    for (int count = 0; count < participantes; count++) {
        variancia += (static_cast<double>(ticks[count]) - media) * (static_cast<double>(ticks[count]) - media);
    }
    variancia /= static_cast<double>(participantes);
    return std::sqrt(variancia) / media;
}

float OpenCLWrapper::ComputarNorma(const float *cargasAntigas, const float *cargasNovas, int participantes) {
    float retorno = 0.0;
    for (int count = 0; count < participantes; count++) {
        retorno += (cargasAntigas[count] - cargasNovas[count]) * (cargasAntigas[count] - cargasNovas[count]);
    }
    return std::sqrt(retorno);
}

void OpenCLWrapper::initializeLengthOffset(int offset, int length, int deviceIndex) {
    this->offset[deviceIndex] = offset;
    this->length[deviceIndex] = length;
}

int OpenCLWrapper::Maximum(int a, int b) {
    return (a > b) ? a : b;
}

int OpenCLWrapper:: GetMemoryObjectPosition( int devicePosition, int memoryObjectID)
{
	for (int count = 0; count < devices[devicePosition].numberOfMemoryObjects; count++)
	{
		if (devices[devicePosition].memoryObjectID[count] == memoryObjectID)
		{
			return count;
		}
	}
	return -1;
	
}


int OpenCLWrapper::GetKernelPosition(int devicePosition, int kernelID)
{
	for (int count = 0; count < devices[devicePosition].numberOfKernels; count++)
	{
		if (devices[devicePosition].kernelID[count] == kernelID)
		{
			return count;
		}
	}
	return -1;
}

void OpenCLWrapper::SynchronizeEvent(int eventPosition) {
    clWaitForEvents(1, &devices[deviceIndex].events[eventPosition]);
}

long int OpenCLWrapper::GetEventTaskOverheadTicks(int devicePosition, int eventPosition)
{
	long int ticksStart;
	long int ticksEnd;

	clGetEventProfilingInfo(devices[devicePosition].events[eventPosition], CL_PROFILING_COMMAND_QUEUED, sizeof(long int), &ticksStart, NULL);
	clGetEventProfilingInfo(devices[devicePosition].events[eventPosition], CL_PROFILING_COMMAND_START, sizeof(long int), &ticksEnd, NULL);
	return (ticksEnd - ticksStart);
}


long int OpenCLWrapper::GetEventTaskTicks(int devicePosition, int eventPosition)
{
	long int ticksStart;
	long int ticksEnd;

	clGetEventProfilingInfo(devices[devicePosition].events[eventPosition], CL_PROFILING_COMMAND_START, sizeof(long int), &ticksStart, NULL);
	clGetEventProfilingInfo(devices[devicePosition].events[eventPosition], CL_PROFILING_COMMAND_END, sizeof(long int), &ticksEnd, NULL);
	return (ticksEnd - ticksStart);
}

cl_device_type OpenCLWrapper::GetDeviceType() {
    return devices[deviceIndex].deviceType;
}

size_t OpenCLWrapper::GetDeviceMaxWorkItemsPerWorkGroup() {
    return devices[deviceIndex].deviceMaxWorkItemsPerWorkGroup;
}

cl_uint OpenCLWrapper::GetDeviceComputeUnits() {
    return devices[deviceIndex].deviceComputeUnits;
}

bool OpenCLWrapper::isDeviceCPU(int devicePosition) {
    return devices[devicePosition].deviceType == CL_DEVICE_TYPE_CPU ? true : false;
}
bool OpenCLWrapper::RemoveKernel(int devicePosition, int kernelID) {
    int kernelPosition = GetKernelPosition(devicePosition, kernelID);
	if (kernelPosition != -1)
	{
		clReleaseKernel(devices[devicePosition].kernels[kernelPosition]);
		memcpy(devices[devicePosition].kernels + kernelPosition, devices[devicePosition].kernels + kernelPosition + 1, sizeof(cl_kernel) * (devices[devicePosition].numberOfKernels - 1));
		memcpy(devices[devicePosition].kernelID + kernelPosition, devices[devicePosition].kernelID + kernelPosition + 1, sizeof(cl_kernel) * (devices[devicePosition].numberOfKernels - 1));
		devices[devicePosition].numberOfKernels -= 1;
		return true;
	}
	return false;
}

bool OpenCLWrapper:: RemoveMemoryObject(int devicePosition, int memoryObjectID)
{
	int memoryObjectPosition = GetMemoryObjectPosition(devicePosition, memoryObjectID);
	if (memoryObjectPosition != -1)
	{
		clReleaseMemObject(devices[devicePosition].memoryObjects[memoryObjectPosition]);
		memcpy(devices[devicePosition].memoryObjects + memoryObjectPosition, devices[devicePosition].memoryObjects + memoryObjectPosition + 1, sizeof(cl_mem) * (devices[devicePosition].numberOfMemoryObjects - 1));
		memcpy(devices[devicePosition].memoryObjectID + memoryObjectPosition, devices[devicePosition].memoryObjectID + memoryObjectPosition + 1, sizeof(cl_mem) * (devices[devicePosition].numberOfMemoryObjects - 1));
		devices[devicePosition].numberOfMemoryObjects -= 1;
		return true;
	}
	return false;
}

int OpenCLWrapper::WriteToMemoryObject(int devicePosition, int memoryObjectID, const char *data, int offset, int size) {
    cl_int state;
	int memoryObjectPosition = GetMemoryObjectPosition(devicePosition, memoryObjectID);
	if (memoryObjectPosition != -1 && devices[devicePosition].numberOfEvents < maxEvents)
	{
		state = clEnqueueWriteBuffer(devices[devicePosition].dataCommandQueue, devices[devicePosition].memoryObjects[memoryObjectPosition], CL_FALSE, offset, size, data, 0, NULL, &devices[devicePosition].events[devices[devicePosition].numberOfEvents]);
		if (state != CL_SUCCESS)
		{
			printf("Error writing to memory object %i.\n", state);
		}
		else
		{
			clFlush(devices[devicePosition].dataCommandQueue);
			devices[devicePosition].numberOfEvents += 1;
			return devices[devicePosition].numberOfEvents - 1;
		}
	}

	printf("Error! Couldn't find memory object position %i or number of events %i exceeded limit.\n", memoryObjectPosition, devices[devicePosition].numberOfEvents);
	return -1;
}

int OpenCLWrapper::ReadFromMemoryObject(int devicePosition, int memoryObjectID, char *data, int offset, int size)
{
	

    cl_int state;
    int memoryObjectPosition = GetMemoryObjectPosition(devicePosition, memoryObjectID);
    if (memoryObjectPosition != -1 && devices[devicePosition].numberOfEvents < maxEvents)
    { 

        state = clEnqueueReadBuffer(devices[devicePosition].dataCommandQueue, 
                                    devices[devicePosition].memoryObjects[memoryObjectPosition], 
                                    CL_FALSE, 
                                    offset, 
                                    size, 
                                    data, 
                                    0, 
                                    NULL, 
                                    &devices[devicePosition].events[devices[devicePosition].numberOfEvents]);
        if (state != CL_SUCCESS)
        {
            printf("Error reading from memory object %i.\n", state);
            return -1;
        }
        else
        {
            clFinish(devices[devicePosition].dataCommandQueue); // Garantir que a operação está completa
            devices[devicePosition].numberOfEvents += 1;
            return devices[devicePosition].numberOfEvents - 1;
        }
    }

    printf("Error! Couldn't find memory object position %i or number of events %i exceeded limit.\n", memoryObjectPosition, devices[devicePosition].numberOfEvents);
    return -1;    
}




int OpenCLWrapper::getMaxNumberOfPlatforms() const {
    return maxNumberOfPlatforms;
}

void OpenCLWrapper::setMaxNumberOfPlatforms(int value) {
    maxNumberOfPlatforms = value;
}

int OpenCLWrapper::getMaxNumberOfDevices() const {
    return maxNumberOfDevices;
}

void OpenCLWrapper::setMaxNumberOfDevices(int value) {
    maxNumberOfDevices = value;
}

int OpenCLWrapper::getMaxNumberOfDevicesPerPlatform() const {
    return maxNumberOfDevicesPerPlatform;
}

void OpenCLWrapper::setMaxNumberOfDevicesPerPlatform(int value) {
    maxNumberOfDevicesPerPlatform = value;
}

int OpenCLWrapper::getMaxMemoryObjects() const {
    return maxMemoryObjects;
}

void OpenCLWrapper::setMaxMemoryObjects(int value) {
    maxMemoryObjects = value;
}

int OpenCLWrapper::getMaxKernels() const {
    return maxKernels;
}

void OpenCLWrapper::setMaxKernels(int value) {
    maxKernels = value;
}

int OpenCLWrapper::getMaxEvents() const {
    return maxEvents;
}

void OpenCLWrapper::setMaxEvents(int value) {
    maxEvents = value;
}

void OpenCLWrapper::FinishParallelProcessor()
{
	for (int count = 0; count < numberOfDevices; count++)
	{
		clFlush(devices[count].kernelCommandQueue);
		clFinish(devices[count].kernelCommandQueue);

		clFlush(devices[count].dataCommandQueue);
		clFinish(devices[count].dataCommandQueue);

		for (int count2 = 0; count2 < devices[count].numberOfKernels; count2++)
		{
			clReleaseKernel(devices[count].kernels[count2]);
		}
		clReleaseProgram(devices[count].program);
		for (int count2 = 0; count2 < devices[count].numberOfMemoryObjects; count2++)
		{
			clReleaseMemObject(devices[count].memoryObjects[count2]);
		}
		clReleaseCommandQueue(devices[count].kernelCommandQueue);
		clReleaseCommandQueue(devices[count].dataCommandQueue);

		delete[] devices[count].memoryObjects;
		delete[] devices[count].kernels;
		devices[count].memoryObjects = NULL;
		devices[count].kernels = NULL;

		delete[] devices[count].memoryObjectID;
		delete[] devices[count].kernelID;
		devices[count].memoryObjectID = NULL;
		devices[count].kernelID = NULL;

		delete[] devices[count].events;
		devices[count].events = NULL;

		devices[count].numberOfMemoryObjects = 0;
		devices[count].numberOfKernels = 0;
		devices[count].numberOfEvents = 0;
	}
	delete[] devices;
	devices = NULL;
}

int OpenCLWrapper::AllocateMemoryObject(size_t _size, cl_mem_flags _flags, void* _host_ptr) {
    int globalMemObjID = globalMemoryObjectIDCounter;
    globalMemoryObjectIDCounter++;
    memoryObjectIDs->emplace(globalMemObjID, std::vector<int>(todosDispositivos, -1)); // Inicializa com -1 para indicar que ainda não foi setado
//  for(int count2 = 0; count2 < world_size; count2++)
// 	{
// 		if(count2 == world_rank)
// 		{
    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            int deviceMemObjID = CreateMemoryObject(count - meusDispositivosOffset, _size, _flags, _host_ptr);
            (*memoryObjectIDs)[globalMemObjID][count] = deviceMemObjID;
        }
    }
        
    return globalMemObjID;
}

//}

//}


int OpenCLWrapper::GetDeviceMemoryObjectID(int globalMemObjID, int deviceIndex) {
   if (memoryObjectIDs->find(globalMemObjID) != memoryObjectIDs->end()) {
        return (*memoryObjectIDs)[globalMemObjID][deviceIndex];
    }
    return -1; // Erro se o ID não for encontrado
}

void OpenCLWrapper::setAttribute(int attribute, int globalMemoryObjectID) {
 for(int count2 = 0; count2 < world_size; count2++)
	{
		if(count2 == world_rank)
		{
	for(int count = 0; count < todosDispositivos; count++)
			{
				if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
				{
        int memoryObjectID = GetDeviceMemoryObjectID(globalMemoryObjectID, count - meusDispositivosOffset);
        SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], attribute, memoryObjectID);
    }
}
}

}
}


int OpenCLWrapper::WriteObject(int GlobalObjectID, const char *data, int offset, int size) {

int returnF;

 for(int count2 = 0; count2 < world_size; count2++)
	{
		if(count2 == world_rank)
		{
for(int count = 0; count < todosDispositivos; count++)
			{
				if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
				{
        int memoryObjectID = GetDeviceMemoryObjectID(GlobalObjectID, count - meusDispositivosOffset);
       returnF = WriteToMemoryObject(count - meusDispositivosOffset, memoryObjectID, data,offset, size);
    }
}

        }

}

return returnF;
}