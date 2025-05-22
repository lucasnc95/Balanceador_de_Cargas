#include "OpenCLWrapper.h"
#include <cstring>
#include <cmath>

OpenCLWrapper::OpenCLWrapper(int &argc, char** &argv) {
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
}

OpenCLWrapper::~OpenCLWrapper() {
  
//    for (int count = 0; count < numberOfDevices; count++) {
//         if (devices[count].context != nullptr) {
//             clReleaseContext(devices[count].context);
//         }
//         for (int i = 0; i < devices[count].numberOfKernels; i++) {
//             clReleaseKernel(devices[count].kernels[i]);
//         }
//         for (int j = 0; j < devices[count].numberOfMemoryObjects; j++) {
//             clReleaseMemObject(devices[count].memoryObjects[j]);
//         }
//         clReleaseCommandQueue(devices[count].kernelCommandQueue);
//         clReleaseCommandQueue(devices[count].dataCommandQueue);
        
//         delete[] devices[count].memoryObjects;
//         delete[] devices[count].kernels;
//         delete[] devices[count].memoryObjectID;
//         delete[] devices[count].kernelID;
//         delete[] devices[count].events;
//     }
//     delete[] devices;
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


// int OpenCLWrapper::InitParallelProcessor()
// {
//     cl_int state;

//     // Alocação de memória para IDs de plataforma
//     platformIDs = new cl_platform_id[maxNumberOfPlatforms];
//     if (!platformIDs) {
//         printf("Memory allocation failed for platformIDs.\n");
//         return -1;
//     }

//     // Obtendo as plataformas disponíveis
//     cl_uint numberOfPlatforms = 0;
//     state = clGetPlatformIDs(maxNumberOfPlatforms, platformIDs, &numberOfPlatforms);
//     if (state != CL_SUCCESS || numberOfPlatforms == 0) {
//         printf("OpenCL Error: Platform couldn't be found.\n");
//         delete[] platformIDs;
//         return -1;
//     }
//     printf("%u platform(s) found.\n", numberOfPlatforms);

//     // Alocação de memória para os dispositivos
//     maxNumberOfDevices = 10;
//     devices = new Device[maxNumberOfDevices];
//     if (!devices) {
//         printf("Memory allocation failed for devices.\n");
//         delete[] platformIDs;
//         return -1;
//     }

//     numberOfDevices = 0;

//     // Determinar o tipo de dispositivo com base em device_type
//     cl_device_type selectedDeviceType = CL_DEVICE_TYPE_ALL;
//     if (device_types == "GPU_DEVICES") {
//         selectedDeviceType = CL_DEVICE_TYPE_GPU;
//     } else if (device_types == "CPU_DEVICES") {
//         selectedDeviceType = CL_DEVICE_TYPE_CPU;
//     }

//     for (cl_uint i = 0; i < numberOfPlatforms; i++) {
//         cl_uint numberOfDevicesOfPlatform = 0;
//         cl_device_id deviceList[maxNumberOfDevices];

//         // Obtendo os dispositivos da plataforma atual de acordo com selectedDeviceType
//         state = clGetDeviceIDs(platformIDs[i], selectedDeviceType, maxNumberOfDevices, deviceList, &numberOfDevicesOfPlatform);
//         if (state != CL_SUCCESS || numberOfDevicesOfPlatform == 0) {
//             printf("OpenCL Error: Devices couldn't be resolved on platform %u.\n", i);
//             continue;
//         }

//         for (cl_uint j = 0; j < numberOfDevicesOfPlatform; j++) {
//             if (numberOfDevices >= maxNumberOfDevices) {
//                 printf("Maximum number of devices reached (%u).\n", maxNumberOfDevices);
//                 break;
//             }

//             devices[numberOfDevices].deviceID = deviceList[j];

//             // Obter e imprimir o nome do dispositivo
//             char deviceName[128];
//             clGetDeviceInfo(devices[numberOfDevices].deviceID, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
//             printf("Device (%u) name: %s\n", numberOfDevices, deviceName);

//             // Obter e imprimir o tipo de dispositivo
//             cl_device_type deviceType;
//             clGetDeviceInfo(devices[numberOfDevices].deviceID, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
//             const char* deviceTypeName = (deviceType == CL_DEVICE_TYPE_CPU) ? "CPU" :
//                                          (deviceType == CL_DEVICE_TYPE_GPU) ? "GPU" :
//                                          (deviceType == CL_DEVICE_TYPE_ACCELERATOR) ? "Accelerator" :
//                                          (deviceType == CL_DEVICE_TYPE_DEFAULT) ? "Default" : "Unknown";
//             printf("Device (%u) type: %s\n", numberOfDevices, deviceTypeName);

//             // Criando um contexto para cada dispositivo
//             cl_context_properties contextProperties[] = {
//                 CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[i],
//                 0
//             };
//             devices[numberOfDevices].context = clCreateContext(contextProperties, 1, &devices[numberOfDevices].deviceID, NULL, NULL, &state);
//             if (state != CL_SUCCESS) {
//                 printf("OpenCL Error: Context couldn't be created for device %u.\n", numberOfDevices);
//                 devices[numberOfDevices].context = NULL;
//                 continue;
//             }

//             // Obtendo e imprimindo a versão do OpenCL suportada pelo dispositivo
//             char versionStr[128];
//             clGetDeviceInfo(devices[numberOfDevices].deviceID, CL_DEVICE_VERSION, sizeof(versionStr), versionStr, NULL);
//             printf("Device (%u) supports OpenCL version: %s\n", numberOfDevices, versionStr);

//             // Obter e imprimir o número máximo de unidades de computação
//             cl_uint maxComputeUnits;
//             clGetDeviceInfo(devices[numberOfDevices].deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
//             printf("Device (%u) max compute units: %u\n", numberOfDevices, maxComputeUnits);

//             // Criando filas de comando com fallback para versões mais antigas
//             int majorVersion = 0, minorVersion = 0;
//             sscanf(versionStr, "OpenCL %d.%d", &majorVersion, &minorVersion);

//             if (majorVersion >= 2) {
//                 cl_queue_properties properties[] = {
//                     CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
//                     0
//                 };
//                 devices[numberOfDevices].kernelCommandQueue = clCreateCommandQueueWithProperties(devices[numberOfDevices].context, devices[numberOfDevices].deviceID, properties, &state);
//                 devices[numberOfDevices].dataCommandQueue = clCreateCommandQueueWithProperties(devices[numberOfDevices].context, devices[numberOfDevices].deviceID, properties, &state);
//             } else {
//                 devices[numberOfDevices].kernelCommandQueue = clCreateCommandQueue(devices[numberOfDevices].context, devices[numberOfDevices].deviceID, CL_QUEUE_PROFILING_ENABLE, &state);
//                 devices[numberOfDevices].dataCommandQueue = clCreateCommandQueue(devices[numberOfDevices].context, devices[numberOfDevices].deviceID, CL_QUEUE_PROFILING_ENABLE, &state);
//             }

//             if (state != CL_SUCCESS) {
//                 printf("OpenCL Error: Command queues couldn't be created for device %u.\n", numberOfDevices);
//                 clReleaseContext(devices[numberOfDevices].context);
//                 devices[numberOfDevices].context = NULL;
//                 continue;
//             }

//             // Inicializando arrays de objetos e eventos
//             devices[numberOfDevices].numberOfMemoryObjects = 0;
//             devices[numberOfDevices].numberOfKernels = 0;
//             devices[numberOfDevices].numberOfEvents = 0;

//             devices[numberOfDevices].memoryObjects = new cl_mem[maxMemoryObjects];
//             devices[numberOfDevices].kernels = new cl_kernel[maxKernels];
//             memset(devices[numberOfDevices].memoryObjects, 0, sizeof(cl_mem) * maxMemoryObjects);
//             memset(devices[numberOfDevices].kernels, 0, sizeof(cl_kernel) * maxKernels);

//             devices[numberOfDevices].memoryObjectID = new int[maxMemoryObjects];
//             devices[numberOfDevices].kernelID = new int[maxKernels];
//             memset(devices[numberOfDevices].memoryObjectID, 0, sizeof(int) * maxMemoryObjects);
//             memset(devices[numberOfDevices].kernelID, 0, sizeof(int) * maxKernels);

//             devices[numberOfDevices].events = new cl_event[maxEvents];
//             memset(devices[numberOfDevices].events, 0, sizeof(cl_event) * maxEvents);

//             devices[numberOfDevices].program = 0;

//             numberOfDevices++;
//         }
//     }

//     if (numberOfDevices == 0) {
//         printf("No OpenCL devices available.\n");
//         delete[] platformIDs;
//         delete[] devices;
//         return -1;
//     }

//     delete[] platformIDs;
//     return numberOfDevices;
// }


// int OpenCLWrapper::InitParallelProcessor()
// {
//     cl_int state;

//     // 1) Obter plataformas disponíveis
//     platformIDs = (cl_platform_id*)malloc(sizeof(cl_platform_id) * maxNumberOfPlatforms);
//     if (!platformIDs) {
//         printf("Memory allocation failed for platformIDs.\n");
//         return -1;
//     }
//     cl_uint numPlatforms = 0;
//     state = clGetPlatformIDs(maxNumberOfPlatforms, platformIDs, &numPlatforms);
//     if (state != CL_SUCCESS || numPlatforms == 0) {
//         printf("OpenCL Error: Platforms couldn't be found.\n");
//         free(platformIDs);
//         return -1;
//     }
//     numberOfPlatforms = numPlatforms;
//     printf("%u platform(s) found.\n", numberOfPlatforms);

//     // 2) Determinar tipo de dispositivo desejado
//     cl_device_type selectedType = CL_DEVICE_TYPE_ALL;
//     if (strcmp(device_types.c_str(), "GPU_DEVICES") == 0) selectedType = CL_DEVICE_TYPE_GPU;
//     else if (strcmp(device_types.c_str(), "CPU_DEVICES") == 0) selectedType = CL_DEVICE_TYPE_CPU;
//     printf("Selecting device types... DONE!\n");

//     // Preparar array de devices
//     devices = (Device*)malloc(sizeof(Device) * maxNumberOfDevices);
//     numberOfDevices = 0;

//     // 3) Para cada plataforma, coletar seus dispositivos e criar um contexto
//     for (cl_uint i = 0; i < numberOfPlatforms; ++i) {
//         cl_uint cnt = 0;
//         cl_device_id tmpList[maxNumberOfDevicesPerPlatform];
//         state = clGetDeviceIDs(
//             platformIDs[i],
//             selectedType,
//             maxNumberOfDevicesPerPlatform,
//             tmpList,
//             &cnt
//         );
//         if (state != CL_SUCCESS || cnt == 0) {
//             printf("Platform %u has no matching devices.\n", i);
//             continue;
//         }
//         printf("Platform %u: %u device(s) found.\n", i, cnt);

//         // 4) Criar contexto único para este grupo de dispositivos
//         cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[i], 0 };
//         cl_context ctx = clCreateContext(
//             props,
//             cnt,
//             tmpList,
//             NULL,
//             NULL,
//             &state
//         );
//         if (state != CL_SUCCESS || !ctx) {
//             printf("OpenCL Error: Context couldn't be created for platform %u.\n", i);
//             continue;
//         }
//         printf("Context created for platform %u with %u device(s).\n", i, cnt);

//         // 5) Para cada device neste contexto, configurar Device struct
//         for (cl_uint j = 0; j < cnt; ++j) {
//             int d = numberOfDevices;
//             devices[d].deviceID = tmpList[j];
//             devices[d].context  = ctx;

//             // Obter nome, tipo e unidades de computação
//             char name[128] = {0};
//             cl_device_type dt;
//             cl_uint cu;
//             clGetDeviceInfo(tmpList[j], CL_DEVICE_NAME, sizeof(name), name, NULL);
//             clGetDeviceInfo(tmpList[j], CL_DEVICE_TYPE, sizeof(dt), &dt, NULL);
//             clGetDeviceInfo(tmpList[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
//             devices[d].deviceType = dt;
//             devices[d].deviceComputeUnits = cu;
//             printf("Device (%d) on platform %u: name = %s, type = %s, computeUnits = %u\n",
//                    d, i, name,
//                    (dt==CL_DEVICE_TYPE_CPU)?"CPU":(dt==CL_DEVICE_TYPE_GPU)?"GPU":"Other",
//                    cu);

//             // Criar filas de comando
//             int major=0, minor=0;
//             char ver[128] = {0};
//             clGetDeviceInfo(tmpList[j], CL_DEVICE_VERSION, sizeof(ver), ver, NULL);
//             sscanf(ver, "OpenCL %d.%d", &major, &minor);
//             if (major >= 2) {
//                 cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
//                 devices[d].kernelCommandQueue =
//                     clCreateCommandQueueWithProperties(ctx, tmpList[j], qp, &state);
//                 devices[d].dataCommandQueue =
//                     clCreateCommandQueueWithProperties(ctx, tmpList[j], qp, &state);
//             } else {
//                 devices[d].kernelCommandQueue =
//                     clCreateCommandQueue(ctx, tmpList[j], CL_QUEUE_PROFILING_ENABLE, &state);
//                 devices[d].dataCommandQueue =
//                     clCreateCommandQueue(ctx, tmpList[j], CL_QUEUE_PROFILING_ENABLE, &state);
//             }
//             if (state != CL_SUCCESS) {
//                 printf("OpenCL Error: Command queues couldn't be created for device %d.\n", d);
//             }

//             // Inicializar vetores e contadores
//             devices[d].numberOfMemoryObjects = 0;
//             devices[d].numberOfKernels      = 0;
//             devices[d].numberOfEvents       = 0;
//             devices[d].memoryObjects = (cl_mem*)malloc(sizeof(cl_mem) * maxMemoryObjects);
//             devices[d].kernels       = (cl_kernel*)malloc(sizeof(cl_kernel) * maxKernels);
//             memset(devices[d].memoryObjects, 0, sizeof(cl_mem) * maxMemoryObjects);
//             memset(devices[d].kernels,       0, sizeof(cl_kernel) * maxKernels);
//             devices[d].memoryObjectID = (int*)malloc(sizeof(int) * maxMemoryObjects);
//             devices[d].kernelID       = (int*)malloc(sizeof(int) * maxKernels);
//             memset(devices[d].memoryObjectID, 0, sizeof(int) * maxMemoryObjects);
//             memset(devices[d].kernelID,       0, sizeof(int) * maxKernels);
//             devices[d].events = (cl_event*)malloc(sizeof(cl_event) * maxEvents);
//             memset(devices[d].events, 0, sizeof(cl_event) * maxEvents);
//             devices[d].program = NULL;

//             numberOfDevices++;
//             if (numberOfDevices >= maxNumberOfDevices) break;
//         }
//         if (numberOfDevices >= maxNumberOfDevices) break;
//     }

//     // Limpeza
//     free(platformIDs);
//     return numberOfDevices;
// }

int OpenCLWrapper::InitParallelProcessor()
{
    cl_int state;

    // 1) obter plataformas
    platformIDs = (cl_platform_id*)malloc(sizeof(cl_platform_id) * maxNumberOfPlatforms);
    cl_uint numPlatforms = 0;
    state = clGetPlatformIDs(maxNumberOfPlatforms, platformIDs, &numPlatforms);
    if (state != CL_SUCCESS || numPlatforms == 0) {
        printf("OpenCL Error: Platforms couldn't be found.\n");
        free(platformIDs);
        return -1;
    }
    numberOfPlatforms = numPlatforms;
    printf("%u platform(s) found.\n", numberOfPlatforms);

    // 2) escolher tipo
    cl_device_type selType = CL_DEVICE_TYPE_ALL;
    if (!device_types.compare("GPU_DEVICES")) selType = CL_DEVICE_TYPE_GPU;
    else if (!device_types.compare("CPU_DEVICES")) selType = CL_DEVICE_TYPE_CPU;

    // 3) preparar array de devices
    devices = (Device*)malloc(sizeof(Device) * maxNumberOfDevices);
    numberOfDevices = 0;

    // 4) para cada plataforma, coletar dispositivos e criar contexto+fila
    for (cl_uint p = 0; p < numberOfPlatforms; ++p) {
        cl_uint cnt = 0;
        cl_device_id tmp[maxNumberOfDevicesPerPlatform];
        state = clGetDeviceIDs(platformIDs[p], selType,
                               maxNumberOfDevicesPerPlatform,
                               tmp, &cnt);
        if (state != CL_SUCCESS || cnt == 0) continue;

        // criar um contexto para todos esses devices
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)platformIDs[p],
            0
        };
        cl_context ctx = clCreateContext(props, cnt, tmp, NULL, NULL, &state);
        if (state != CL_SUCCESS) {
            printf("Error creating context on platform %u\n", p);
            continue;
        }

        // criar uma única fila de comando para este contexto
        cl_command_queue queue;
        // OpenCL 2.0+:
        queue = clCreateCommandQueueWithProperties(ctx, tmp[0],
                    (cl_queue_properties[]){CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,0},
                    &state);
        if (state != CL_SUCCESS) {
            // fallback OpenCL 1.2
            queue = clCreateCommandQueue(ctx, tmp[0],
                        CL_QUEUE_PROFILING_ENABLE, &state);
        }

        // agora, para cada device dentro desse contexto, copiamos ctx+queue
        for (cl_uint j = 0; j < cnt && numberOfDevices < maxNumberOfDevices; ++j) {
            int d = numberOfDevices++;
            devices[d].deviceID = tmp[j];
            devices[d].context  = ctx;

            // vinculamos **a mesma** fila a kernelCommandQueue E dataCommandQueue
            devices[d].kernelCommandQueue = queue;
            devices[d].dataCommandQueue   = queue;

            // ... o restante permanece **inalterado** ...
            // obter nome/tipo/unidades e inicializar memoryObjects, kernels, etc.
            char name[128] = {0};
            clGetDeviceInfo(tmp[j], CL_DEVICE_NAME, sizeof(name), name, NULL);
            cl_device_type dt;
            cl_uint cu;
            clGetDeviceInfo(tmp[j], CL_DEVICE_TYPE, sizeof(dt), &dt, NULL);
            clGetDeviceInfo(tmp[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(cu), &cu, NULL);
            devices[d].deviceType = dt;
            devices[d].deviceComputeUnits = cu;
            printf("Device (%d): %s (%s), CUs=%u\n",
                   d, name,
                   dt==CL_DEVICE_TYPE_GPU ? "GPU" : "CPU",
                   cu);

            devices[d].numberOfMemoryObjects = 0;
            devices[d].numberOfKernels      = 0;
            devices[d].numberOfEvents       = 0;
            devices[d].memoryObjects = (cl_mem*)malloc(sizeof(cl_mem)*maxMemoryObjects);
            devices[d].kernels       = (cl_kernel*)malloc(sizeof(cl_kernel)*maxKernels);
            memset(devices[d].memoryObjects,0,sizeof(cl_mem)*maxMemoryObjects);
            memset(devices[d].kernels,0,sizeof(cl_kernel)*maxKernels);
            devices[d].memoryObjectID = (int*)malloc(sizeof(int)*maxMemoryObjects);
            devices[d].kernelID       = (int*)malloc(sizeof(int)*maxKernels);
            memset(devices[d].memoryObjectID,0,sizeof(int)*maxMemoryObjects);
            memset(devices[d].kernelID,0,sizeof(int)*maxKernels);
            devices[d].events = (cl_event*)malloc(sizeof(cl_event)*maxEvents);
            memset(devices[d].events,0,sizeof(cl_event)*maxEvents);
            devices[d].program = NULL;
        }
    }

    free(platformIDs);
    return numberOfDevices;
}



// void OpenCLWrapper::setKernel(const std::string &sourceFile, const std::string &kernelName) {
//     kernelSourceFile = sourceFile;
//     kernelFunctionName = kernelName;
//     kernelDispositivo = new int[todosDispositivos];
	
//     for(int count = 0; count < todosDispositivos; count++)
// 	{
// 		if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
// 		{
			
// 			kernelDispositivo[count] = CreateKernel(count-meusDispositivosOffset, kernelSourceFile.c_str(), kernelFunctionName.c_str());
// 			}
		
// 	}
    
    
//     kernelSet = true;
// }
// int OpenCLWrapper::CreateKernel(int devicePosition, const char *source, const char *kernelName)
// {
// 	if (devices[devicePosition].program != 0)
// 	{
// 		clReleaseProgram(devices[devicePosition].program);
// 	}
// 	devices[devicePosition].program = 0;

// 	cl_int state;

// 	// Read kernel file.
// 	FILE *fileHandle;
// 	char *sourceBuffer = (char *)malloc(sizeof(char) * MAX_SOURCE_BUFFER_LENGTH);
// 	if ((fileHandle = fopen(source, "r")) == NULL)
// 	{
// 		printf("Error reading %s\n!", source);
// 		return -1;
// 	}
// 	size_t sourceBufferLength = fread(sourceBuffer, 1, sizeof(char) * MAX_SOURCE_BUFFER_LENGTH, fileHandle);

// 	// Create program.
// 	devices[devicePosition].program = clCreateProgramWithSource(devices[devicePosition].context, 1, (const char **)&sourceBuffer, (const size_t *)&sourceBufferLength, &state);

// 	// Close kernel file.
// 	fclose(fileHandle);
// 	fileHandle = NULL;
// 	free(sourceBuffer);
// 	sourceBuffer = NULL;

// 	// Program created?
// 	if (state != CL_SUCCESS)
// 	{
// 		printf("Error creating program!\n");
// 		return -1;
// 	}

// 	// Compile program.
// 	state = clBuildProgram(devices[devicePosition].program, 1, &devices[devicePosition].deviceID, NULL, NULL, NULL);
// 	if (state != CL_SUCCESS)
// 	{
// 		printf("Error compiling program!\n");
// 		return -1;
// 	}

// 	// Create kernel.
// 	devices[devicePosition].kernels[devices[devicePosition].numberOfKernels] = clCreateKernel(devices[devicePosition].program, kernelName, &state);
// 	if (state != CL_SUCCESS)
// 	{
// 		printf("Error creating kernel!\n");
// 		return -1;
// 	}
// 	devices[devicePosition].kernelID[devices[devicePosition].numberOfKernels] = automaticNumber;
// 	devices[devicePosition].numberOfKernels += 1;
// 	automaticNumber += 1;
// 	return automaticNumber - 1;
// }


void OpenCLWrapper::setKernel(const std::string &sourceFile,
                              const std::string &kernelName)
{
    kernelSourceFile   = sourceFile;
    kernelFunctionName = kernelName;

    // Aloca vetor global de IDs de kernel
    kernelDispositivo = (int*)malloc(sizeof(int) * todosDispositivos);

    // Constrói e cria todos os kernels
    BuildAndCreateKernels(kernelSourceFile.c_str(),
                          kernelFunctionName.c_str());

    kernelSet = true;
}

int OpenCLWrapper::BuildAndCreateKernels(const char *sourcePath,
                                         const char *kernelName)
{
    cl_int state;

    // 1) Ler o código-fonte do kernel
    FILE *fh = fopen(sourcePath, "r");
    if (!fh) {
        printf("Error reading %s\n", sourcePath);
        return -1;
    }
    char *src = (char*)malloc(MAX_SOURCE_BUFFER_LENGTH);
    size_t srcLen = fread(src, 1, MAX_SOURCE_BUFFER_LENGTH, fh);
    fclose(fh);

    // 2) Para cada contexto distinto, compilamos e criamos kernels
    //    (na prática, um único contexto se você inicializou assim)
    for (int d = 0; d < numberOfDevices; ++d) {
        cl_context ctx = devices[d].context;
        // Já compilado para este contexto?
        if (devices[d].program != NULL) continue;

        // 2a) Contar quantos devices usam este contexto
        int cnt = 0;
        for (int e = 0; e < numberOfDevices; ++e)
            if (devices[e].context == ctx)
                ++cnt;

        // 2b) Alocar e preencher lista de cl_device_id
        cl_device_id *devIDs = (cl_device_id*)malloc(sizeof(cl_device_id) * cnt);
        int idx = 0;
        for (int e = 0; e < numberOfDevices; ++e) {
            if (devices[e].context == ctx)
                devIDs[idx++] = devices[e].deviceID;
        }

        // 2c) Criar o programa
        cl_program program = clCreateProgramWithSource(
            ctx, 1, (const char**)&src, (const size_t*)&srcLen, &state
        );
        if (state != CL_SUCCESS || program == NULL) {
            printf("Error creating program for context %p\n", (void*)ctx);
            free(devIDs);
            free(src);
            return -1;
        }

        // 2d) Compilar para todos devices desse contexto
        state = clBuildProgram(program, cnt, devIDs, NULL, NULL, NULL);
        if (state != CL_SUCCESS) {
            // Mostrar logs de build
            for (int k = 0; k < cnt; ++k) {
                size_t logSize = 0;
                clGetProgramBuildInfo(
                    program, devIDs[k],
                    CL_PROGRAM_BUILD_LOG,
                    0, NULL, &logSize
                );
                char *log = (char*)malloc(logSize+1);
                clGetProgramBuildInfo(
                    program, devIDs[k],
                    CL_PROGRAM_BUILD_LOG,
                    logSize, log, NULL
                );
                log[logSize] = '\0';
                printf("Build log (device %d):\\n%s\\n", k, log);
                free(log);
            }
            clReleaseProgram(program);
            free(devIDs);
            free(src);
            return -1;
        }

        // 2e) Criar kernels em cada device local desse contexto
        for (int e = 0; e < numberOfDevices; ++e) {
            if (devices[e].context != ctx) continue;
            cl_kernel k = clCreateKernel(program, kernelName, &state);
            if (state != CL_SUCCESS) {
                printf("Error creating kernel on device %d\\n", e);
                clReleaseProgram(program);
                free(devIDs);
                free(src);
                return -1;
            }
            int slot = devices[e].numberOfKernels;
            devices[e].kernels[slot]  = k;
            devices[e].kernelID[slot] = automaticNumber;
            devices[e].numberOfKernels++;

            // Mapear no vetor global
            int globalIdx = meusDispositivosOffset + e;
            kernelDispositivo[globalIdx] = automaticNumber;
            automaticNumber++;
        }

        // 2f) Guardar programa para liberar depois
        for (int e = 0; e < numberOfDevices; ++e) {
            if (devices[e].context == ctx)
                devices[e].program = program;
        }

        free(devIDs);
    }

    // 3) Liberar buffer fonte UMA ÚNICA VEZ
    free(src);
    return 0;
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


// int OpenCLWrapper::CreateMemoryObject(int devicePosition, size_t size, cl_mem_flags memoryType, void *hostMemory) {
// 	cl_int state;
// 	if(devices[devicePosition].numberOfMemoryObjects < maxMemoryObjects)
// 	{
// 		devices[devicePosition].memoryObjects[devices[devicePosition].numberOfMemoryObjects] = clCreateBuffer(devices[devicePosition].context, memoryType, size, hostMemory, &state);
// 		if(state != CL_SUCCESS)
// 		{
// 			printf("Error creating memory object!\n");
// 			return -1;
// 		}
// 		else
// 		{
// 			devices[devicePosition].memoryObjectID[devices[devicePosition].numberOfMemoryObjects] = automaticNumber;
// 			devices[devicePosition].numberOfMemoryObjects += 1;
// 		}
// 		automaticNumber += 1;
// 		return automaticNumber-1;
// 	}
// 	printf("Error creating memory object, limit exceeded!");
// 	return -1;



// }


int OpenCLWrapper::CreateMemoryObject(int devicePosition,size_t size,cl_mem_flags memoryType, void *hostMemory)                                    
{
    cl_int state;
    Device &dev = devices[devicePosition];

    // 1) criar buffer no contexto deste dispositivo
    cl_mem buf = clCreateBuffer(
        dev.context,
        memoryType,
        size,
        hostMemory,
        &state
    );
    if (state != CL_SUCCESS) {
        printf("Error creating memory object! %d\n", state);
        return -1;
    }

    // 2) armazenar no primeiro slot livre
    int slot = dev.numberOfMemoryObjects;
    if (slot >= maxMemoryObjects) {
        printf("Error: maxMemoryObjects exceeded\n");
        clReleaseMemObject(buf);
        return -1;
    }
    dev.memoryObjects[slot]    = buf;
    dev.memoryObjectID[slot]   = automaticNumber;
    dev.numberOfMemoryObjects += 1;

    // 3) devolve o ID lógico e incrementa
    return automaticNumber++;
}


void OpenCLWrapper::ExecuteKernel() {
  
 if(!sdSet){
    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {

            
            
            int deviceIndex2 = count - meusDispositivosOffset;
            if (deviceIndex2 >= 0 && deviceIndex2 < todosDispositivos) {
                std::cout<<"Dispositivo "<<count<<std::endl;
                kernelEventoDispositivo[deviceIndex2] = RunKernel(deviceIndex2, kernelDispositivo[deviceIndex2], offset[deviceIndex2], length[deviceIndex2], isDeviceCPU(deviceIndex2)? 8 : 64);
               // SynchronizeCommandQueue(deviceIndex2);
            } else {
                std::cerr << "Invalid device index: " << deviceIndex2 << std::endl;
            }
        }
    }
 
}

else {
    //Computação interna.
   
			for(int count = 0; count < todosDispositivos; count++)
			{
				if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
				{	
                    std::cout<<"(Interna) Dispositivo "<<count<<std::endl;
                    std::cout<<"offset "<<offset[count]+(sdSize)<<std::endl;
                    std::cout<<"length "<<length[count]-(sdSize)<<std::endl;
					RunKernel(count-meusDispositivosOffset, kernelDispositivo[count], offset[count]+(sdSize), length[count]-(sdSize), isDeviceCPU(count-meusDispositivosOffset) ? 8 :  64);
                    SynchronizeCommandQueue(count-meusDispositivosOffset);
				}
			}

			// //Sincronizacao da computação interna.
			// for(int count = 0; count < todosDispositivos; count++)
			// {
			// 	if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
			// 	{
			// 		SynchronizeCommandQueue(count-meusDispositivosOffset);
			// 	}
			// }

                
                Comms();

            //Sincronizacao da comunicacao.
			for(int count = 0; count < todosDispositivos; count++)
			{
				if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
				{
					SynchronizeCommandQueue(count-meusDispositivosOffset);
				}
			}

		
			
			// Computação das bordas.
for (int count = 0; count < todosDispositivos; count++) {
    if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
        int auxCount = 0;
        std::cout<<"(Borda 1)Dispositivo "<<count<<std::endl;
        std::cout<<"offset "<<offset[count]<<std::endl;
        std::cout<<"length "<<sdSize<<std::endl;
        RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count], sdSize, isDeviceCPU(count - meusDispositivosOffset) ? 8 : 64);
        SynchronizeCommandQueue(count-meusDispositivosOffset);
        std::cout<<"(Borda 2)Dispositivo "<<count<<std::endl;
        std::cout<<"offset "<<offset[count]+ length[count] - (sdSize)<<std::endl;
        std::cout<<"length "<<sdSize<<std::endl;
        RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count] + length[count] - (sdSize), sdSize, isDeviceCPU(count - meusDispositivosOffset) ? 8 : 64);
        SynchronizeCommandQueue(count-meusDispositivosOffset);    
    }
}




 }
    
    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            int deviceIndex2 = count - meusDispositivosOffset;
            if (deviceIndex2 >= 0 && deviceIndex2 < todosDispositivos) {
                SynchronizeCommandQueue(deviceIndex2);
            } else {
                std::cerr << "Invalid device index: " << deviceIndex2 << std::endl;
            }
        }
    }
    
   itCounter++;
}

// int OpenCLWrapper::RunKernel(int devicePosition,
//                              int kernelID,
//                              int parallelDataOffset,
//                              size_t parallelData,
//                              int workGroupSize)
// {
//     int kernelPosition = GetKernelPosition(devicePosition, kernelID);
//     if (kernelPosition < 0 ||
//         devices[devicePosition].numberOfEvents >= maxEvents)
//     {
//         printf("Error! Couldn't find kernel position %i or events %i exceeded limit.\n",
//                kernelPosition,
//                devices[devicePosition].numberOfEvents);
//         return -1;
//     }

//     // Compute global offset and size
//     size_t globalOffset = (parallelDataOffset > 0) ? (size_t)parallelDataOffset : 0;
//     size_t globalSize   = parallelData;

//     // Prepare NDRange arguments
//     size_t offsetArr[1] = { globalOffset };
//     size_t sizeArr  [1] = { globalSize };
//     // We intentionally pass NULL for local_work_size so OpenCL picks
//     // a suitable work-group size and allows exact globalSize
//     size_t *localArr = NULL;

//     // Enqueue kernel
//     cl_int err = clEnqueueNDRangeKernel(
//         devices[devicePosition].kernelCommandQueue,
//         devices[devicePosition].kernels[kernelPosition],
//         1,              // work_dim
//         offsetArr,      // global_work_offset
//         sizeArr,        // global_work_size
//         localArr,       // local_work_size = NULL
//         0, NULL,
//         &devices[devicePosition].events[
//             devices[devicePosition].numberOfEvents]
//     );
//     if (err != CL_SUCCESS) {
//         printf("Error queueing task! %d\n", err);
//         return -1;
//     }

//     // Synchronize immediately if desired
//     clFlush(devices[devicePosition].kernelCommandQueue);
//     clFinish(devices[devicePosition].kernelCommandQueue);

//     int evtIndex = devices[devicePosition].numberOfEvents;
//     devices[devicePosition].numberOfEvents += 1;
//     return evtIndex;
// }



int OpenCLWrapper::RunKernel(int devicePosition,
                             int kernelID,
                             int parallelDataOffset,
                             size_t parallelData,
                             int workGroupSize)
{
    // 1) Localiza o índice interno do kernel
    int kernelPosition = GetKernelPosition(devicePosition, kernelID);
    if (kernelPosition < 0 ||
        devices[devicePosition].numberOfEvents >= maxEvents)
    {
        printf("Error! Couldn't find kernel position %i or events %i exceeded limit.\n",
               kernelPosition,
               devices[devicePosition].numberOfEvents);
        return -1;
    }

    // 2) Define global offset e tamanho exato (sem arredondamento)
    size_t globalOffset = (parallelDataOffset > 0) ? (size_t)parallelDataOffset : 0;
    size_t globalSize   = parallelData;

    // 3) Use work‐group size = 1 para garantir ids sequenciais
    //    e cobertura exata de [offset, offset+length-1]
    size_t localSize    = 1;

    size_t offsetArr[1] = { globalOffset };
    size_t sizeArr  [1] = { globalSize };
    size_t localArr [1] = { localSize };

    // 4) Enfileira com offset global
    cl_int err = clEnqueueNDRangeKernel(
        devices[devicePosition].kernelCommandQueue,
        devices[devicePosition].kernels[kernelPosition],
        1,              // work_dim
        offsetArr,      // global_work_offset
        sizeArr,        // global_work_size
        localArr,       // local_work_size = 1
        0, NULL,
        &devices[devicePosition].events[
            devices[devicePosition].numberOfEvents]
    );
    if (err != CL_SUCCESS) {
        printf("Error queueing kernel %i on device %i: %d\n",
               kernelID, devicePosition, err);
        return -1;
    }

    // 5) Flush + finish para sincronizar
    clFlush(devices[devicePosition].kernelCommandQueue);
    clFinish(devices[devicePosition].kernelCommandQueue);

    // 6) Retorna índice do evento
    int evtIndex = devices[devicePosition].numberOfEvents;
    devices[devicePosition].numberOfEvents += 1;
    return evtIndex;
}




void OpenCLWrapper::SynchronizeCommandQueue(int devicePosition)
{
    
	clFinish(devices[devicePosition].kernelCommandQueue);
	clFinish(devices[devicePosition].dataCommandQueue);
	devices[devicePosition].numberOfEvents = 0;
}

void OpenCLWrapper::GatherResults(int dataIndex, void *resultData) {
   
			// std::cout<<"Entrando em gather results"<<std::endl;
			for(int count = 0; count < todosDispositivos; count++)
			{	

				if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
				{	
					int id = GetDeviceMemoryObjectID(dataIndex, count);
					
					ReadFromMemoryObject(count-meusDispositivosOffset, id, (char *)resultData+(offset[count]*elementSize*unitsPerElement), offset[count]*elementSize*unitsPerElement, length[count]*elementSize*unitsPerElement);
				
					SynchronizeCommandQueue(count-meusDispositivosOffset);
				}
			}
		
   
}

void OpenCLWrapper::setLoadBalancer(size_t _elementSize, int N_Elements, int units_per_elements, int _divisionSize) {
    
    
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
    nElements = N_Elements;
	unitsPerElement = units_per_elements;
    memset(ticks, 0, sizeof(long int) * todosDispositivos);    
    memset(tempos_por_carga, 0, sizeof(double) * todosDispositivos);
    memset(cargasNovas, 0, sizeof(float) * todosDispositivos);
    memset(cargasAntigas, 0, sizeof(float) * todosDispositivos);
	divisionSize = _divisionSize;
    offsetComputacao = 0;
    lengthComputacao = (nElements / todosDispositivos);
    elementSize = _elementSize;
    if (kernelSet)  {
        for (int count = 0; count < todosDispositivos; count++) {
            if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
               
			    initializeLengthOffset(offsetComputacao, (count + 1 == todosDispositivos) ? (nElements - offsetComputacao) : lengthComputacao, count);
                SynchronizeCommandQueue(count - meusDispositivosOffset);
            }
            offsetComputacao= offsetComputacao + lengthComputacao;
			
				
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


void OpenCLWrapper::Probing()
{
   // std::cout << "Iniciando balanceamento..." << std::endl;

    double tempoInicioProbing = MPI_Wtime();
    double localLatencia = 0.0, localBanda = 0.0;
	double localwriteByte1 = 0.0;
    double localwriteByte2 = 0.0;
    if (nElements <= 0 || unitsPerElement <= 0 || elementSize <= 0) {
        std::cerr << "Erro: Valores inválidos para nElements, unitsPerElement ou elementSize." << std::endl;
        return;
    }

    char *auxData = new char[nElements * unitsPerElement * elementSize];
    
    if (!auxData) {
        std::cerr << "Erro: Falha ao alocar memória para auxData." << std::endl;
        return;
    }

    //GatherResults(balancingTargetID, auxData);

    int somaLengthAntes = 0;
    for (int i = 0; i < todosDispositivos; i++) {
        somaLengthAntes += length[i];
    }
    std::cout << "Soma do length antes do probing: " << somaLengthAntes << std::endl;

    PrecisaoBalanceamento();
   
    for (int count = 0; count < todosDispositivos; count++)
    {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
        {
            int overlapNovoOffset = static_cast<int>(round(count == 0 ? 0.0f : cargasNovas[count - 1] * static_cast<float>(nElements)));

            int overlapNovoLength;
            if (count == todosDispositivos - 1) {
                // Último dispositivo pega todos os elementos restantes
                overlapNovoLength = nElements - overlapNovoOffset;
            } else {
                overlapNovoLength = static_cast<int>(round(cargasNovas[count] * static_cast<float>(nElements)) - round(count == 0 ? 0.0f : cargasNovas[count - 1] * static_cast<float>(nElements)));
            }

            // Verificações e logs antes de usar os valores calculados
            if (overlapNovoOffset < 0 || overlapNovoOffset >= nElements || 
                overlapNovoLength < 0 || overlapNovoOffset + overlapNovoLength > nElements) {
                std::cerr << "Erro: valores inválidos para overlapNovoOffset ou overlapNovoLength." << std::endl;
                std::cerr << "  overlapNovoOffset: " << overlapNovoOffset 
                          << ", overlapNovoLength: " << overlapNovoLength 
                          << ", nElements: " << nElements << std::endl;
                delete[] auxData;
                return;
            }

            for (int count2 = 0; count2 < todosDispositivos; count2++)
            {
                if (count > count2)
                {
                    if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
                    {
                        int overlap[2];
                        int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
                        char *malha = auxData;
                        int dataDevice = GetDeviceMemoryObjectID(balancingTargetID, count);

                        MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        if (overlap[1] > 0 && overlap[0] >= 0 && overlap[0] + overlap[1] <= nElements)
                        {
                            ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice, malha + (overlap[0] * unitsPerElement), overlap[0] * unitsPerElement * elementSize, overlap[1] * unitsPerElement * elementSize);
                            SynchronizeCommandQueue(count - meusDispositivosOffset);

                            int sizeCarga = overlap[1] * unitsPerElement;

                            double tempoInicioBanda = MPI_Wtime();
                            MPI_Ssend(malha + (overlap[0] * unitsPerElement), sizeCarga, MPI_CHAR, alvo, 0, MPI_COMM_WORLD);
                            double aux = (MPI_Wtime() - tempoInicioBanda) / sizeCarga;
                            localBanda = aux > localBanda ? aux : localBanda;
                        }
                    }
                }
                else if (count < count2)
                {
                    int overlapAntigoOffset = static_cast<int>(round(count == 0 ? 0.0f : cargasNovas[count - 1] * static_cast<float>(nElements)));
                    int overlapAntigoLength = static_cast<int>(round(cargasNovas[count] * static_cast<float>(nElements)) - round(count == 0 ? 0.0f : cargasNovas[count - 1] * static_cast<float>(nElements)));

                    if (overlapAntigoOffset < 0 || overlapAntigoOffset >= nElements || 
                        overlapAntigoLength < 0 || overlapAntigoOffset + overlapAntigoLength > nElements) {
                        std::cerr << "Erro: valores inválidos para overlapAntigoOffset ou overlapAntigoLength." << std::endl;
                        delete[] auxData;
                        return;
                    }

                    int intersecaoOffset;
                    int intersecaoLength;

                    if (ComputarIntersecao(overlapAntigoOffset, overlapAntigoLength, overlapNovoOffset, overlapNovoLength, &intersecaoOffset, &intersecaoLength))
                    {
                        if (count2 >= meusDispositivosOffset && count2 < meusDispositivosOffset + meusDispositivosLength)
                        {
                            char *malha = auxData;
                            int dataDevice[2] = {GetDeviceMemoryObjectID(balancingTargetID, count), GetDeviceMemoryObjectID(balancingTargetID, count2)};
                          	double tempoIniciowriteByte = MPI_Wtime();
						    ReadFromMemoryObject(count2 - meusDispositivosOffset, dataDevice[1], malha + (intersecaoOffset * unitsPerElement), intersecaoOffset * unitsPerElement * elementSize, intersecaoLength * unitsPerElement * elementSize);
                            SynchronizeCommandQueue(count2 - meusDispositivosOffset);

                            WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], malha + (intersecaoOffset * unitsPerElement), intersecaoOffset * unitsPerElement * elementSize, intersecaoLength * unitsPerElement * elementSize);
                            SynchronizeCommandQueue(count - meusDispositivosOffset);
							double tempoFimwriteByte = MPI_Wtime() - tempoIniciowriteByte;
							localwriteByte1 = tempoFimwriteByte > localwriteByte1 ? tempoFimwriteByte : localwriteByte1;
                        }
                        else
                        {
                            if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
                            {
                                int overlap[2] = {intersecaoOffset, intersecaoLength};
                                int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
                                char *malha = auxData;
                                int dataDevice = GetDeviceMemoryObjectID(balancingTargetID, count);

                                SynchronizeCommandQueue(count - meusDispositivosOffset);
                                double tempoInicioLatencia = MPI_Wtime();
                                MPI_Ssend(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
                                double aux = (MPI_Wtime() - tempoInicioLatencia) / 2;
                                localLatencia = aux > localLatencia ? aux : localLatencia;

                                MPI_Recv(malha + (overlap[0] * unitsPerElement), overlap[1] * unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
								double tempoIniciowriteByte2 = MPI_Wtime();
                                WriteToMemoryObject(count - meusDispositivosOffset, dataDevice, malha + (overlap[0] * unitsPerElement), overlap[0] * unitsPerElement * elementSize, overlap[1] * unitsPerElement * elementSize);
                                SynchronizeCommandQueue(count - meusDispositivosOffset);
								double tempoFimwriteByte = MPI_Wtime() - tempoIniciowriteByte2;
								localwriteByte2 = tempoFimwriteByte > localwriteByte2 ? tempoFimwriteByte : localwriteByte2;
                            }
                        }
                    }
                }
            }

            offset[count] = overlapNovoOffset;
            length[count] = overlapNovoLength;
            SynchronizeCommandQueue(count - meusDispositivosOffset);
        }
    }

    int somaLengthDepois = 0;
    for (int i = 0; i < todosDispositivos; i++) {
        somaLengthDepois += length[i];
    }
    std::cout << "Soma do length depois do probing: " << somaLengthDepois << std::endl;

    std::cout << "Após o probing: " << std::endl;
    for (int i = 0; i < todosDispositivos; i++)
        std::cout << " Offset[" << i << "] = " << offset[i] << " length[" << i << "] = " << length[i] << " ";
    std::cout << "\n";

    memcpy(cargasAntigas, cargasNovas, sizeof(float) * todosDispositivos);
	double writeByte1, writeByte2;
    MPI_Allreduce(&localLatencia, &latencia, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localBanda, &banda, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&localwriteByte1, &writeByte1, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&localwriteByte2, &writeByte2, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    writeByte = writeByte1 + writeByte2;
	MPI_Barrier(MPI_COMM_WORLD);
    double tempoFimProbing = MPI_Wtime();
    tempoBalanceamento += tempoFimProbing - tempoInicioProbing;
    fatorErro = tempoBalanceamento;
    delete[] auxData;
}


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
				
				kernelEventoDispositivo[count - meusDispositivosOffset] = RunKernel(count - meusDispositivosOffset, kernelDispositivo[count- meusDispositivosOffset], offset[count - meusDispositivosOffset], length[count- meusDispositivosOffset], isDeviceCPU(count - meusDispositivosOffset)? 8 : 256);
			}
		}
	

	
	// // Ticks.
	for (int count = 0; count < todosDispositivos; count++)
	{	
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{	
			SynchronizeCommandQueue(count - meusDispositivosOffset);
			
            long tickEvent = GetEventTaskTicks(count - meusDispositivosOffset, kernelEventoDispositivo[count]);          
            ticks[count] += tickEvent;
			
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
			if(count == 0)
	 		tempos[count] = ((float)ticks[count]) / (((float)cargasNovas[count]));
            else
			tempos[count] = ((float)ticks[count]) / (((float)cargasNovas[count] - (float)cargasNovas[count - 1]));
	 	}
	}
	float tempos_root[todosDispositivos];
	MPI_Allreduce(tempos, tempos_root, todosDispositivos, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	memcpy(tempos, tempos_root, sizeof(float) * todosDispositivos);
  


}



void OpenCLWrapper::LoadBalancing()
{
	double tempoInicioBalanceamento = MPI_Wtime();
    double tempoComputacaoInterna = tempos[0];
	PrecisaoBalanceamento();
    char *auxData = new char[nElements * unitsPerElement * elementSize];
	// Computar novas cargas.
	double localTempoCB;
	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			SynchronizeCommandQueue(count - meusDispositivosOffset);
			if (tempoComputacaoInterna < tempos[count] )
				tempoComputacaoInterna = tempos[count];
			if(count == 0)
				localTempoCB = cargasNovas[count] * (tempos[count]);

			else
				localTempoCB = (cargasNovas[count] - cargasNovas[count - 1]) * (tempos[count]);
		}
	}
	MPI_Allreduce(&tempoCB, &localTempoCB, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	tempoCB *= nElements;
	std::cout<<"TempoCB: "<<tempoCB<<std::endl;
	std::cout<<"writeByte: "<<writeByte<<std::endl;
	std::cout<<"tempo calculado: "<<((latencia) + ComputarNorma(cargasAntigas, cargasNovas, todosDispositivos) * ((writeByte) + (banda)) + (tempoCB))<<std::endl;
	std::cout<<"Tempo anterior: "<<tempoComputacaoInterna<<std::endl;
	if ((latencia + ComputarNorma(cargasAntigas, cargasNovas, todosDispositivos) * (writeByte + banda) + tempoCB) < tempoComputacaoInterna)
	{
		for (int count = 0; count < todosDispositivos; count++)
		{
			if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
			{
				int overlapNovoOffset = static_cast<int>(round(((count == 0) ? 0.0f : cargasNovas[count - 1]) * (static_cast<float>(nElements))));
				int overlapNovoLength;
            if (count == todosDispositivos - 1) {
                // Último dispositivo pega todos os elementos restantes
                overlapNovoLength = nElements - overlapNovoOffset;
            } else {
                overlapNovoLength = static_cast<int>(round(cargasNovas[count] * static_cast<float>(nElements)) - round(count == 0 ? 0.0f : cargasNovas[count - 1] * static_cast<float>(nElements)));
            }
				for (int count2 = 0; count2 < todosDispositivos; count2++)
				{
					if (count > count2)
					{
						// Atender requisicoes de outros processos.
						if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
						{
							int overlap[2];
							int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
							char *malha = auxData;
							int malhaDevice = GetDeviceMemoryObjectID(balancingTargetID, count);
							MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
							// Podem ocorrer requisicoes vazias.
							if (overlap[1] > 0)
							{
								ReadFromMemoryObject(count - meusDispositivosOffset, malhaDevice, (char *)(malha + (overlap[0] * unitsPerElement)), overlap[0] * unitsPerElement * elementSize, overlap[1] * unitsPerElement * elementSize);
								SynchronizeCommandQueue(count - meusDispositivosOffset);
								int sizeCarga = overlap[1] * unitsPerElement;
								MPI_Send(malha + (overlap[0] * unitsPerElement), sizeCarga, MPI_CHAR, alvo, 0, MPI_COMM_WORLD);
							}
						}
					}
					else if (count < count2)
					{
						// Fazer requisicoes a outros processos.
						int overlapAntigoOffset = static_cast<int>(round(((count2 == 0) ? 0 : cargasAntigas[count2 - 1]) * (nElements)));
						int overlapAntigoLength = static_cast<int>(round(((count2 == 0) ? cargasAntigas[count2] - 0.0f : cargasAntigas[count2] - cargasAntigas[count2 - 1]) * (nElements)));

						int intersecaoOffset;
						int intersecaoLength;

						if (((overlapAntigoOffset <= overlapNovoOffset - divisionSize) && ComputarIntersecao(overlapAntigoOffset, overlapAntigoLength, overlapNovoOffset - divisionSize, overlapNovoLength + divisionSize, &intersecaoOffset, &intersecaoLength)) ||
								((overlapAntigoOffset > overlapNovoOffset - divisionSize) && ComputarIntersecao(overlapNovoOffset - divisionSize, overlapNovoLength + divisionSize, overlapAntigoOffset, overlapAntigoLength, &intersecaoOffset, &intersecaoLength)))
						{
							if (count2 >= meusDispositivosOffset && count2 < meusDispositivosOffset + meusDispositivosLength)
							{    
                                char *malha = auxData;  
								int malhaDevice[2] = {GetDeviceMemoryObjectID(balancingTargetID, count), GetDeviceMemoryObjectID(balancingTargetID, count2)};
								ReadFromMemoryObject(count2 - meusDispositivosOffset, malhaDevice[1], (char *)(malha + (intersecaoOffset * unitsPerElement)), intersecaoOffset * unitsPerElement * sizeof(float), intersecaoLength * unitsPerElement * sizeof(float));
								SynchronizeCommandQueue(count2 - meusDispositivosOffset);
								WriteToMemoryObject(count - meusDispositivosOffset, malhaDevice[0], (char *)(malha + (intersecaoOffset * unitsPerElement)), intersecaoOffset * unitsPerElement * sizeof(float), intersecaoLength * unitsPerElement * sizeof(float));
								SynchronizeCommandQueue(count - meusDispositivosOffset);
							}
							else
							{
								// Fazer uma requisicao.
								if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
								{
									int overlap[2] = {intersecaoOffset, intersecaoLength};
									int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
									char *malha = auxData;
							        int malhaDevice = GetDeviceMemoryObjectID(balancingTargetID, count);
									MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
									MPI_Recv(malha + (overlap[0] * unitsPerElement), overlap[1] * unitsPerElement, MPI_FLOAT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
									WriteToMemoryObject(count - meusDispositivosOffset, malhaDevice, (char *)(malha + (overlap[0] * unitsPerElement)), overlap[0] * unitsPerElement * sizeof(float), overlap[1] * unitsPerElement * sizeof(float));
									SynchronizeCommandQueue(count - meusDispositivosOffset);
								}
							}
						}
						else
						{
							// Fazer uma requisicao vazia.
							if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
							{
								int overlap[2] = {0, 0};
								int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
								char *malha = auxData;
								MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
							}
						}
					}
				}

				offset[count] = overlapNovoOffset;
				length[count] = overlapNovoLength;

				//WriteToMemoryObject(count - meusDispositivosOffset, parametrosMalhaDispositivo[count], (char *)parametrosMalha[count], 0, sizeof(int) * NUMERO_PARAMETROS_MALHA);
				SynchronizeCommandQueue(count - meusDispositivosOffset);
			}
		}
		memcpy(cargasAntigas, cargasNovas, sizeof(float) * todosDispositivos);

         int somaLengthDepois = 0;
    for (int i = 0; i < todosDispositivos; i++) {
        somaLengthDepois += length[i];
    }
   // std::cout << "Soma do length depois do balanceamento: " << somaLengthDepois << std::endl;


    for (int i = 0; i < todosDispositivos; i++)
        std::cout << " Dispositivo[" << i << "] = " << ((double)length[i]/(double)somaLengthDepois)*100 << "% " << std::endl;
    std::cout << "\n";
		MPI_Barrier(MPI_COMM_WORLD);
		double tempoFimBalanceamento = MPI_Wtime();
		tempoBalanceamento += tempoFimBalanceamento - tempoInicioBalanceamento;
	}
	delete[]auxData;
}





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

int OpenCLWrapper::WriteToMemoryObject(int devicePosition, int memoryObjectID, const char *data, int offset, size_t size) {
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

int OpenCLWrapper::ReadFromMemoryObject(int devicePosition, int memoryObjectID, char *data, int offset, size_t size)
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

// int OpenCLWrapper::AllocateMemoryObject(size_t _size, cl_mem_flags _flags, void* _host_ptr) {
//     int globalMemObjID = globalMemoryObjectIDCounter;
//     globalMemoryObjectIDCounter++;
//     memoryObjectIDs->emplace(globalMemObjID, std::vector<int>(todosDispositivos, -1)); // Inicializa com -1 para indicar que ainda não foi setado

//     for (int count = 0; count < todosDispositivos; count++) {
//         if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
//             int deviceMemObjID = CreateMemoryObject(count - meusDispositivosOffset, _size, _flags, _host_ptr);
//             (*memoryObjectIDs)[globalMemObjID][count] = deviceMemObjID;
            
//         }
//     }
    
// //     for(int count = 0; count < todosDispositivos; count++)
// // 			{
// // 		if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
// // 		{
// //          SynchronizeCommandQueue(count-meusDispositivosOffset);
// //     }
// // }  



//     return globalMemObjID;
// }

int OpenCLWrapper::AllocateMemoryObject(size_t _size,
                                        cl_mem_flags _flags,
                                        void* _host_ptr)
{
    // 1) Cria um novo ID global
    int globalMemObjID = globalMemoryObjectIDCounter++;
    // inicializa a linha do mapa com -1
    memoryObjectIDs->emplace(globalMemObjID,std::vector<int>(todosDispositivos, -1)); 
                             
    
    // 2) Para cada contexto distinto, alocar UMA vez
    for (int d = 0; d < numberOfDevices; ++d) {
        cl_context ctx = devices[d].context;
        // Já criamos para este contexto?
        bool created = false;
        for (int e = 0; e < d; ++e) {
            if (devices[e].context == ctx &&
                (*memoryObjectIDs)[globalMemObjID][
                    meusDispositivosOffset + e] != -1)
            {
                // reutiliza o mesmo cl_mem
                int existingLocalID =
                    (*memoryObjectIDs)[globalMemObjID][
                        meusDispositivosOffset + e] ;
                // registra este localID em d também
                (*memoryObjectIDs)[globalMemObjID][
                    meusDispositivosOffset + d] = existingLocalID;
                created = true;
                break;
            }
        }
        if (created) continue;

        // 3) Não criada ainda para este contexto: vamos criar
        int localPos = d;  // índice dentro do array devices[]
        int localMemObjID = CreateMemoryObject(
            localPos, _size, _flags, _host_ptr
        );
        if (localMemObjID < 0) return -1;

        // 4) Registrar em **todos** os dispositivos com este mesmo contexto
        for (int e = 0; e < numberOfDevices; ++e) {
            if (devices[e].context == ctx) {
                (*memoryObjectIDs)[globalMemObjID][
                    meusDispositivosOffset + e] = localMemObjID;
            }
        }
    }

    return globalMemObjID;
}


int OpenCLWrapper::GetDeviceMemoryObjectID(int globalMemObjID, int deviceIndex) {
   if (memoryObjectIDs->find(globalMemObjID) != memoryObjectIDs->end()) {
        return (*memoryObjectIDs)[globalMemObjID][deviceIndex];
    }
    return -1; // Erro se o ID não for encontrado
}

void OpenCLWrapper::setAttribute(int attribute, int globalMemoryObjectID) {
 
	for(int count = 0; count < todosDispositivos; count++)
			{
				if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
				{
        int memoryObjectID = GetDeviceMemoryObjectID(globalMemoryObjectID, count - meusDispositivosOffset);
        SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], attribute, memoryObjectID);
    }
}

for(int count = 0; count < todosDispositivos; count++)
			{
		if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
		{
         SynchronizeCommandQueue(count-meusDispositivosOffset);
    }
}     



}




int OpenCLWrapper::WriteObject(int GlobalObjectID, const char *data, int offset, size_t size) {

int returnF;

 
for(int count = 0; count < todosDispositivos; count++)
			{
		if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
		{
        int memoryObjectID = GetDeviceMemoryObjectID(GlobalObjectID, count - meusDispositivosOffset);
       returnF = WriteToMemoryObject(count - meusDispositivosOffset, memoryObjectID, data,offset, size);
        
    }
}

 for(int count = 0; count < todosDispositivos; count++)
			{
		if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
		{
         SynchronizeCommandQueue(count-meusDispositivosOffset);
    }
}       



return returnF;

}

void OpenCLWrapper::setBalancingTargetID(int targetID)
{

balancingTargetID = targetID;


}


void OpenCLWrapper::setSubdomainBoundary(size_t _sdSize, int _nArgs, int* _args) {
    sdSize = _sdSize; // Tamanho da borda
    nArgs = _nArgs;
    args = new int[nArgs]; // Inicializando corretamente o array args
    for (int i = 0; i < nArgs; i++) {
        args[i] = _args[i]; // Copiando os valores do array _args
    }
    sdSet = true; // Borda definida
}




void OpenCLWrapper::Comms(){
size_t tamanhoBorda = sdSize;
char *malha = new char[nElements * elementSize * unitsPerElement];
//GatherResults(balancingTargetID, malha);
int *malhaDevice = new int[2];
int *borda = new int[2];
int alvo;
int *dataEventoDispositivo = new int[todosDispositivos];
MPI_Request sendRequest, receiveRequest;
//Transferencia de bordas, feita em quatro passos.
			for(int passo = 0; passo < 4; passo++)
			{
				for(int count = 0; count < todosDispositivos; count++)
				{
					if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
					{
						
						
						//Entre processos diferentes, no quarto passo.
						if(passo == 3)
						{
							if(count == meusDispositivosOffset && count > 0)
							{
								
								malhaDevice[0] = GetDeviceMemoryObjectID(balancingTargetID, count);
								borda[0] = int(offset[count]-(tamanhoBorda));
								borda[0] = (borda[0] < 0) ? 0 : borda[0];
								borda[1] = int(offset[count]);
								alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count-1);

								if(alvo%2 == 0)
								{
									MPI_Irecv(malha+(borda[0]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, &receiveRequest);

									dataEventoDispositivo[count] = ReadFromMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[1]*unitsPerElement)), borda[1]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);
									MPI_Isend(malha+(borda[1]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, &sendRequest);
									MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
									MPI_Wait(&receiveRequest, MPI_STATUS_IGNORE);
									
									WriteToMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[0]*unitsPerElement)), borda[0]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);

								}
							}
							if(count == meusDispositivosOffset+meusDispositivosLength-1 && count < todosDispositivos-1)
							{
								malhaDevice[0]= GetDeviceMemoryObjectID(balancingTargetID, count);
			
								borda[0] = int((offset[count]+length[count])-(tamanhoBorda));
								borda[0] = (borda[0] < 0) ? 0 : borda[0];
								borda[1] = int((offset[count]+length[count]));
								alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count+1);

								if(alvo%2 == 1)
								{
									dataEventoDispositivo[count] = ReadFromMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[1]*unitsPerElement)), borda[1]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);
									MPI_Isend(malha+(borda[1]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, &sendRequest);
									MPI_Irecv(malha+(borda[0]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, &receiveRequest);
									MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
									MPI_Wait(&receiveRequest, MPI_STATUS_IGNORE);
									WriteToMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[0]*unitsPerElement)), borda[0]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);

								}
							}
						}

						//Entre processos diferentes, no terceiro passo.
						if(passo == 2)
						{
							if(count == meusDispositivosOffset && count > 0)
							{
								//malha = ((simulacao%2)==0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];
								malhaDevice[0] = GetDeviceMemoryObjectID(balancingTargetID, count);
								borda[0] = offset[count]-(tamanhoBorda);
								borda[0] = (borda[0] < 0) ? 0 : borda[0];
								borda[1] = offset[count];
								alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count-1);

								if(alvo%2 == 1)
								{
									MPI_Irecv(malha+(borda[0]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR , alvo, 0, MPI_COMM_WORLD, &receiveRequest);

									dataEventoDispositivo[count] = ReadFromMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[1]*unitsPerElement)), borda[1]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);
									MPI_Isend(malha+(borda[1]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, &sendRequest);
									MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
									MPI_Wait(&receiveRequest, MPI_STATUS_IGNORE);

									WriteToMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[0]*unitsPerElement)), borda[0]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);

								}
							}
							if(count == meusDispositivosOffset+meusDispositivosLength-1 && count < todosDispositivos-1)
							{
								
								malhaDevice[0] = GetDeviceMemoryObjectID(balancingTargetID, count);
								borda[0] = (offset[count]+length[count])-(tamanhoBorda);
								borda[0] = (borda[0] < 0) ? 0 : borda[0];
								borda[1] = (offset[count]+length[count]);
								alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count+1);

								if(alvo%2 == 0)
								{
									dataEventoDispositivo[count] = ReadFromMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[1]*unitsPerElement)), borda[1]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);
									MPI_Isend(malha+(borda[1]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, &sendRequest);
									MPI_Irecv(malha+(borda[0]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, &receiveRequest);
									MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
									MPI_Wait(&receiveRequest, MPI_STATUS_IGNORE);

									WriteToMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[0]*unitsPerElement)), borda[0]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);

								}
							}
						}

						//No mesmo processo, no primeiro passo.
						if(passo == 0 && count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength-1)
						{
							//malha = ((simulacao%2)==0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];
							malhaDevice[0] = GetDeviceMemoryObjectID(balancingTargetID, count);
							malhaDevice[1] = GetDeviceMemoryObjectID(balancingTargetID, count+1);
							borda[0] = int(offset[count+1]-(tamanhoBorda));
							borda[0] = (borda[0] < 0) ? 0 : borda[0];
							borda[1] = int(offset[count+1]);
							dataEventoDispositivo[count+0] = ReadFromMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[0]*unitsPerElement)), borda[0]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
							SynchronizeCommandQueue(count+0-meusDispositivosOffset);

							dataEventoDispositivo[count+1] = ReadFromMemoryObject(count+1-meusDispositivosOffset, malhaDevice[1], (char *)(malha+(borda[1]*unitsPerElement)), borda[1]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
							SynchronizeCommandQueue(count+1-meusDispositivosOffset);

						}

						//No mesmo processo, no segundo passo.
						if(passo == 1 && count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength-1)
						{
							//malha = ((simulacao%2)==0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];
							malhaDevice[0] = GetDeviceMemoryObjectID(balancingTargetID, count);
							malhaDevice[1] = GetDeviceMemoryObjectID(balancingTargetID, count+1);
							borda[0] = offset[count+1]-(tamanhoBorda);
							borda[0] = (borda[0] < 0) ? 0 : borda[0];
							borda[1] = offset[count+1];

							WriteToMemoryObject(count+0-meusDispositivosOffset, malhaDevice[0], (malha+(borda[1]*unitsPerElement)), borda[1]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
							SynchronizeCommandQueue(count+0-meusDispositivosOffset);

							WriteToMemoryObject(count+1-meusDispositivosOffset, malhaDevice[1], (malha+(borda[0]*unitsPerElement)), borda[0]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
							SynchronizeCommandQueue(count+1-meusDispositivosOffset);
						}
					}
				}
			}

delete[] malha;
}

