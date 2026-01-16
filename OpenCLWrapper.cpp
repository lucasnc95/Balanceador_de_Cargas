#include "OpenCLWrapper.h"
#include <algorithm> 
#include <cstring>
#include <cmath>
#include <iomanip> 

OpenCLWrapper::OpenCLWrapper(int &argc, char** &argv) {
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   comm = MPI_COMM_WORLD;
    
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

int OpenCLWrapper::getWorldRank(){


return world_rank;


}

int OpenCLWrapper::getComm(){

//return comm;
return 1;

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
 printf("todos dispositivos: %i \n", todosDispositivos);   
 return todosDispositivos;

}

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
        // queue = clCreateCommandQueueWithProperties(ctx, tmp[0],
        //             (cl_queue_properties[]){CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,0},
        //             &state);
        // if (state != CL_SUCCESS) {
        //     // fallback OpenCL 1.2
        //     queue = clCreateCommandQueue(ctx, tmp[0],
        //                 CL_QUEUE_PROFILING_ENABLE, &state);
        // }
        cl_queue_properties queue_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,0};

        queue = clCreateCommandQueueWithProperties(ctx, tmp[0], queue_properties, &state);

        if (state != CL_SUCCESS) {
            // fallback OpenCL 1.2
            queue = clCreateCommandQueue(ctx, tmp[0], CL_QUEUE_PROFILING_ENABLE, &state);
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



int OpenCLWrapper::CreateMemoryObject(int devicePosition,int size,cl_mem_flags memoryType, void *hostMemory)                                    
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


// void OpenCLWrapper::ExecuteKernel() {
//     if(!sdSet) {
//         printf("erro");
//     } else {
//         MPI_Barrier(MPI_COMM_WORLD);
//      //   printf("\n--- INÍCIO DA ITERAÇÃO %ld ---\n", itCounter);

//         // 1. Computação dos PONTOS INTERNOS
//         for(int count = 0; count < todosDispositivos; count++) {
//             if(count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
//                 int local_idx = count - meusDispositivosOffset;
//                 int internal_offset = offset[count] + sdSize;
//                 int internal_length = length[count] - 2*sdSize; 
                
//                 // Print para a computação interna
//               //  printf("[Rank %d] INTERNO:   Dispositivo Global %d (Local %d) | Offset: %d, Length: %d (Índices %d a %d)\n",
//                       // world_rank, count, local_idx, internal_offset, internal_length, internal_offset, internal_offset + internal_length - 1);

//                 if (internal_length > 0) {
//                     RunKernel(local_idx, kernelDispositivo[count], internal_offset, internal_length, isDeviceCPU(local_idx) ? 8 : 64);
//                     SynchronizeCommandQueue(local_idx);
//                 }
//             }
//         }
        
//         // 2. Comunicação das BORDAS (HALO EXCHANGE)
//         MPI_Barrier(MPI_COMM_WORLD);
//       //  printf("--- INICIANDO Comms() ---\n");
//         Comms();
//         MPI_Barrier(MPI_COMM_WORLD);
//       //  printf("--- FINALIZANDO Comms() ---\n");

//         // 3. Computação das BORDAS
//         for (int count = 0; count < todosDispositivos; count++) {
//             if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
//                 int local_idx = count - meusDispositivosOffset;
                
//                 // Print para a borda esquerda
//              //   printf("[Rank %d] BORDA ESQ: Dispositivo Global %d (Local %d) | Offset: %d, Length: %d (Índice %d)\n",
//                       // world_rank, count, local_idx, offset[count], sdSize, offset[count]);
//                 RunKernel(local_idx, kernelDispositivo[count], offset[count], sdSize, isDeviceCPU(local_idx) ? 8 : 64);
//                 SynchronizeCommandQueue(local_idx);
                
//                 // Print para a borda direita
//                 int right_border_offset = offset[count] + length[count] - sdSize;
//              //   printf("[Rank %d] BORDA DIR: Dispositivo Global %d (Local %d) | Offset: %d, Length: %d (Índice %d)\n",
//                       // world_rank, count, local_idx, right_border_offset, sdSize, right_border_offset);
//                 RunKernel(local_idx, kernelDispositivo[count], right_border_offset, sdSize, isDeviceCPU(local_idx) ? 8 : 64);
//                 SynchronizeCommandQueue(local_idx);
//             }
//         }
//     	MPI_Barrier(MPI_COMM_WORLD);
//        // printf("--- FIM DA ITERAÇÃO %ld ---\n", itCounter);
//     }
    
//     // Sincronização final
//     for (int i = 0; i < meusDispositivosLength; ++i) {
//         SynchronizeCommandQueue(i);
//     }
//     itCounter++;
// }




void OpenCLWrapper::ExecuteKernel() {
    if(!sdSet) {
        printf("erro: subdomain boundary not set\n");
        return;
    } 
    
    MPI_Barrier(MPI_COMM_WORLD);

    // 1. PONTOS INTERNOS
    for(int count = 0; count < todosDispositivos; count++) {
        if(count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            int local_idx = count - meusDispositivosOffset;
            int internal_offset = offset[count] + sdSize;
            int internal_length = length[count] - 2*sdSize; 
            
            if (internal_length > 0) {
                int evtIdx = RunKernel(local_idx, kernelDispositivo[count], internal_offset, internal_length, isDeviceCPU(local_idx) ? 8 : 64);
                
                // CAPTURA DO INÍCIO (Só na primeira iteração após o balanceamento)
                if (captureStartEvent) {
                    SaveEvent(devices[local_idx].events[evtIdx], startEvents[count]);
                }
                SynchronizeCommandQueue(local_idx);
            }
        }
    }
    
    // Desliga a captura de início até o próximo LoadBalancing
    if (captureStartEvent) captureStartEvent = false;

    // 2. COMUNICAÇÃO
    MPI_Barrier(MPI_COMM_WORLD);
    Comms();
    MPI_Barrier(MPI_COMM_WORLD);

    // 3. BORDAS
    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            int local_idx = count - meusDispositivosOffset;
            
            // Borda Esquerda
            RunKernel(local_idx, kernelDispositivo[count], offset[count], sdSize, isDeviceCPU(local_idx) ? 8 : 64);
            SynchronizeCommandQueue(local_idx);
            
            // Borda Direita (Última operação -> Fim do ciclo)
            int right_border_offset = offset[count] + length[count] - sdSize;
            int evtIdx = RunKernel(local_idx, kernelDispositivo[count], right_border_offset, sdSize, isDeviceCPU(local_idx) ? 8 : 64);
            
            // SEMPRE SALVA O FIM da iteração atual
            SaveEvent(devices[local_idx].events[evtIdx], endEvents[count]);
            
            SynchronizeCommandQueue(local_idx);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Limpa fila, mas nossos eventos start/end estão seguros via clRetainEvent
    for (int i = 0; i < meusDispositivosLength; ++i) {
        SynchronizeCommandQueue(i);
    }
    itCounter++;
}





int OpenCLWrapper::RunKernel(int devicePosition,
                             int kernelID,
                             int parallelDataOffset,
                             int parallelData,
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
    // Número de bytes por elemento de work (unitsPerElement * elementSize)
    int elemBytes = elementSize * unitsPerElement;

    // 1) Calcula quantos bytes este rank enviará (soma de all local lengths)
    int localElems = 0;
    for (int dev = meusDispositivosOffset; dev < meusDispositivosOffset + meusDispositivosLength; ++dev) {
        localElems += length[dev];
    }
    int localBytes = localElems * elemBytes;

    // 2) Aloca buffer local e copia todas as fatias de device para ele
    char *localBuf = (char*)malloc(localBytes);
    size_t pos = 0;
    for (int dev = meusDispositivosOffset; dev < meusDispositivosOffset + meusDispositivosLength; ++dev) {
        int localIdx = dev - meusDispositivosOffset;
        int memObj   = GetDeviceMemoryObjectID(dataIndex, dev);
        size_t byteLen = size_t(length[dev] * elemBytes);
        size_t byteOff = size_t(offset[dev]   * elemBytes);

        // Lê do device para localBuf[pos]
        ReadFromMemoryObject(
            localIdx,
            memObj,
            localBuf + pos,
            byteOff,
            byteLen
        );
        SynchronizeCommandQueue(localIdx);
        pos += byteLen;
    }

    // 3) Prepara arrays para MPI_Allgatherv
    int *recvCounts = (int*)malloc(sizeof(int) * world_size);
    int *displs     = (int*)malloc(sizeof(int) * world_size);

    // Cada rank informa localBytes
    MPI_Allgather(
        &localBytes, 1, MPI_INT,
        recvCounts,  1, MPI_INT,
        MPI_COMM_WORLD
    );

    // Deslocamentos a partir de recvCounts
    displs[0] = 0;
    for (int i = 1; i < world_size; ++i) {
        displs[i] = displs[i-1] + recvCounts[i-1];
    }

    // 4) Allgatherv para que todos os ranks recebam o vetor completo
    MPI_Allgatherv(
        localBuf, localBytes,   MPI_CHAR,
        resultData, recvCounts, displs, MPI_CHAR,
        MPI_COMM_WORLD
    );

    // 5) Limpeza
    free(localBuf);
    free(recvCounts);
    free(displs);
}

void OpenCLWrapper::SaveEvent(cl_event source, cl_event &destination) {
    if (destination != NULL) {
        clReleaseEvent(destination); // Solta o evento antigo
    }
    destination = source;
    if (destination != NULL) {
        clRetainEvent(destination); // Segura o novo evento para não ser deletado
    }
}


void OpenCLWrapper::setLoadBalancer(int _elementSize, int N_Elements, int units_per_elements, int _divisionSize) {
    ticks = new double[todosDispositivos];
    for(int i=0; i<todosDispositivos; i++) ticks[i] = 0.0; 
    startEvents = new cl_event[todosDispositivos];
    endEvents = new cl_event[todosDispositivos];
    for(int i = 0; i < todosDispositivos; i++) {
        startEvents[i] = NULL;
        endEvents[i] = NULL;
    }
    captureStartEvent = true;
    firstLoadBalancing = true;
    tempos_por_carga = new double[todosDispositivos];    
    cargasNovas = new float[todosDispositivos]; 
    cargasAntigas = new float[todosDispositivos]; 
    swapBufferDispositivo = new int*[todosDispositivos]; 
    memObjects = new int[todosDispositivos];  
    tempos = new double[todosDispositivos];    
    offset = new int[todosDispositivos];   
    length = new int[todosDispositivos];   
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

  if (kernelSet) {
    if (world_rank == 0) {
        offsetComputacao = 0;
        lengthComputacao = (nElements / todosDispositivos);

        for (int count = 0; count < todosDispositivos; count++) {
            initializeLengthOffset(offsetComputacao, (count + 1 == todosDispositivos) ? (nElements - offsetComputacao) : lengthComputacao, count);
            offsetComputacao += lengthComputacao;
        }
    }

    MPI_Bcast(offset, todosDispositivos, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(length, todosDispositivos, MPI_INT, 0, MPI_COMM_WORLD);

    loadBalancerSet = true;
    } else {
        std::cerr << "Error: Kernel is not initialized." << std::endl;
    }

    for (int i = 0; i < world_size; i++) {
        for (int count = 0; count < todosDispositivos; count++) {
            if (world_rank == i) {
                cargasNovas[count] = static_cast<float>(count + 1) * (1.0f / static_cast<float>(todosDispositivos));
                cargasAntigas[count] = cargasNovas[count];
                tempos[count] = 1;
                std::cout << "Carga: " << cargasNovas[count] << " rank: " << world_rank << std::endl;
            }
        }
    }


}


// void OpenCLWrapper::Probing()
// {
//     // Etapa 0: Medir o desempenho e calcular as novas cargas ideais.
//     PrecisaoBalanceamento();
//     CollectOverheads();
//     // --- ETAPA 1: CALCULAR AS NOVAS PARTIÇÕES (de forma robusta) ---
//     int* novosOffsets = new int[todosDispositivos + 1];
//     int* novosLengths = new int[todosDispositivos];
    
//     novosOffsets[0] = 0;
//     for (int i = 0; i < todosDispositivos; i++) {
//         // Calcula o ponto final da partição e arredonda.
//         novosOffsets[i + 1] = static_cast<int>(round(cargasNovas[i] * static_cast<float>(nElements)));
//     }
//     // Garante que a última partição vá até o final, corrigindo possíveis erros de arredondamento.
//     novosOffsets[todosDispositivos] = nElements;

//     for (int i = 0; i < todosDispositivos; i++) {
//         // O tamanho é a diferença entre o início da próxima partição e o início da atual.
//         novosLengths[i] = novosOffsets[i + 1] - novosOffsets[i];
//     }

//     // --- ETAPA 2: COLETAR TODOS OS DADOS (Gather) ---

//     // Aloca um buffer global no host para conter uma cópia de todos os dados.
//     int elemBytes = elementSize * unitsPerElement;
//     char* globalDataSnapshot = new char[(size_t)nElements * elemBytes];
    
//     // Usa a função GatherResults que já existe para coletar os dados de todos os dispositivos.
//     // Ela junta os dados de forma ordenada no buffer 'globalDataSnapshot'.
//     GatherResults(balancingTargetID, globalDataSnapshot);
    
//     // --- ETAPA 3: DISTRIBUIR OS DADOS PARA AS NOVAS PARTIÇÕES (Scatter) ---
    
//     // Cada processo agora itera sobre seus dispositivos locais...
//     for (int count = meusDispositivosOffset; count < meusDispositivosOffset + meusDispositivosLength; ++count) {
//         int localIdx = count - meusDispositivosOffset;
//         int memObj = GetDeviceMemoryObjectID(balancingTargetID, count);
        
//         // ...e escreve a fatia correta do snapshot global para o dispositivo,
//         // de acordo com a NOVA partição calculada.
//         size_t new_offset_bytes = (size_t)novosOffsets[count] * elemBytes;
//         size_t new_length_bytes = (size_t)novosLengths[count] * elemBytes;
        
//         if (new_length_bytes > 0) {
//             WriteToMemoryObject(
//                 localIdx,
//                 memObj,
//                 globalDataSnapshot + new_offset_bytes, 
//                 new_offset_bytes,                    
//                 new_length_bytes
//             );
//         }
//     }
    
//     // Atualiza os arrays globais de offset e length com as novas partições
//     memcpy(this->offset, novosOffsets, todosDispositivos * sizeof(int));
//     memcpy(this->length, novosLengths, todosDispositivos * sizeof(int));
//     memcpy(this->cargasAntigas, this->cargasNovas, todosDispositivos * sizeof(float));

//     // Limpeza
//     delete[] novosOffsets;
//     delete[] novosLengths;
//     delete[] globalDataSnapshot;

//     // Sincronização e verificação (opcional, mas bom para depuração)
//     MPI_Barrier(MPI_COMM_WORLD);
//     if (world_rank == 0) {
//         int somaLengthDepois = 0;
//         for (int i = 0; i < todosDispositivos; i++) {
//             somaLengthDepois += length[i];
//         }
//         std::cout << "Soma do length depois do probing: " << somaLengthDepois << " (Total esperado: " << nElements << ")" << std::endl;
//         std::cout << "Partições após o probing: " << std::endl;
//         for (int i = 0; i < todosDispositivos; i++) {
//             std::cout << "  Dispositivo[" << i << "]: Offset=" << offset[i] << ", Length=" << length[i] << std::endl;
//         }
//     }
// }




// Necessário para std::fixed e std::setprecision

// void OpenCLWrapper::PrecisaoBalanceamento() {
//     // 1) INICIALIZAÇÃO LIMPA (Evita lixo de memória)
//     // Zera os vetores locais antes de começar
//     std::fill(ticks, ticks + todosDispositivos, 0.0);
//     std::fill(tempos, tempos + todosDispositivos, 0.0);
    
//     // Arrays auxiliares para índices de eventos
//     int *startEventIdx = new int[meusDispositivosLength];
//     int *endEventIdx = new int[meusDispositivosLength];

//     // Limpa a fila de comandos para garantir que não haja eventos pendentes
//     for (int i = 0; i < meusDispositivosLength; i++) {
//         SynchronizeCommandQueue(i);
//     }

//     // 2) MEDIÇÃO (Loop de precisão)
//     for (int iter = 0; iter < precision; ++iter) {
        
//         for (int count = 0; count < todosDispositivos; ++count) {
//             // Apenas executa nos meus dispositivos
//             if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
//                 int localIdx = count - meusDispositivosOffset;
                
//                 // Só executa se tiver trabalho (length > 0) para evitar erros do OpenCL
//                 if (length[count] > 0) {
//                     int evtIndex = RunKernel(
//                         localIdx,
//                         kernelDispositivo[count],
//                         offset[count],
//                         length[count],
//                         isDeviceCPU(localIdx) ? 8 : 256
//                     );

//                     // Captura o primeiro evento
//                     if (iter == 0) {
//                         startEventIdx[localIdx] = evtIndex;
//                     }
//                     // Atualiza o último evento
//                     endEventIdx[localIdx] = evtIndex;
//                 }
//             }
//         }
//     }

//     // 3) LEITURA LOCAL DOS TEMPOS
//     for (int count = 0; count < todosDispositivos; ++count) {
//         if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
//             int localIdx = count - meusDispositivosOffset;
            
//             if (length[count] > 0) {
//                 clFinish(devices[localIdx].kernelCommandQueue);

//                 // Obtém o tempo total da bateria de testes em SEGUNDOS
//                 double totalSeconds = GetEventTaskTicks(localIdx, startEventIdx[localIdx], endEventIdx[localIdx]);
                
//                 ticks[count] = totalSeconds;
                
//                 // Reset de eventos
//                 devices[localIdx].numberOfEvents = 0;
//             } else {
//                 // Se o dispositivo não rodou nada (carga 0), definimos um tempo muito pequeno mas não zero
//                 // para evitar divisão por zero na ComputarCargas.
//                 ticks[count] = 0.000001; 
//             }
//         }
//     }

//     delete[] startEventIdx;
//     delete[] endEventIdx;

//     // 4) SINCRONIZAÇÃO GLOBAL (Evita que outros processos tenham lixo)
//     // Usamos um buffer temporário para garantir que o Allreduce funcione corretamente
//     double *ticksGlobal = new double[todosDispositivos];
    
//     // Soma os tempos de todos os processos. Como cada processo só preencheu seus devices (e zerou os outros),
//     // a soma resultará no vetor completo correto em TODOS os processos.
//     MPI_Allreduce(ticks, ticksGlobal, todosDispositivos, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
//     // Atualiza o vetor ticks da classe com os dados globais
//     std::copy(ticksGlobal, ticksGlobal + todosDispositivos, ticks);
//     delete[] ticksGlobal;

//     // 5) CÁLCULO DE TEMPOS MÉDIOS E SANITY CHECK (Igual em todos os processos)
//     for (int i = 0; i < todosDispositivos; ++i) {
//         // Média por iteração
//         double mediaSegundos = ticks[i] / (double)precision;

//         // --- FILTRO DE VALORES INVÁLIDOS (Clamp) ---
//         // Se for muito pequeno (quase zero ou negativo), fixa num mínimo (1 nanossegundo)
//         if (mediaSegundos < 1.0e-9) mediaSegundos = 1.0e-9;
        
//         // Se for absurdamente grande (ex: > 1000 segundos por iteração, ou lixo 1e35), 
//         // limita a um valor alto mas "real" para não quebrar a matemática.
//         if (mediaSegundos > 1000.0) mediaSegundos = 1000.0;

//         tempos[i] = mediaSegundos;
//     }

//     // 6) CÁLCULO DAS CARGAS (Agora seguro pois tempos[] está limpo e sincronizado)
//     ComputarCargas(tempos, cargasAntigas, cargasNovas, todosDispositivos);

//     // Opcional: Print de Debug apenas no Rank 0 com precisão controlada
    
//     if (world_rank == 0) {
//         std::cout << "--- Tempos Medidos (s) ---" << std::endl;
//         std::cout << std::fixed << std::setprecision(5); // Limita visualização a 5 casas
//         for(int i=0; i<todosDispositivos; i++) {
//             std::cout << "Dev " << i << ": " << tempos[i] << " s | Carga: " << cargasNovas[i] << std::endl;
//         }
//         std::cout.unsetf(std::ios_base::floatfield); // Reseta formatação
//     }
    
// }

void OpenCLWrapper::PrecisaoBalanceamento() {
    // 1. INICIALIZAÇÃO
    // Zera os acumuladores
    std::fill(ticks, ticks + todosDispositivos, 0.0);
    std::fill(tempos, tempos + todosDispositivos, 0.0);

    // Arrays locais para segurar os eventos (start e end) com segurança
    cl_event *localStartEvents = new cl_event[meusDispositivosLength];
    cl_event *localEndEvents = new cl_event[meusDispositivosLength];

    // Limpa filas e inicializa ponteiros
    for (int i = 0; i < meusDispositivosLength; i++) {
        localStartEvents[i] = NULL;
        localEndEvents[i] = NULL;
        SynchronizeCommandQueue(i); 
    }

    // 2. LOOP DE MEDIÇÃO (Benchmark)
    for (int iter = 0; iter < precision; ++iter) {
        
        for (int count = 0; count < todosDispositivos; ++count) {
            // Verifica se o dispositivo é local
            if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
                int localIdx = count - meusDispositivosOffset;
                
                // Só executa se tiver carga
                if (length[count] > 0) {
                    // Executa e pega o índice do evento na lista temporária
                    int evtIdx = RunKernel(
                        localIdx, 
                        kernelDispositivo[count], 
                        offset[count], 
                        length[count], 
                        isDeviceCPU(localIdx) ? 8 : 256
                    );
                    
                    cl_event currentEvt = devices[localIdx].events[evtIdx];

                    // Se for a primeira iteração, guarda como START
                    if (iter == 0) {
                        localStartEvents[localIdx] = currentEvt;
                        clRetainEvent(localStartEvents[localIdx]); // Importante: Impede que o evento seja destruído
                    }

                    // Atualiza sempre o END (no final do loop será o último)
                    if (localEndEvents[localIdx] != NULL) {
                        clReleaseEvent(localEndEvents[localIdx]); // Solta o anterior
                    }
                    localEndEvents[localIdx] = currentEvt;
                    clRetainEvent(localEndEvents[localIdx]); // Segura o novo
                }
            }
        }
    }

    // 3. CÁLCULO DOS TEMPOS
    for (int count = 0; count < todosDispositivos; ++count) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            int localIdx = count - meusDispositivosOffset;
            
            if (length[count] > 0) {
                clFinish(devices[localIdx].kernelCommandQueue);

                // Calcula o tempo TOTAL decorrido (em Segundos)
                // Usa a nova assinatura que aceita (cl_event, cl_event)
                ticks[count] = GetEventTaskTicks(localStartEvents[localIdx], localEndEvents[localIdx]);
            } else {
                ticks[count] = 1.0e-9; // Tempo mínimo para evitar divisão por zero
            }

            // Limpeza: Libera os eventos retidos
            if (localStartEvents[localIdx]) clReleaseEvent(localStartEvents[localIdx]);
            if (localEndEvents[localIdx])   clReleaseEvent(localEndEvents[localIdx]);
            
            // Reseta a contagem de eventos da classe para não estourar o limite
            devices[localIdx].numberOfEvents = 0;
        }
    }
    
    delete[] localStartEvents;
    delete[] localEndEvents;

    // 4. SINCRONIZAÇÃO GLOBAL
    // Soma os tempos de todos os processos (Gather implícito via Sum)
    double *ticksGlobal = new double[todosDispositivos];
    MPI_Allreduce(ticks, ticksGlobal, todosDispositivos, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    std::copy(ticksGlobal, ticksGlobal + todosDispositivos, ticks);
    delete[] ticksGlobal;

    // 5. CÁLCULO DAS CARGAS
    for (int i = 0; i < todosDispositivos; ++i) {
        // O vetor 'ticks' contém o tempo TOTAL da bateria de testes.
        // A média por iteração é ticks / precision.
        tempos[i] = ticks[i] / (double)precision;
        
        // Sanity Check
        if (tempos[i] < 1.0e-9) tempos[i] = 1.0e-9;
    }

    // Calcula as novas cargas ideais com base nos tempos médios
    ComputarCargas(tempos, cargasAntigas, cargasNovas, todosDispositivos);
}

// void OpenCLWrapper::LoadBalancing()
// {
//     // --- ETAPA 1: Medição e Decisão ---
    
//     // 1.1 Mede o desempenho atual e calcula 'cargasNovas' (Ideal)
//     PrecisaoBalanceamento();
    
//     // (Opcional) Coleta overheads se necessário para outros modelos
//     // CollectOverheads(); 

//     // 1.2 Verificação de Sanidade dos Tempos (Evita bugs numéricos)
//     bool temposValidos = true;
//     for (int i = 0; i < todosDispositivos; ++i) {
//         if (std::isnan(tempos[i]) || std::isinf(tempos[i]) || tempos[i] > 1e15) {
//             temposValidos = false;
//         }
//     }
//     if (!temposValidos) {
//         if (world_rank == 0) std::cerr << "[LB] Aviso: Tempos inválidos detectados. Cancelando balanceamento." << std::endl;
//         return;
//     }

//     // 1.3 CÁLCULO DA NORMA E VERIFICAÇÃO DO THRESHOLD
//     // Calcula o quanto as cargas ideais (Novas) diferem das atuais (Antigas)
//     float threshold = 0.000025f; 
//     float norma = ComputarNorma(cargasAntigas, cargasNovas, todosDispositivos);

//     // Se a mudança for muito pequena, não vale a pena pagar o custo de mover dados.
//     if (norma <= threshold) {
//         if (world_rank == 0) {
//             // Comentado para não poluir o log, descomente se quiser ver
//             std::cout << "[LB] Estável. Norma (" << norma << ") <= Threshold. Ignorando." << std::endl;
//         }
//         return; // <--- PONTO DE SAÍDA ANTECIPADA: Mantém tudo como está.
//     }

//     if (world_rank == 0) {
//         std::cout << "[LB] Rebalanceando... Norma (" << norma << ") > Threshold." << std::endl;
//     }

//     // --- ETAPA 2: Calcular Novo Layout (Se passou do threshold) ---
    
//     int* novosOffsets = new int[todosDispositivos + 1];
//     int* novosLengths = new int[todosDispositivos];

//     novosOffsets[0] = 0;
//     for (int i = 0; i < todosDispositivos; i++) {
//         // Calcula o fim da partição baseado na carga acumulada 'cargasNovas'
//         int fim = static_cast<int>(round(cargasNovas[i] * static_cast<float>(nElements)));
        
//         // O último sempre termina em nElements para fechar a conta
//         if (i == todosDispositivos - 1) fim = nElements;
        
//         novosOffsets[i + 1] = fim;
//         novosLengths[i] = novosOffsets[i + 1] - novosOffsets[i];
//     }

//     // --- ETAPA 3: Redistribuição via Interseção (Mover Dados) ---
    
//     size_t bytesTotal = (size_t)nElements * unitsPerElement * elementSize;
//     char* auxData = new char[bytesTotal]; 

//     // Buffers que precisam ser atualizados (Ping e Pong)
//     std::vector<int> buffersParaAtualizar;
//     buffersParaAtualizar.push_back(balancingTargetID);
//     if (enableSwapBuffer) buffersParaAtualizar.push_back(swapBufferID);

//     // Itera sobre todas as combinações (Origem -> Destino)
//     for (int src = 0; src < todosDispositivos; ++src) {
//         for (int dst = 0; dst < todosDispositivos; ++dst) {
            
//             int intersecaoOffset, intersecaoLength;
//             bool haIntersecao = ComputarIntersecao(
//                 offset[src], length[src],             // Layout Antigo
//                 novosOffsets[dst], novosLengths[dst], // Layout Novo
//                 &intersecaoOffset, &intersecaoLength
//             );

//             if (haIntersecao && intersecaoLength > 0) {
                
//                 int rankSrc = RecuperarPosicaoHistograma(dispositivosWorld, world_size, src);
//                 int rankDst = RecuperarPosicaoHistograma(dispositivosWorld, world_size, dst);
                
//                 size_t byteOffset = (size_t)intersecaoOffset * unitsPerElement * elementSize;
//                 size_t byteLength = (size_t)intersecaoLength * unitsPerElement * elementSize;

//                 for (int memObjID : buffersParaAtualizar) {
//                     int idSrc = GetDeviceMemoryObjectID(memObjID, src);
//                     int idDst = GetDeviceMemoryObjectID(memObjID, dst);

//                     // A: Cópia Local (Mesmo Rank)
//                     if (rankSrc == world_rank && rankDst == world_rank) {
//                         if (src != dst) {
//                             int localSrc = src - meusDispositivosOffset;
//                             int localDst = dst - meusDispositivosOffset;
                            
//                             ReadFromMemoryObject(localSrc, idSrc, auxData, (int)byteOffset, (int)byteLength);
//                             SynchronizeCommandQueue(localSrc);
                            
//                             WriteToMemoryObject(localDst, idDst, auxData, (int)byteOffset, (int)byteLength);
//                             SynchronizeCommandQueue(localDst);
//                         }
//                     }
//                     // B: Envio (Sou Origem)
//                     else if (world_rank == rankSrc) {
//                         int localSrc = src - meusDispositivosOffset;
//                         ReadFromMemoryObject(localSrc, idSrc, auxData, (int)byteOffset, (int)byteLength);
//                         SynchronizeCommandQueue(localSrc);
                        
//                         int tag = 5000 + (src * todosDispositivos + dst); 
//                         MPI_Send(auxData, (int)byteLength, MPI_BYTE, rankDst, tag, MPI_COMM_WORLD);
//                     }
//                     // C: Recebimento (Sou Destino)
//                     else if (world_rank == rankDst) {
//                         int localDst = dst - meusDispositivosOffset;
//                         int tag = 5000 + (src * todosDispositivos + dst);
                        
//                         MPI_Recv(auxData, (int)byteLength, MPI_BYTE, rankSrc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        
//                         WriteToMemoryObject(localDst, idDst, auxData, (int)byteOffset, (int)byteLength);
//                         SynchronizeCommandQueue(localDst);
//                     }
//                 }
//             }
//         }
//     }

//     // --- ETAPA 4: Atualizar Estado da Classe ---
//     // Só atualizamos 'cargasAntigas', 'offset' e 'length' SE o rebalanceamento ocorreu
//     memcpy(this->offset, novosOffsets, todosDispositivos * sizeof(int));
//     memcpy(this->length, novosLengths, todosDispositivos * sizeof(int));
//     memcpy(this->cargasAntigas, this->cargasNovas, todosDispositivos * sizeof(float));

//     delete[] novosOffsets;
//     delete[] novosLengths;
//     delete[] auxData;

//     MPI_Barrier(MPI_COMM_WORLD);
// }


void OpenCLWrapper::Probing()
{
    if (world_rank == 0) std::cout << "[Probing] Iniciando balanceamento inicial..." << std::endl;

    // 1. MEDIÇÃO INTENSIVA (Executa kernel 'precision' vezes)
    PrecisaoBalanceamento(); 
    
    // (Opcional) Coleta overheads
    // CollectOverheads(); 

    // Verificação de Sanidade
    for (int i = 0; i < todosDispositivos; ++i) {
        if (std::isnan(tempos[i]) || std::isinf(tempos[i]) || tempos[i] > 1e15) {
             if (world_rank == 0) std::cerr << "[Probing] Erro: Tempos inválidos." << std::endl;
             return;
        }
    }

    // 2. CÁLCULO DO NOVO LAYOUT
    int* novosOffsets = new int[todosDispositivos + 1];
    int* novosLengths = new int[todosDispositivos];
    novosOffsets[0] = 0;
    for (int i = 0; i < todosDispositivos; i++) {
        int fim = static_cast<int>(round(cargasNovas[i] * static_cast<float>(nElements)));
        if (i == todosDispositivos - 1) fim = nElements;
        novosOffsets[i + 1] = fim;
        novosLengths[i] = novosOffsets[i + 1] - novosOffsets[i];
    }

    // 3. REDISTRIBUIÇÃO (Lógica Eficiente de Interseção)
    size_t bytesTotal = (size_t)nElements * unitsPerElement * elementSize;
    char* auxData = new char[bytesTotal]; 
    std::vector<int> buffersParaAtualizar;
    buffersParaAtualizar.push_back(balancingTargetID);
    if (enableSwapBuffer) buffersParaAtualizar.push_back(swapBufferID);

    for (int src = 0; src < todosDispositivos; ++src) {
        for (int dst = 0; dst < todosDispositivos; ++dst) {
            int intersecaoOffset, intersecaoLength;
            bool haIntersecao = ComputarIntersecao(offset[src], length[src], novosOffsets[dst], novosLengths[dst], &intersecaoOffset, &intersecaoLength);

            if (haIntersecao && intersecaoLength > 0) {
                int rankSrc = RecuperarPosicaoHistograma(dispositivosWorld, world_size, src);
                int rankDst = RecuperarPosicaoHistograma(dispositivosWorld, world_size, dst);
                size_t byteOffset = (size_t)intersecaoOffset * unitsPerElement * elementSize;
                size_t byteLength = (size_t)intersecaoLength * unitsPerElement * elementSize;

                for (int memObjID : buffersParaAtualizar) {
                    int idSrc = GetDeviceMemoryObjectID(memObjID, src);
                    int idDst = GetDeviceMemoryObjectID(memObjID, dst);

                    if (rankSrc == world_rank && rankDst == world_rank) {
                        if (src != dst) {
                            int localSrc = src - meusDispositivosOffset;
                            int localDst = dst - meusDispositivosOffset;
                            ReadFromMemoryObject(localSrc, idSrc, auxData, (int)byteOffset, (int)byteLength);
                            SynchronizeCommandQueue(localSrc);
                            WriteToMemoryObject(localDst, idDst, auxData, (int)byteOffset, (int)byteLength);
                            SynchronizeCommandQueue(localDst);
                        }
                    } else if (world_rank == rankSrc) {
                        int localSrc = src - meusDispositivosOffset;
                        ReadFromMemoryObject(localSrc, idSrc, auxData, (int)byteOffset, (int)byteLength);
                        SynchronizeCommandQueue(localSrc);
                        int tag = 5000 + (src * todosDispositivos + dst); 
                        MPI_Send(auxData, (int)byteLength, MPI_BYTE, rankDst, tag, MPI_COMM_WORLD);
                    } else if (world_rank == rankDst) {
                        int localDst = dst - meusDispositivosOffset;
                        int tag = 5000 + (src * todosDispositivos + dst);
                        MPI_Recv(auxData, (int)byteLength, MPI_BYTE, rankSrc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        WriteToMemoryObject(localDst, idDst, auxData, (int)byteOffset, (int)byteLength);
                        SynchronizeCommandQueue(localDst);
                    }
                }
            }
        }
    }

    // 4. ATUALIZAÇÃO
    memcpy(this->offset, novosOffsets, todosDispositivos * sizeof(int));
    memcpy(this->length, novosLengths, todosDispositivos * sizeof(int));
    memcpy(this->cargasAntigas, this->cargasNovas, todosDispositivos * sizeof(float));

    delete[] novosOffsets;
    delete[] novosLengths;
    delete[] auxData;

    MPI_Barrier(MPI_COMM_WORLD);
    
    // PREPARAÇÃO PARA O FUTURO
    firstLoadBalancing = false; // Probing feito!
    captureStartEvent = true;   // Começar a monitorar execução real agora
}


void OpenCLWrapper::LoadBalancing()
{
    // --- DECISÃO: É a primeira vez? ---
    if (firstLoadBalancing) {
        Probing(); // Chama a versão pesada (benchmark)
        return;
    }

    // --- MODO LEVE: Usar eventos da execução real ---
    
    // 1. Calcular Tempos Reais (Sem rodar kernels extras)
    std::fill(ticks, ticks + todosDispositivos, 0.0);
    std::fill(tempos, tempos + todosDispositivos, 0.0);

    for (int count = 0; count < todosDispositivos; ++count) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            
            // Calcula tempo entre o Start (início do ciclo) e End (fim do ciclo atual)
            double totalSeconds = GetEventTaskTicks(startEvents[count], endEvents[count]);
            
            // Se o dispositivo não tem elementos, tempo é quase zero
            if (length[count] == 0) totalSeconds = 1.0e-9;
            
            ticks[count] = totalSeconds;
        }
    }

    // Sincronização Global dos Ticks Reais
    double *ticksGlobal = new double[todosDispositivos];
    MPI_Allreduce(ticks, ticksGlobal, todosDispositivos, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    std::copy(ticksGlobal, ticksGlobal + todosDispositivos, ticks);
    delete[] ticksGlobal;

    // Converte para tempos e Cargas
    for (int i = 0; i < todosDispositivos; ++i) {
        double t = ticks[i];
        if (t < 1.0e-9) t = 1.0e-9;
        tempos[i] = t;
    }

    ComputarCargas(tempos, cargasAntigas, cargasNovas, todosDispositivos);

    // 2. Verificar Threshold
    float threshold = 0.000025f; 
    float norma = ComputarNorma(cargasAntigas, cargasNovas, todosDispositivos);

    if (norma <= threshold) {
        // Se não precisar balancear, apenas rearmamos o gatilho para medir o próximo intervalo
        captureStartEvent = true;
        return;
    }

    if (world_rank == 0) {
        std::cout << "[LB] Rebalanceando... Norma (" << norma << ") > Threshold." << std::endl;
    }

    // 3. Redistribuição (Código duplicado do Probing para garantir isolamento)
    // Nota: Poderia ser uma função separada 'Redistribute()', mas aqui está inline conforme pedido
    int* novosOffsets = new int[todosDispositivos + 1];
    int* novosLengths = new int[todosDispositivos];
    novosOffsets[0] = 0;
    for (int i = 0; i < todosDispositivos; i++) {
        int fim = static_cast<int>(round(cargasNovas[i] * static_cast<float>(nElements)));
        if (i == todosDispositivos - 1) fim = nElements;
        novosOffsets[i + 1] = fim;
        novosLengths[i] = novosOffsets[i + 1] - novosOffsets[i];
    }

    size_t bytesTotal = (size_t)nElements * unitsPerElement * elementSize;
    char* auxData = new char[bytesTotal]; 
    std::vector<int> buffersParaAtualizar;
    buffersParaAtualizar.push_back(balancingTargetID);
    if (enableSwapBuffer) buffersParaAtualizar.push_back(swapBufferID);

    for (int src = 0; src < todosDispositivos; ++src) {
        for (int dst = 0; dst < todosDispositivos; ++dst) {
            int intersecaoOffset, intersecaoLength;
            bool haIntersecao = ComputarIntersecao(offset[src], length[src], novosOffsets[dst], novosLengths[dst], &intersecaoOffset, &intersecaoLength);

            if (haIntersecao && intersecaoLength > 0) {
                int rankSrc = RecuperarPosicaoHistograma(dispositivosWorld, world_size, src);
                int rankDst = RecuperarPosicaoHistograma(dispositivosWorld, world_size, dst);
                size_t byteOffset = (size_t)intersecaoOffset * unitsPerElement * elementSize;
                size_t byteLength = (size_t)intersecaoLength * unitsPerElement * elementSize;

                for (int memObjID : buffersParaAtualizar) {
                    int idSrc = GetDeviceMemoryObjectID(memObjID, src);
                    int idDst = GetDeviceMemoryObjectID(memObjID, dst);

                    if (rankSrc == world_rank && rankDst == world_rank) {
                        if (src != dst) {
                            int localSrc = src - meusDispositivosOffset;
                            int localDst = dst - meusDispositivosOffset;
                            ReadFromMemoryObject(localSrc, idSrc, auxData, (int)byteOffset, (int)byteLength);
                            SynchronizeCommandQueue(localSrc);
                            WriteToMemoryObject(localDst, idDst, auxData, (int)byteOffset, (int)byteLength);
                            SynchronizeCommandQueue(localDst);
                        }
                    } else if (world_rank == rankSrc) {
                        int localSrc = src - meusDispositivosOffset;
                        ReadFromMemoryObject(localSrc, idSrc, auxData, (int)byteOffset, (int)byteLength);
                        SynchronizeCommandQueue(localSrc);
                        int tag = 5000 + (src * todosDispositivos + dst); 
                        MPI_Send(auxData, (int)byteLength, MPI_BYTE, rankDst, tag, MPI_COMM_WORLD);
                    } else if (world_rank == rankDst) {
                        int localDst = dst - meusDispositivosOffset;
                        int tag = 5000 + (src * todosDispositivos + dst);
                        MPI_Recv(auxData, (int)byteLength, MPI_BYTE, rankSrc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        WriteToMemoryObject(localDst, idDst, auxData, (int)byteOffset, (int)byteLength);
                        SynchronizeCommandQueue(localDst);
                    }
                }
            }
        }
    }

    memcpy(this->offset, novosOffsets, todosDispositivos * sizeof(int));
    memcpy(this->length, novosLengths, todosDispositivos * sizeof(int));
    memcpy(this->cargasAntigas, this->cargasNovas, todosDispositivos * sizeof(float));

    delete[] novosOffsets;
    delete[] novosLengths;
    delete[] auxData;

    MPI_Barrier(MPI_COMM_WORLD);
    
    // IMPORTANTE: Habilita captura para o próximo ciclo de medição
    captureStartEvent = true; 
}




// void OpenCLWrapper::ComputarCargas(const double *ticks, const float *cargasAntigas, float *cargasNovas, int participantes) {
//     if (participantes == 1) {
//         cargasNovas[0] = 1.0f;
//         return;
//     }

//     float cargaTotal = 0.0f;
//     for (int count = 0; count < participantes; count++) {
//         cargaTotal += ((count == 0) ? (cargasAntigas[count] - 0.0f) : (cargasAntigas[count] - cargasAntigas[count - 1])) * ((count == 0) ? 1.0f : static_cast<float>(ticks[0]) / static_cast<float>(ticks[count]));
//     }

//     for (int count = 0; count < participantes; count++) {
//         float cargaNova = (((count == 0) ? (cargasAntigas[count] - 0.0f) : (cargasAntigas[count] - cargasAntigas[count - 1])) * ((count == 0) ? 1.0f : static_cast<float>(ticks[0]) / static_cast<float>(ticks[count]))) / cargaTotal;
//         cargasNovas[count] = ((count == 0) ? cargaNova : cargasNovas[count - 1] + cargaNova);
//     }
// }


void OpenCLWrapper::ComputarCargas(const double *temposMedidos, const float *cargasAntigas, float *cargasNovas, int participantes) {
    if (participantes <= 1) {
        cargasNovas[0] = 1.0f;
        return;
    }

    // Usamos double para o cálculo intermediário para máxima precisão
    double *capacidade = new double[participantes];
    double capacidadeTotal = 0.0;

    for(int i=0; i < participantes; i++) {
        // Capacidade é o inverso do tempo (1/t)
        // Como já filtramos temposMedidos na função anterior (min 1e-9), não haverá divisão por zero.
        capacidade[i] = 1.0 / temposMedidos[i];
        capacidadeTotal += capacidade[i];
    }

    double cargaAcumulada = 0.0;
    for(int i=0; i < participantes; i++) {
        double fatia = capacidade[i] / capacidadeTotal;
        cargaAcumulada += fatia;
        
        // Atribui ao vetor de cargas (convertendo para float no final)
        cargasNovas[i] = (float)cargaAcumulada;
    }
    
    // Força fechamento exato para evitar erros de ponto flutuante (0.99999...)
    cargasNovas[participantes - 1] = 1.0f;

    delete[] capacidade;
}



/*
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
*/


int OpenCLWrapper::RecuperarPosicaoHistograma(int *histograma, int tamanho, int indice) {
    // Verifica se o índice está dentro do range total
    int total = 0;
    for (int i = 0; i < tamanho; i++) {
        total += histograma[i];
    }
    
    if (indice < 0 || indice >= total) {
        std::cerr << "Índice inválido: " << indice 
                  << " (total de dispositivos: " << total << ")" << std::endl;
        return -1;
    }

    // Encontra o processo responsável pelo dispositivo
    int offset = 0;
    for (int rank = 0; rank < tamanho; rank++) {
        if (indice < offset + histograma[rank]) {
            return rank;  // Retorna o RANK do processo dono
        }
        offset += histograma[rank];
    }

    std::cerr << "Erro inesperado: dispositivo não encontrado" << std::endl;
    return -1;
}


bool OpenCLWrapper::ComputarIntersecao(int offset1, int length1, int offset2, int length2, int *intersecaoOffset, int *intersecaoLength) {
    // Calcula o início e o fim de cada intervalo
    int end1 = offset1 + length1;
    int end2 = offset2 + length2;

    // O início da interseção é o maior dos dois inícios
    int start_intersecao = std::max(offset1, offset2);
    
    // O fim da interseção é o menor dos dois fins
    int end_intersecao = std::min(end1, end2);

    // O tamanho da interseção é a diferença
    int len_intersecao = end_intersecao - start_intersecao;

    // Se o tamanho for positivo, há uma interseção
    if (len_intersecao > 0) {
        *intersecaoOffset = start_intersecao;
        *intersecaoLength = len_intersecao;
        return true;
    }

    // Caso contrário, não há sobreposição
    *intersecaoOffset = 0;
    *intersecaoLength = 0;
    return false;
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


// double OpenCLWrapper::GetEventTaskTicks(int devicePosition, int startEventPosition, int endEventPosition)
// {
//     cl_ulong ticksStart = 0;
//     cl_ulong ticksEnd = 0;

//     // Obtém timestamps brutos (nanossegundos)
//     cl_int err1 = clGetEventProfilingInfo(devices[devicePosition].events[startEventPosition], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ticksStart, NULL);
//     cl_int err2 = clGetEventProfilingInfo(devices[devicePosition].events[endEventPosition], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ticksEnd, NULL);

//     if (err1 != CL_SUCCESS || err2 != CL_SUCCESS) {
//         // Se falhar em pegar o evento, retorna 0 para não quebrar o cálculo com lixo
//         return 0.0;
//     }

//     // Proteção contra Underflow (se start > end por erro de driver)
//     if (ticksEnd < ticksStart) return 0.000001; 

//     // Converte Nanossegundos -> Segundos
//     return (double)(ticksEnd - ticksStart) * 1.0e-9;
// }



double OpenCLWrapper::GetEventTaskTicks(cl_event startEvent, cl_event endEvent) {
    if (startEvent == NULL || endEvent == NULL) return 0.0;

    cl_ulong ticksStart = 0;
    cl_ulong ticksEnd = 0;

    cl_int err1 = clGetEventProfilingInfo(startEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ticksStart, NULL);
    cl_int err2 = clGetEventProfilingInfo(endEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ticksEnd, NULL);

    if (err1 != CL_SUCCESS || err2 != CL_SUCCESS) return 0.0;
    if (ticksEnd < ticksStart) return 1.0e-9; // Proteção

    return (double)(ticksEnd - ticksStart) * 1.0e-9; // Retorna Segundos
}


cl_device_type OpenCLWrapper::GetDeviceType() {
    return devices[deviceIndex].deviceType;
}

int OpenCLWrapper::GetDeviceMaxWorkItemsPerWorkGroup() {
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
    // --- DEBUG PRINT START ---
    // if (size < 0 || offset < 0) {
       // printf("[DEBUG WRITE] Rank %d | Device %d | MemObjID: %d | Offset: %d | Size: %d\n", 
              // world_rank, devicePosition, memoryObjectID, offset, size);
      //  fflush(stdout);
    // }
    // --- DEBUG PRINT END ---

    if (size == 0) {
        printf("[DEBUG WRITE] Ignorando escrita de tamanho 0.\n");
        return 0;
    }

    cl_int state;
    int memoryObjectPosition = GetMemoryObjectPosition(devicePosition, memoryObjectID);
    
    if (memoryObjectPosition != -1 && devices[devicePosition].numberOfEvents < maxEvents)
    {
        if (data == nullptr) {
             printf("[ERROR WRITE] Rank %d | Device %d: Host pointer 'data' is NULL!\n", world_rank, devicePosition);
             return -1;
        }

        state = clEnqueueWriteBuffer(devices[devicePosition].dataCommandQueue, 
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
            printf("[ERROR CL WRITE] Rank %d | Device %d | Error Code: %d\n", world_rank, devicePosition, state);
            printf("  -> Tentativa: MemObjPos: %d, Offset: %d, Size: %d\n", memoryObjectPosition, offset, size);
        }
        else
        {
            clFlush(devices[devicePosition].dataCommandQueue);
            devices[devicePosition].numberOfEvents += 1;
            return devices[devicePosition].numberOfEvents - 1;
        }
    } else {
        printf("[ERROR WRITE] Rank %d | Device %d: Couldn't find memory object position %d (MemObjID: %d) or events limit exceeded.\n", 
               world_rank, devicePosition, memoryObjectPosition, memoryObjectID);
    }
    return -1;
}

int OpenCLWrapper::ReadFromMemoryObject(int devicePosition, int memoryObjectID, char *data, int offset, int size)
{
    // --- DEBUG PRINT START ---
    // Filtra prints para não poluir demais, ou remova o if para ver tudo
    // if (size < 0 || offset < 0) { 
      //  printf("[DEBUG READ] Rank %d | Device %d | MemObjID: %d | Offset: %d | Size: %d\n", 
      //         world_rank, devicePosition, memoryObjectID, offset, size);
      //  fflush(stdout);
    // }
    // --- DEBUG PRINT END ---

    if (size == 0) {
        printf("[DEBUG READ] Ignorando leitura de tamanho 0.\n");
        return 0; // Evita erro -30 se a implementação não gostar de size 0
    }

    cl_int state;
    int memoryObjectPosition = GetMemoryObjectPosition(devicePosition, memoryObjectID);
    
    if (memoryObjectPosition != -1 && devices[devicePosition].numberOfEvents < maxEvents)
    { 
        // Verificação extra de sanidade do ponteiro
        if (data == nullptr) {
             printf("[ERROR READ] Rank %d | Device %d: Host pointer 'data' is NULL!\n", world_rank, devicePosition);
             return -1;
        }

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
            printf("[ERROR CL READ] Rank %d | Device %d | Error Code: %d\n", world_rank, devicePosition, state);
            printf("  -> Tentativa: MemObjPos: %d, Offset: %d, Size: %d\n", memoryObjectPosition, offset, size);
            return -1;
        }
        else
        {
            clFinish(devices[devicePosition].dataCommandQueue); 
            devices[devicePosition].numberOfEvents += 1;
            return devices[devicePosition].numberOfEvents - 1;
        }
    }

    printf("[ERROR READ] Rank %d | Device %d: Couldn't find memory object position %d (MemObjID: %d) or events limit (%d/%d) reached.\n", 
           world_rank, devicePosition, memoryObjectPosition, memoryObjectID, devices[devicePosition].numberOfEvents, maxEvents);
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


int OpenCLWrapper::AllocateMemoryObject(int _size, cl_mem_flags _flags, void* _host_ptr) {
    int globalMemObjID = 0;

    // Rank 0 gera o ID e incrementa
    if (world_rank == 0) {
        globalMemObjID = globalMemoryObjectIDCounter++;
    }

    // Broadcast para todos os processos
    MPI_Bcast(&globalMemObjID, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Todos os ranks devem alocar seus dispositivos locais
    memoryObjectIDs->emplace(globalMemObjID, std::vector<int>(todosDispositivos, -1));

    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            int deviceMemObjID = CreateMemoryObject(count - meusDispositivosOffset, _size, _flags, _host_ptr);
            (*memoryObjectIDs)[globalMemObjID][count] = deviceMemObjID;
            
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
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
        int memoryObjectID = GetDeviceMemoryObjectID(globalMemoryObjectID, count);
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



int OpenCLWrapper::WriteObject(int GlobalObjectID, const char *data, int offset, int size) {
    bool success = true;

    // Parâmetros da escrita solicitada (já em bytes)
    long long global_write_offset_bytes = offset;
    long long global_write_size_bytes = size;

    // Bytes por elemento (posição)
    int elemBytes = elementSize * unitsPerElement;

    // Itera sobre todos os dispositivos
    for (int count = 0; count < todosDispositivos; count++) {
        
        // Apenas o rank que gerencia o dispositivo 'count' executa a lógica de escrita
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            
            // Converte a partição do dispositivo (que está em elementos) para BYTES
            long long device_offset_bytes = (long long)this->offset[count] * elemBytes;
            long long device_len_bytes = (long long)this->length[count] * elemBytes;
            long long device_end_bytes = device_offset_bytes + device_len_bytes;

            // Calcula a interseção em BYTES entre a escrita solicitada e a partição deste dispositivo
            long long start_write_pos_bytes = std::max(global_write_offset_bytes, device_offset_bytes);
            long long end_write_pos_bytes = std::min(global_write_offset_bytes + global_write_size_bytes, device_end_bytes);

            long long len_to_write_bytes = end_write_pos_bytes - start_write_pos_bytes;

            // Se há uma porção a ser escrita neste dispositivo
            if (len_to_write_bytes > 0) {
                int local_device_idx = count - meusDispositivosOffset;
                int memoryObjectID = GetDeviceMemoryObjectID(GlobalObjectID, count);

                if (memoryObjectID != -1) {
                    // Offset de LEITURA no buffer do HOST 'data'.
                    // Onde, dentro do buffer 'data', estão os bytes que precisamos enviar.
                    long long host_read_offset_bytes = start_write_pos_bytes - global_write_offset_bytes;

                    // --- CORREÇÃO PRINCIPAL AQUI ---
                    // Offset de ESCRITA no buffer do DISPOSITIVO.
                    // Seguindo o modelo da GatherResults, usamos o offset GLOBAL absoluto em bytes.
                    long long device_write_offset_bytes = start_write_pos_bytes;

                    // Executa a escrita com os parâmetros corretos
                    int result = WriteToMemoryObject(
                        local_device_idx,
                        memoryObjectID,
                        data + host_read_offset_bytes, // Ponteiro para o início dos dados corretos no host
                        device_write_offset_bytes,     // Onde escrever no buffer do dispositivo (offset global)
                        len_to_write_bytes             // Quantos bytes escrever
                    );
                    
                    if (result == -1) {
                        success = false;
                    }
                } else {
                    fprintf(stderr, "[Rank %d] Erro: ID de objeto de memória inválido para dispositivo global %d.\n", world_rank, count);
                    success = false;
                }
            }
        }
    }

    // Sincronização final
    for (int i = 0; i < meusDispositivosLength; i++) {
        SynchronizeCommandQueue(i);
    }

    return success ? 0 : -1;
}

void OpenCLWrapper::setBalancingTargetID(int targetID)
{

balancingTargetID = targetID;


}


void OpenCLWrapper::setSubdomainBoundary(int _sdSize, int _nArgs, int* _args) {
    sdSize = _sdSize; // Tamanho da borda
    nArgs = _nArgs;
    args = new int[nArgs]; // Inicializando corretamente o array args
    for (int i = 0; i < nArgs; i++) {
        args[i] = _args[i]; // Copiando os valores do array _args
    }
    sdSet = true; // Borda definida
}


void OpenCLWrapper::Comms() {
    size_t tamanhoBorda = sdSize;
    size_t bytes_borda = tamanhoBorda * elementSize * unitsPerElement;

    char *sendBuff = new char[bytes_borda];
    char *recBuff = new char[bytes_borda];
    
    const int TAG_D1_PARA_D2 = 201; // Tag para comunicação ->
    const int TAG_D2_PARA_D1 = 202; // Tag para comunicação <-

    // Itera por todas as fronteiras entre dispositivos adjacentes
    for (int d1_idx = 0; d1_idx < todosDispositivos - 1; ++d1_idx) {
        int d2_idx = d1_idx + 1;

        int d1_rank = RecuperarPosicaoHistograma(dispositivosWorld, world_size, d1_idx);
        int d2_rank = RecuperarPosicaoHistograma(dispositivosWorld, world_size, d2_idx);

        // Offsets de DADOS A SEREM ENVIADOS (as bordas)
        size_t offset_borda_d1 = (offset[d1_idx] + length[d1_idx] - tamanhoBorda) * unitsPerElement * elementSize;
        size_t offset_borda_d2 = offset[d2_idx] * unitsPerElement * elementSize;

        // Offsets de ONDE ESCREVER OS DADOS RECEBIDOS (os halos)
        size_t offset_halo_d1 = (offset[d1_idx] + length[d1_idx]) * unitsPerElement * elementSize;
        size_t offset_halo_d2 = (offset[d2_idx] - tamanhoBorda) * unitsPerElement * elementSize;

        int ID_Device_d1 = GetDeviceMemoryObjectID(balancingTargetID, d1_idx);
        int ID_Device_d2 = GetDeviceMemoryObjectID(balancingTargetID, d2_idx);
        if (enableSwapBuffer){
        ID_Device_d1 = GetDeviceMemoryObjectID(swapBufferID, d1_idx);
        ID_Device_d2 = GetDeviceMemoryObjectID(swapBufferID, d2_idx);

        }

        // CASO 1: Comunicação INTER-PROCESSO (entre ranks diferentes)
        if (d1_rank != d2_rank) {
            MPI_Request req_s, req_r;
            // Lógica para o Rank que controla d1
            if (world_rank == d1_rank) {
                int d1_local_idx = d1_idx - meusDispositivosOffset;
                ReadFromMemoryObject(d1_local_idx, ID_Device_d1, sendBuff, offset_borda_d1, bytes_borda);
                SynchronizeCommandQueue(d1_local_idx);
                MPI_Isend(sendBuff, bytes_borda, MPI_BYTE, d2_rank, TAG_D1_PARA_D2, MPI_COMM_WORLD, &req_s);
                MPI_Irecv(recBuff, bytes_borda, MPI_BYTE, d2_rank, TAG_D2_PARA_D1, MPI_COMM_WORLD, &req_r);
                MPI_Wait(&req_s, MPI_STATUS_IGNORE);
                MPI_Wait(&req_r, MPI_STATUS_IGNORE);
                WriteToMemoryObject(d1_local_idx, ID_Device_d1, recBuff, offset_halo_d1, bytes_borda);
                SynchronizeCommandQueue(d1_local_idx);
            }
            // Lógica para o Rank que controla d2
            else if (world_rank == d2_rank) {
                int d2_local_idx = d2_idx - meusDispositivosOffset;
                ReadFromMemoryObject(d2_local_idx, ID_Device_d2, sendBuff, offset_borda_d2, bytes_borda);
                SynchronizeCommandQueue(d2_local_idx);
                MPI_Isend(sendBuff, bytes_borda, MPI_BYTE, d1_rank, TAG_D2_PARA_D1, MPI_COMM_WORLD, &req_s);
                MPI_Irecv(recBuff, bytes_borda, MPI_BYTE, d1_rank, TAG_D1_PARA_D2, MPI_COMM_WORLD, &req_r);
                MPI_Wait(&req_s, MPI_STATUS_IGNORE);
                MPI_Wait(&req_r, MPI_STATUS_IGNORE);
                WriteToMemoryObject(d2_local_idx, ID_Device_d2, recBuff, offset_halo_d2, bytes_borda);
                SynchronizeCommandQueue(d2_local_idx);
            }
        }
        // CASO 2: Comunicação INTRA-PROCESSO (dispositivos no mesmo rank)
        else if (world_rank == d1_rank) {
            int d1_local_idx = d1_idx - meusDispositivosOffset;
            int d2_local_idx = d2_idx - meusDispositivosOffset;
            ReadFromMemoryObject(d1_local_idx, ID_Device_d1, sendBuff, offset_borda_d1, bytes_borda);
            ReadFromMemoryObject(d2_local_idx, ID_Device_d2, recBuff, offset_borda_d2, bytes_borda);
            SynchronizeCommandQueue(d1_local_idx);
            SynchronizeCommandQueue(d2_local_idx);
            WriteToMemoryObject(d1_local_idx, ID_Device_d1, recBuff, offset_halo_d1, bytes_borda);
            WriteToMemoryObject(d2_local_idx, ID_Device_d2, sendBuff, offset_halo_d2, bytes_borda);
            SynchronizeCommandQueue(d1_local_idx);
            SynchronizeCommandQueue(d2_local_idx);
        }
    }
    delete[] sendBuff;
    delete[] recBuff;
}


        

void OpenCLWrapper::setSwapBufferID(int swapID) {


    swapBufferID = swapID;
    enableSwapBuffer = true;


}




void OpenCLWrapper::CollectOverheadsPerDevice(int deviceID, double &lat, double &ban, double &rd, double &wr)
{
    size_t totalBytes = (size_t)nElements * unitsPerElement * elementSize;
    char *benchBuf = new char[totalBytes];
    char *recvBuf  = new char[totalBytes];
    char dummySend = 0, dummyRecv = 0;

    // 1) LATÊNCIA: ping de 1 byte
    lat = 0.0;
    for (int p = 0; p < precision; ++p) {
        int alvo = RecuperarPosicaoHistograma(dispositivosWorld, todosDispositivos, deviceID);
        double t0 = MPI_Wtime();
        MPI_Sendrecv(
            &dummySend, 1, MPI_CHAR, alvo, 0,
            &dummyRecv, 1, MPI_CHAR, alvo, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        lat += (MPI_Wtime() - t0);
    }
    lat /= precision;

    // 2) BANDA: troca completa de benchBuf <-> recvBuf
    ban = 0.0;
    for (int p = 0; p < precision; ++p) {
        int alvo = RecuperarPosicaoHistograma(dispositivosWorld, todosDispositivos, deviceID);
        double t0 = MPI_Wtime();
        MPI_Sendrecv(
            benchBuf, (int)totalBytes, MPI_CHAR, alvo, 0,
            recvBuf,  (int)totalBytes, MPI_CHAR, alvo, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        ban += (MPI_Wtime() - t0) / (double)totalBytes;
    }
    ban /= precision;

    // 3) READ BUFFER (device → host)
    {
        int localIdx = deviceID - meusDispositivosOffset;
        int memObj   = GetDeviceMemoryObjectID(balancingTargetID, deviceID);
        rd = 0.0;
        for (int p = 0; p < precision; ++p) {
            double t0 = MPI_Wtime();
            ReadFromMemoryObject(localIdx, memObj, benchBuf, 0, totalBytes);
            SynchronizeCommandQueue(localIdx);
            rd += (MPI_Wtime() - t0) / (double)totalBytes;
        }
        rd /= precision;
    }

    // 4) WRITE BUFFER (host → device)
    {
        int localIdx = deviceID - meusDispositivosOffset;
        int memObj   = GetDeviceMemoryObjectID(balancingTargetID, deviceID);
        wr = 0.0;
        for (int p = 0; p < precision; ++p) {
            double t0 = MPI_Wtime();
            WriteToMemoryObject(localIdx, memObj, benchBuf, 0, totalBytes);
            SynchronizeCommandQueue(localIdx);
            wr += (MPI_Wtime() - t0) / (double)totalBytes;
        }
        wr /= precision;
    }

    delete[] benchBuf;
    delete[] recvBuf;
}




void OpenCLWrapper::CollectOverheads() {
    // 1. Cada rank encontra o overhead máximo APENAS entre seus dispositivos locais
    double max_local_lat = 0.0;
    double max_local_ban = 0.0;
    double max_local_rd = 0.0;
    double max_local_wr = 0.0;

    for (int i = 0; i < meusDispositivosLength; ++i) {
        int deviceID = meusDispositivosOffset + i;
        double current_lat, current_ban, current_rd, current_wr;
        
        CollectOverheadsPerDevice(deviceID, current_lat, current_ban, current_rd, current_wr);

        max_local_lat = std::max(max_local_lat, current_lat);
        max_local_ban = std::max(max_local_ban, current_ban);
        max_local_rd = std::max(max_local_rd, current_rd);
        max_local_wr = std::max(max_local_wr, current_wr);
    }

    // 2. Todos os ranks usam Allreduce para encontrar o máximo global (o pior caso do sistema)
    MPI_Allreduce(&max_local_lat, &latencia, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_local_ban, &banda, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_local_rd, &readByte, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_local_wr, &writeByte, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Opcional: imprimir os valores globais finais apenas no rank 0
    if (world_rank == 0) {
        std::cout << "--- Overheads Globais (Pior Caso) ---" << std::endl;
        std::cout << "Latencia (s): " << latencia << std::endl;
        std::cout << "Banda (s/byte): " << banda << std::endl;
        std::cout << "Read (s/byte): " << readByte << std::endl;
        std::cout << "Write (s/byte): " << writeByte << std::endl;
        std::cout << "------------------------------------" << std::endl;
    }
}
// NOVA SetKernelAttribute sobrecarregada para valores
void OpenCLWrapper::SetKernelAttribute(int devicePosition, int kernelID, int attribute, void* data, size_t size)
{
    int kernelPosition = GetKernelPosition(devicePosition, kernelID);
    if (kernelPosition != -1) {
        // Define o argumento do kernel passando o tamanho e um ponteiro para o valor
        cl_int state = clSetKernelArg(devices[devicePosition].kernels[kernelPosition], attribute, size, data);
        if (state != CL_SUCCESS) {
            printf("Error setting kernel argument by value! Error code: %d\n", state);
        }
    } else {
        printf("Error setting kernel argument: Kernel ID=%i does not exist!\n", kernelID);
    }
}

// NOVA setAttribute para valores (int, float, etc.)
void OpenCLWrapper::setAttribute(int attribute, void* data, size_t size) {
    // Itera sobre todos os dispositivos gerenciados por este processo
    for (int count = meusDispositivosOffset; count < meusDispositivosOffset + meusDispositivosLength; count++) {
        int local_idx = count - meusDispositivosOffset;
        
        // Chama a função de baixo nível para definir o argumento do kernel
        SetKernelAttribute(local_idx, kernelDispositivo[count], attribute, data, size);
    }

    // Sincroniza para garantir que os argumentos foram definidos
    for (int i = 0; i < meusDispositivosLength; i++) {
        SynchronizeCommandQueue(i);
    }
}