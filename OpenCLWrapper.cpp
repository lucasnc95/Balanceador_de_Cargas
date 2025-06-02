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
            
                kernelEventoDispositivo[deviceIndex2] = RunKernel(deviceIndex2, kernelDispositivo[deviceIndex2], offset[deviceIndex2], length[deviceIndex2], isDeviceCPU(deviceIndex2)? 8 : 64);
              
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

					kernelEventoDispositivo[count] = RunKernel(count-meusDispositivosOffset, kernelDispositivo[count], offset[count]+(sdSize), length[count]-(sdSize)-1, isDeviceCPU(count-meusDispositivosOffset) ? 8 :  64);
                    
				}
			}
                
                Comms();

            //Sincronizacao da comunicacao.
			for(int count = 0; count < todosDispositivos; count++)
			{
				if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
				{
				    long tickEvent = GetEventTaskTicks(count - meusDispositivosOffset, kernelEventoDispositivo[count]);          
                    ticks[count] += tickEvent;
                    SynchronizeCommandQueue(count-meusDispositivosOffset);

				}
			}

		
			
			// Computação das bordas.
for (int count = 0; count < todosDispositivos; count++) {
    if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
        RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count], sdSize, isDeviceCPU(count - meusDispositivosOffset) ? 8 : 64);
        RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count] + length[count] - (sdSize), sdSize, isDeviceCPU(count - meusDispositivosOffset) ? 8 : 64);
         
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
    tempos = new double[todosDispositivos];    
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
    double localLat = 0, localBan = 0, localW = 0;
    char *auxData = new char[nElements*unitsPerElement*elementSize];
    int somaLengthAntes = 0;
    
    CollectOverheads();
   

    PrecisaoBalanceamento();

    //double localBanda,tempoInicioProbing, localLatencia, localwriteByte1, localwriteByte2;
   
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
                         
						    ReadFromMemoryObject(count2 - meusDispositivosOffset, dataDevice[1], malha + (intersecaoOffset * unitsPerElement), intersecaoOffset * unitsPerElement * elementSize, intersecaoLength * unitsPerElement * elementSize);
                            SynchronizeCommandQueue(count2 - meusDispositivosOffset);

                            WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], malha + (intersecaoOffset * unitsPerElement), intersecaoOffset * unitsPerElement * elementSize, intersecaoLength * unitsPerElement * elementSize);
                            SynchronizeCommandQueue(count - meusDispositivosOffset);

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
 

                                MPI_Recv(malha + (overlap[0] * unitsPerElement), overlap[1] * unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
								
                                WriteToMemoryObject(count - meusDispositivosOffset, dataDevice, malha + (overlap[0] * unitsPerElement), overlap[0] * unitsPerElement * elementSize, overlap[1] * unitsPerElement * elementSize);
                                SynchronizeCommandQueue(count - meusDispositivosOffset);
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
	

 
	
    // double tempoFimProbing = MPI_Wtime();
    // tempoBalanceamento += tempoFimProbing - tempoInicioProbing;
    // fatorErro = tempoBalanceamento;
    delete[] auxData;
   
}


// void OpenCLWrapper::PrecisaoBalanceamento() {
  
  
//   	memset(ticks, 0, sizeof(long int) * todosDispositivos);
// 	memset(tempos, 0, sizeof(float) * todosDispositivos);

// 	for (int precisao = 0; precisao < precision; precisao++)
// 	{
		
// 		// Computação.
// 		for (int count = 0; count < todosDispositivos; count++)
// 		{
			
// 			if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
// 			{
				
// 				kernelEventoDispositivo[count] = RunKernel(count - meusDispositivosOffset, kernelDispositivo[count- meusDispositivosOffset], offset[count - meusDispositivosOffset], length[count- meusDispositivosOffset], isDeviceCPU(count - meusDispositivosOffset)? 8 : 256);
// 			}
// 		}
	

	
// 	// // Ticks.
// 	for (int count = 0; count < todosDispositivos; count++)
// 	{	
// 		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
// 		{	
// 			SynchronizeCommandQueue(count - meusDispositivosOffset);
			
//             long tickEvent = GetEventTaskTicks(count - meusDispositivosOffset, kernelEventoDispositivo[count]);          
//             ticks[count] += tickEvent;
			
// 		}
// 	}
	
// }	
// 	// Reduzir ticks.
	
// 	long int ticks_root[todosDispositivos];
// 	MPI_Allreduce(ticks, ticks_root, todosDispositivos, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
// 	memcpy(ticks, ticks_root, sizeof(long int) * todosDispositivos);
// 	ComputarCargas(ticks, cargasAntigas, cargasNovas, todosDispositivos);
// 	for (int count = 0; count < todosDispositivos; count++)
// 	{
// 		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
// 	 	{
// 	 		SynchronizeCommandQueue(count - meusDispositivosOffset);
// 			if(count == 0)
// 	 		tempos[count] = ((float)ticks[count]);
//             else
// 			tempos[count] = ((float)ticks[count]);
// 	 	}
// 	}
// 	float tempos_root[todosDispositivos];
// 	MPI_Allreduce(tempos, tempos_root, todosDispositivos, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
// 	memcpy(tempos, tempos_root, sizeof(float) * todosDispositivos);
  


// }



void OpenCLWrapper::PrecisaoBalanceamento() {
    // 1) Zera o array de ticks
    memset(ticks, 0, sizeof(long int) * todosDispositivos);

    // 2) Executa 'precision' rodadas de medição
    for (int iter = 0; iter < precision; ++iter) {
        // Dispara todos os kernels nos nossos dispositivos locais
        for (int count = 0; count < todosDispositivos; ++count) {
            if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
                int localIdx = count - meusDispositivosOffset;
                kernelEventoDispositivo[localIdx] = RunKernel(
                    localIdx,
                    kernelDispositivo[count],
                    offset[count],
                    length[count],
                    isDeviceCPU(localIdx) ? 8 : 256
                );
            }
        }
        // Sincroniza e acumula ticks retornados pelo profiling
        for (int count = 0; count < todosDispositivos; ++count) {
            if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
                int localIdx = count - meusDispositivosOffset;
                SynchronizeCommandQueue(localIdx);
                long tickEvent = GetEventTaskTicks(localIdx, kernelEventoDispositivo[localIdx]);
                ticks[count] += tickEvent;
            }
        }
    }

    // 3) Reduz (MPI_Allreduce) os ticks entre todos os ranks
    long *ticksRoot = (long*) malloc(sizeof(long) * todosDispositivos);
    MPI_Allreduce(ticks, ticksRoot, todosDispositivos, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    // Copia de volta para ticks[]
    for (int i = 0; i < todosDispositivos; ++i) {
        ticks[i] = ticksRoot[i];
    }
    free(ticksRoot);

    // 4) Recalcula as cargas com base nos ticks totais
    ComputarCargas(ticks, cargasAntigas, cargasNovas, todosDispositivos);

    // 5) Converte ticks (ns) → tempo em segundos e média por iteração
    //    tempos[] já existe como float[todosDispositivos]
    for (int count = 0; count < todosDispositivos; ++count) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            tempos[count] = (double)ticks[count] / (1e9 * precision);
        } else {
            tempos[count] = 0.0f;
        }
    }

    // 6) (Opcional) reduz tempos[] entre ranks para ter soma global
    float *temposRoot = (float*) malloc(sizeof(float) * todosDispositivos);
    MPI_Allreduce(tempos, temposRoot, todosDispositivos, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    for (int i = 0; i < todosDispositivos; ++i) {
        tempos[i] = temposRoot[i];
    }
    free(temposRoot);
}




void OpenCLWrapper::LoadBalancing()
{   tempoCB = 0.0f;
	double tempoInicioBalanceamento = MPI_Wtime();
    // double tempoComputacaoInterna = tempos[0];
	PrecisaoBalanceamento();
    char *auxData = new char[nElements * unitsPerElement * elementSize];
	// Computar novas cargas.
	// double localTempoCB;
	// for (int count = 0; count < todosDispositivos; count++)
	// {
	// 	if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
	// 	{
	// 		SynchronizeCommandQueue(count - meusDispositivosOffset);
	// 		if (tempoComputacaoInterna < tempos[count] )
	// 			tempoComputacaoInterna = tempos[count];
	// 		if(count == 0)
	// 			localTempoCB = cargasNovas[count] * (tempos[count]);

	// 		else
	// 			localTempoCB = (cargasNovas[count] - cargasNovas[count - 1]) * (tempos[count]);
	// 	}
	// }
	// MPI_Allreduce(&tempoCB, &localTempoCB, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	// tempoCB *= nElements*elementSize*unitsPerElement;
	// std::cout<<"TempoCB: "<<tempoCB<<std::endl;
	// std::cout<<"writeByte: "<<writeByte<<std::endl;
    // std::cout<<"readByte: "<<readByte<<std::endl;
    // std::cout<<"banda: "<<banda<<std::endl;
    // std::cout<<"latência: "<<latencia<<std::endl;
	// std::cout<<"tempo calculado: "<<((latencia) + ComputarNorma(cargasAntigas, cargasNovas, todosDispositivos) * nElements*elementSize*unitsPerElement * ((writeByte) + (banda)) + (tempoCB))<<std::endl;
	// std::cout<<"Tempo anterior: "<<tempoComputacaoInterna<<std::endl;
	// if ((latencia + ComputarNorma(cargasAntigas, cargasNovas, todosDispositivos) * (writeByte + banda) + tempoCB) < tempoComputacaoInterna)
    
        double tempoComputacaoInterna = 0.0;
    for (int i = 0; i < todosDispositivos; ++i) {
        tempoComputacaoInterna = std::max(tempoComputacaoInterna, tempos[i]);
    }

    // 2) compute the predicted balanced compute time:
    //    each device will take a fraction of the work
    double localTempoCB = 0.0;
    for (int i = 0; i < todosDispositivos; ++i) {
        float frac = (i == 0 ? cargasNovas[0]
                            : cargasNovas[i] - cargasNovas[i-1]);
        localTempoCB = std::max(localTempoCB, frac * tempoComputacaoInterna);
    }

    // 3) global worst‐case across MPI ranks
    double globalTempoCB = 0.0;
    MPI_Allreduce(&localTempoCB, &globalTempoCB, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // 4) store it so future decisions see it
    tempoCB = globalTempoCB;  // in seconds

    // 5) compute the rest of your cost model (all in seconds)
    double totalBytes = double(nElements) * unitsPerElement * elementSize;
    double overheadComm = latencia + ComputarNorma(cargasAntigas, cargasNovas, todosDispositivos)* totalBytes * (writeByte + banda + readByte);
    // 6) print a sane table
    std::cout << "=== LoadBalancing ===\n";
    std::cout << "Tempo computacao interna (s): " << tempoComputacaoInterna << "\n";
    std::cout << "Latency (s): " << latencia << "\n";
    std::cout << "Read (s/byte): " << readByte << "\n";
    std::cout << "Write (s/byte): " << writeByte << "\n";
    std::cout << "Bandwidth (s/byte): " << banda << "\n";
    std::cout << "Total bytes: " << totalBytes << "\n";
    std::cout << "OverheadComm (s): " << overheadComm << "\n";
    std::cout << "Norma: " << ComputarNorma(cargasAntigas, cargasNovas, todosDispositivos) << "\n";
    std::cout << "TempoCB (s): " << tempoCB << "\n";
    std::cout << "Custo previsto (s): " << overheadComm << "\n";

    // 7) compare and rebalance
    if (overheadComm + tempoCB < tempoComputacaoInterna) 
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
int *malhaDevice = new int[2];
int *borda = new int[4];
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

									dataEventoDispositivo[count] = ReadFromMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[1]*unitsPerElement*elementSize)), borda[1]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);
									MPI_Isend(malha+(borda[1]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, &sendRequest);
									MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
									MPI_Wait(&receiveRequest, MPI_STATUS_IGNORE);
									
									WriteToMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[0]*unitsPerElement*elementSize)), borda[0]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
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
									dataEventoDispositivo[count] = ReadFromMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[1]*unitsPerElement*elementSize)), borda[1]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);
									MPI_Isend(malha+(borda[1]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, &sendRequest);
									MPI_Irecv(malha+(borda[0]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, &receiveRequest);
									MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
									MPI_Wait(&receiveRequest, MPI_STATUS_IGNORE);
									WriteToMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[0]*unitsPerElement*elementSize)), borda[0]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);

								}
							}
						}

						//Entre processos diferentes, no terceiro passo.
						if(passo == 2)
						{
							if(count == meusDispositivosOffset && count > 0)
							{
								malhaDevice[0] = GetDeviceMemoryObjectID(balancingTargetID, count);
								borda[0] = offset[count]-(tamanhoBorda);
								borda[0] = (borda[0] < 0) ? 0 : borda[0];
								borda[1] = offset[count];
								alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count-1);

								if(alvo%2 == 1)
								{
									MPI_Irecv(malha+(borda[0]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR , alvo, 0, MPI_COMM_WORLD, &receiveRequest);

									dataEventoDispositivo[count] = ReadFromMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[1]*unitsPerElement*elementSize)), borda[1]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);
									MPI_Isend(malha+(borda[1]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, &sendRequest);
									MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
									MPI_Wait(&receiveRequest, MPI_STATUS_IGNORE);

									WriteToMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[0]*unitsPerElement*elementSize)), borda[0]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
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
									dataEventoDispositivo[count] = ReadFromMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[1]*unitsPerElement*elementSize)), borda[1]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);
									MPI_Isend(malha+(borda[1]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, &sendRequest);
									MPI_Irecv(malha+(borda[0]*unitsPerElement), tamanhoBorda*unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, &receiveRequest);
									MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
									MPI_Wait(&receiveRequest, MPI_STATUS_IGNORE);

									WriteToMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[0]*unitsPerElement*elementSize)), borda[0]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
									SynchronizeCommandQueue(count-meusDispositivosOffset);

								}
							}
						}

						//No mesmo processo, no primeiro passo.
						if(passo == 0 && count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength-1)
						{   if(!enableSwapBuffer)
                            {	
                            malhaDevice[0] = GetDeviceMemoryObjectID(balancingTargetID, count);
							malhaDevice[1] = GetDeviceMemoryObjectID(balancingTargetID, count+1);
                            }
                            else
                            {
                            malhaDevice[0] = GetDeviceMemoryObjectID(swapBufferID, count);
							malhaDevice[1] = GetDeviceMemoryObjectID(swapBufferID, count+1);
                            }
							borda[0] = int(offset[count+1]-(tamanhoBorda));
							borda[0] = (borda[0] < 0) ? 0 : borda[0];
							borda[1] = int(offset[count+1]);
                           
							dataEventoDispositivo[count+0] = ReadFromMemoryObject(count-meusDispositivosOffset, malhaDevice[0], (malha+(borda[0]*unitsPerElement*elementSize)), borda[0]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
							SynchronizeCommandQueue(count+0-meusDispositivosOffset);

							dataEventoDispositivo[count+1] = ReadFromMemoryObject(count+1-meusDispositivosOffset, malhaDevice[1], (malha+(borda[1]*unitsPerElement*elementSize)), borda[1]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
							SynchronizeCommandQueue(count+1-meusDispositivosOffset);

 
						}

						//No mesmo processo, no segundo passo.
						if(passo == 1 && count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength-1)
						{
                        if(!enableSwapBuffer)
                            {	
                            malhaDevice[0] = GetDeviceMemoryObjectID(balancingTargetID, count);
							malhaDevice[1] = GetDeviceMemoryObjectID(balancingTargetID, count+1);
                            }
                            else
                            {
                            malhaDevice[0] = GetDeviceMemoryObjectID(swapBufferID, count);
							malhaDevice[1] = GetDeviceMemoryObjectID(swapBufferID, count+1);
                            }
							borda[0] = offset[count+1]-(tamanhoBorda);
							borda[0] = (borda[0] < 0) ? 0 : borda[0];
							borda[1] = offset[count+1];
                         

							WriteToMemoryObject(count+0-meusDispositivosOffset, malhaDevice[0], (malha+(borda[1]*unitsPerElement*elementSize)), borda[1]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
							SynchronizeCommandQueue(count+0-meusDispositivosOffset);

							WriteToMemoryObject(count+1-meusDispositivosOffset, malhaDevice[1], (malha+(borda[0]*unitsPerElement*elementSize)), borda[0]*unitsPerElement*elementSize, tamanhoBorda*unitsPerElement*elementSize);
							SynchronizeCommandQueue(count+1-meusDispositivosOffset);

    
						}
					}
				}
			}

delete[] malha;
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




// 2) Para cada dispositivo local mede e depois faz MPI_Allreduce para todos
void OpenCLWrapper::CollectOverheads() {
    // aloca arrays locais e globais
double *localLat = new double[meusDispositivosLength];
double  *localBan = new double[meusDispositivosLength];
double  *localRd  = new double[meusDispositivosLength];
double   *localWr  = new double[meusDispositivosLength];

globLat = new double[meusDispositivosLength];
globBan = new double[meusDispositivosLength];
globRd  = new double[meusDispositivosLength];
globWr  = new double[meusDispositivosLength];

    // 1) cada rank mede só seus dispositivos
    for (int i = 0; i < meusDispositivosLength; ++i) {
        int deviceID = meusDispositivosOffset + i;
        CollectOverheadsPerDevice(deviceID, localLat[i], localBan[i], localRd [i], localWr [i]);
    }

    // 2) faz Allreduce máxima sobre cada elemento do array local
    MPI_Allreduce(localLat, globLat, meusDispositivosLength,MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(localBan, globBan, meusDispositivosLength,MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(localRd,  globRd,  meusDispositivosLength,MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(localWr,  globWr,  meusDispositivosLength,MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        for (int i = 0; i < meusDispositivosLength; ++i) {
        latencia  = std::max(latencia, globLat[i]);
        banda     = std::max(banda,    globBan[i]);
        readByte  = std::max(readByte, globRd [i]);
        writeByte = std::max(writeByte,globWr [i]);
    }

    std::cout<<"latencia: "<<latencia<<std::endl;
    std::cout<<"banda: "<<banda<<std::endl;
    std::cout<<"readByte: "<<readByte<<std::endl;
    std::cout<<"writeByte: "<<writeByte<<std::endl;

    // libera arrays temporários
    delete[] localLat; delete[] localBan;
    delete[] localRd;  delete[] localWr;
}

