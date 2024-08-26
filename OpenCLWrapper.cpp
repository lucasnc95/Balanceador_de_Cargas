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

    // Alocação de memória para IDs de plataforma
    platformIDs = new cl_platform_id[maxNumberOfPlatforms];
    if (!platformIDs) {
        printf("Memory allocation failed for platformIDs.\n");
        return -1;
    }

    // Obtendo as plataformas disponíveis
    cl_uint numberOfPlatforms = 0;
    state = clGetPlatformIDs(maxNumberOfPlatforms, platformIDs, &numberOfPlatforms);
    if (state != CL_SUCCESS || numberOfPlatforms == 0) {
        printf("OpenCL Error: Platform couldn't be found.\n");
        return -1;
    }
    printf("%u platform(s) found.\n", numberOfPlatforms);

    // Alocação de memória para os dispositivos
    devices = new Device[maxNumberOfDevices];
    if (!devices) {
        printf("Memory allocation failed for devices.\n");
        return -1;
    }

    numberOfDevices = 0;

    for (cl_uint i = 0; i < numberOfPlatforms; i++) {
        cl_uint numberOfDevicesOfPlatform = 0;
        cl_device_id deviceList[maxNumberOfDevices];

        // Obtendo os dispositivos da plataforma atual
        state = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL, maxNumberOfDevices, deviceList, &numberOfDevicesOfPlatform);
        if (state != CL_SUCCESS || numberOfDevicesOfPlatform == 0) {
            printf("OpenCL Error: Devices couldn't be resolved on platform %u.\n", i);
            continue;
        }

        for (cl_uint j = 0; j < numberOfDevicesOfPlatform; j++) {
            devices[numberOfDevices].deviceID = deviceList[j];

            // Criando um contexto para cada dispositivo
            cl_context_properties contextProperties[] = {
                CL_CONTEXT_PLATFORM, (cl_context_properties)platformIDs[i],
                0
            };
            devices[numberOfDevices].context = clCreateContext(contextProperties, 1, &devices[numberOfDevices].deviceID, NULL, NULL, &state);
            if (state != CL_SUCCESS) {
                printf("OpenCL Error: Context couldn't be created for device %u.\n", numberOfDevices);
                devices[numberOfDevices].context = NULL;
                continue;
            }

            // Obtendo a versão do OpenCL suportada pelo dispositivo
            char versionStr[128];
            clGetDeviceInfo(devices[numberOfDevices].deviceID, CL_DEVICE_VERSION, sizeof(versionStr), versionStr, NULL);

            int majorVersion = 0, minorVersion = 0;
            sscanf(versionStr, "OpenCL %d.%d", &majorVersion, &minorVersion);
            printf("Device (%u) supports OpenCL version: %d.%d\n", numberOfDevices, majorVersion, minorVersion);

            // Criando filas de comando para o dispositivo
            devices[numberOfDevices].kernelCommandQueue = clCreateCommandQueue(devices[numberOfDevices].context, devices[numberOfDevices].deviceID, CL_QUEUE_PROFILING_ENABLE, &state);
            if (state != CL_SUCCESS) {
                printf("OpenCL Error: Kernel command queue couldn't be created for device %u.\n", numberOfDevices);
                clReleaseContext(devices[numberOfDevices].context);
                devices[numberOfDevices].context = NULL;
                continue;
            }

            devices[numberOfDevices].dataCommandQueue = clCreateCommandQueue(devices[numberOfDevices].context, devices[numberOfDevices].deviceID, CL_QUEUE_PROFILING_ENABLE, &state);
            if (state != CL_SUCCESS) {
                printf("OpenCL Error: Data command queue couldn't be created for device %u.\n", numberOfDevices);
                clReleaseCommandQueue(devices[numberOfDevices].kernelCommandQueue);
                clReleaseContext(devices[numberOfDevices].context);
                devices[numberOfDevices].context = NULL;
                continue;
            }

            // Inicializando arrays de objetos e eventos
            devices[numberOfDevices].numberOfMemoryObjects = 0;
            devices[numberOfDevices].numberOfKernels = 0;
            devices[numberOfDevices].numberOfEvents = 0;

            devices[numberOfDevices].memoryObjects = new cl_mem[maxMemoryObjects];
            devices[numberOfDevices].kernels = new cl_kernel[maxKernels];
            memset(devices[numberOfDevices].memoryObjects, 0, sizeof(cl_mem) * maxMemoryObjects);
            memset(devices[numberOfDevices].kernels, 0, sizeof(cl_kernel) * maxKernels);

            devices[numberOfDevices].memoryObjectID = new int[maxMemoryObjects];
            devices[numberOfDevices].kernelID = new int[maxKernels];
            memset(devices[numberOfDevices].memoryObjectID, 0, sizeof(int) * maxMemoryObjects);
            memset(devices[numberOfDevices].kernelID, 0, sizeof(int) * maxKernels);

            devices[numberOfDevices].events = new cl_event[maxEvents];
            memset(devices[numberOfDevices].events, 0, sizeof(cl_event) * maxEvents);

            devices[numberOfDevices].program = 0;

            numberOfDevices++;
        }
    }

    if (numberOfDevices == 0) {
        printf("No OpenCL devices available.\n");
        return -1;
    }

    return numberOfDevices;
}






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
		
        // Ajustar o tamanho do grupo de trabalho para não exceder o limite e ser um divisor de globalItems
        if (localItems > maxWorkGroupSize) {
            localItems = maxWorkGroupSize;
        }
        while (globalItems % localItems != 0) {
            localItems--;
        }
		
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

void OpenCLWrapper::SynchronizeCommandQueue(int devicePosition)
{
    
	clFinish(devices[devicePosition].kernelCommandQueue);
	clFinish(devices[devicePosition].dataCommandQueue);
	devices[devicePosition].numberOfEvents = 0;
}

void OpenCLWrapper::GatherResults(int dataIndex, void *resultData) {
   
			
			for(int count = 0; count < todosDispositivos; count++)
			{	

				if(count >= meusDispositivosOffset && count < meusDispositivosOffset+meusDispositivosLength)
				{	
					int id = GetDeviceMemoryObjectID(dataIndex, count);
					
					ReadFromMemoryObject(count-meusDispositivosOffset, id, (char *)resultData+(offset[count]*elementSize), offset[count]*elementSize, length[count]*elementSize);
				
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

	//Probing();


}

// void OpenCLWrapper::Probing()
// {
//     std::cout << "Iniciando probing..." << std::endl;

//     double tempoInicioProbing = MPI_Wtime();
//     double localLatencia = 0, localBanda = 0;
    
//     // Verificação antes da alocação de memória
//     if (nElements <= 0 || unitsPerElement <= 0 || elementSize <= 0) {
//         std::cerr << "Erro: Valores inválidos para nElements, unitsPerElement ou elementSize." << std::endl;
//         return;
//     }

//     char *auxData = new char[nElements * unitsPerElement * elementSize];
    
//     if (!auxData) {
//         std::cerr << "Erro: Falha ao alocar memória para auxData." << std::endl;
//         return;
//     }

//     GatherResults(balancingTargetID, auxData);

//     // Soma do length antes do probing
//     int somaLengthAntes = 0;
//     for (int i = 0; i < todosDispositivos; i++) {
//         if (i < 0 || i >= todosDispositivos) {
//             std::cerr << "Erro: Índice 'i' fora do limite ao calcular somaLengthAntes." << std::endl;
//             delete[] auxData;
//             return;
//         }
//         somaLengthAntes += length[i];
//     }
//     std::cout << "Soma do length antes do probing: " << somaLengthAntes << std::endl;

//     PrecisaoBalanceamento();

//     for (int count = 0; count < todosDispositivos; count++)
//     {
//         if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
//         {
//             int overlapNovoOffset = static_cast<int>((count == 0 ? 0.0f : cargasNovas[count - 1]) * static_cast<float>(nElements));
//             int overlapNovoLength = static_cast<int>((cargasNovas[count] - (count == 0 ? 0.0f : cargasNovas[count - 1])) * static_cast<float>(nElements));

//             // Verificações e logs antes de usar os valores calculados
//             if (overlapNovoOffset < 0 || overlapNovoOffset >= nElements || 
//                 overlapNovoLength < 0 || overlapNovoOffset + overlapNovoLength > nElements) {
//                 std::cerr << "Erro: valores inválidos para overlapNovoOffset ou overlapNovoLength." << std::endl;
//                 std::cerr << "  overlapNovoOffset: " << overlapNovoOffset 
//                           << ", overlapNovoLength: " << overlapNovoLength 
//                           << ", nElements: " << nElements << std::endl;
//                 delete[] auxData;
//                 return;
//             }

//             for (int count2 = 0; count2 < todosDispositivos; count2++)
//             {
//                 if (count > count2)
//                 {
//                     if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
//                     {
//                         int overlap[2];
//                         int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
//                         char *malha = auxData;
//                         int dataDevice = GetDeviceMemoryObjectID(balancingTargetID, count);

//                         // Verificação antes de MPI_Recv
//                         if (alvo < 0 || alvo >= world_size) {
//                             std::cerr << "Erro: alvo fora dos limites no MPI_Recv." << std::endl;
//                             delete[] auxData;
//                             return;
//                         }

//                         MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//                         if (overlap[1] > 0 && overlap[0] >= 0 && overlap[0] + overlap[1] <= nElements)
//                         {
//                             ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice, malha + (overlap[0] * unitsPerElement), overlap[0] * unitsPerElement * elementSize, overlap[1] * unitsPerElement * elementSize);
//                             SynchronizeCommandQueue(count - meusDispositivosOffset);

//                             int sizeCarga = overlap[1] * unitsPerElement;

//                             double tempoInicioBanda = MPI_Wtime();
//                             MPI_Ssend(malha + (overlap[0] * unitsPerElement), sizeCarga, MPI_CHAR, alvo, 0, MPI_COMM_WORLD);
//                             double aux = (MPI_Wtime() - tempoInicioBanda) / sizeCarga;
//                             localBanda = aux > localBanda ? aux : localBanda;
//                         }
//                     }
//                 }
//                 else if (count < count2)
//                 {
//                     int overlapAntigoOffset = static_cast<int>((count2 == 0 ? 0 : cargasAntigas[count2 - 1]) * static_cast<float>(nElements));
//                     int overlapAntigoLength = static_cast<int>((cargasAntigas[count2] - (count2 == 0 ? 0.0f : cargasAntigas[count2 - 1])) * static_cast<float>(nElements));

//                     // Verificação dos valores antigos
//                     if (overlapAntigoOffset < 0 || overlapAntigoOffset >= nElements || 
//                         overlapAntigoLength < 0 || overlapAntigoOffset + overlapAntigoLength > nElements) {
//                         std::cerr << "Erro: valores inválidos para overlapAntigoOffset ou overlapAntigoLength." << std::endl;
//                         delete[] auxData;
//                         return;
//                     }

//                     int intersecaoOffset;
//                     int intersecaoLength;

//                     if (ComputarIntersecao(overlapAntigoOffset, overlapAntigoLength, overlapNovoOffset, overlapNovoLength, &intersecaoOffset, &intersecaoLength))
//                     {
//                         if (count2 >= meusDispositivosOffset && count2 < meusDispositivosOffset + meusDispositivosLength)
//                         {
//                             char *malha = auxData;
//                             int dataDevice[2] = {GetDeviceMemoryObjectID(balancingTargetID, count), GetDeviceMemoryObjectID(balancingTargetID, count2)};
//                             ReadFromMemoryObject(count2 - meusDispositivosOffset, dataDevice[1], malha + (intersecaoOffset * unitsPerElement), intersecaoOffset * unitsPerElement * elementSize, intersecaoLength * unitsPerElement * elementSize);
//                             SynchronizeCommandQueue(count2 - meusDispositivosOffset);

//                             WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], malha + (intersecaoOffset * unitsPerElement), intersecaoOffset * unitsPerElement * elementSize, intersecaoLength * unitsPerElement * elementSize);
//                             SynchronizeCommandQueue(count - meusDispositivosOffset);
//                         }
//                         else
//                         {
//                             if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
//                             {
//                                 int overlap[2] = {intersecaoOffset, intersecaoLength};
//                                 int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
//                                 char *malha = auxData;
//                                 int dataDevice = GetDeviceMemoryObjectID(balancingTargetID, count);

//                                 // Verificação antes de MPI_Ssend
//                                 if (alvo < 0 || alvo >= world_size) {
//                                     std::cerr << "Erro: alvo fora dos limites no MPI_Ssend." << std::endl;
//                                     delete[] auxData;
//                                     return;
//                                 }

//                                 SynchronizeCommandQueue(count - meusDispositivosOffset);
//                                 double tempoInicioLatencia = MPI_Wtime();
//                                 MPI_Ssend(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
//                                 double aux = (MPI_Wtime() - tempoInicioLatencia) / 2;
//                                 localLatencia = aux > localLatencia ? aux : localLatencia;

//                                 MPI_Recv(malha + (overlap[0] * unitsPerElement), overlap[1] * unitsPerElement, MPI_CHAR, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//                                 WriteToMemoryObject(count - meusDispositivosOffset, dataDevice, malha + (overlap[0] * unitsPerElement), overlap[0] * unitsPerElement * elementSize, overlap[1] * unitsPerElement * elementSize);
//                                 SynchronizeCommandQueue(count - meusDispositivosOffset);
//                             }
//                         }
//                     }
//                 }
//             }

//             offset[count] = overlapNovoOffset;
//             length[count] = overlapNovoLength;
//             SynchronizeCommandQueue(count - meusDispositivosOffset);
//         }
//     }

//     // Soma do length depois do probing
//     int somaLengthDepois = 0;
//     for (int i = 0; i < todosDispositivos; i++) {
        
//         somaLengthDepois += length[i];
//     }
//     std::cout << "Soma do length depois do probing: " << somaLengthDepois << std::endl;

//     std::cout << "Após o probing: " << std::endl;
//     for (int i = 0; i < todosDispositivos; i++)
//         std::cout << " Offset[" << i << "] = " << offset[i] << " length[" << i << "] = " << length[i] << " ";
//     std::cout << "\n";

//     memcpy(cargasAntigas, cargasNovas, sizeof(float) * todosDispositivos);

//     MPI_Allreduce(&localLatencia, &latencia, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//     MPI_Allreduce(&localBanda, &banda, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

//     MPI_Barrier(MPI_COMM_WORLD);
//     double tempoFimProbing = MPI_Wtime();
//     tempoBalanceamento += tempoFimProbing - tempoInicioProbing;
//     fatorErro = tempoBalanceamento;
//     delete[] auxData;
// }

void OpenCLWrapper::Probing()
{
    std::cout << "Iniciando probing..." << std::endl;

    double tempoInicioProbing = MPI_Wtime();
    double localLatencia = 0, localBanda = 0;
    
    if (nElements <= 0 || unitsPerElement <= 0 || elementSize <= 0) {
        std::cerr << "Erro: Valores inválidos para nElements, unitsPerElement ou elementSize." << std::endl;
        return;
    }

    char *auxData = new char[nElements * unitsPerElement * elementSize];
    
    if (!auxData) {
        std::cerr << "Erro: Falha ao alocar memória para auxData." << std::endl;
        return;
    }

    GatherResults(balancingTargetID, auxData);

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
            int overlapNovoOffset = static_cast<int>((count == 0 ? 0.0f : cargasNovas[count - 1]) * static_cast<float>(nElements));

            // Corrigido cálculo de overlapNovoLength
            int overlapNovoLength;
            if (count == todosDispositivos - 1) {
                // Último dispositivo pega todos os elementos restantes
                overlapNovoLength = nElements - overlapNovoOffset;
            } else {
                overlapNovoLength = static_cast<int>((cargasNovas[count] - (count == 0 ? 0.0f : cargasNovas[count - 1])) * static_cast<float>(nElements));
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
                    int overlapAntigoOffset = static_cast<int>((count2 == 0 ? 0 : cargasAntigas[count2 - 1]) * static_cast<float>(nElements));
                    int overlapAntigoLength = static_cast<int>((cargasAntigas[count2] - (count2 == 0 ? 0.0f : cargasAntigas[count2 - 1])) * static_cast<float>(nElements));

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
                                double aux = (MPI_Wtime() - tempoInicioLatencia) / 2;
                                localLatencia = aux > localLatencia ? aux : localLatencia;

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

    MPI_Allreduce(&localLatencia, &latencia, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localBanda, &banda, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

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
				
				kernelEventoDispositivo[count] = RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count], length[count], isDeviceCPU(deviceIndex)? 8 : 64);
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
	 		tempos[count] = ((float)ticks[count]) / (((float)cargasNovas[count])*10.0f);
	 	}
	}
	float tempos_root[todosDispositivos];
	MPI_Allreduce(tempos, tempos_root, todosDispositivos, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	memcpy(tempos, tempos_root, sizeof(float) * todosDispositivos);
  


}


// void OpenCLWrapper::Probing()
// {
//    	std::cout<<"Iniciando probing..."<<std::endl;
// 	double tempoInicioProbing = MPI_Wtime();
// 	double localLatencia = 0, localBanda = 0;
// 	void *auxData = new float[nElements*unitsPerElement];
// 	GatherResults(balancingTargetID, (auxData));
	
   
//     for (int count = 0; count < todosDispositivos; count++)
//     {
//         if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
//         {
//             int overlapNovoOffset = (count == 0 ? 0 : cargasNovas[count - 1]) * nElements;
//             int overlapNovoLength = (count == 0 ? cargasNovas[count] : cargasNovas[count] - cargasNovas[count - 1]) * nElements;

//             // Cast para char* para operar com offset
//             char* malhaChar = static_cast<char*>(auxData);

//             for (int count2 = 0; count2 < todosDispositivos; count2++)
//             {
//                 if (count > count2)
//                 {
//                     // Atender requisicoes de outros processos
//                     if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
//                     {
//                         int overlap[2];
//                         int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
//                         void* malha = auxData; 
//                         int dataDevice = GetDeviceMemoryObjectID(balancingTargetID, count);
//                         MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//                         if (overlap[1] > 0)
//                         {
//                             // Calcular o endereço base para leitura
//                             char* leituraEndereco = malhaChar + (overlap[0] * unitsPerElement);
//                             ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice, leituraEndereco, overlap[0] * unitsPerElement * elementSize, overlap[1] * unitsPerElement * elementSize);
//                             SynchronizeCommandQueue(count - meusDispositivosOffset);

//                             int sizeCarga = overlap[1] * unitsPerElement;

//                             double tempoInicioBanda = MPI_Wtime();
//                             MPI_Ssend(leituraEndereco, sizeCarga, MPI_BYTE, alvo, 0, MPI_COMM_WORLD);
//                             double aux = (MPI_Wtime() - tempoInicioBanda) / sizeCarga;
//                             localBanda = aux > localBanda ? aux : localBanda;
//                         }
//                     }
//                 }
//                 else if (count < count2)
//                 {
//                     int overlapAntigoOffset = (count2 == 0 ? 0 : cargasAntigas[count2 - 1]) * nElements;
//                     int overlapAntigoLength = (count2 == 0 ? cargasAntigas[count2] : cargasAntigas[count2] - cargasAntigas[count2 - 1]) * nElements;

//                     int intersecaoOffset;
//                     int intersecaoLength;

//                     if (((overlapAntigoOffset <= overlapNovoOffset - divisionSize) && ComputarIntersecao(overlapAntigoOffset, overlapAntigoLength, overlapNovoOffset - divisionSize, overlapNovoLength + divisionSize, &intersecaoOffset, &intersecaoLength)) ||
//                         ((overlapAntigoOffset > overlapNovoOffset - divisionSize) && ComputarIntersecao(overlapNovoOffset - divisionSize, overlapNovoLength + divisionSize, overlapAntigoOffset, overlapAntigoLength, &intersecaoOffset, &intersecaoLength)))
//                     {
//                         if (count2 >= meusDispositivosOffset && count2 < meusDispositivosOffset + meusDispositivosLength)
//                         {
//                             void* malha = auxData; 
//                             int dataDevice[2] = {GetDeviceMemoryObjectID(balancingTargetID, count), GetDeviceMemoryObjectID(balancingTargetID, count2)};

//                             char* malhaChar = static_cast<char*>(auxData);

//                             char* leituraEndereco = malhaChar + (intersecaoOffset * unitsPerElement);
//                             ReadFromMemoryObject(count2 - meusDispositivosOffset, dataDevice[1], leituraEndereco, intersecaoOffset * unitsPerElement * elementSize, intersecaoLength * unitsPerElement * elementSize);
//                             SynchronizeCommandQueue(count2 - meusDispositivosOffset);

//                             WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], leituraEndereco, intersecaoOffset * unitsPerElement * elementSize, intersecaoLength * unitsPerElement * elementSize);
//                             SynchronizeCommandQueue(count - meusDispositivosOffset);
//                         }
//                         else
//                         {
//                             if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
//                             {
//                                 int overlap[2] = {intersecaoOffset, intersecaoLength};
//                                 int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
//                                 void* malha = auxData;
//                                 int dataDevice = GetDeviceMemoryObjectID(balancingTargetID, count);
//                                 SynchronizeCommandQueue(count - meusDispositivosOffset);
//                                 double tempoInicioLatencia = MPI_Wtime();
//                                 MPI_Ssend(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
//                                 double aux = (MPI_Wtime() - tempoInicioLatencia) / 2;
//                                 localLatencia = aux > localLatencia ? aux : localLatencia;

//                                 MPI_Recv(malhaChar + (overlap[0] * unitsPerElement), overlap[1] * unitsPerElement, MPI_BYTE, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//                                 WriteToMemoryObject(count - meusDispositivosOffset, dataDevice, malhaChar + (overlap[0] * unitsPerElement), overlap[0] * unitsPerElement * elementSize, overlap[1] * unitsPerElement * elementSize);
//                                 SynchronizeCommandQueue(count - meusDispositivosOffset);
//                             }
//                         }
//                     }
//                     else
//                     {
//                         if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
//                         {
//                             int overlap[2] = {0, 0};
//                             int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
//                             void* malha = auxData;
//                             MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
//                         }
//                     }
//                 }
//             }

//             offset[count] = overlapNovoOffset;
//             length[count] = overlapNovoLength;
            
//             SynchronizeCommandQueue(count - meusDispositivosOffset);
//         }
//     }

//     std::cout <<"Após o probing: "<< std::endl;
//     for (int i = 0; i < todosDispositivos; i++)
//         std::cout << "Offset[" << i << "] = " << offset[i] << " length[" << i << "] = " << length[i] << " ";
//     std::cout << "\n";
// 	auxData = nullptr;
// 	delete[] auxData;
//     memcpy(cargasAntigas, cargasNovas, sizeof(float) * todosDispositivos);

//     MPI_Allreduce(&localLatencia, &latencia, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//     MPI_Allreduce(&localBanda, &banda, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

//     MPI_Barrier(MPI_COMM_WORLD);
//     double tempoFimProbing = MPI_Wtime();
//     tempoBalanceamento += tempoFimProbing - tempoInicioProbing;
//     fatorErro = tempoBalanceamento;
// }




// void OpenCLWrapper::LoadBalancing(){
//     double tempoInicioBalanceamento = MPI_Wtime();
//     double localTempoCB;


// 	PrecisaoBalanceamento();

//     for (int count = 0; count < todosDispositivos; count++) {
//         if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
//             SynchronizeCommandQueue(count - meusDispositivosOffset);
//             localTempoCB = cargasNovas[count] * tempos[count];
//         }
//     }
//     MPI_Allreduce(&localTempoCB, &tempoCB, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//     tempoCB *= nElements;

//     if (latencia + ComputarNorma(cargasAntigas, cargasNovas, todosDispositivos) * (writeByte + banda) + tempoCB < tempoComputacaoInterna) {
//         for (int count = 0; count < todosDispositivos; count++) {
//             if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
//                 int overlapNovoOffset = ((count == 0 ? 0.0f : cargasNovas[count - 1]) * (nElements));
//                 int overlapNovoLength = ((count == 0 ? cargasNovas[count] - 0.0f : cargasNovas[count] - cargasNovas[count - 1]) * (nElements));
//                 for (int count2 = 0; count2 < todosDispositivos; count2++) {
//                     if (count > count2) {
//                         if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
//                             int overlap[2];
//                             int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
//                             T *data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
//                             int dataDevice = (simulacao % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
//                             MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                             if (overlap[1] > 0) {
//                                 ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice, (char *)(data + overlap[0] * Element_size), overlap[0] * Element_size * elementSize, overlap[1] * Element_size * sizeof(float));
//                                 SynchronizeCommandQueue(count - meusDispositivosOffset);
//                                 size_t sizeCarga = overlap[1] * Element_size;
//                                 MPI_Send(data + overlap[0] * Element_size, sizeCarga, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD);
//                             }
//                         }
//                     } else if (count < count2) {
//                         int overlapAntigoOffset = ((count2 == 0 ? 0 : cargasAntigas[count2 - 1]) * nElements);
//                         int overlapAntigoLength = ((count2 == 0 ? cargasAntigas[count2] - 0.0f : cargasAntigas[count2] - cargasAntigas[count2 - 1]) * nElements);

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








// void OpenCLWrapper::ComputarCargas(const long int *ticks, const float *cargasAntigas, float *cargasNovas, int participantes) {
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


void OpenCLWrapper::ComputarCargas(const long int *ticks, const float *cargasAntigas, float *cargasNovas, int participantes) {
    if (participantes == 1) {
        cargasNovas[0] = 1.0f;
        return;
    }

    // Imprimir as cargas antigas antes da modificação
    std::cout << "Cargas antigas:" << std::endl;
    for (int i = 0; i < participantes; i++) {
        std::cout << "cargasAntigas[" << i << "] = " << cargasAntigas[i] << std::endl;
    }

    // Cálculo de cargaTotal
    float cargaTotal = 0.0f;
    for (int count = 0; count < participantes; count++) {
        float cargaAtual = (count == 0 ? cargasAntigas[count] : cargasAntigas[count] - cargasAntigas[count - 1]);
        float fatorTicks = (count == 0 || ticks[count] == 0) ? 1.0f : static_cast<float>(ticks[0]) / static_cast<float>(ticks[count]);
        cargaTotal += cargaAtual * fatorTicks;
    }

    // Verificação para evitar divisão por zero
    if (cargaTotal == 0.0f) {
        std::cerr << "Erro: cargaTotal é zero, não é possível computar cargas." << std::endl;
        return;
    }

    // Cálculo de cargas novas
    for (int count = 0; count < participantes; count++) {
        float cargaAtual = (count == 0 ? cargasAntigas[count] : cargasAntigas[count] - cargasAntigas[count - 1]);
        float fatorTicks = (count == 0 || ticks[count] == 0) ? 1.0f : static_cast<float>(ticks[0]) / static_cast<float>(ticks[count]);
        float cargaNova = (cargaAtual * fatorTicks) / cargaTotal;
        cargasNovas[count] = (count == 0 ? cargaNova : cargasNovas[count - 1] + cargaNova);
    }

    // Garantir que o último valor em cargasNovas seja 1.0
    cargasNovas[participantes - 1] = 1.0f ;

    // Imprimir as cargas novas após a modificação
    std::cout << "Cargas novas:" << std::endl;
    for (int i = 0; i < participantes; i++) {
        std::cout << "cargasNovas[" << i << "] = " << cargasNovas[i] << std::endl;
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



// bool OpenCLWrapper::ComputarIntersecao(int offset1, int length1, int offset2, int length2, int *intersecaoOffset, int *intersecaoLength)
// {
//    // Calcular os finais dos intervalos
//     int final1 = offset1 + length1;
//     int final2 = offset2 + length2;

//     // Verificar se há sobreposição
//     if (final1 <= offset2 || final2 <= offset1) {
//         return false; // Sem interseção
//     }

//     // Calcular o início da interseção
//     *intersecaoOffset = std::max(offset1, offset2);

//     // Calcular o final da interseção
//     int intersecaoFinal = std::min(final1, final2);

//     // Calcular o comprimento da interseção
//     *intersecaoLength = intersecaoFinal - *intersecaoOffset;

//     // Garantir que a interseção seja válida e evitar buracos
//     if (*intersecaoLength > 0 && *intersecaoOffset >= 0) {
//         return true;
//     }

//     return false;
// }



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


void OpenCLWrapper::setBalancingTargetID(int targetID)
{

balancingTargetID = targetID;


}