#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "OpenCLWrapper.h"
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iostream>

using namespace std;
template<typename T, typename U>
class Balanceador {
public:
    Balanceador(int argc, char *argv[], T* data, const size_t Element_sz, const unsigned long int N_Element, U *DTK, const size_t div_size, const unsigned int interv, const unsigned long int _units_per_elements, const std::string& kernelSource, const std::string& kernelFunction);    
    ~Balanceador();
    void setCustomDatatype(MPI_Datatype custom_type);
    void runKernelOperations();
    void gatherData(T *data_dest);
    void balanceLoad();
    

private:
    void initMPI(int argc, char *argv[]);
    void createKernel(const std::string& kernelSource, const std::string& kernelFunction);
    void ComputarCargas(const long int *ticks, const float *cargasAntigas, float *cargas, int participantes);
    bool ComputarIntersecao(int offset1, int length1, int offset2, int length2, int *intersecaoOffset, int *intersecaoLength);
    int RecuperarPosicaoHistograma(int *histograma, int tamanho, int indice);
    float ComputarDesvioPadraoPercentual(const long int *ticks, int participantes);
    float ComputarNorma(const float *cargasAntigas, const float *cargasNovas, int participantes);
    inline void InicializarLenghtOffset(unsigned int offsetComputacao, unsigned int lengthComputacao, int count);
    void initializeDevices();
    void initializeKernel();
    void synchronizeAndCollectExecutionTimes();
    void Probing();

    T* Data;
    long int Iterations;
    size_t Element_size;
    unsigned long int N_Elements;
    U* DataToKernel;
    size_t DataToKernel_Size;
    unsigned int interv_balance;
    unsigned long int units_per_elements;
    unsigned int* offset;
    unsigned long int* length;
    //cl_device_type deviceType;
    MPI_Datatype mpi_custom_type;
    bool custom_type_set;
    bool kernel_set;
    cl_command_queue commandQueue;
    std::vector<cl_device_id> devices;
    std::vector<cl_context> contexts;
    std::vector<cl_program> programs;
    std::vector<cl_kernel> kernels;
    std::vector<cl_mem> memObjects;
    int world_size;
    int world_rank;
    int meusDispositivosOffset;
    int meusDispositivosLength;
    int todosDispositivos;
    long int* ticks;
    float* cargasNovas;
    float* cargasAntigas;
    double* tempos_por_carga;
    int* DataToKernelDispositivo;
    int* kernelDispositivo;
    int** swapBufferDispositivo;
    double latencia;
    double banda;
    double tempoComputacaoInterna;
    double tempoComputacaoBorda;
    double tempoBalanceamento;
    double tempoCB;
    double fatorErro;
    cl_event kernelEvento;
    cl_event dataEvento;
    cl_event* kernelEventos;
    int dispositivos;
    int *dispositivosLocal;
    int *dispositivosWorld;
    int offsetComputacao; 
    int lengthComputacao;
    T **swapBuffer;
    std::string kernelName;
    std::string functionName;
    double *tempos;
    int *dataEventoDispositivo;
    int *kernelEventoDispositivo;
    double tempoInicio;
    double writeByte;
    int CPU_WORK_GROUP_SIZE;
    int GPU_WORK_GROUP_SIZE;
    int PRECISAO_BALANCEAMENTO = 10;
    MPI_Datatype mpi_data_type; // Tipo MPI correspondente a T
    int MAX_NUMBER_OF_PLATFORMS = 10;
    int MAX_NUMBER_OF_DEVICES_PER_PLATFORM = 10;
  
};

template<typename T>
MPI_Datatype GetMPIType() {
    MPI_Datatype mpi_type;
    if (std::is_same<T, char>::value) {
        mpi_type = MPI_CHAR;
    } else if (std::is_same<T, int>::value) {
        mpi_type = MPI_INT;
    } else if (std::is_same<T, float>::value) {
        mpi_type = MPI_FLOAT;
    } else if (std::is_same<T, double>::value) {
        mpi_type = MPI_DOUBLE;
    } else {
        MPI_Type_match_size(MPI_TYPECLASS_REAL, sizeof(T), &mpi_type);
    }
    return mpi_type;
}


template<typename T, typename U>
Balanceador<T, U>::Balanceador(int argc, char *argv[], T* data, const size_t Element_sz, const unsigned long int N_Element, U *DTK, const size_t div_size, const unsigned int interv, const unsigned long int _units_per_elements, const std::string& kernelSource, const std::string& kernelFunction)
    : Data(data), Element_size(Element_sz), N_Elements(N_Element), DataToKernel(DTK), DataToKernel_Size(div_size), interv_balance(interv), units_per_elements(_units_per_elements), custom_type_set(false), kernel_set(false), Iterations(0) {
    cout<<"Init MPI"<<endl;
    initMPI(argc, argv);
  
    
   
    
    cout<<"createKernel"<<endl;
    createKernel(kernelSource, kernelFunction);
    cout<<"initializedevices"<<endl;
    initializeDevices();
    cout<<"initializeKernel"<<endl;
   initializeKernel();
   cout<<"End constructor"<<endl;
}

template<typename T, typename U>
void Balanceador<T, U>::initMPI(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}



template<typename T, typename U>
void Balanceador<T, U>::createKernel(const std::string& kernelSource, const std::string& kernelFunction) {
    cl_int status;
    size_t sourceSize;
    size_t MAX_SOURCE_BUFFER_LENGTH = 2048;
    char* sourceStr;
    FILE* fp = fopen(kernelSource.c_str(), "r");
    if (!fp) {
        throw std::runtime_error("Failed to load kernel");
    }
    sourceStr = (char*)malloc(MAX_SOURCE_BUFFER_LENGTH);
    sourceSize = fread(sourceStr, 1, MAX_SOURCE_BUFFER_LENGTH, fp);
    fclose(fp);

    for (size_t i = 0; i < devices.size(); ++i) {
        contexts[i] = clCreateContext(NULL, 1, &devices[i], NULL, NULL, &status);
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Failed to create context");
        }

        cl_program program = clCreateProgramWithSource(contexts[i], 1, (const char**)&sourceStr, &sourceSize, &status);
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Failed to create program");
        }

        status = clBuildProgram(program, 1, &devices[i], NULL, NULL, NULL);
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Failed to build program");
        }

        cl_kernel kernel = clCreateKernel(program, kernelFunction.c_str(), &status);
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Failed to create kernel");
        }

        programs.push_back(program);
        kernels.push_back(kernel);
    }

    kernel_set = true;
    free(sourceStr);
}

template<typename T, typename U>
void Balanceador<T, U>::initializeDevices() {
    
    
    int dispositivos = InitParallelProcessor();
    dispositivosWorld = new int[world_size];
	int dispositivosLocal[world_size];
	memset(dispositivosLocal, 0, sizeof(int) * world_size);
	dispositivosLocal[world_rank] = dispositivos;
	MPI_Allreduce(dispositivosLocal, dispositivosWorld, world_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	todosDispositivos = 0;
	
	
	for (int count = 0; count < world_size; count++)
	{
		if (count == world_rank)
		{
			meusDispositivosOffset = todosDispositivos;
			meusDispositivosLength = dispositivosWorld[count];
		}
		todosDispositivos += dispositivosWorld[count];
	}
    offsetComputacao = 0;
    lengthComputacao = (N_Elements) / todosDispositivos;
    ticks = new long int[todosDispositivos];
    tempos_por_carga = new double[todosDispositivos];
    cargasNovas = new float[todosDispositivos];
    cargasAntigas = new float[todosDispositivos];
    DataToKernelDispositivo = new int[todosDispositivos];
    swapBufferDispositivo = new int*[todosDispositivos];
   
    for (int i = 0; i < todosDispositivos; ++i) {
        swapBufferDispositivo[i] = new int[2];
    }
    memset(ticks, 0, sizeof(long int) * todosDispositivos);
    memset(tempos_por_carga, 0, sizeof(double) * todosDispositivos);
    memset(cargasNovas, 0, sizeof(float) * todosDispositivos);
    memset(cargasAntigas, 0, sizeof(float) * todosDispositivos);
    offset = new unsigned int[todosDispositivos];
    length = new unsigned long int[todosDispositivos];
    cout<<"loop"<<endl;
    cout<<"lengthComputacao = "<<lengthComputacao<<endl;
    cout<<"TodosDispostivos = "<<todosDispositivos<<endl;
    double localWriteByte = 0;
    cout<<"1"<<endl;
    for (int count = 0; count < todosDispositivos; count++) {
       cout<<"count = "<<count<<endl;
       cout<<"meusDispositivosOffset = "<<meusDispositivosOffset<<endl;
       cout<<"meusDispositivosOffset+meusDispositivosLength = "<<meusDispositivosOffset+meusDispositivosLength<<endl;
    
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            cout<<"Dentro do if"<<endl;
            InicializarLenghtOffset(offsetComputacao, (count + 1 == todosDispositivos) ? (N_Elements - offsetComputacao) : lengthComputacao, count);           
            DataToKernelDispositivo[count] = CreateMemoryObject(count - meusDispositivosOffset, DataToKernel_Size, CL_MEM_READ_ONLY, NULL);
            size_t sizeCarga = Element_size * N_Elements;
            swapBufferDispositivo[count][0] = CreateMemoryObject(count - meusDispositivosOffset, sizeCarga, CL_MEM_READ_WRITE, NULL);
            swapBufferDispositivo[count][1] = CreateMemoryObject(count - meusDispositivosOffset, sizeCarga, CL_MEM_READ_WRITE, NULL);
            cout<<"WriteToMemoryObject(count - meusDispositivosOffset) = "<<count - meusDispositivosOffset<<endl;
            for(int i = 0; i < 1024; i++)
            cout<<DataToKernel[i]<<" ";
            cout<<endl;
            WriteToMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)DataToKernel, 0, DataToKernel_Size);
            cout<<"---1----"<<endl;
            SynchronizeCommandQueue(count - meusDispositivosOffset);
            cout<<"----2---"<<endl;
            tempoInicio = MPI_Wtime();
            cout<<"----3---"<<endl;
            WriteToMemoryObject(count - meusDispositivosOffset, swapBufferDispositivo[count][0], (char *)swapBuffer[0], 0, sizeCarga);
            cout<<"---4----"<<endl;
            WriteToMemoryObject(count - meusDispositivosOffset, swapBufferDispositivo[count][1], (char *)swapBuffer[1], 0, sizeCarga);
            cout<<"----5---"<<endl;
            double aux = (MPI_Wtime() - tempoInicio) / sizeCarga / 2;
            cout<<"----6---"<<endl;
            localWriteByte = aux > localWriteByte ? aux : localWriteByte;
            cout<<"----7---"<<endl;
            kernelDispositivo[count] = CreateKernel(count - meusDispositivosOffset, kernelName.c_str(), functionName.c_str());
            SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, swapBufferDispositivo[count][0]);
            SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, swapBufferDispositivo[count][1]);
            SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 2, DataToKernelDispositivo[count]);
        }

        offsetComputacao += lengthComputacao;
    }
    cout<<"fim loop"<<endl;
    MPI_Allreduce(&localWriteByte, &writeByte, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
     cout<<"end"<<endl;
}

template<typename T, typename U>
void Balanceador<T, U>::initializeKernel() {
    cl_int status;
    for (size_t i = 0; i < devices.size(); ++i) {
        cl_mem memObject = clCreateBuffer(contexts[i], CL_MEM_READ_WRITE,  Element_size * N_Elements * units_per_elements, NULL, &status);
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Failed to create buffer");
        }

        memObjects.push_back(memObject);
        status = clSetKernelArg(kernels[i], 0, sizeof(cl_mem), &memObject);
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Failed to set kernel argument");
        }
    }
}

template<typename T, typename U>
inline void Balanceador<T,U>::InicializarLenghtOffset(unsigned int offsetComputacao, unsigned int lengthComputacao, int count)
{
	offset[count] = offsetComputacao;
	length[count] = lengthComputacao;
}

template<typename T, typename U>
void Balanceador<T, U>::runKernelOperations() {
    cl_int status;
    for (size_t i = 0; i < devices.size(); ++i) {
        size_t globalWorkSize = N_Elements;
        status = clEnqueueNDRangeKernel(commandQueue, kernels[i], 1, NULL, &globalWorkSize, NULL, 0, NULL, &kernelEvento);
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue kernel");
        }
    }

    clFinish(commandQueue);

    MPI_Barrier(MPI_COMM_WORLD);

    synchronizeAndCollectExecutionTimes();
}
template<typename T, typename U>
Balanceador<T,U>::~Balanceador() {
    FinishParallelProcessor();
    MPI_Finalize();
}
template<typename T, typename U>
void Balanceador<T, U>::synchronizeAndCollectExecutionTimes() {
    for (size_t i = 0; i < devices.size(); ++i) {
        cl_ulong startTime, endTime;
        clGetEventProfilingInfo(kernelEvento, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
        clGetEventProfilingInfo(kernelEvento, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
        double executionTime = (endTime - startTime) * 1e-6; // Convert to milliseconds
        tempos_por_carga[i] = executionTime;
        cout<<"ExecutionTime: "<<executionTime<<endl;
    }
}

template<typename T, typename U>
void Balanceador<T, U>::gatherData(T* data_dest) {
    for (size_t i = 0; i < devices.size(); ++i) {
        clEnqueueReadBuffer(commandQueue, memObjects[i], CL_TRUE, 0,  Element_size * N_Elements * units_per_elements, Data, 0, NULL, &dataEvento);
    }

    clFinish(commandQueue);
    MPI_Barrier(MPI_COMM_WORLD);
}

template<typename T, typename U>
void Balanceador<T, U>::setCustomDatatype(MPI_Datatype custom_type) {
    mpi_custom_type = custom_type;
    custom_type_set = true;
}

template<typename T, typename U>
void Balanceador<T, U>::balanceLoad() {
    double tempoInicioBalanceamento = MPI_Wtime();
    double localTempoCB;
    PrecisaoBalanceamento();

    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            SynchronizeCommandQueue(count - meusDispositivosOffset);
            localTempoCB = cargasNovas[count] * tempos[count];
        }
    }
    MPI_Allreduce(&localTempoCB, &tempoCB, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    tempoCB *= N_Elements;

    if (latencia + ComputarNorma(cargasAntigas, cargasNovas, todosDispositivos) * (writeByte + banda) + tempoCB < tempoComputacaoInterna) {
        for (int count = 0; count < todosDispositivos; count++) {
            if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
                int overlapNovoOffset = ((count == 0 ? 0.0f : cargasNovas[count - 1]) * (N_Elements));
                int overlapNovoLength = ((count == 0 ? cargasNovas[count] - 0.0f : cargasNovas[count] - cargasNovas[count - 1]) * (N_Elements));
                for (int count2 = 0; count2 < todosDispositivos; count2++) {
                    if (count > count2) {
                        if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
                            int overlap[2];
                            int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
                            T *data = (Iteration % 2) == 0 ? swapBuffer[0] : swapBuffer[1];
                            int dataDevice = (Iteration % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
                            MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            if (overlap[1] > 0) {
                                ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice, (char *)(data + overlap[0] * Element_size), overlap[0] * Element_size * sizeof(float), overlap[1] * Element_size * sizeof(float));
                                SynchronizeCommandQueue(count - meusDispositivosOffset);
                                size_t sizeCarga = overlap[1] * Element_size;
                                MPI_Send(data + overlap[0] * Element_size, sizeCarga, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD);
                            }
                        }
                    } else if (count < count2) {
                        int overlapAntigoOffset = ((count2 == 0 ? 0 : cargasAntigas[count2 - 1]) * N_Elements);
                        int overlapAntigoLength = ((count2 == 0 ? cargasAntigas[count2] - 0.0f : cargasAntigas[count2] - cargasAntigas[count2 - 1]) * N_Elements);

                        int intersecaoOffset;
                        int intersecaoLength;

                        if ((overlapAntigoOffset <= overlapNovoOffset - interv_balance && ComputarIntersecao(overlapAntigoOffset, overlapAntigoLength, overlapNovoOffset - interv_balance, overlapNovoLength + interv_balance, &intersecaoOffset, &intersecaoLength)) ||
                            (overlapAntigoOffset > overlapNovoOffset - interv_balance && ComputarIntersecao(overlapNovoOffset - interv_balance, overlapNovoLength + interv_balance, overlapAntigoOffset, overlapAntigoLength, &intersecaoOffset, &intersecaoLength))) {
                            if (count2 >= meusDispositivosOffset && count2 < meusDispositivosOffset + meusDispositivosLength) {
                                T *data = (Iteration % 2) == 0 ? swapBuffer[0] : swapBuffer[1];
                                int dataDevice[2] = {(Iteration % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1],
                                                     (Iteration % 2) == 0 ? swapBufferDispositivo[count2][0] : swapBufferDispositivo[count2][1]};

                                ReadFromMemoryObject(count2 - meusDispositivosOffset, dataDevice[1], (char *)(data + intersecaoOffset * Element_size), intersecaoOffset * Element_size * sizeof(float), intersecaoLength * Element_size * sizeof(float));
                                SynchronizeCommandQueue(count2 - meusDispositivosOffset);
                                WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(data + intersecaoOffset * Element_size), intersecaoOffset * Element_size * sizeof(float), intersecaoLength * Element_size * sizeof(float));
                                SynchronizeCommandQueue(count - meusDispositivosOffset);
                            } else {
                                if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
                                    int overlap[2] = {intersecaoOffset, intersecaoLength};
                                    int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
                                    T *data = (Iteration % 2) == 0 ? swapBuffer[0] : swapBuffer[1];
                                    int dataDevice = (Iteration % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
                                    MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
                                    MPI_Recv(data + overlap[0] * Element_size, overlap[1] * Element_size, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                    WriteToMemoryObject(count - meusDispositivosOffset, dataDevice, (char *)(data + overlap[0] * Element_size), overlap[0] * Element_size * sizeof(float), overlap[1] * Element_size * sizeof(float));
                                    SynchronizeCommandQueue(count - meusDispositivosOffset);
                                }
                            }
                        } else {
                            if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
                                int overlap[2] = {0, 0};
                                int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
                                T *data = (Iteration % 2) == 0 ? swapBuffer[0] : swapBuffer[1];
                                MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
                            }
                        }
                    }
                }
                offset[count] = overlapNovoOffset;
                length[count] = overlapNovoLength;
                WriteToMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)DataToKernel, 0, sizeof(int) * units_per_elements);
                SynchronizeCommandQueue(count - meusDispositivosOffset);
            }
        }
        memcpy(cargasAntigas, cargasNovas, sizeof(float) * todosDispositivos);

        MPI_Barrier(MPI_COMM_WORLD);
        double tempoFimBalanceamento = MPI_Wtime();
        tempoBalanceamento += tempoFimBalanceamento - tempoInicioBalanceamento;
    }
}



template<typename T, typename U>
void Balanceador<T,U>::Probing()
{
	

	double tempoInicioProbing = MPI_Wtime();
	double localLatencia = 0, localBanda = 0;
	PrecisaoBalanceamento();

	// Computar novas cargas.

	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			int overlapNovoOffset = ((int)(((count == 0) ? 0.0f : cargasNovas[count - 1]) * ((float)(N_Elements))));
			int overlapNovoLength = ((int)(((count == 0) ? cargasNovas[count] - 0.0f : cargasNovas[count] - cargasNovas[count - 1]) * ((float)(N_Elements))));
			for (int count2 = 0; count2 < todosDispositivos; count2++)
			{
				if (count > count2)
				{
					// Atender requisicoes de outros processos.
					if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
					{
						int overlap[2];
						int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
						T *Data = ((Iteration % 2) == 0) ? swapBuffer[0] : swapBuffer[1];
						int dataDevice = ((Iteration % 2) == 0) ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
						MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						// Podem ocorrer requisicoes vazias.
						if (overlap[1] > 0)
						{
							ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice, (char *)(Data + (overlap[0] * units_per_elements)), overlap[0] * units_per_elements * sizeof(float), overlap[1] * units_per_elements * sizeof(float));
							SynchronizeCommandQueue(count - meusDispositivosOffset);

							sizeCarga = overlap[1] * units_per_elements;

							double tempoInicioBanda = MPI_Wtime();
							MPI_Ssend(Data + (overlap[0] * units_per_elements), sizeCarga, MPI_FLOAT, alvo, 0, MPI_COMM_WORLD);
							double aux = (MPI_Wtime() - tempoInicioBanda) / sizeCarga;
							localBanda = aux > localBanda ? aux : localBanda;
						}
					}
				}
				else if (count < count2)
				{
					
					int overlapAntigoOffset = ((int)(((count2 == 0) ? 0 : cargasAntigas[count2 - 1]) * (N_Elements)));
					int overlapAntigoLength = ((int)(((count2 == 0) ? cargasAntigas[count2] - 0.0f : cargasAntigas[count2] - cargasAntigas[count2 - 1]) * (N_Elements)));

					int intersecaoOffset;
					int intersecaoLength;

					if (((overlapAntigoOffset <= overlapNovoOffset - (interv_balance)) && ComputarIntersecao(overlapAntigoOffset, overlapAntigoLength, overlapNovoOffset - (interv_balance), overlapNovoLength + (interv_balance), &intersecaoOffset, &intersecaoLength)) ||
							((overlapAntigoOffset > overlapNovoOffset - (interv_balance)) && ComputarIntersecao(overlapNovoOffset - (interv_balance), overlapNovoLength + (interv_balance), overlapAntigoOffset, overlapAntigoLength, &intersecaoOffset, &intersecaoLength)))
					{
						if (count2 >= meusDispositivosOffset && count2 < meusDispositivosOffset + meusDispositivosLength)
						{
							T *Data = ((Iteration % 2) == 0) ? swapBuffer[0] : swapBuffer[1];

							int dataDevice[2] = {((Iteration % 2) == 0) ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1],
																		((Iteration % 2) == 0) ? swapBufferDispositivo[count2][0] : swapBufferDispositivo[count2][1]};

							ReadFromMemoryObject(count2 - meusDispositivosOffset, dataDevice[1], (char *)(Data + (intersecaoOffset * units_per_elements)), intersecaoOffset * units_per_elements * sizeof(float), intersecaoLength * units_per_elements * sizeof(float));
							SynchronizeCommandQueue(count2 - meusDispositivosOffset);

							WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(Data + (intersecaoOffset * units_per_elements)), intersecaoOffset * units_per_elements * sizeof(float), intersecaoLength * units_per_elements * sizeof(float));
							SynchronizeCommandQueue(count - meusDispositivosOffset);
						}
						else
						{
							// Fazer uma requisicao.
							if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
							{
								int overlap[2] = {intersecaoOffset, intersecaoLength};
								int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
								T *Data = ((Iteration % 2) == 0) ? swapBuffer[0] : swapBuffer[1];
								int dataDevice = ((Iteration % 2) == 0) ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
								SynchronizeCommandQueue(count - meusDispositivosOffset);
								double tempoInicioLatencia = MPI_Wtime();
								MPI_Ssend(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
								double aux = (MPI_Wtime() - tempoInicioLatencia) / 2;
								localLatencia = aux > localLatencia ? aux : localLatencia;

								MPI_Recv(Data + (overlap[0] * units_per_elements), overlap[1] * units_per_elements, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

								WriteToMemoryObject(count - meusDispositivosOffset, dataDevice, (char *)(Data + (overlap[0] * units_per_elements)), overlap[0] * units_per_elements * sizeof(float), overlap[1] * units_per_elements * sizeof(float));
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
							T *Data = ((Iteration % 2) == 0) ? swapBuffer[0] : swapBuffer[1];
							MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
						}
					}
				}
			}

			offset[count] = overlapNovoOffset;
			length[count]= overlapNovoLength;

			WriteToMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)DataToKernel, 0, DataToKernel_Size);
			SynchronizeCommandQueue(count - meusDispositivosOffset);
		}
	}
	memcpy(cargasAntigas, cargasNovas, sizeof(float) * todosDispositivos);

	MPI_Allreduce(&localLatencia, &latencia, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&localBanda, &banda, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	double tempoFimProbing = MPI_Wtime();
	tempoBalanceamento += tempoFimProbing - tempoInicioProbing;
	fatorErro = tempoBalanceamento;
}


template<typename T, typename U>
void Balanceador<T,U>::ComputarCargas(const long int *ticks, const float *cargasAntigas, float *cargas, int participantes)
{
	if (participantes == 1)
	{
		cargas[0] = 1.0f;
		return;
	}

	float cargaTotal = 0.0f;
	for (int count = 0; count < participantes; count++)
	{
		cargaTotal += ((count == 0) ? (cargasAntigas[count] - 0.0f) : (cargasAntigas[count] - cargasAntigas[count - 1])) * ((count == 0) ? 1.0f : ((float)ticks[0]) / ((float)ticks[count]));
	}

	for (int count = 0; count < participantes; count++)
	{
		float cargaNova = (((count == 0) ? (cargasAntigas[count] - 0.0f) : (cargasAntigas[count] - cargasAntigas[count - 1])) * ((count == 0) ? 1.0f : ((float)ticks[0]) / ((float)ticks[count]))) / cargaTotal;
		cargas[count] = ((count == 0) ? cargaNova : cargas[count - 1] + cargaNova);
	}
}

template<typename T, typename U>
bool Balanceador<T,U>::ComputarIntersecao(int offset1, int length1, int offset2, int length2, int *intersecaoOffset, int *intersecaoLength)
{
	if (offset1 + length1 <= offset2)
	{
		return false;
	}

	if (offset1 + length1 > offset2 + length2)
	{
		*intersecaoOffset = offset2;
		*intersecaoLength = length2;
	}
	else
	{
		*intersecaoOffset = offset2;
		*intersecaoLength = (offset1 + length1) - offset2;
	}
	return true;
}

template<typename T, typename U>
int Balanceador<T,U>::RecuperarPosicaoHistograma(int *histograma, int tamanho, int indice)
{
	int offset = 0;
	for (int count = 0; count < tamanho; count++)
	{
		if (indice >= offset && indice < offset + histograma[count])
		{
			return count;
		}
		offset += histograma[count];
	}
	return -1;
}

template<typename T, typename U>
float Balanceador<T,U>::ComputarDesvioPadraoPercentual(const long int *ticks, int participantes)
{
	double media = 0.0;
	for (int count = 0; count < participantes; count++)
	{
		media += (double)ticks[count];
	}
	media /= (double)participantes;

	double variancia = 0.0;
	for (int count = 0; count < participantes; count++)
	{
		variancia += ((double)ticks[count] - media) * ((double)ticks[count] - media);
	}
	variancia /= (double)participantes;
	return sqrt(variancia) / media;
}

template<typename T, typename U>
float Balanceador<T,U>::ComputarNorma(const float *cargasAntigas, const float *cargasNovas, int participantes)
{
	float retorno = 0.0;
	for (int count = 0; count < participantes; count++)
	{
		retorno += (cargasAntigas[count] - cargasNovas[count]) * (cargasAntigas[count] - cargasNovas[count]);
	}
	return sqrt(retorno);
}


template<typename T, typename U>
void Balanceador<T,U>::PrecisaoBalanceamento() {
    // Precisao do balanceamento.

	
	for (int precisao = 0; precisao < PRECISAO_BALANCEAMENTO; precisao++)
	{
		
		// Computação.
		for (int count = 0; count < todosDispositivos; count++)
		{
			
			if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
			{
				if ((Iteration % 2) == 0)
				{
					
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, swapBufferDispositivo[count][0]);
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, swapBufferDispositivo[count][1]);
				}
				else
				{
					
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, swapBufferDispositivo[count][1]);
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, swapBufferDispositivo[count][0]);
				}

				kernelEventoDispositivo[count] = RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count], length[count], isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
			}
		}
	}

	
	// // Ticks.
	for (int count = 0; count < todosDispositivos; count++)
	{	cout<<count<<endl;
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{	
			SynchronizeCommandQueue(count - meusDispositivosOffset);
			
			ticks[count] += GetEventTaskTicks(count - meusDispositivosOffset, kernelEventoDispositivo[count]);
			
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





int main(int argc, char *argv[]) {
    // Inicializando dados de exemplo
    const size_t N = 1024;
    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
       
    }

    // Kernel information
    const std::string kernel_name = "kernel.cl";
    const std::string function_name = "vector_add";

    // Criando o balanceador
    Balanceador<float, float> balanceador(argc, argv, a, sizeof(float), N, b, N*sizeof(float), 128, 1, kernel_name, function_name);
    // Balanceador(int argc, char *argv[],data, Element_sz, N_Element, U *DTK, div_size,  interv,  _units_per_elements, kernelSource, kernelFunction, cl_device_type deviceType)
    // Inicializando dispositivos
    // balanceador.inicializaDispositivos();

    // Inicializando o kernel
    // balanceador.inicializaKernel();

    // Executando a computação
    balanceador.runKernelOperations(0);

    // Juntando os dados na memória principal
    balanceador.gatherData(c);

    // Verificando os resultados
    for (size_t i = 0; i < N; ++i) {
      //  if (c[i] != a[i] + b[i]) {
        //    std::cerr << "Erro na computação no índice " << i << std::endl;
          //  return -1;
       // }
       cout<<"a["<<i<<"] = "<<c[i]<<" ";
       if((i+1)%32 == 0 )
        cout<<endl;
    }

    std::cout << "Computação bem-sucedida!" << std::endl;

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}


