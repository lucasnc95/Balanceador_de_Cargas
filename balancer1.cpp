#include <iostream>
#include <vector>
#include <string>
#include <CL/cl.h>
#include <mpi.h>
#include "OpenCLWrapper.h"

template<typename T, typename U>
class Balanceador {
public:
    Balanceador(int argc, char *argv[], T* data, size_t element_size, unsigned long n_elements, unsigned int interv_balance, unsigned long units_per_elements);
    ~Balanceador();

    void initMPI(int argc, char *argv[]);
    void createKernel(const std::string& kernelSource, const std::string& kernelFunction);
    void initializeDevices();
    void initializeKernel();
    void setKernelAttribute(int devicePosition, int kernelID, int attribute, int memoryObjectID);
    void computaKernel(int kernelID);
    void gatherData(T* data_dest);
    void setCustomDatatype(MPI_Datatype custom_type);
    void balanceLoad();
    void Probing();

private:
    T* Data;
    size_t Element_size;
    unsigned long N_Elements;
    unsigned int interv_balance;
    unsigned long units_per_elements;
    bool custom_type_set;
    bool kernel_set;
    unsigned int Iterations;
    std::vector<cl_device_id> devices;
    std::vector<cl_context> contexts;
    std::vector<cl_program> programs;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> commandQueues;
    std::vector<cl_mem> memObjects;
    cl_event kernelEvento;
    MPI_Datatype mpi_custom_type;
    int world_size, world_rank, meusDispositivosOffset, meusDispositivosLength, todosDispositivos;
    int* dispositivosWorld;
    unsigned int* offset;
    unsigned long* length;
    long int* ticks;
    double* tempos_por_carga;
    float* cargasNovas;
    float* cargasAntigas;
    double tempoBalanceamento;
    double tempoComputacaoInterna;
    double tempoInicio;
    double banda, writeByte, latencia, tempoCB, fatorErro;

    void PrecisaoBalanceamento();
    void InicializarLenghtOffset(unsigned int offsetComputacao, unsigned int lengthComputacao, int count);
    void ComputarCargas(const long int *ticks, const float *cargasAntigas, float *cargas, int participantes);
    bool ComputarIntersecao(int offset1, int length1, int offset2, int length2, int *intersecaoOffset, int *intersecaoLength);
    int RecuperarPosicaoHistograma(int *histograma, int tamanho, int indice);
    float ComputarDesvioPadraoPercentual(const long int *ticks, int participantes);
    float ComputarNorma(const float *cargasAntigas, const float *cargasNovas, int participantes);
    void SynchronizeCommandQueue(int devicePosition);
};



#include "Balanceador.h"

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
Balanceador<T, U>::Balanceador(int argc, char *argv[], T* data, size_t element_size, unsigned long n_elements, unsigned int interv_balance, unsigned long units_per_elements)
    : Data(data), Element_size(element_size), N_Elements(n_elements), interv_balance(interv_balance), units_per_elements(units_per_elements), custom_type_set(false), kernel_set(false), Iterations(0) {
    initMPI(argc, argv);
    initializeDevices();
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
        cl_context context = contexts[i];
        cl_device_id device = devices[i];

        cl_program program = clCreateProgramWithSource(context, 1, (const char**)&sourceStr, &sourceSize, &status);
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Failed to create program");
        }

        status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
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

    for (int count = 0; count < world_size; count++) {
        if (count == world_rank) {
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
    int dataDevice[todosDispositivos];
   
    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            InicializarLenghtOffset(offsetComputacao, (count + 1 == todosDispositivos) ? (N_Elements - offsetComputacao) : lengthComputacao, count);           
            dataDevice[count] = CreateMemoryObject(count - meusDispositivosOffset, Element_size * N_Elements * units_per_elements, CL_MEM_READ_WRITE, NULL);
        }

        offsetComputacao += lengthComputacao;
    }
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
void Balanceador<T, U>::computaKernel(int kernelID) {
    cl_int status;
    for (size_t i = 0; i < devices.size(); ++i) {
        size_t globalWorkSize = N_Elements;
        status = clEnqueueNDRangeKernel(commandQueues[i], kernels[i], 1, NULL, &globalWorkSize, NULL, 0, NULL, &kernelEvento);
        if (status != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue kernel");
        }
    }

    for (size_t i = 0; i < devices.size(); ++i) {
        clFinish(commandQueues[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    synchronizeAndCollectExecutionTimes();
}

template<typename T, typename U>
void Balanceador<T, U>::setKernelAttribute(int devicePosition, int kernelID, int attribute, int memoryObjectID) {
    int kernelPosition = kernelID;
    if (kernelPosition != -1) {
        cl_int state = clSetKernelArg(kernels[devicePosition], attribute, sizeof(cl_mem), (void *)&memObjects[memoryObjectID]);
        if (state != CL_SUCCESS) {
            throw std::runtime_error("Error setting kernel argument");
        }
    } else {
        throw std::runtime_error("Kernel ID or Memory Object not found");
    }
}

template<typename T, typename U>
void Balanceador<T, U>::gatherData(T* data_dest) {
    for (size_t i = 0; i < devices.size(); ++i) {
        clEnqueueReadBuffer(commandQueues[i], memObjects[i], CL_TRUE, 0,  Element_size * N_Elements * units_per_elements, Data, 0, NULL, NULL);
    }

    for (size_t i = 0; i < devices.size(); ++i) {
        clFinish(commandQueues[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

template<typename T, typename U>
void Balanceador<T, U>::setCustomDatatype(MPI_Datatype custom_type) {
    mpi_custom_type = custom_type;
    custom_type_set = true;
}

template<typename T, typename U>
Balanceador<T,U>::~Balanceador() {
    FinishParallelProcessor();
    MPI_Finalize();
}

template<typename T, typename U>
void Balanceador<T, U>::PrecisaoBalanceamento() {
    for (int precisao = 0; precisao < PRECISAO_BALANCEAMENTO; precisao++) {
        for (int count = 0; count < todosDispositivos; count++) {
            if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
                kernelEventoDispositivo[count] = RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count], length[count], isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
            }
        }
    }

    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            SynchronizeCommandQueue(count - meusDispositivosOffset);
            ticks[count] += GetEventTaskTicks(count - meusDispositivosOffset, kernelEventoDispositivo[count]);
        }
    }

    long int ticks_root[todosDispositivos];
    MPI_Allreduce(ticks, ticks_root, todosDispositivos, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    memcpy(ticks, ticks_root, sizeof(long int) * todosDispositivos);
    ComputarCargas(ticks, cargasAntigas, cargasNovas, todosDispositivos);
    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            SynchronizeCommandQueue(count - meusDispositivosOffset);
            tempos[count] = ((float)ticks[count]) / ((float)cargasNovas[count]);
        }
    }
    float tempos_root[todosDispositivos];
    MPI_Allreduce(tempos, tempos_root, todosDispositivos, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    memcpy(tempos, tempos_root, sizeof(float) * todosDispositivos);
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
                            T *data = Data;
                            int dataDevice = dataDevice[count];
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
                                T *data = Data;
                                int dataDevice[2] = {dataDevice[count], dataDevice[count2]};

                                ReadFromMemoryObject(count2 - meusDispositivosOffset, dataDevice[1], (char *)(data + (intersecaoOffset * units_per_elements)), intersecaoOffset * units_per_elements * sizeof(float), intersecaoLength * units_per_elements * sizeof(float));
                                SynchronizeCommandQueue(count2 - meusDispositivosOffset);
                                WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(data + (intersecaoOffset * units_per_elements)), intersecaoOffset * units_per_elements * sizeof(float), intersecaoLength * units_per_elements * sizeof(float));
                                SynchronizeCommandQueue(count - meusDispositivosOffset);
                            } else {
                                if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
                                    int overlap[2] = {intersecaoOffset, intersecaoLength};
                                    int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
                                    T *data = Data;
                                    int dataDevice = dataDevice[count];
                                    MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
                                    MPI_Recv(data + (overlap[0] * units_per_elements), overlap[1] * units_per_elements, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                    WriteToMemoryObject(count - meusDispositivosOffset, dataDevice, (char *)(data + (overlap[0] * units_per_elements)), overlap[0] * units_per_elements * sizeof(float), overlap[1] * units_per_elements * sizeof(float));
                                    SynchronizeCommandQueue(count - meusDispositivosOffset);
                                }
                            }
                        } else {
                            if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
                                int overlap[2] = {0, 0};
                                int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
                                T *data = Data;
                                MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
                            }
                        }
                    }
                }
                offset[count] = overlapNovoOffset;
                length[count] = overlapNovoLength;
                WriteToMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)Data, 0, sizeof(int) * units_per_elements);
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
void Balanceador<T, U>::Probing() {
    double tempoInicioProbing = MPI_Wtime();
    double localLatencia = 0, localBanda = 0;
    PrecisaoBalanceamento();

    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            int overlapNovoOffset = ((int)(((count == 0) ? 0.0f : cargasNovas[count - 1]) * ((float)(N_Elements))));
            int overlapNovoLength = ((int)(((count == 0) ? cargasNovas[count] - 0.0f : cargasNovas[count] - cargasNovas[count - 1]) * ((float)(N_Elements))));
            for (int count2 = 0; count2 < todosDispositivos; count2++) {
                if (count > count2) {
                    if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
                        int overlap[2];
                        int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
                        T *Data = Data;
                        int dataDevice = dataDevice[count];
                        MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        if (overlap[1] > 0) {
                            ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice, (char *)(Data + (overlap[0] * units_per_elements)), overlap[0] * units_per_elements * sizeof(float), overlap[1] * units_per_elements * sizeof(float));
                            SynchronizeCommandQueue(count - meusDispositivosOffset);
                            sizeCarga = overlap[1] * units_per_elements;
                            double tempoInicioBanda = MPI_Wtime();
                            MPI_Ssend(Data + (overlap[0] * units_per_elements), sizeCarga, MPI_FLOAT, alvo, 0, MPI_COMM_WORLD);
                            double aux = (MPI_Wtime() - tempoInicioBanda) / sizeCarga;
                            localBanda = aux > localBanda ? aux : localBanda;
                        }
                    }
                } else if (count < count2) {
                    int overlapAntigoOffset = ((int)(((count2 == 0) ? 0 : cargasAntigas[count2 - 1]) * (N_Elements)));
                    int overlapAntigoLength = ((int)(((count2 == 0) ? cargasAntigas[count] - 0.0f : cargasAntigas[count2] - cargasAntigas[count2 - 1]) * (N_Elements)));

                    int intersecaoOffset;
                    int intersecaoLength;

                    if (((overlapAntigoOffset <= overlapNovoOffset - (interv_balance)) && ComputarIntersecao(overlapAntigoOffset, overlapAntigoLength, overlapNovoOffset - (interv_balance), overlapNovoLength + (interv_balance), &intersecaoOffset, &intersecaoLength)) ||
                            ((overlapAntigoOffset > overlapNovoOffset - (interv_balance)) && ComputarIntersecao(overlapNovoOffset - (interv_balance), overlapNovoLength + (interv_balance), overlapAntigoOffset, overlapAntigoLength, &intersecaoOffset, &intersecaoLength))) {
                        if (count2 >= meusDispositivosOffset && count2 < meusDispositivosOffset + meusDispositivosLength) {
                            T *Data = Data;
                            int dataDevice[2] = {dataDevice[count], dataDevice[count2]};
                            ReadFromMemoryObject(count2 - meusDispositivosOffset, dataDevice[1], (char *)(Data + (intersecaoOffset * units_per_elements)), intersecaoOffset * units_per_elements * sizeof(float), intersecaoLength * units_per_elements * sizeof(float));
                            SynchronizeCommandQueue(count2 - meusDispositivosOffset);
                            WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(Data + (intersecaoOffset * units_per_elements)), intersecaoOffset * units_per_elements * sizeof(float), intersecaoLength * units_per_elements * sizeof(float));
                            SynchronizeCommandQueue(count - meusDispositivosOffset);
                        } else {
                            if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
                                int overlap[2] = {intersecaoOffset, intersecaoLength};
                                int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
                                T *Data = Data;
                                int dataDevice = dataDevice[count];
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
                    } else {
                        if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
                            int overlap[2] = {0, 0};
                            int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
                            T *Data = Data;
                            MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
                        }
                    }
                }
            }

            offset[count] = overlapNovoOffset;
            length[count]= overlapNovoLength;
            WriteToMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)Data, 0, sizeof(int) * units_per_elements);
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
void Balanceador<T, U>::SynchronizeCommandQueue(int devicePosition) {
    clFinish(commandQueues[devicePosition]);
}

using namespace std;
template<typename T, typename U>

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
Balanceador<T, U>::Balanceador(int argc, char *argv[], T* data, const size_t Element_sz, const unsigned long int N_Element, const unsigned int interv, const unsigned long int _units_per_elements)
    : Data(data), Element_size(Element_sz), N_Elements(N_Element), interv_balance(interv), units_per_elements(_units_per_elements), custom_type_set(false), kernel_set(false), Iterations(0) {
    cout<<"Init MPI"<<endl;
    initMPI(argc, argv);
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
void Balanceador<T, U>::initializeDevices(const std::string& probing, int iterations) {
    int dispositivos = InitParallelProcessor();
    dispositivosWorld = new int[world_size];
	int dispositivosLocal[world_size];
	memset(dispositivosLocal, 0, sizeof(int) * world_size);
	dispositivosLocal[world_rank] = dispositivos;
	MPI_Allreduce(dispositivosLocal, dispositivosWorld, world_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	todosDispositivos = 0;
	
	for (int count = 0; count < world_size; count++) {
		if (count == world_rank) {
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
    
    memset(ticks, 0, sizeof(long int) * todosDispositivos);
    memset(tempos_por_carga, 0, sizeof(double) * todosDispositivos);
    memset(cargasNovas, 0, sizeof(float) * todosDispositivos);
    memset(cargasAntigas, 0, sizeof(float) * todosDispositivos);
    offset = new unsigned int[todosDispositivos];
    length = new unsigned long int[todosDispositivos];

    double localWriteByte = 0;
    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            InicializarLenghtOffset(offsetComputacao, (count + 1 == todosDispositivos) ? (N_Elements - offsetComputacao) : lengthComputacao, count);
            DataToKernelDispositivo[count] = CreateMemoryObject(count - meusDispositivosOffset, Element_size * N_Elements, CL_MEM_READ_WRITE, NULL);
            WriteToMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)Data, 0, Element_size * N_Elements);
            SynchronizeCommandQueue(count - meusDispositivosOffset);
            double aux = (MPI_Wtime() - tempoInicio) / Element_size * N_Elements / 2;
            localWriteByte = aux > localWriteByte ? aux : localWriteByte;
        }
        offsetComputacao += lengthComputacao;
    }
    MPI_Allreduce(&localWriteByte, &writeByte, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (probing == "none") {
        DistribuicaoUniformeDeCarga();
    } else if (probing == "generic") {
        // Implementar lógica de probing genérico, como multiplicação de matrizes
    } else if (probing == "true") {
        Iterations = iterations;
        Probing();
    }
}

template<typename T, typename U>
inline void Balanceador<T,U>::InicializarLenghtOffset(unsigned int offsetComputacao, unsigned int lengthComputacao, int count) {
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
void Balanceador<T, U>::gatherData(T* data_dest) {
    for (size_t i = 0; i < devices.size(); ++i) {
        clEnqueueReadBuffer(commandQueue, DataToKernelDispositivo[i], CL_TRUE, 0, Element_size * N_Elements * units_per_elements, Data, 0, NULL, &dataEvento);
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
                            T *data = Data;
                            MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            if (overlap[1] > 0) {
                                ReadFromMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)(data + overlap[0] * Element_size), overlap[0] * Element_size * sizeof(T), overlap[1] * Element_size * sizeof(T));
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
                                T *data = Data;
                                int dataDevice[2] = {DataToKernelDispositivo[count], DataToKernelDispositivo[count2]};

                                ReadFromMemoryObject(count2 - meusDispositivosOffset, dataDevice[1], (char *)(data + intersecaoOffset * Element_size), intersecaoOffset * Element_size * sizeof(T), intersecaoLength * Element_size * sizeof(T));
                                SynchronizeCommandQueue(count2 - meusDispositivosOffset);
                                WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(data + intersecaoOffset * Element_size), intersecaoOffset * Element_size * sizeof(T), intersecaoLength * Element_size * sizeof(T));
                                SynchronizeCommandQueue(count - meusDispositivosOffset);
                            } else {
                                if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
                                    int overlap[2] = {intersecaoOffset, intersecaoLength};
                                    int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
                                    T *data = Data;
                                    MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
                                    MPI_Recv(data + overlap[0] * Element_size, overlap[1] * Element_size, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                    WriteToMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)(data + overlap[0] * Element_size), overlap[0] * Element_size * sizeof(T), overlap[1] * Element_size * sizeof(T));
                                    SynchronizeCommandQueue(count - meusDispositivosOffset);
                                }
                            }
                        } else {
                            if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
                                int overlap[2] = {0, 0};
                                int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
                                T *data = Data;
                                MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
                            }
                        }
                    }
                }
                offset[count] = overlapNovoOffset;
                length[count] = overlapNovoLength;
                WriteToMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)Data, 0, sizeof(int) * units_per_elements);
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
void Balanceador<T,U>::Probing() {
    double tempoInicioProbing = MPI_Wtime();
	double localLatencia = 0, localBanda = 0;
	PrecisaoBalanceamento();

	for (int count = 0; count < todosDispositivos; count++) {
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
			int overlapNovoOffset = ((int)(((count == 0) ? 0.0f : cargasNovas[count - 1]) * ((float)(N_Elements))));
			int overlapNovoLength = ((int)(((count == 0) ? cargasNovas[count] - 0.0f : cargasNovas[count] - cargasNovas[count - 1]) * ((float)(N_Elements))));
			for (int count2 = 0; count2 < todosDispositivos; count2++) {
				if (count > count2) {
					if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
						int overlap[2];
						int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
						T *Data = Data;
						int dataDevice = DataToKernelDispositivo[count];
						MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						if (overlap[1] > 0) {
							ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice, (char *)(Data + (overlap[0] * units_per_elements)), overlap[0] * units_per_elements * sizeof(T), overlap[1] * units_per_elements * sizeof(T));
							SynchronizeCommandQueue(count - meusDispositivosOffset);

							sizeCarga = overlap[1] * units_per_elements;

							double tempoInicioBanda = MPI_Wtime();
							MPI_Ssend(Data + (overlap[0] * units_per_elements), sizeCarga, mpi_data_type, alvo, 0, MPI_COMM_WORLD);
							double aux = (MPI_Wtime() - tempoInicioBanda) / sizeCarga;
							localBanda = aux > localBanda ? aux : localBanda;
						}
					}
				} else if (count < count2) {
					int overlapAntigoOffset = ((int)(((count2 == 0) ? 0 : cargasAntigas[count2 - 1]) * (N_Elements)));
					int overlapAntigoLength = ((int)(((count2 == 0) ? cargasAntigas[count2] - 0.0f : cargasAntigas[count2] - cargasAntigas[count2 - 1]) * (N_Elements)));

					int intersecaoOffset;
					int intersecaoLength;

					if (((overlapAntigoOffset <= overlapNovoOffset - (interv_balance)) && ComputarIntersecao(overlapAntigoOffset, overlapAntigoLength, overlapNovoOffset - (interv_balance), overlapNovoLength + (interv_balance), &intersecao
