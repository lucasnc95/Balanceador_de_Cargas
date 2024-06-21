//#include "Balanceador.h"
#include "OpenCLWrapper.h"
#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <cstdlib>

// Função auxiliar para obter o tipo MPI correspondente ao tipo de dado C++
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
class Balanceador {
private:
    int world_size;
    int world_rank;
    int dispositivos;
    int meusDispositivosOffset;
    int meusDispositivosLength;
    int todosDispositivos;
    int *dispositivosLocal;
    int *dispositivosWorld;
    int offsetComputacao; 
    int lengthComputacao;
    T *Data;
    U *DataToKernel;
    T **SwapBuffer;
    long int *ticks;
    double *tempos;
    float *cargasNovas;
    float *cargasAntigas;
    int *DataToKernelDispositivo;
    int **swapBufferDispositivo;
    int *kernelDispositivo;
    int *dataEventoDispositivo;
    int *kernelEventoDispositivo;
    unsigned int *offset;
    unsigned long int *length;
    unsigned int Element_size;
    unsigned long int N_Elements;
    unsigned int DataToKernel_Size;
    double tempoInicio;
    double tempoComputacaoInterna;
    double tempoBalanceamento;
    double tempoTrocaBorda;
    double tempoComputacaoBorda;
    double tempoCB;
    double writeByte;
    double banda;
    double latencia;
    double fatorErro;
    bool HABILITAR_BENCHMARK;
    bool HABILITAR_ESTATICO;
    bool HABILITAR_DINAMICO;
    int CPU_WORK_GROUP_SIZE;
    int GPU_WORK_GROUP_SIZE;
    int PRECISAO_BALANCEAMENTO;
    unsigned int interv_balance;
    MPI_Datatype mpi_data_type; // Tipo MPI correspondente a T
    MPI_Datatype mpi_custom_type; // Tipo MPI customizado definido pelo usuário
    bool custom_type_set; // Flag para indicar se o tipo customizado foi definido
    MPI_Request receiveRequest;
    MPI_Request sendRequest;
public:
    Balanceador(int argc, char *argv[], T *data, const size_t Element_sz, const unsigned long int N_Element, U *DTK, const size_t div_size, const unsigned int interv);
    ~Balanceador();
    void setCustomDatatype(MPI_Datatype custom_type);
    void InicializaDispositivos(int meusDispositivosOffset, int meusDispositivosLength, int offsetComputacao, int lengthComputacao);
    void DistribuicaoUniformeDeCarga();
    void PrecisaoBalanceamento(int simulacao);
    void BalanceamentoDeCarga(int simulacao);
    void Probing(int simulacao);
    void ComputaKernel(int simulacao);
    void TrocaDeBordas(int simulacao);
    void computacaoDeBordas(int simulacao); 
    void ComputarCargas(const long int *ticks, const float *cargasAntigas, float *cargas, int participantes);
    bool ComputarIntersecao(int offset1, int length1, int offset2, int length2, int *intersecaoOffset, int *intersecaoLength);
    int RecuperarPosicaoHistograma(int *histograma, int tamanho, int indice);
    float ComputarDesvioPadraoPercentual(const long int *ticks, int participantes);
    float ComputarNorma(const float *cargasAntigas, const float *cargasNovas, int participantes);
    void InicializarLenghtOffset(unsigned int offsetComputacao, unsigned int lengthComputacao, int count);
    void computacaoInterna(int simulacao); 
    void run_multi_step_balancer();
};

template<typename T, typename U>
Balanceador<T,U>::Balanceador(int argc, char *argv[], T *data, const size_t Element_sz, const unsigned long int N_Element, U *DTK, const size_t div_size, const unsigned int interv)
    : Data(data), interv_balance(interv), custom_type_set(false) {
    // Inicialização do MPI
    MPI_Init(&argc, &argv);

    // Definições iniciais
    HABILITAR_BENCHMARK = false;
    HABILITAR_ESTATICO = true;
    HABILITAR_DINAMICO = false;
    CPU_WORK_GROUP_SIZE = 8;
    GPU_WORK_GROUP_SIZE = 64;
    PRECISAO_BALANCEAMENTO = 10;

    // Inicialização dos parâmetros
    Element_size = Element_sz;
    N_Elements = N_Element;
    DataToKernel_Size = div_size;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    mpi_data_type = GetMPIType<T>();

    int dispositivos = InitParallelProcessor();

    dispositivosLocal = new int[world_size];
    dispositivosWorld = new int[world_size];

    memset(dispositivosLocal, 0, sizeof(int) * world_size);
    dispositivosLocal[world_rank] = dispositivos;

    MPI_Allreduce(&dispositivosLocal, &dispositivosWorld, world_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    todosDispositivos = 0;
    for (int count = 0; count < world_size; count++) {
        if (count == world_rank) {
            meusDispositivosOffset = todosDispositivos;
            meusDispositivosLength = dispositivosWorld[count];
        }
        todosDispositivos += dispositivosWorld[count];
    }

    DataToKernel = new U;
    memcpy(DataToKernel, DTK, DataToKernel_Size);

    SwapBuffer = new T *[2];
    SwapBuffer[0] = new T[(Element_sz * N_Element)];
    SwapBuffer[1] = new T[(Element_sz * N_Element)];

    ticks = new long int[todosDispositivos];
    tempos = new double[todosDispositivos];
    cargasNovas = new float[todosDispositivos];
    cargasAntigas = new float[todosDispositivos];
    DataToKernelDispositivo = new int[todosDispositivos];
    swapBufferDispositivo = new int *[todosDispositivos];
    for (int i = 0; i < todosDispositivos; ++i) {
        swapBufferDispositivo[i] = new int[2];
    }
    kernelDispositivo = new int[todosDispositivos];
    dataEventoDispositivo = new int[todosDispositivos];
    kernelEventoDispositivo = new int[todosDispositivos];
    offset = new unsigned int[todosDispositivos];
    length = new unsigned long int[todosDispositivos];
    offset = new unsigned int[todosDispositivos];
    length = new unsigned long int[todosDispositivos];

    offsetComputacao = 0;
    lengthComputacao = (N_Elements) / todosDispositivos;

    InicializaDispositivos(meusDispositivosOffset, meusDispositivosLength, offsetComputacao, lengthComputacao);
    DistribuicaoUniformeDeCarga();
   // PrecisaoBalanceamento(0);
   // PrecisaoBalanceamento(1);
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
void Balanceador<T,U>::ComputaKernel(int simulacao)
{

	//double tempoInicio = MPI_Wtime();

	// Computação interna.
	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			kernelEventoDispositivo[count] = RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count]+ interv_balance, length[count] - interv_balance, isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
		}
	}

	// Sincronizacao da computação interna.
	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			SynchronizeCommandQueue(count - meusDispositivosOffset);
            ticks[count] += GetEventTaskTicks(count - meusDispositivosOffset, kernelEventoDispositivo[count]);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	tempoComputacaoInterna += MPI_Wtime() - tempoInicio;

	// Computação das bordas.
	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			if ((simulacao % 2) == 0)
			{
				SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, swapBufferDispositivo[count][0]);
				SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, swapBufferDispositivo[count][1]);
			}
			else
			{
				SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, swapBufferDispositivo[count][1]);
				SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, swapBufferDispositivo[count][0]);
			}

			RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count], interv_balance, isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
			RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count] + length[count] - interv_balance, interv_balance, isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
		}
	}

	// Sincronizacao da computação das borda
	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			SynchronizeCommandQueue(count - meusDispositivosOffset);
		}
	}
}






template<typename T, typename U>
void Balanceador<T,U>::InicializarLenghtOffset(unsigned int offsetComputacao, unsigned int lengthComputacao, int count)
{

	offset[count] = offsetComputacao;
	length[count] = lengthComputacao;
}

template<typename T, typename U>
Balanceador<T,U>::~Balanceador() {
    FinishParallelProcessor();
    MPI_Finalize();
}

template<typename T, typename U>
void Balanceador<T,U>::setCustomDatatype(MPI_Datatype custom_type) {
    mpi_custom_type = custom_type;
    custom_type_set = true;
}

template<typename T, typename U>
void Balanceador<T,U>::InicializaDispositivos(int meusDispositivosOffset, int meusDispositivosLength, int offsetComputacao, int lengthComputacao) {
    double localWriteByte = 0;

    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            InicializarLenghtOffset(offsetComputacao, (count + 1 == todosDispositivos) ? (N_Elements - offsetComputacao) : lengthComputacao, count);
            DataToKernelDispositivo[count] = CreateMemoryObject(count - meusDispositivosOffset, DataToKernel_Size, CL_MEM_READ_ONLY, NULL);
            size_t sizeCarga = Element_size * N_Elements;
            swapBufferDispositivo[count][0] = CreateMemoryObject(count - meusDispositivosOffset, sizeCarga, CL_MEM_READ_WRITE, NULL);
            swapBufferDispositivo[count][1] = CreateMemoryObject(count - meusDispositivosOffset, sizeCarga, CL_MEM_READ_WRITE, NULL);
            WriteToMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)DataToKernel, 0, DataToKernel_Size);
            SynchronizeCommandQueue(count - meusDispositivosOffset);
            tempoInicio = MPI_Wtime();
            WriteToMemoryObject(count - meusDispositivosOffset, swapBufferDispositivo[count][0], (char *)SwapBuffer[0], 0, sizeCarga);
            WriteToMemoryObject(count - meusDispositivosOffset, swapBufferDispositivo[count][1], (char *)SwapBuffer[1], 0, sizeCarga);
            double aux = (MPI_Wtime() - tempoInicio) / sizeCarga / 2;
            localWriteByte = aux > localWriteByte ? aux : localWriteByte;

            kernelDispositivo[count] = CreateKernel(count - meusDispositivosOffset, "kernels.cl", "ProcessarPontos");
            SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, swapBufferDispositivo[count][0]);
            SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, swapBufferDispositivo[count][1]);
            SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 2, DataToKernelDispositivo[count]);
        }

        offsetComputacao += lengthComputacao;
    }
    MPI_Allreduce(&localWriteByte, &writeByte, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

template<typename T, typename U>
void Balanceador<T,U>::PrecisaoBalanceamento(int simulacao) {
    // Precisao do balanceamento.
using namespace std;
	memset(ticks, 0, sizeof(long int) * todosDispositivos);
	memset(tempos, 0, sizeof(float) * todosDispositivos);

	for (int precisao = 0; precisao < PRECISAO_BALANCEAMENTO; precisao++)
	{
		std::cout << "Precisão: " << precisao << std::endl;
		// Computação.
		for (int count = 0; count < todosDispositivos; count++)
		{
			std::cout << "Count: " << count << std::endl;
			if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
			{
				if ((simulacao % 2) == 0)
				{
					std::cout << "If " << count << std::endl;
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, swapBufferDispositivo[count][0]);
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, swapBufferDispositivo[count][1]);
				}
				else
				{
					std::cout << "else " << count << std::endl;
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, swapBufferDispositivo[count][1]);
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, swapBufferDispositivo[count][0]);
				}

				kernelEventoDispositivo[count] = RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count], length[count], isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
			}
		}
	}

	cout<<"Ticks"<<endl;
	// // Ticks.
	for (int count = 0; count < todosDispositivos; count++)
	{	cout<<count<<endl;
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{	cout<<"if"<<endl;
			SynchronizeCommandQueue(count - meusDispositivosOffset);
			cout<<"offset disp = "<<meusDispositivosOffset<<endl;
			ticks[count] += GetEventTaskTicks(count - meusDispositivosOffset, kernelEventoDispositivo[count]);
			cout<<"tick"<<endl;
		}
	}

	// Reduzir ticks.
	std::cout << "redução Ticks" << std::endl;
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

template<typename T, typename U>
void Balanceador<T,U>::BalanceamentoDeCarga(int simulacao) {
    double tempoInicioBalanceamento = MPI_Wtime();
    double localTempoCB;

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
                            T *data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
                            int dataDevice = (simulacao % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
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
                                T *data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
                                int dataDevice[2] = {(simulacao % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1],
                                                     (simulacao % 2) == 0 ? swapBufferDispositivo[count2][0] : swapBufferDispositivo[count2][1]};

                                ReadFromMemoryObject(count2 - meusDispositivosOffset, dataDevice[1], (char *)(data + intersecaoOffset * Element_size), intersecaoOffset * Element_size * sizeof(float), intersecaoLength * Element_size * sizeof(float));
                                SynchronizeCommandQueue(count2 - meusDispositivosOffset);
                                WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(data + intersecaoOffset * Element_size), intersecaoOffset * Element_size * sizeof(float), intersecaoLength * Element_size * sizeof(float));
                                SynchronizeCommandQueue(count - meusDispositivosOffset);
                            } else {
                                if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2)) {
                                    int overlap[2] = {intersecaoOffset, intersecaoLength};
                                    int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
                                    T *data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
                                    int dataDevice = (simulacao % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
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
                                T *data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
                                MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
                            }
                        }
                    }
                }
                offset[count] = overlapNovoOffset;
                length[count] = overlapNovoLength;
                WriteToMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)DataToKernel, 0, sizeof(int) * 8);
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
void Balanceador<T,U>::DistribuicaoUniformeDeCarga()
{

	for (int count = 0; count < todosDispositivos; count++)
	{
		cargasNovas[count] = ((float)(count + 1)) * (1.0f / ((float)todosDispositivos));
		cargasAntigas[count] = cargasNovas[count];
		tempos[count] = 1;
	}
}






template<typename T, typename U>
void Balanceador<T,U>::TrocaDeBordas(int simulacao) {
    for (int passo = 0; passo < 4; passo++) {
        for (int count = 0; count < todosDispositivos; count++) {
            if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
                int tamanhoBorda = interv_balance;
                T *data;
                int dataDevice[2];
                int borda[2];
                int alvo;

                if (passo == 3) {
                    if (count == meusDispositivosOffset && count > 0) {
                        data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
                        dataDevice[0] = (simulacao % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
                        borda[0] = offset[count] - tamanhoBorda;
                        borda[0] = borda[0] < 0 ? 0 : borda[0];
                        borda[1] = offset[count];
                        alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count - 1);

                        if (alvo % 2 == 0) {
                            MPI_Irecv(data + borda[0] * Element_size, tamanhoBorda * Element_size, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, &receiveRequest);
                            dataEventoDispositivo[count] = ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(data + borda[1] * Element_size), borda[1] * Element_size * sizeof(float), tamanhoBorda * Element_size * sizeof(float));
                            SynchronizeCommandQueue(count - meusDispositivosOffset);
                            MPI_Isend(data + borda[1] * Element_size, tamanhoBorda * Element_size, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, &sendRequest);
                            MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
                            MPI_Wait(&receiveRequest, MPI_STATUS_IGNORE);
                            WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(data + borda[0] * Element_size), borda[0] * Element_size * sizeof(float), tamanhoBorda * Element_size * sizeof(float));
                            SynchronizeCommandQueue(count - meusDispositivosOffset);
                        }
                    }
                    if (count == meusDispositivosOffset + meusDispositivosLength - 1 && count < todosDispositivos - 1) {
                        data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
                        dataDevice[0] = (simulacao % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
                        borda[0] = (offset[count] + length[count]) - tamanhoBorda;
                        borda[0] = borda[0] < 0 ? 0 : borda[0];
                        borda[1] = offset[count] + length[count];
                        alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count + 1);

                        if (alvo % 2 == 1) {
                            dataEventoDispositivo[count] = ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(data + borda[1] * Element_size), borda[1] * Element_size * sizeof(float), tamanhoBorda * Element_size * sizeof(float));
                            SynchronizeCommandQueue(count - meusDispositivosOffset);
                            MPI_Isend(data + borda[1] * Element_size, tamanhoBorda * Element_size, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, &sendRequest);
                            MPI_Irecv(data + borda[0] * Element_size, tamanhoBorda * Element_size, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, &receiveRequest);
                            MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
                            MPI_Wait(&receiveRequest, MPI_STATUS_IGNORE);
                            WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(data + borda[0] * Element_size), borda[0] * Element_size * sizeof(float), tamanhoBorda * Element_size * sizeof(float));
                            SynchronizeCommandQueue(count - meusDispositivosOffset);
                        }
                    }
                }

                if (passo == 2) {
                    if (count == meusDispositivosOffset && count > 0) {
                        data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
                        dataDevice[0] = (simulacao % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
                        borda[0] = offset[count] - tamanhoBorda;
                        borda[0] = borda[0] < 0 ? 0 : borda[0];
                        borda[1] = offset[count];
                        alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count - 1);

                        if (alvo % 2 == 1) {
                            MPI_Irecv(data + borda[0] * Element_size, tamanhoBorda * Element_size, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, &receiveRequest);
                            dataEventoDispositivo[count] = ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(data + borda[1] * Element_size), borda[1] * Element_size * sizeof(float), tamanhoBorda * Element_size * sizeof(float));
                            SynchronizeCommandQueue(count - meusDispositivosOffset);
                            MPI_Isend(data + borda[1] * Element_size, tamanhoBorda * Element_size, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, &sendRequest);
                            MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
                            MPI_Wait(&receiveRequest, MPI_STATUS_IGNORE);
                            WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(data + borda[0] * Element_size), borda[0] * Element_size * sizeof(float), tamanhoBorda * Element_size * sizeof(float));
                            SynchronizeCommandQueue(count - meusDispositivosOffset);
                        }
                    }
                    if (count == meusDispositivosOffset + meusDispositivosLength - 1 && count < todosDispositivos - 1) {
                        data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
                        dataDevice[0] = (simulacao % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
                        borda[0] = (offset[count] + length[count]) - tamanhoBorda;
                        borda[0] = borda[0] < 0 ? 0 : borda[0];
                        borda[1] = offset[count] + length[count];
                        alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count + 1);

                        if (alvo % 2 == 0) {
                            dataEventoDispositivo[count] = ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(data + borda[1] * Element_size), borda[1] * Element_size * sizeof(float), tamanhoBorda * Element_size * sizeof(float));
                            SynchronizeCommandQueue(count - meusDispositivosOffset);
                            MPI_Isend(data + borda[1] * Element_size, tamanhoBorda * Element_size, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, &sendRequest);
                            MPI_Irecv(data + borda[0] * Element_size, tamanhoBorda * Element_size, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, &receiveRequest);
                            MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
                            MPI_Wait(&receiveRequest, MPI_STATUS_IGNORE);
                            WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(data + borda[0] * Element_size), borda[0] * Element_size * sizeof(float), tamanhoBorda * Element_size * sizeof(float));
                            SynchronizeCommandQueue(count - meusDispositivosOffset);
                        }
                    }
                }

                if (passo == 0 && count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength - 1) {
                    data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
                    dataDevice[0] = (simulacao % 2) == 0 ? swapBufferDispositivo[count + 0][0] : swapBufferDispositivo[count + 0][1];
                    dataDevice[1] = (simulacao % 2) == 0 ? swapBufferDispositivo[count + 1][0] : swapBufferDispositivo[count + 1][1];
                    borda[0] = offset[count + 1] - tamanhoBorda;
                    borda[0] = borda[0] < 0 ? 0 : borda[0];
                    borda[1] = offset[count + 1];

                    dataEventoDispositivo[count + 0] = ReadFromMemoryObject(count + 0 - meusDispositivosOffset, dataDevice[0], (char *)(data + borda[0] * Element_size), borda[0] * Element_size * sizeof(float), tamanhoBorda * Element_size * sizeof(float));
                    SynchronizeCommandQueue(count + 0 - meusDispositivosOffset);

                    dataEventoDispositivo[count + 1] = ReadFromMemoryObject(count + 1 - meusDispositivosOffset, dataDevice[1], (char *)(data + borda[1] * Element_size), borda[1] * Element_size * sizeof(float), tamanhoBorda * Element_size * sizeof(float));
                    SynchronizeCommandQueue(count + 1 - meusDispositivosOffset);
                }

                if (passo == 1 && count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength - 1) {
                    data = (simulacao % 2) == 0 ? SwapBuffer[0] : SwapBuffer[1];
                    dataDevice[0] = (simulacao % 2) == 0 ? swapBufferDispositivo[count + 0][0] : swapBufferDispositivo[count + 0][1];
                    dataDevice[1] = (simulacao % 2) == 0 ? swapBufferDispositivo[count + 1][0] : swapBufferDispositivo[count + 1][1];
                    borda[0] = offset[count + 1] - tamanhoBorda;
                    borda[0] = borda[0] < 0 ? 0 : borda[0];
                    borda[1] = offset[count + 1];

                    WriteToMemoryObject(count + 0 - meusDispositivosOffset, dataDevice[0], (char *)(data + borda[1] * Element_size), borda[1] * Element_size * sizeof(float), tamanhoBorda * Element_size * sizeof(float));
                    SynchronizeCommandQueue(count + 0 - meusDispositivosOffset);
                    WriteToMemoryObject(count + 1 - meusDispositivosOffset, dataDevice[1], (char *)(data + borda[0] * Element_size), borda[0] * Element_size * sizeof(float), tamanhoBorda * Element_size * sizeof(float));
                    SynchronizeCommandQueue(count + 1 - meusDispositivosOffset);
                }
            }
        }
    }

    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            SynchronizeCommandQueue(count - meusDispositivosOffset);
        }
    }
}



template<typename T, typename U>
void Balanceador<T,U>::computacaoInterna(int simulacao) {
    
    double tempoInicio = MPI_Wtime();
    

    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count] + interv_balance, length[count] - interv_balance, isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
        }
    }

    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            SynchronizeCommandQueue(count - meusDispositivosOffset);
        }
    }

    
    MPI_Barrier(MPI_COMM_WORLD);
    double tempoFim = MPI_Wtime();
    tempoComputacaoInterna += tempoFim - tempoInicio;
   
}






template<typename T, typename U>
void Balanceador<T,U>::computacaoDeBordas(int simulacao) {
    #ifdef HABILITAR_BENCHMARK
    double tempoInicio = MPI_Wtime();
    #endif

    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            if ((simulacao % 2) == 0) {
                SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, swapBufferDispositivo[count][0]);
                SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, swapBufferDispositivo[count][1]);
            } else {
                SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, swapBufferDispositivo[count][1]);
                SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, swapBufferDispositivo[count][0]);
            }

            RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count], interv_balance, isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
            RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], offset[count] + length[count] - interv_balance, interv_balance, isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
        }
    }

    for (int count = 0; count < todosDispositivos; count++) {
        if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength) {
            SynchronizeCommandQueue(count - meusDispositivosOffset);
        }
    }

    #ifdef HABILITAR_BENCHMARK
    MPI_Barrier(MPI_COMM_WORLD);
    double tempoFim = MPI_Wtime();
    tempoComputacaoBorda += tempoFim - tempoInicio;
    #endif
}











template<typename T, typename U>
void Balanceador<T,U>::run_multi_step_balancer(){

bool balanceamento = true;
int SIMULACOES = 1000;    
int INTERVALO_BALANCEAMENTO  = 100;

    for (int simulacao = 0; simulacao < SIMULACOES; simulacao++) {
        // Balanceamento de carga
        if (balanceamento && ((simulacao == 0) || (simulacao == 1) || (simulacao % INTERVALO_BALANCEAMENTO == 0))) {
            if (HABILITAR_ESTATICO)
            if (simulacao > 1) {
                balanceamento = false;
            }
            

            BalanceamentoDeCarga(simulacao);
        } else {
            computacaoInterna(simulacao);
            TrocaDeBordas(simulacao);
            //computacaoDeBordas(simulacao);
        }
    }

    
    }






using namespace std;

#define CELULA_A 0
#define CELULA_MR 1
#define CELULA_MA 2
#define CELULA_N 3
#define CELULA_CH 4
#define CELULA_ND 5
#define CELULA_G 6
#define CELULA_CA 7
#define MALHA_TOTAL_CELULAS 8

#define OFFSET_COMPUTACAO 0
#define LENGTH_COMPUTACAO 1
#define COMPRIMENTO_GLOBAL_X 2
#define COMPRIMENTO_GLOBAL_Y 3
#define COMPRIMENTO_GLOBAL_Z 4
#define MALHA_DIMENSAO_POSICAO_Z 5
#define MALHA_DIMENSAO_POSICAO_Y 6
#define MALHA_DIMENSAO_POSICAO_X 7
#define MALHA_DIMENSAO_CELULAS 8
#define NUMERO_PARAMETROS_MALHA 9


void LerPontosHIS(const float *malha, const int *parametrosMalha)
{
	for(unsigned int x = 0; x < parametrosMalha[COMPRIMENTO_GLOBAL_X]; x++)
	{
		for(unsigned int y = 0; y < parametrosMalha[COMPRIMENTO_GLOBAL_Y]; y++)
		{
			for(unsigned int z = 0; z < parametrosMalha[COMPRIMENTO_GLOBAL_Z]; z++)
			{
				if((CELULA_A * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X]) >= parametrosMalha[OFFSET_COMPUTACAO]*MALHA_TOTAL_CELULAS && (CELULA_A * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X]) < (parametrosMalha[OFFSET_COMPUTACAO]+parametrosMalha[LENGTH_COMPUTACAO])*MALHA_TOTAL_CELULAS)
				{
					printf("%f ", malha[(CELULA_A * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X])]);
				}
				else
				{
					printf("%f ", 0.0f);
				}
			}
			printf("\n");
		}
	}
}




int main(int argc, char *argv[])
{

	int *parametrosMalha = new int[9];

	(parametrosMalha)[0] = 0;
	(parametrosMalha)[1] = 0;
	(parametrosMalha)[2] = 50;
	(parametrosMalha)[3] = 50;
	(parametrosMalha)[4] = 320;
	(parametrosMalha)[5] = 50*50*8;
	(parametrosMalha)[6] = 50*MALHA_TOTAL_CELULAS;
	(parametrosMalha)[7] = MALHA_TOTAL_CELULAS;
	(parametrosMalha)[8] = 1;

	// float *malha = (float*)malloc(parametrosMalha[7] * parametrosMalha[6] * parametrosMalha[5] * 8 * sizeof(float));
	float *malha = new float[parametrosMalha[COMPRIMENTO_GLOBAL_X] * parametrosMalha[COMPRIMENTO_GLOBAL_Y] * parametrosMalha[COMPRIMENTO_GLOBAL_Z] * MALHA_TOTAL_CELULAS];
	float x = (parametrosMalha[7] * parametrosMalha[6] * parametrosMalha[5] * 8 * sizeof(float));
	for (unsigned int x = 0; x < parametrosMalha[COMPRIMENTO_GLOBAL_X]; x++)
	{
		for (unsigned int y = 0; y < parametrosMalha[COMPRIMENTO_GLOBAL_Y]; y++)
		{
			for (unsigned int z = 0; z < parametrosMalha[COMPRIMENTO_GLOBAL_Z]; z++)
			{
				if (z >= (0.75f * parametrosMalha[COMPRIMENTO_GLOBAL_Z]))
				{
					(malha)[(CELULA_A * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 100.0f;
				}
				else
				{
					(malha)[(CELULA_A * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
				}

				(malha)[(CELULA_MR * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 1.0f;
				(malha)[(CELULA_MA * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
				(malha)[(CELULA_N * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
				(malha)[(CELULA_CH * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
				(malha)[(CELULA_ND * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
				(malha)[(CELULA_G * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
				(malha)[(CELULA_CA * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
			}
		}
	}

	



	const size_t N = 50*50*320;
	const size_t Element_sz = sizeof(float);
	const size_t div_size = 9 * sizeof(int);
    const unsigned int interv_balance = 50*50;
	cout << "N = " << N << endl;
	cout << "Element_sz = " << Element_sz << endl;
	cout << "Malha_sz = " << sizeof(malha[0]) * N << endl;
	
	//LerPontosHIS(malha, parametrosMalha);
	//Balanceador<float, int> balancer(argc, argv, malha, Element_sz, N, parametrosMalha, div_size);
    Balanceador<float, int> balanceador(argc, argv, malha, Element_sz, N, parametrosMalha, div_size, interv_balance);
	cout<<"Running multi step..."<<endl;
    balanceador.run_multi_step_balancer();
	
	LerPontosHIS(malha, parametrosMalha);
	cout << "Fim do programa..." << endl;

	return 0;
}