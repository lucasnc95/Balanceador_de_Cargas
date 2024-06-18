#include "Balanceador.h"
#include "OpenCLWrapper.h"
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <iostream>

using namespace std;

template<typename T, typename U>
Balanceador<T,U>::Balanceador(int argc, char *argv[], T *data, const size_t Element_sz, const unsigned long int N_Element, U *DTK, const size_t div_size)
{

	MPI_Init(&argc, &argv);

	HABILITAR_BENCHMARK = false;
	HABILITAR_ESTATICO = true;
	HABILITAR_DINAMICO = false;

	CPU_WORK_GROUP_SIZE = 8;
	GPU_WORK_GROUP_SIZE = 64;
	PRECISAO_BALANCEAMENTO = 10;

	Element_size = Element_sz;
	N_Elements = N_Element;
	DataToKernel_Size = div_size;

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int dispositivos = InitParallelProcessor();

	dispositivosLocal = new int[world_size];
	dispositivosWorld = new int[world_size];

	memset(dispositivosLocal, 0, sizeof(int) * world_size);

	dispositivosLocal[world_rank] = dispositivos;

	MPI_Allreduce(&dispositivosLocal, &dispositivosWorld, world_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

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
	Data = new T[8*8*8*8];
  	DataToKernel = new U[8];


	//DataToKernel = malloc(sizeof(int)*8);
	memcpy(DataToKernel, DTK, (sizeof(int)*8));

	//Data = malloc(Element_sz * N_Element);
	memcpy(Data, data, Element_sz * N_Element);
	
	
	cout << "Disp offset = " << meusDispositivosOffset << endl;
	SwapBuffer = new T *[2];

	SwapBuffer[0] = new T[(Element_sz * N_Element)];
	//memcpy(SwapBuffer[0], data, Element_sz * N_Element);
	SwapBuffer[1] = new T[(Element_sz * N_Element)];
	//memcpy(SwapBuffer[1], data, Element_sz * N_Element);

	ticks = new long int[todosDispositivos];
	tempos = new double[todosDispositivos];
	cargasNovas = new float[todosDispositivos];
	cargasAntigas = new float[todosDispositivos];
	DataToKernelDispositivo = new int[todosDispositivos];
	SwapBufferDispositivo = new int *[todosDispositivos];
	for (int i = 0; i < todosDispositivos; ++i)
	{
		SwapBufferDispositivo[i] = new int[2];
	}
	kernelDispositivo = new int[todosDispositivos];
	dataEventoDispositivo = new int[todosDispositivos];
	kernelEventoDispositivo = new int[todosDispositivos];
	OFFSET_COMPUTACAO = new unsigned int[todosDispositivos];
	LENGTH_COMPUTACAO = new unsigned long int[todosDispositivos];

	offsetComputacao = 0;
	lengthComputacao = (N_Elements) / todosDispositivos;

	InicializaDispositivos(meusDispositivosOffset, meusDispositivosLength, offsetComputacao, lengthComputacao);

	DistribuicaoUniformeDeCarga();
	cout << "Balanceamento 1" << endl;
	 PrecisaoBalanceamento(0);
	cout << "Balanceamento 2" << endl;
	PrecisaoBalanceamento(1);
}

template<typename T, typename U>
Balanceador<T,U>::~Balanceador()
{
	FinishParallelProcessor();
	MPI_Finalize();
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
void Balanceador<T,U>::InicializarLenghtOffset(unsigned int offsetComputacao, unsigned int lengthComputacao, int count)
{

	OFFSET_COMPUTACAO[count] = offsetComputacao;
	LENGTH_COMPUTACAO[count] = lengthComputacao;
}


template<typename T, typename U>
void Balanceador<T,U>::InicializaDispositivos(int meusDispositivosOffset, int meusDispositivosLenght, int offsetComputacao, int lengthComputacao)
{
	double localWriteByte = 0;

	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{

			cout << "Disp offset = " << meusDispositivosOffset << endl;
			InicializarLenghtOffset(offsetComputacao, (count + 1 == todosDispositivos) ? (N_Elements)-offsetComputacao : lengthComputacao, count);
			DataToKernelDispositivo[count] = CreateMemoryObject(count - meusDispositivosOffset, DataToKernel_Size, CL_MEM_READ_ONLY, NULL);
			cout << "N_Elements: " << N_Elements << " Element_size: " << Element_size << " DatatoKernel size: " << DataToKernel_Size << endl;
			sizeCarga = Element_size * N_Elements;
			SwapBufferDispositivo[count][0] = CreateMemoryObject(count - meusDispositivosOffset, sizeCarga, CL_MEM_READ_WRITE, NULL);
			SwapBufferDispositivo[count][1] = CreateMemoryObject(count - meusDispositivosOffset, sizeCarga, CL_MEM_READ_WRITE, NULL);
			WriteToMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)DataToKernel, 0, DataToKernel_Size);
			SynchronizeCommandQueue(count - meusDispositivosOffset);
			tempoInicio = MPI_Wtime();
			WriteToMemoryObject(count - meusDispositivosOffset, SwapBufferDispositivo[count][0], (char *)SwapBuffer[0], 0, sizeCarga);
			WriteToMemoryObject(count - meusDispositivosOffset, SwapBufferDispositivo[count][1], (char *)SwapBuffer[1], 0, sizeCarga);
			double aux = (MPI_Wtime() - tempoInicio) / sizeCarga / 2;
			localWriteByte = aux > localWriteByte ? aux : localWriteByte;
			cout << "Local Writebyte: " << localWriteByte << endl;

			kernelDispositivo[count] = CreateKernel(count - meusDispositivosOffset, "kernels.cl", "ProcessarPontos");
			SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, SwapBufferDispositivo[count][0]);
			SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, SwapBufferDispositivo[count][1]);
			SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 2, DataToKernelDispositivo[count]);
		}

		offsetComputacao += lengthComputacao;
	}
	MPI_Allreduce(&localWriteByte, &writeByte, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	cout << "Writebyte: " << writeByte << endl;
}

template<typename T, typename U>
void Balanceador<T,U>::PrecisaoBalanceamento(int simulacao)
{
	// Precisao do balanceamento.

	memset(ticks, 0, sizeof(long int) * todosDispositivos);
	memset(tempos, 0, sizeof(double) * todosDispositivos);

	for (int precisao = 0; precisao < PRECISAO_BALANCEAMENTO; precisao++)
	{
		
		// Computação.
		for (int count = 0; count < todosDispositivos; count++)
		{
			
			if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
			{
				if ((simulacao % 2) == 0)
				{
					
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, SwapBufferDispositivo[count][0]);
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, SwapBufferDispositivo[count][1]);
				}
				else
				{
					
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, SwapBufferDispositivo[count][1]);
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, SwapBufferDispositivo[count][0]);
				}

				kernelEventoDispositivo[count] = RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], OFFSET_COMPUTACAO[count], LENGTH_COMPUTACAO[count], isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
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
	 		tempos[count] = ticks[count] / (GetMaxFrequency(count) * PRECISAO_BALANCEAMENTO) / cargasNovas[count];
			cout<<"Tempos: "<<tempos[count]<<" ";
	 	}
		cout<<endl;
	 }
	 
	 double tempos_root[todosDispositivos];
	 MPI_Allreduce(tempos, tempos_root, todosDispositivos, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	 memcpy(tempos, tempos_root, sizeof(double) * todosDispositivos);
	 cout<<"tempos_root: "<<tempos_root[0]<<endl;
}
/*
void Balanceador::BalanceamentoDeCarga(int simulacao)
{
	double tempoInicioBalanceamento = MPI_Wtime();

	//PrecisaoBalanceamento(simulacao);

	// Computar novas cargas.
	localTempoCB;
	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			SynchronizeCommandQueue(count - meusDispositivosOffset);
			localTempoCB = cargasNovas[count] * tempos[count];
		}
	}
	MPI_Allreduce(tempoCB, localTempoCB, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	tempoCB *= xMalhaLength * yMalhaLength * zMalhaLength;

	if (latencia + ComputarNorma(cargasAntigas, cargasNovas, todosDispositivos) * (writeByte + banda) + tempoCB < tempoComputacaoInterna)
	{
		for (int count = 0; count < todosDispositivos; count++)
		{
			if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
			{
				int overlapNovoOffset = ((int)(((count == 0) ? 0.0f : cargasNovas[count - 1]) * ((float)(xMalhaLength * yMalhaLength * zMalhaLength))));
				int overlapNovoLength = ((int)(((count == 0) ? cargasNovas[count] - 0.0f : cargasNovas[count] - cargasNovas[count - 1]) * ((float)(xMalhaLength * yMalhaLength * zMalhaLength))));
				for (int count2 = 0; count2 < todosDispositivos; count2++)
				{
					if (count > count2)
					{
						// Atender requisicoes de outros processos.
						if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
						{
							int overlap[2];
							int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
							float *malha = ((simulacao % 2) == 0) ? SwapBuffer[0] : SwapBuffer[1];
							int malhaDevice = ((simulacao % 2) == 0) ? SwapBufferDispositivo[count][0] : SwapBufferDispositivo[count][1];
							MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
							// Podem ocorrer requisicoes vazias.
							if (overlap[1] > 0)
							{
								ReadFromMemoryObject(count - meusDispositivosOffset, malhaDevice, (char *)(malha + (overlap[0] * MALHA_TOTAL_CELULAS)), overlap[0] * MALHA_TOTAL_CELULAS * sizeof(float), overlap[1] * MALHA_TOTAL_CELULAS * sizeof(float));
								SynchronizeCommandQueue(count - meusDispositivosOffset);

								sizeCarga = overlap[1] * MALHA_TOTAL_CELULAS;

								MPI_Send(malha + (overlap[0] * MALHA_TOTAL_CELULAS), sizeCarga, MPI_FLOAT, alvo, 0, MPI_COMM_WORLD);
							}
						}
					}
					else if (count < count2)
					{
						// Fazer requisicoes a outros processos.
						int overlapAntigoOffset = ((int)(((count2 == 0) ? 0 : cargasAntigas[count2 - 1]) * (xMalhaLength * yMalhaLength * zMalhaLength)));
						int overlapAntigoLength = ((int)(((count2 == 0) ? cargasAntigas[count2] - 0.0f : cargasAntigas[count2] - cargasAntigas[count2 - 1]) * (xMalhaLength * yMalhaLength * zMalhaLength)));

						int intersecaoOffset;
						int intersecaoLength;

						if (((overlapAntigoOffset <= overlapNovoOffset - (xMalhaLength * yMalhaLength)) && ComputarIntersecao(overlapAntigoOffset, overlapAntigoLength, overlapNovoOffset - (xMalhaLength * yMalhaLength), overlapNovoLength + (xMalhaLength * yMalhaLength), &intersecaoOffset, &intersecaoLength)) ||
								((overlapAntigoOffset > overlapNovoOffset - (xMalhaLength * yMalhaLength)) && ComputarIntersecao(overlapNovoOffset - (xMalhaLength * yMalhaLength), overlapNovoLength + (xMalhaLength * yMalhaLength), overlapAntigoOffset, overlapAntigoLength, &intersecaoOffset, &intersecaoLength)))
						{
							if (count2 >= meusDispositivosOffset && count2 < meusDispositivosOffset + meusDispositivosLength)
							{
								float *malha = ((simulacao % 2) == 0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];

								int malhaDevice[2] = {((simulacao % 2) == 0) ? malhaSwapBufferDispositivo[count][0] : malhaSwapBufferDispositivo[count][1],
																			((simulacao % 2) == 0) ? malhaSwapBufferDispositivo[count2][0] : malhaSwapBufferDispositivo[count2][1]};

								ReadFromMemoryObject(count2 - meusDispositivosOffset, malhaDevice[1], (char *)(malha + (intersecaoOffset * MALHA_TOTAL_CELULAS)), intersecaoOffset * MALHA_TOTAL_CELULAS * sizeof(float), intersecaoLength * MALHA_TOTAL_CELULAS * sizeof(float));
								SynchronizeCommandQueue(count2 - meusDispositivosOffset);

								WriteToMemoryObject(count - meusDispositivosOffset, malhaDevice[0], (char *)(malha + (intersecaoOffset * MALHA_TOTAL_CELULAS)), intersecaoOffset * MALHA_TOTAL_CELULAS * sizeof(float), intersecaoLength * MALHA_TOTAL_CELULAS * sizeof(float));
								SynchronizeCommandQueue(count - meusDispositivosOffset);
							}
							else
							{
								// Fazer uma requisicao.
								if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
								{
									int overlap[2] = {intersecaoOffset, intersecaoLength};
									int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
									float *malha = ((simulacao % 2) == 0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];
									int malhaDevice = ((simulacao % 2) == 0) ? malhaSwapBufferDispositivo[count][0] : malhaSwapBufferDispositivo[count][1];

									MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);

									MPI_Recv(malha + (overlap[0] * MALHA_TOTAL_CELULAS), overlap[1] * MALHA_TOTAL_CELULAS, MPI_FLOAT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

									WriteToMemoryObject(count - meusDispositivosOffset, malhaDevice, (char *)(malha + (overlap[0] * MALHA_TOTAL_CELULAS)), overlap[0] * MALHA_TOTAL_CELULAS * sizeof(float), overlap[1] * MALHA_TOTAL_CELULAS * sizeof(float));
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
								float *malha = ((simulacao % 2) == 0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];
								MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
							}
						}
					}
				}

				parametrosMalha[count][OFFSET_COMPUTACAO] = overlapNovoOffset;
				parametrosMalha[count][LENGTH_COMPUTACAO] = overlapNovoLength;

				WriteToMemoryObject(count - meusDispositivosOffset, parametrosMalhaDispositivo[count], (char *)parametrosMalha[count], 0, sizeof(int) * NUMERO_PARAMETROS_MALHA);
				SynchronizeCommandQueue(count - meusDispositivosOffset);
			}
		}
		memcpy(cargasAntigas, cargasNovas, sizeof(float) * todosDispositivos);

		MPI_Barrier(MPI_COMM_WORLD);
		double tempoFimBalanceamento = MPI_Wtime();
		tempoBalanceamento += tempoFimBalanceamento - tempoInicioBalanceamento;
	}
}
/*
void Balanceador::Probing(int simulacao)
{
	// if (balanceamento && ((simulacao == 0) || (simulacao == 1) ))

	double tempoInicioProbing = MPI_Wtime();
	double localLatencia = 0, localBanda = 0;
	PrecisaoBalanceamento(simulacao);

	// Computar novas cargas.

	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			int overlapNovoOffset = ((int)(((count == 0) ? 0.0f : cargasNovas[count - 1]) * ((float)(xMalhaLength * yMalhaLength * zMalhaLength))));
			int overlapNovoLength = ((int)(((count == 0) ? cargasNovas[count] - 0.0f : cargasNovas[count] - cargasNovas[count - 1]) * ((float)(xMalhaLength * yMalhaLength * zMalhaLength))));
			for (int count2 = 0; count2 < todosDispositivos; count2++)
			{
				if (count > count2)
				{
					// Atender requisicoes de outros processos.
					if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
					{
						int overlap[2];
						int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
						float *malha = ((simulacao % 2) == 0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];
						int malhaDevice = ((simulacao % 2) == 0) ? malhaSwapBufferDispositivo[count][0] : malhaSwapBufferDispositivo[count][1];
						MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						// Podem ocorrer requisicoes vazias.
						if (overlap[1] > 0)
						{
							ReadFromMemoryObject(count - meusDispositivosOffset, malhaDevice, (char *)(malha + (overlap[0] * MALHA_TOTAL_CELULAS)), overlap[0] * MALHA_TOTAL_CELULAS * sizeof(float), overlap[1] * MALHA_TOTAL_CELULAS * sizeof(float));
							SynchronizeCommandQueue(count - meusDispositivosOffset);

							sizeCarga = overlap[1] * MALHA_TOTAL_CELULAS;

							double tempoInicioBanda = MPI_Wtime();
							MPI_Ssend(malha + (overlap[0] * MALHA_TOTAL_CELULAS), sizeCarga, MPI_FLOAT, alvo, 0, MPI_COMM_WORLD);
							double aux = (MPI_Wtime() - tempoInicioBanda) / sizeCarga;
							localBanda = aux > localBanda ? aux : localBanda;
						}
					}
				}
				else if (count < count2)
				{
					// Fazer requisicoes a outros processos.
					int overlapAntigoOffset = ((int)(((count2 == 0) ? 0 : cargasAntigas[count2 - 1]) * (xMalhaLength * yMalhaLength * zMalhaLength)));
					int overlapAntigoLength = ((int)(((count2 == 0) ? cargasAntigas[count2] - 0.0f : cargasAntigas[count2] - cargasAntigas[count2 - 1]) * (xMalhaLength * yMalhaLength * zMalhaLength)));

					int intersecaoOffset;
					int intersecaoLength;

					if (((overlapAntigoOffset <= overlapNovoOffset - (xMalhaLength * yMalhaLength)) && ComputarIntersecao(overlapAntigoOffset, overlapAntigoLength, overlapNovoOffset - (xMalhaLength * yMalhaLength), overlapNovoLength + (xMalhaLength * yMalhaLength), &intersecaoOffset, &intersecaoLength)) ||
							((overlapAntigoOffset > overlapNovoOffset - (xMalhaLength * yMalhaLength)) && ComputarIntersecao(overlapNovoOffset - (xMalhaLength * yMalhaLength), overlapNovoLength + (xMalhaLength * yMalhaLength), overlapAntigoOffset, overlapAntigoLength, &intersecaoOffset, &intersecaoLength)))
					{
						if (count2 >= meusDispositivosOffset && count2 < meusDispositivosOffset + meusDispositivosLength)
						{
							float *malha = ((simulacao % 2) == 0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];

							int malhaDevice[2] = {((simulacao % 2) == 0) ? malhaSwapBufferDispositivo[count][0] : malhaSwapBufferDispositivo[count][1],
																		((simulacao % 2) == 0) ? malhaSwapBufferDispositivo[count2][0] : malhaSwapBufferDispositivo[count2][1]};

							ReadFromMemoryObject(count2 - meusDispositivosOffset, malhaDevice[1], (char *)(malha + (intersecaoOffset * MALHA_TOTAL_CELULAS)), intersecaoOffset * MALHA_TOTAL_CELULAS * sizeof(float), intersecaoLength * MALHA_TOTAL_CELULAS * sizeof(float));
							SynchronizeCommandQueue(count2 - meusDispositivosOffset);

							WriteToMemoryObject(count - meusDispositivosOffset, malhaDevice[0], (char *)(malha + (intersecaoOffset * MALHA_TOTAL_CELULAS)), intersecaoOffset * MALHA_TOTAL_CELULAS * sizeof(float), intersecaoLength * MALHA_TOTAL_CELULAS * sizeof(float));
							SynchronizeCommandQueue(count - meusDispositivosOffset);
						}
						else
						{
							// Fazer uma requisicao.
							if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
							{
								int overlap[2] = {intersecaoOffset, intersecaoLength};
								int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
								float *malha = ((simulacao % 2) == 0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];
								int malhaDevice = ((simulacao % 2) == 0) ? malhaSwapBufferDispositivo[count][0] : malhaSwapBufferDispositivo[count][1];
								SynchronizeCommandQueue(count - meusDispositivosOffset);
								double tempoInicioLatencia = MPI_Wtime();
								MPI_Ssend(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
								double aux = (MPI_Wtime() - tempoInicioLatencia) / 2;
								localLatencia = aux > localLatencia ? aux : localLatencia;

								MPI_Recv(malha + (overlap[0] * MALHA_TOTAL_CELULAS), overlap[1] * MALHA_TOTAL_CELULAS, MPI_FLOAT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

								WriteToMemoryObject(count - meusDispositivosOffset, malhaDevice, (char *)(malha + (overlap[0] * MALHA_TOTAL_CELULAS)), overlap[0] * MALHA_TOTAL_CELULAS * sizeof(float), overlap[1] * MALHA_TOTAL_CELULAS * sizeof(float));
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
							float *malha = ((simulacao % 2) == 0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];
							MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
						}
					}
				}
			}

			parametrosMalha[count][OFFSET_COMPUTACAO] = overlapNovoOffset;
			parametrosMalha[count][LENGTH_COMPUTACAO] = overlapNovoLength;

			WriteToMemoryObject(count - meusDispositivosOffset, parametrosMalhaDispositivo[count], (char *)parametrosMalha[count], 0, sizeof(int) * NUMERO_PARAMETROS_MALHA);
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

void Balanceador::ComputaKernel(int simulacao)
{

	double tempoInicio = MPI_Wtime();

	// Computação interna.
	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], parametrosMalha[count][OFFSET_COMPUTACAO] + (xMalhaLength * yMalhaLength), parametrosMalha[count][LENGTH_COMPUTACAO] - (xMalhaLength * yMalhaLength), isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
		}
	}

	// Sincronizacao da computação interna.
	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			SynchronizeCommandQueue(count - meusDispositivosOffset);
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
				SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, malhaSwapBufferDispositivo[count][0]);
				SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, malhaSwapBufferDispositivo[count][1]);
			}
			else
			{
				SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, malhaSwapBufferDispositivo[count][1]);
				SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, malhaSwapBufferDispositivo[count][0]);
			}

			RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], parametrosMalha[count][OFFSET_COMPUTACAO], xMalhaLength * yMalhaLength, isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
			RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], parametrosMalha[count][OFFSET_COMPUTACAO] + parametrosMalha[count][LENGTH_COMPUTACAO] - (xMalhaLength * yMalhaLength), xMalhaLength * yMalhaLength, isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
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
*/
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
T* Balanceador<T,U>::returnData(){

Data[0] = (99.9);

Data[10] = (99.9);
return Data;


}



void Balancer::run_mult_step_balancer() {
   
    balanceamento = true;
   

    for (int simulacao = 0; simulacao < SIMULACOES; simulacao++) {
        // Balanceamento de carga
        if (balanceamento && ((simulacao == 0) || (simulacao == 1) || (simulacao % INTERVALO_BALANCEAMENTO == 0))) {
            if (HABILITAR_ESTATICO)
            if (simulacao > 1) {
                balanceamento = false;
            }
            #endif

            balanceamentoDeCarga(simulacao);
        } else {
            computacaoInterna(simulacao);
            trocaDeBordas(simulacao);
            computacaoDeBordas(simulacao);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        gettimeofday(&timeEnd, NULL);
        printf("Overall ticks (1tick->1ms): %lu\n", (timeEnd.tv_sec - timeStart.tv_sec) * 1000000 + (timeEnd.tv_usec - timeStart.tv_usec));

        if (HABILITAR_BENCHMARK){
        printf("Internal computation (s): %f\n", tempoComputacaoInterna);
        printf("Border swap time (s): %f\n", tempoTrocaBorda);
        printf("Border computation time (s): %f\n", tempoComputacaoBorda);
        printf("Balancing time (s): %f\n", tempoBalanceamento);
        }

        for (int count = 0; count < todosDispositivos; count++) {
            printf("Tempo dispositivo (1tick->1nanosegundo) %i: %li\n", count, tempos[count]);
        }
    }

    
}

