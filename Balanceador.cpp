#include "Balanceador.h"
#include "OpenCLWrapper.h"
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>

Balanceador::Balanceador(int argc, char *argv[], int malha[])
{
	printf("Iniciando MPI....\n");
	// MPI_Init(&argc, &argv);
	MPI_Init(NULL, NULL);
	xMalhaLength = malha[0];
	yMalhaLength = malha[1];
	zMalhaLength = malha[2];

	HABILITAR_BENCHMARK = false;
	HABILITAR_ESTATICO = true;
	HABILITAR_DINAMICO = false;

	CPU_WORK_GROUP_SIZE = 8;
	GPU_WORK_GROUP_SIZE = 64;
	SIMULACOES = 10000;
	INTERVALO_BALANCEAMENTO = 1000;
	PRECISAO_BALANCEAMENTO = 10;

	OFFSET_COMPUTACAO = 0;
	LENGTH_COMPUTACAO = 1;
	COMPRIMENTO_GLOBAL_X = 2;
	COMPRIMENTO_GLOBAL_Y = 3;
	COMPRIMENTO_GLOBAL_Z = 4;
	MALHA_DIMENSAO_POSICAO_Z = 5;
	MALHA_DIMENSAO_POSICAO_Y = 6;
	MALHA_DIMENSAO_POSICAO_X = 7;
	MALHA_DIMENSAO_CELULAS = 8;
	NUMERO_PARAMETROS_MALHA = 9;

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	printf("Fim da inicialização...\n");

	// MPI_Request sendRequest, receiveRequest;		Variáveis usadas na borda, porém esta parte foi removida do código
	dispositivosLocal = new int[world_size];
	printf("Inicializando openCL...\n");
	dispositivos = InitParallelProcessor();
	printf("0...\n");
	// dispositivosLocal[world_size];
	printf("Disp. Local...\n");
	// dispositivosWorld[world_size];
	dispositivosWorld = new int[world_size];
	printf("Disp. World...\n");
	memset(dispositivosLocal, 0, sizeof(int) * world_size);
	printf("Memset...\n");
	dispositivosLocal[world_rank] = dispositivos;
	printf("Disp[]...\n");
	MPI_Allreduce(dispositivosLocal, dispositivosWorld, world_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	printf("MPI_ALLreduce\n");
	todosDispositivos = 0;
	printf("for...\n");
	for (int count = 0; count < world_size; count++)
	{
		printf("2...\n");
		if (count == world_rank)
		{
			meusDispositivosOffset = todosDispositivos;
			meusDispositivosLength = dispositivosWorld[count];
		}
		todosDispositivos += dispositivosWorld[count];
	}
	printf("Meus dispositivos offset = %d\n", meusDispositivosOffset);
	printf("Todos dispositivos = %d\n", todosDispositivos);
	printf("Fim da inicialização...\n");

	*malhaSwapBuffer[2];
	tempos[todosDispositivos];
	cargasNovas[todosDispositivos];
	cargasAntigas[todosDispositivos];
	double LocalWriteByte = 0;

	parametrosMalhaDispositivo[todosDispositivos];
	malhaSwapBufferDispositivo[todosDispositivos][2];
	kernelDispositivo[todosDispositivos];
	dataEventoDispositivo[todosDispositivos];
	kernelEventoDispositivo[todosDispositivos];

	offsetComputacao = 0;
	lengthComputacao = (xMalhaLength * yMalhaLength * zMalhaLength) / todosDispositivos;
	printf("Inicialização dos dispositivos...\n");
	InicializaDispositivos();
	printf("Fim da inicialização dos dispositivos...\n");
	printf("Distribuição de carga...\n");
	DstribuicaoDeCarga();
	printf("Fim...\n");
}

Balanceador::~Balanceador()
{
	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			delete parametrosMalha;
			parametrosMalha[count] = NULL;
		}
	}
	delete malhaSwapBuffer;
	malhaSwapBuffer[0] = NULL;
	malhaSwapBuffer[1] = NULL;

	FinishParallelProcessor();
	MPI_Finalize();
}

void Balanceador::ComputarCargas(const long int *ticks, const float *cargasAntigas, float *cargas, int participantes)
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

// Offset1 deve ser sempre menor ou igual que offset2.
bool Balanceador::ComputarIntersecao(int offset1, int length1, int offset2, int length2, int *intersecaoOffset, int *intersecaoLength)
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

int Balanceador::RecuperarPosicaoHistograma(int *histograma, int tamanho, int indice)
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

float Balanceador::ComputarDesvioPadraoPercentual(const long int *ticks, int participantes)
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

float Balanceador::ComputarNorma(const float *cargasAntigas, const float *cargasNovas, int participantes)
{
	float retorno = 0.0;
	for (int count = 0; count < participantes; count++)
	{
		retorno += (cargasAntigas[count] - cargasNovas[count]) * (cargasAntigas[count] - cargasNovas[count]);
	}
	return sqrt(retorno);
}

void Balanceador::InicializarParametrosMalha(int **parametrosMalha, unsigned int offsetComputacao, unsigned int lengthComputacao, unsigned int xMalhaLength, unsigned int yMalhaLength, unsigned int zMalhaLength)
{
	printf("Dentro da Inicialização da malha...\n");
	*parametrosMalha = new int[NUMERO_PARAMETROS_MALHA];

	(*parametrosMalha)[OFFSET_COMPUTACAO] = offsetComputacao;
	(*parametrosMalha)[LENGTH_COMPUTACAO] = lengthComputacao;
	(*parametrosMalha)[COMPRIMENTO_GLOBAL_X] = xMalhaLength;
	(*parametrosMalha)[COMPRIMENTO_GLOBAL_Y] = yMalhaLength;
	(*parametrosMalha)[COMPRIMENTO_GLOBAL_Z] = zMalhaLength;
	(*parametrosMalha)[MALHA_DIMENSAO_POSICAO_Z] = yMalhaLength * xMalhaLength * MALHA_TOTAL_CELULAS;
	(*parametrosMalha)[MALHA_DIMENSAO_POSICAO_Y] = xMalhaLength * MALHA_TOTAL_CELULAS;
	(*parametrosMalha)[MALHA_DIMENSAO_POSICAO_X] = MALHA_TOTAL_CELULAS;
	(*parametrosMalha)[MALHA_DIMENSAO_CELULAS] = 1;
	printf("Fim Inicialização da malha...\n");
}

// Função de inicialização em todos os dispositivos
void Balanceador::InicializaDispositivos()
{
	printf("Dentro da Inicialização dos dispositivos...\n");
	//*parametrosMalha = new int[todosDispositivos];
	parametrosMalhaDispositivo = new int *[todosDispositivos];
	malhaSwapBufferDispositivo = new float **[todosDispositivos];
	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			printf("inicialização dos parametros da malha...\n");
			InicializarParametrosMalha(&parametrosMalha[count], offsetComputacao, (count + 1 == todosDispositivos) ? (xMalhaLength * yMalhaLength * zMalhaLength) - offsetComputacao : lengthComputacao, xMalhaLength, yMalhaLength, zMalhaLength);
			parametrosMalhaDispositivo[count] = new int[NUMERO_PARAMETROS_MALHA];
			malhaSwapBufferDispositivo[count] = new float *[2];
			printf("Criando objeto 1...\n");
			printf("Meus dispositivos offset (2) = %d\n", meusDispositivosOffset);
			parametrosMalhaDispositivo[count] = CreateMemoryObject(count - meusDispositivosOffset, sizeof(int) * NUMERO_PARAMETROS_MALHA, CL_MEM_READ_ONLY, NULL);
			printf("Criando objeto 2...\n");
			malhaSwapBufferDispositivo[count][0] = CreateMemoryObject(count - meusDispositivosOffset, sizeof(float) * (xMalhaLength * yMalhaLength * zMalhaLength * MALHA_TOTAL_CELULAS), CL_MEM_READ_WRITE, NULL);
			printf("Criando objeto 3...\n");
			malhaSwapBufferDispositivo[count][1] = CreateMemoryObject(count - meusDispositivosOffset, sizeof(float) * (xMalhaLength * yMalhaLength * zMalhaLength * MALHA_TOTAL_CELULAS), CL_MEM_READ_WRITE, NULL);
			printf("Escrevendo objeto 1...\n");
			WriteToMemoryObject(count - meusDispositivosOffset, parametrosMalhaDispositivo[count], (char *)parametrosMalha[count], 0, sizeof(int) * NUMERO_PARAMETROS_MALHA);
			sizeCarga = sizeof(float) * (xMalhaLength * yMalhaLength * zMalhaLength * MALHA_TOTAL_CELULAS);
			printf("Escrevendo objeto 2...\n");
			WriteToMemoryObject(count - meusDispositivosOffset, malhaSwapBufferDispositivo[count][0], (char *)malhaSwapBuffer[0], 0, sizeCarga);
			printf("Escrevendo objeto 3...\n");
			WriteToMemoryObject(count - meusDispositivosOffset, malhaSwapBufferDispositivo[count][1], (char *)malhaSwapBuffer[1], 0, sizeCarga);
			LocalWriteByte = sizeCarga / (tempoFim - tempoInicio) * 2;
			SynchronizeCommandQueue(count - meusDispositivosOffset);

			MPI_Allreduce(&LocalWriteByte, &writeByte, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

			kernelDispositivo[count] = CreateKernel(count - meusDispositivosOffset, "kernels.cl", "ProcessarPontos");
			SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, malhaSwapBufferDispositivo[count][0]);
			SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, malhaSwapBufferDispositivo[count][1]);
			SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 2, parametrosMalhaDispositivo[count]);
		}
		printf("Soma offset...\n");
		offsetComputacao += lengthComputacao;
	}
	printf("Fim da função...\n");
}

void Balanceador::PrecisaoBalanceamento(int &simulacao)
{
	// Precisao do balanceamento.
	memset(ticks, 0, sizeof(long int) * todosDispositivos);
	memset(tempos, 0, sizeof(double) * todosDispositivos);
	for (int precisao = 0; precisao < PRECISAO_BALANCEAMENTO; precisao++)
	{
		// Computação.
		if ((simulacao % 2) == 0)
		{
			for (int count = 0; count < todosDispositivos; count++)
			{
				if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
				{
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, malhaSwapBufferDispositivo[count][0]);
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, malhaSwapBufferDispositivo[count][1]);
				}
			}
		}
		else
		{
			for (int count = 0; count < todosDispositivos; count++)
			{
				if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
				{
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 0, malhaSwapBufferDispositivo[count][1]);
					SetKernelAttribute(count - meusDispositivosOffset, kernelDispositivo[count], 1, malhaSwapBufferDispositivo[count][0]);
				}

				kernelEventoDispositivo[count] = RunKernel(count - meusDispositivosOffset, kernelDispositivo[count], parametrosMalha[count][OFFSET_COMPUTACAO], parametrosMalha[count][LENGTH_COMPUTACAO], isDeviceCPU(count - meusDispositivosOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
			}
		}

		// Ticks.
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

	long int tempos_root[todosDispositivos];
	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			tempos[count] = ticks[count] / (frequencias[count] * PRECISAO_BALANCEAMENTO) / cargasNovas[count];
		}
	}
	MPI_Allreduce(tempos, tempos_root, todosDispositivos, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	memcpy(tempos, tempos_root, sizeof(long int) * todosDispositivos);
}

void Balanceador::BalanceamentoDeCarga(int simulacao)
{
	double tempoInicioBalanceamento = MPI_Wtime();

	PrecisaoBalanceamento(simulacao);

	// Computar novas cargas.
	double tempoComputacaoBalanceada = cargasNovas[0] * tempos[0];
	for (int count = 1; count < todosDispositivos; count++)
	{
		double aux = cargasNovas[count] * tempos[count];
		if (aux > tempoComputacaoBalanceada)
		{
			tempoComputacaoBalanceada = aux;
		}
	}
	tempoComputacaoBalanceada *= xMalhaLength * yMalhaLength * zMalhaLength;

	if (latencia + ComputarNorma(cargasAntigas, cargasNovas, todosDispositivos) * (writeByte + banda) + tempoComputacaoBalanceada < tempoComputacaoInterna)
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
							float *malha = ((simulacao % 2) == 0) ? malhaSwapBuffer[0] : malhaSwapBuffer[1];
							int malhaDevice = ((simulacao % 2) == 0) ? malhaSwapBufferDispositivo[count][0] : malhaSwapBufferDispositivo[count][1];
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

void Balanceador::Probing(int simulacao)
{
	// if (balanceamento && ((simulacao == 0) || (simulacao == 1) ))

	double tempoInicioProbing = MPI_Wtime();
	double LocalLatencia, LocalBanda;
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
							LocalBanda = (MPI_Wtime() - tempoInicioBanda) / sizeCarga;
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

								double tempoInicioLatencia = MPI_Wtime();
								MPI_Ssend(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
								LocalLatencia = (MPI_Wtime() - tempoInicioLatencia) / 2;

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

	MPI_Allreduce(&LocalLatencia, &latencia, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&LocalBanda, &banda, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

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

/* void Balanceador::DstribuicaoUniformeDeCarga()
{

	for (int count = 0; count < todosDispositivos; count++)
	{
		cargasNovas[count] = ((float)(count + 1)) * (1.0f / ((float)todosDispositivos));
		cargasAntigas[count] = cargasNovas[count];
		tempos[count] = 1;
	}
} */

void Balanceador::SetMalha(double **malha[])
{

	printf("TESTE\n");
}
