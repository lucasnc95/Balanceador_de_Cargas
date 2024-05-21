#ifndef BALANCEADOR_H
#define BALANCEADOR_H
#include "OpenCLWrapper.h"
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>




class Balanceador
{

private:
  MPI_Request sendRequest, receiveRequest, recvNotification;
  unsigned int CPU_WORK_GROUP_SIZE;
  unsigned int GPU_WORK_GROUP_SIZE;
  unsigned int SIMULACOES;
  bool HABILITAR_ESTATICO;
  bool HABILITAR_DINAMICO;
  unsigned int HABILITAR_BALANCEAMENTO;
  unsigned int INTERVALO_BALANCEAMENTO;
  unsigned int PRECISAO_BALANCEAMENTO;
  float BALANCEAMENTO_THRESHOLD;
  bool HABILITAR_BENCHMARK;
  
  void *Data;
  void *DataToKernel;
  
  int todosDispositivos;
  double tempoInicio, tempoFim;
  double tempoComputacaoInterna;
  double tempoTrocaBorda;
  double tempoComputacaoBorda;
  double tempoBalanceamento;
  
  void **SwapBuffer;
  long int *ticks;      
  double *tempos;       
  float *cargasNovas;   
  float *cargasAntigas; 
  double writeByte;
  double fatorErro;
  size_t Element_size;
  size_t DataToKernel_Size;
  long int N_Elements;
  int *DataToKernelDispositivo;    
  int **SwapBufferDispositivo; 
  int *kernelDispositivo;             
  int *dataEventoDispositivo;         
  int *kernelEventoDispositivo;       
  int offsetComputacao;
  int lengthComputacao;
  int sizeCarga;
  bool balanceamento;
  double latencia;
  double banda;
  double *frequencias;
  
  int meusDispositivosOffset;
  int meusDispositivosLength;
  int world_size;
  int world_rank;
  int *dispositivosLocal;
  int *dispositivosWorld;
  unsigned int *OFFSET_COMPUTACAO;
  unsigned long int *LENGTH_COMPUTACAO;
  

  void ComputarCargas(const long int *tempos, const float *cargasAntigas, float *cargas, int participantes);
  bool ComputarIntersecao(int offset1, int length1, int offset2, int length2, int *intersecaoOffset, int *intersecaoLength);
  int RecuperarPosicaoHistograma(int *histograma, int tamanho, int indice);
  float ComputarDesvioPadraoPercentual(const long int *tempos, int participantes);
  float ComputarNorma(const float *cargasAntigas, const float *cargaNovas, int participantes);
  void InicializarParametrosMalha(int **parametrosMalha, unsigned int offsetComputacao, unsigned int lengthComputacao, unsigned int xMalhaLength, unsigned int yMalhaLength, unsigned int zMalhaLength);
  void LerPontos(const double *malha, const int *parametrosMalha);
  void InicializaDispositivos(int meusDispositivosOffset, int meusDispositivosLenght, int offsetComputacao, int lengthComputacao);
  void DistribuicaoUniformeDeCarga();
  void BalanceamentoDeCarga(int simulacao);
  void ComputaKernel(int simulacao);
  void Probing(int simulacao);
  void PrecisaoBalanceamento(int &simulacao);
  void ExecutarBalanceamento();
  void InicializarLenghtOffset(unsigned int offsetComputacao, unsigned int lengthComputacao, int count);

public:
  Balanceador(int argc, char *argv[], void *data, const size_t Element_sz, const unsigned long int N_Element, void *DTK, const size_t div_size );
  ~Balanceador();
  
};
#endif
