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
  double **malha;
  int xMalhaLength, yMalhaLength, zMalhaLength;
  int todosDispositivos;
  double tempoInicio, tempoFim;
  double tempoComputacaoInterna;
  double tempoTrocaBorda;
  double tempoComputacaoBorda;
  double tempoBalanceamento;
  int **parametrosMalha; //[todosDispositivos]
  float *malhaSwapBuffer[2];
  long int *ticks;      //[todosDispositivos]
  double *tempos;       //[todosDispositivos]
  float *cargasNovas;   //[todosDispositivos]
  float *cargasAntigas; //[todosDispositivos]
  double writeByte;
  double fatorErro;
  double tempoComputacaoBalanceada;
  int *parametrosMalhaDispositivo;    //[todosDispositivos]
  float **malhaSwapBufferDispositivo; //[todosDispositivos][2]
  int *kernelDispositivo;             //[todosDispositivos]
  int *dataEventoDispositivo;         //[todosDispositivos]
  int *kernelEventoDispositivo;       //[todosDispositivos]
  int offsetComputacao;
  int lengthComputacao;
  int sizeCarga;
  bool balanceamento;
  double latencia;
  double banda;
  int dispositivos;
  int meusDispositivosOffset;
  int meusDispositivosLength;
  bool setMalha;
  bool setMalhaSwapbuffer;
  int world_size;
  int world_rank;
  int *dispositivosLocal;
  int *dispositivosWorld;
  unsigned int OFFSET_COMPUTACAO;
  unsigned int LENGTH_COMPUTACAO;
  unsigned int COMPRIMENTO_GLOBAL_X;
  unsigned int COMPRIMENTO_GLOBAL_Y;
  unsigned int COMPRIMENTO_GLOBAL_Z;
  unsigned int MALHA_DIMENSAO_POSICAO_Z;
  unsigned int MALHA_DIMENSAO_POSICAO_Y;
  unsigned int MALHA_DIMENSAO_POSICAO_X;
  unsigned int MALHA_DIMENSAO_CELULAS = 1;
  unsigned int NUMERO_PARAMETROS_MALHA;
  unsigned int MALHA_TOTAL_CELULAS;

  void ComputarCargas(const long int *tempos, const float *cargasAntigas, float *cargas, int participantes);
  bool ComputarIntersecao(int offset1, int length1, int offset2, int length2, int *intersecaoOffset, int *intersecaoLength);
  int RecuperarPosicaoHistograma(int *histograma, int tamanho, int indice);
  float ComputarDesvioPadraoPercentual(const long int *tempos, int participantes);
  float ComputarNorma(const float *cargasAntigas, const float *cargaNovas, int participantes);
  void InicializarParametrosMalha(int **parametrosMalha, unsigned int offsetComputacao, unsigned int lengthComputacao, unsigned int xMalhaLength, unsigned int yMalhaLength, unsigned int zMalhaLength);
  void LerPontos(const double *malha, const int *parametrosMalha);
  void InicializaDispositivos();
  void DstribuicaoUniformeDeCarga();
  void BalanceamentoDeCarga(int simulacao);
  void ComputaKernel(int simulacao);
  void Probing(int simulacao);
  void PrecisaoBalanceamento(int &simulacao);
  void ExecutarBalanceamento();

public:
  Balanceador(int argc, char *argv[], int malha[]);
  ~Balanceador();
  void SetmalhaSwapBuffer(double *malhaSwapBuffer[]);
  void SetMalha(double **malha[]);
};
#endif
