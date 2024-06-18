#include <stdio.h>
#include <iostream>
#include <mpi.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include "Balanceador.h"
#include "Balanceador.cpp"
#include "OpenCLWrapper.h"

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
	(parametrosMalha)[2] = 8;
	(parametrosMalha)[3] = 8;
	(parametrosMalha)[4] = 8;
	(parametrosMalha)[5] = 8 * 8 * 8;
	(parametrosMalha)[6] = 8 * 8;
	(parametrosMalha)[7] = 8;
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

	



	const size_t N = 8 * 8 * 8 * 8;
	const size_t Element_sz = sizeof(malha[0]);
	const size_t div_size = 8 * sizeof(int);
	cout << "N = " << N << endl;
	cout << "Element_sz = " << Element_sz << endl;
	cout << "Malha_sz = " << sizeof(malha[0]) * N << endl;
	// Instanciando um objeto da classe Balanceador
	LerPontosHIS(malha, parametrosMalha);
	Balanceador<float, int> balancer(argc, argv, malha, Element_sz, N, parametrosMalha, div_size);
	
	float *teste = balancer.returnData();
	
	LerPontosHIS(teste, parametrosMalha);
	cout << "Fim do programa..." << endl;

	return 0;
}
