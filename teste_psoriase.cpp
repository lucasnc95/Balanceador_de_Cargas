#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "OpenCLWrapper.h"
#include <iostream>

// Cada célula ocupa 5 posições na malha 
#define CELULAS_SINGLE_CELL_SIZEOF 8

// Tamanho total para armazenar todas as células (em posições)
#define CELULAS_SIZEOF 40
#define MALHA_TOTAL_CELULAS 5
// Offsets para as células na malha
#define CELULAS_L_OFFSET 0    // Células T - L
#define CELULAS_M_OFFSET 8    // Células dendríticas - M
#define CELULAS_S_OFFSET 16   // Citocinas - S
#define CELULAS_W_OFFSET 24   // Medicamento imunológico - W
#define CELULAS_K_OFFSET 32   // Queratinócitos - K

//Informacoes de acesso à estrutura "parametrosMalha".
#define OFFSET_COMPUTACAO               0
#define LENGTH_COMPUTACAO               1
#define COMPRIMENTO_GLOBAL_X            2
#define COMPRIMENTO_GLOBAL_Y            3
#define COMPRIMENTO_GLOBAL_Z            4
#define MALHA_DIMENSAO_POSICAO_Z        5
#define MALHA_DIMENSAO_POSICAO_Y        6
#define MALHA_DIMENSAO_POSICAO_X        7
#define MALHA_DIMENSAO_CELULAS          8
#define NUMERO_PARAMETROS_MALHA         9



// Definição de semente fixa para reprodutibilidade
#define SEED 12345

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// Função para inicializar a malha com base no artigo
void InicializarPontosHIS(float **malha, const int *parametrosMalha)
{
    // Alocação da malha em memória
    *malha = (float *)malloc(parametrosMalha[COMPRIMENTO_GLOBAL_X] * 
                            parametrosMalha[COMPRIMENTO_GLOBAL_Y] * 
                            parametrosMalha[COMPRIMENTO_GLOBAL_Z] * 
                            MALHA_TOTAL_CELULAS * sizeof(float));

    // Fixar a semente do gerador de números aleatórios para garantir repetibilidade
    srand(42); // Usar sempre o mesmo valor para garantir que os resultados sejam iguais

    // Calcular o ponto central da malha
    int centroX = parametrosMalha[COMPRIMENTO_GLOBAL_X] / 2;
    int centroY = parametrosMalha[COMPRIMENTO_GLOBAL_Y] / 2;
    int centroZ = parametrosMalha[COMPRIMENTO_GLOBAL_Z] / 2;

    // Preencher a malha
    for (unsigned int x = 0; x < parametrosMalha[COMPRIMENTO_GLOBAL_X]; x++)
    {
        for (unsigned int y = 0; y < parametrosMalha[COMPRIMENTO_GLOBAL_Y]; y++)
        {
            for (unsigned int z = 0; z < parametrosMalha[COMPRIMENTO_GLOBAL_Z]; z++)
            {
                int index = (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) +
                            (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) +
                            (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X]);

                // Definir os valores iniciais para as células com base na presença dos queratinócitos no centro
                if (x == centroX && y == centroY && z == centroZ)
                {
                    // Queratinócitos no centro
                    (*malha)[CELULAS_K_OFFSET + index] = 100.0f;  // Valor inicial alto para queratinócitos
                }
                else
                {
                    (*malha)[CELULAS_K_OFFSET + index] = 0.0f;
                }

                // Valores iniciais aleatórios baseados nos parâmetros do artigo
                (*malha)[CELULAS_L_OFFSET + index] = (float)(rand() % 10) / 10.0f;  // Inicializa L entre 0 e 1
                (*malha)[CELULAS_M_OFFSET + index] = (float)(rand() % 20) / 20.0f;  // Inicializa M entre 0 e 1
                (*malha)[CELULAS_S_OFFSET + index] = (float)(rand() % 15) / 15.0f;  // Inicializa S entre 0 e 1
                (*malha)[CELULAS_W_OFFSET + index] = 0.0f;  // Medicamento inicializado como zero em todas as posições
            }
        }
    }
}


// Função para ler os valores da malha e imprimir os resultados
void LerPontosHIS(const float *malha, const int *parametrosMalha)
{
    for (unsigned int x = 0; x < parametrosMalha[COMPRIMENTO_GLOBAL_X]; x++)
    {
        for (unsigned int y = 0; y < parametrosMalha[COMPRIMENTO_GLOBAL_Y]; y++)
        {
            for (unsigned int z = 0; z < parametrosMalha[COMPRIMENTO_GLOBAL_Z]; z++)
            {
                int index = (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) +
                            (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) +
                            (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X]);

                // Verificar se o ponto está dentro do intervalo de computação
                if (index >= parametrosMalha[OFFSET_COMPUTACAO] * MALHA_TOTAL_CELULAS &&
                    index < (parametrosMalha[OFFSET_COMPUTACAO] + parametrosMalha[LENGTH_COMPUTACAO]) * MALHA_TOTAL_CELULAS)
                {
                    printf("Posição (%d, %d, %d): L=%f M=%f S=%f W=%f K=%f\n", 
                           x, y, z, 
                           malha[CELULAS_L_OFFSET + index],
                           malha[CELULAS_M_OFFSET + index],
                           malha[CELULAS_S_OFFSET + index],
                           malha[CELULAS_W_OFFSET + index],
                           malha[CELULAS_K_OFFSET + index]);
                }
                else
                {
                    printf("Posição (%d, %d, %d): L=0.0 M=0.0 S=0.0 W=0.0 K=0.0\n", x, y, z);
                }
            }
            printf("\n");
        }
    }
}



int main(int argc, char** argv) {
     
    OpenCLWrapper openCL(argc, argv);
    openCL.InitDevices("CPU_DEVICES", 10);  
    openCL.setKernel("kernel_psoriase.cl", "ProcessarPontos");

    int x = 100, y = 100, z = 100;
    int tam = x * y * z * MALHA_TOTAL_CELULAS;  // Tamanho correto da malha
    int *parametros = new int[NUMERO_PARAMETROS_MALHA];
    float *malha = new float[tam];  // Alocar a malha corretamente

    double	tempoInicio = MPI_Wtime();
    InicializarParametrosMalhaHIS(parametros, 0, (x * y * z), x, y, z);

   
    InicializarPontosHIS(malha, parametros);
	int total_elements = x*y*z;

    // Configuração do balanceador de carga no OpenCLWrapper
    size_t sub = x * y  ;
    openCL.setLoadBalancer(sizeof(float), total_elements, MALHA_TOTAL_CELULAS, sub);

    // Alocar objetos de memória OpenCL
    int aMemObj = openCL.AllocateMemoryObject(NUMERO_PARAMETROS_MALHA * sizeof(int), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, parametros);
    int bMemObj = openCL.AllocateMemoryObject(tam * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, malha);
    int cMemObj = openCL.AllocateMemoryObject(tam * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, malha);

    float *malhaAux = new float[tam];  // Para armazenar os resultados
	int *vetArgs = new int[2];
	vetArgs[0] = bMemObj;
	vetArgs[1] = cMemObj;

	
    // Definir atributos de kernel
	openCL.setAttribute(0, bMemObj);
    openCL.setAttribute(1, cMemObj);
	openCL.setAttribute(2, aMemObj);
   // openCL.setSubdomainBoundary(sub, 2, vetArgs);
	openCL.setBalancingTargetID(bMemObj);
	openCL.Probing();
    for (int x = 0; x < 10000; x++) {
		
	// openCL.GatherResults(bMemObj, malhaAux);
	// openCL.WriteObject(cMemObj, (char *) malhaAux, 0, tam*sizeof(float));
		 if (x % 2 == 0) {
            openCL.setAttribute(0, bMemObj);
            openCL.setAttribute(1, cMemObj);
            openCL.setBalancingTargetID(bMemObj);
        } 
		else {
            openCL.setAttribute(0, cMemObj);
            openCL.setAttribute(1, bMemObj);
            openCL.setBalancingTargetID(cMemObj);
        }

		if(x % 1000 == 0)
		openCL.Probing();


		openCL.ExecuteKernel();
		
		
        
    }

	openCL.GatherResults(bMemObj, malhaAux);
	
    LerPontosHIS(malhaAux, parametros);
	  
	double tempoFim = MPI_Wtime();
	std::cout<<"Tempo execução:"<<tempoFim-tempoInicio<<std::endl;
    // Liberar memória alocada
    delete[] parametros;
    delete[] malha;
    delete[] malhaAux;
	
    return 0;
}
