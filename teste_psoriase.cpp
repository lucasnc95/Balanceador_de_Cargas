#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "OpenCLWrapper.h"
#include <iostream>
#include <stdio.h>

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





void InicializarParametrosMalhaHIS(int *parametrosMalha,  int offsetComputacao,  int lengthComputacao,  int xMalhaLength,  int yMalhaLength,  int zMalhaLength)
{
	//parametrosMalha = new int[NUMERO_PARAMETROS_MALHA];

	(parametrosMalha)[OFFSET_COMPUTACAO] = offsetComputacao;
	(parametrosMalha)[LENGTH_COMPUTACAO] = lengthComputacao;
	(parametrosMalha)[COMPRIMENTO_GLOBAL_X] = xMalhaLength;
	(parametrosMalha)[COMPRIMENTO_GLOBAL_Y] = yMalhaLength;
	(parametrosMalha)[COMPRIMENTO_GLOBAL_Z] = zMalhaLength;
	(parametrosMalha)[MALHA_DIMENSAO_POSICAO_Z] = yMalhaLength*xMalhaLength*MALHA_TOTAL_CELULAS;
	(parametrosMalha)[MALHA_DIMENSAO_POSICAO_Y] = xMalhaLength*MALHA_TOTAL_CELULAS;
	(parametrosMalha)[MALHA_DIMENSAO_POSICAO_X] = MALHA_TOTAL_CELULAS;
	(parametrosMalha)[MALHA_DIMENSAO_CELULAS] = 1;
}


#define TAMANHO_FLOAT sizeof(float)

void InicializarPontosHIS(float *malha, int *parametrosMalha)
{
    // Calcular o total de células na malha
    unsigned int comprimentoGlobalX = parametrosMalha[COMPRIMENTO_GLOBAL_X];
    unsigned int comprimentoGlobalY = parametrosMalha[COMPRIMENTO_GLOBAL_Y];
    unsigned int comprimentoGlobalZ = parametrosMalha[COMPRIMENTO_GLOBAL_Z];

    // Fixar a semente do gerador de números aleatórios para garantir repetibilidade
    srand(42);

    // Calcular o ponto central da malha para os queratinócitos
    int centroX = comprimentoGlobalX / 2;
    int centroY = comprimentoGlobalY / 2;
    int centroZ = comprimentoGlobalZ / 2;

    // Preencher a malha
    for (unsigned int x = 0; x < comprimentoGlobalX; x++)
    {
        for (unsigned int y = 0; y < comprimentoGlobalY; y++)
        {
            for (unsigned int z = 0; z < comprimentoGlobalZ; z++)
            {
                // Calcular o índice base para a posição na malha
                int indexBase = (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) +
                                (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) +
                                (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X]);

                // Preencher os valores da célula para cada tipo
                if (x == centroX && y == centroY && z == centroZ)
                {
                    // Queratinócitos apenas no centro
                    malha[CELULAS_K_OFFSET + indexBase] = 16.0f;
                }
                else
                {
                    malha[CELULAS_K_OFFSET + indexBase] = 0.0f;
                }

                // Distribuir valores aleatórios para as outras células
                malha[CELULAS_L_OFFSET + indexBase] = (float)(rand() % 10) / 10.0f;  // Inicializa L entre 0.0 e 1.0
                malha[CELULAS_M_OFFSET + indexBase] = (float)(rand() % 20) / 20.0f;  // Inicializa M entre 0.0 e 1.0
                malha[CELULAS_S_OFFSET + indexBase] = (float)(rand() % 15) / 15.0f;  // Inicializa S entre 0.0 e 1.0
                malha[CELULAS_W_OFFSET + indexBase] = 0.0f;  // Inicializa W como 0.0 em todo o domínio
            }
        }
    }
}


// // Função para ler os valores da malha e imprimir os resultados
// void LerPontosHIS(const float *malha, const int *parametrosMalha)
// {
//     for (unsigned int x = 0; x < parametrosMalha[COMPRIMENTO_GLOBAL_X]; x++)
//     {
//         for (unsigned int y = 0; y < parametrosMalha[COMPRIMENTO_GLOBAL_Y]; y++)
//         {
//             for (unsigned int z = 0; z < parametrosMalha[COMPRIMENTO_GLOBAL_Z]; z++)
//             {
//                 int index = (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) +
//                             (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) +
//                             (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X]);

//                 // Verificar se o ponto está dentro do intervalo de computação
//                 if (index >= parametrosMalha[OFFSET_COMPUTACAO] * MALHA_TOTAL_CELULAS &&
//                     index < (parametrosMalha[OFFSET_COMPUTACAO] + parametrosMalha[LENGTH_COMPUTACAO]) * MALHA_TOTAL_CELULAS)
//                 {
//                     printf("Posição (%d, %d, %d): L=%f M=%f S=%f W=%f K=%f\n", 
//                            x, y, z, 
//                            malha[CELULAS_L_OFFSET + index],
//                            malha[CELULAS_M_OFFSET + index],
//                            malha[CELULAS_S_OFFSET + index],
//                            malha[CELULAS_W_OFFSET + index],
//                            malha[CELULAS_K_OFFSET + index]);
//                 }
//                 else
//                 {
//                     printf("Posição (%d, %d, %d): L=0.0 M=0.0 S=0.0 W=0.0 K=0.0\n", x, y, z);
//                 }
//             }
//             printf("\n");
//         }
//     }
// }


// Função para ler e imprimir os valores da malha, mantendo a lógica original
void LerPontosHIS(float *malha, int *parametrosMalha)
{
    int counter = 0;  // Contador para células impressas
    
    for (unsigned int x = 0; x < parametrosMalha[COMPRIMENTO_GLOBAL_X]; x++)
    {
        for (unsigned int y = 0; y < parametrosMalha[COMPRIMENTO_GLOBAL_Y]; y++)
        {
            for (unsigned int z = 0; z < parametrosMalha[COMPRIMENTO_GLOBAL_Z]; z++)
            {
                // Calcular o índice base da célula na malha
                int indexBase = (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) +
                                (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) +
                                (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X]);

                // Verificar se a posição está dentro do intervalo de computação
                if ((CELULAS_L_OFFSET + indexBase) >= parametrosMalha[OFFSET_COMPUTACAO] * MALHA_TOTAL_CELULAS &&
                    (CELULAS_L_OFFSET + indexBase) < (parametrosMalha[OFFSET_COMPUTACAO] + parametrosMalha[LENGTH_COMPUTACAO]) * MALHA_TOTAL_CELULAS)
                {
                    // Incrementar o contador de células processadas
                    counter++;

                    // Imprimir valores de todas as células como na lógica original
                    printf("%f ", malha[CELULAS_L_OFFSET + indexBase]); // Célula L
                    printf("%f ", malha[CELULAS_M_OFFSET + indexBase]); // Célula M
                    printf("%f ", malha[CELULAS_S_OFFSET + indexBase]); // Célula S
                    printf("%f ", malha[CELULAS_W_OFFSET + indexBase]); // Célula W
                    printf("%f ", malha[CELULAS_K_OFFSET + indexBase]); // Célula K
                }
                else
                {
                    // Imprimir 0.0 para valores fora do intervalo de computação
                    printf("0.0 0.0 0.0 0.0 0.0 ");
                }
            }
            // Quebra de linha ao terminar a impressão para o eixo Z, mantendo o mesmo padrão
            printf("\n");
        }
    }

    // Imprimir o número total de células processadas, igual ao original
    printf("Cells printed: %d\n", counter);
}



int main(int argc, char** argv) {
     
    OpenCLWrapper openCL(argc, argv);
    openCL.InitDevices("CPU_DEVICES", 10);  
    openCL.setKernel("kernel_psoriase.cl", "ProcessarPontos");
    
    int x = 8, y = 8, z = 16;
    int tam = x * y * z * MALHA_TOTAL_CELULAS*2;  // Tamanho correto da malha
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
	
	
    // Definir atributos de kernel
	openCL.setAttribute(0, bMemObj);
    openCL.setAttribute(1, cMemObj);
	openCL.setAttribute(2, aMemObj);
	openCL.setBalancingTargetID(bMemObj);
	//openCL.Probing();
    for (int x = 0; x < 10; x++) {
		
	
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

		if(x  == 0){
        openCL.GatherResults(bMemObj, malhaAux);
        LerPontosHIS(malhaAux, parametros);
		}
        
		openCL.ExecuteKernel();
		
    }

	
	  
	double tempoFim = MPI_Wtime();
	std::cout<<"Tempo execução:"<<tempoFim-tempoInicio<<std::endl;
    // Liberar memória alocada
    delete[] parametros;
    delete[] malha;
    delete[] malhaAux;
	
    return 0;
}
