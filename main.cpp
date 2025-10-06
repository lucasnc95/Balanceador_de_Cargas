/*#include "OpenCLWrapper.h"
#include <iostream>

//Tipos de celulas.
#define CELULA_A		    0
#define CELULA_MR			1
#define CELULA_MA			2
#define CELULA_N		    3
#define CELULA_CH			4
#define CELULA_ND			5
#define CELULA_G		    6
#define CELULA_CA			7
#define MALHA_TOTAL_CELULAS	8

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

void InicializarPontosHIS(float *malha,  int *parametrosMalha)
{
	
	for(unsigned int x = 0; x < parametrosMalha[COMPRIMENTO_GLOBAL_X]; x++)
	{	
		for(unsigned int y = 0; y < parametrosMalha[COMPRIMENTO_GLOBAL_Y]; y++)
		{
			for(unsigned int z = 0; z < parametrosMalha[COMPRIMENTO_GLOBAL_Z]; z++)
			{
				if(z >= (0.75f*parametrosMalha[COMPRIMENTO_GLOBAL_Z]))
				{
					(malha)[(CELULA_A * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 100.0f;
				}
				else
				{
					(malha)[(CELULA_A * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
				}

				(malha)[(CELULA_MR * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 1.0f;
				(malha)[(CELULA_MA * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
				(malha)[(CELULA_N * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
				(malha)[(CELULA_CH * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
				(malha)[(CELULA_ND * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
				(malha)[(CELULA_G * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
				(malha)[(CELULA_CA * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X])] = 0.0f;
			}
		}
	}
}

void LerPontosHIS( float *malha, int *parametrosMalha)
{	int counter = 0;
	for(unsigned int x = 0; x < parametrosMalha[COMPRIMENTO_GLOBAL_X]; x++)
	{
		for(unsigned int y = 0; y < parametrosMalha[COMPRIMENTO_GLOBAL_Y]; y++)
		{
			for(unsigned int z = 0; z < parametrosMalha[COMPRIMENTO_GLOBAL_Z]; z++)
			{
				if((CELULA_A * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X]) >= parametrosMalha[OFFSET_COMPUTACAO]*MALHA_TOTAL_CELULAS && (CELULA_A * parametrosMalha[MALHA_DIMENSAO_CELULAS]) + (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) + (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) + (x *parametrosMalha[MALHA_DIMENSAO_POSICAO_X]) < (parametrosMalha[OFFSET_COMPUTACAO]+parametrosMalha[LENGTH_COMPUTACAO])*MALHA_TOTAL_CELULAS)
				{	counter++;
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

std::cout<<"Cells printed: "<<counter<<std::endl;

}




int main(int argc, char** argv) {
     
    OpenCLWrapper openCL(argc, argv);
    openCL.InitDevices("ALL_DEVICES", 10);  
    openCL.setKernel("kernels.cl", "ProcessarPontos");

    int x = 10, y = 10, z = 10;
    int tam = x * y * z * MALHA_TOTAL_CELULAS;   
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
    openCL.setSubdomainBoundary(sub, 2, vetArgs);
	openCL.setBalancingTargetID(bMemObj);
	openCL.Probing();
    for (int x = 0; x < 10000; x++) {
		
		 if (x % 2 == 0) {
            openCL.setAttribute(0, bMemObj);
            openCL.setAttribute(1, cMemObj);
            openCL.setBalancingTargetID(bMemObj);
            openCL.setSwapBufferID(cMemObj);
        } 
		else {
            openCL.setAttribute(0, cMemObj);
            openCL.setAttribute(1, bMemObj);
            openCL.setBalancingTargetID(cMemObj);
            openCL.setSwapBufferID(bMemObj);
        }

//		if(x > 0 && x % 1000 == 0)
//		openCL.LoadBalancing();


		openCL.ExecuteKernel();


		
		
        
    }

		int rank = openCL.getWorldRank();
        openCL.GatherResults(bMemObj, malhaAux);
//	printf("Meu rank (main): %i", rank);
        if(rank == 0)
        {
        LerPontosHIS(malhaAux, parametros);
        double tempoFim = MPI_Wtime();
        std::cout<<"Tempo execução:"<<tempoFim-tempoInicio<<std::endl;
        }

    delete[] parametros;
    delete[] malha;
    delete[] malhaAux;
    return 0;
}

*/

#include "OpenCLWrapper.h"
#include <iostream>

int main(int argc, char** argv) {

    OpenCLWrapper openCL(argc, argv);
    openCL.InitDevices("ALL_DEVICES", 10);
    openCL.setKernel("kernel_teste.cl", "kernelSomaVizinhos");

    int tam = 20;
    float *malha = new float[tam];

    double	tempoInicio = MPI_Wtime();

	for(int i = 0; i < tam; i++)
	malha[i] = 1;
	
    int sub = 1;
    openCL.setLoadBalancer(sizeof(float), tam, 1, sub);

    int bMemObj = openCL.AllocateMemoryObject(tam * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, malha);
    int cMemObj = openCL.AllocateMemoryObject(tam * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, malha);
	std::cout<<"b memObj: "<<bMemObj<<" c memObj: "<<cMemObj<<std::endl;

    float *malhaAux = new float[tam];  // Para armazenar os resultados
	int *vetArgs = new int[2];
	vetArgs[0] = bMemObj;
	vetArgs[1] = cMemObj;
	float *malhaAux2 = new float[tam];
	int id;
    
	openCL.setSubdomainBoundary(sub, 2, vetArgs);
	
        int rank = openCL.getWorldRank();
    for (int x = 0; x < 3; x++) {
		 
		 if (x % 2 == 0) {

           openCL.setAttribute(0, bMemObj);
           openCL.setAttribute(1, cMemObj);
           openCL.setBalancingTargetID(bMemObj);
           openCL.setSwapBufferID(cMemObj);

        }
	else {
           openCL.setAttribute(0, cMemObj);
    	    openCL.setAttribute(1, bMemObj);
            openCL.setBalancingTargetID(cMemObj);
            openCL.setSwapBufferID(bMemObj);

       }
	if(x == 0)
		openCL.Probing();
	else
		openCL.LoadBalancing();

	openCL.ExecuteKernel();
}
	openCL.GatherResults(bMemObj, malhaAux);
	openCL.GatherResults(cMemObj, malhaAux2);
	if(rank == 0){
        for(int i = 0; i < tam; i++)
            std::cout<<"vet["<<i<<"] = "<<malhaAux[i]<<std::endl;

	double tempoFim = MPI_Wtime();
	std::cout<<"Tempo execução:"<<tempoFim-tempoInicio<<std::endl;
	MPI_Barrier(MPI_COMM_WORLD);
   }
    delete[] malha;
    delete[] malhaAux;
    return 0;
}


