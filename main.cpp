#include "OpenCLWrapper.h"
#include <iostream>

//Tipos de celulas.
#define CELULA_A		0
#define CELULA_MR		1
#define CELULA_MA		2
#define CELULA_N		3
#define CELULA_CH		4
#define CELULA_ND		5
#define CELULA_G		6
#define CELULA_CA		7
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
	//*malha = new float[parametrosMalha[COMPRIMENTO_GLOBAL_X]*parametrosMalha[COMPRIMENTO_GLOBAL_Y]*parametrosMalha[COMPRIMENTO_GLOBAL_Z]*MALHA_TOTAL_CELULAS*sizeof(float)];
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
   
	double	tempoInicio = MPI_Wtime();
   
    OpenCLWrapper openCL(argc, argv);
    openCL.InitDevices("ALL_DEVICES", 10);  
    openCL.setKernel("kernels.cl", "ProcessarPontos");

    int x = 8, y = 8, z = 16;
    int tam = x * y * z * MALHA_TOTAL_CELULAS;  // Tamanho correto da malha
    int *parametros = new int[NUMERO_PARAMETROS_MALHA];
    float *malha = new float[tam];  // Alocar a malha corretamente

   
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
    for (int x = 0; x < 10000; x++) {
		
	// openCL.GatherResults(bMemObj, malhaAux);
	// openCL.WriteObject(cMemObj, (char *) malhaAux, 0, tam*sizeof(float));
		 if (x % 2 == 0) {
            openCL.setAttribute(0, bMemObj);
            openCL.setAttribute(1, cMemObj);
           // openCL.setBalancingTargetID(bMemObj);
        } 
		else {
            openCL.setAttribute(0, cMemObj);
            openCL.setAttribute(1, bMemObj);
          //  openCL.setBalancingTargetID(cMemObj);
        }
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





// return 0;




// }


// int main(int argc, char **argv) {
//     OpenCLWrapper openCL(argc, argv);
    
//     openCL.InitDevices("ALL_DEVICES", 10);  
//     openCL.setKernel("kernel.cl", "vectorAdd");
//     long unsigned int N = 33554432*2*2;
//     float* a = new float[N];
//     float* b = new float[N];
//     float* result = new float[N];
//     float* result2 = new float[N];
 
//     for (int i = 0; i < N; ++i) {
//         a[i] = 1.0f;
//         b[i] = 2.0f;
//         result[i] = 0.0f; 
//     }

//     openCL.setLoadBalancer(sizeof(float), N, 1, 8);
//     int aMemObj = openCL.AllocateMemoryObject(N * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, a);
//     int bMemObj = openCL.AllocateMemoryObject(N * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, b);
//     int resultMemObj = openCL.AllocateMemoryObject(N * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, result);
    
//     openCL.setAttribute(0, aMemObj);
//     openCL.setAttribute(1, bMemObj);
//     openCL.setAttribute(2, resultMemObj);
//     openCL.setBalancingTargetID(resultMemObj);

//     double inicio = MPI_Wtime();
//     for (int x = 0; x < 1000; x++) {
        
// 		if (x%100 == 0)
// 			openCL.Probing();
		
// 		openCL.ExecuteKernel();    
       

//         for (int i = 0; i < N; i++) {
//             a[i] += 1.0f;
//             b[i] += 2.0f;
//         }

//         openCL.WriteObject(aMemObj, (char*)a, 0, N*sizeof(float));
//         openCL.WriteObject(bMemObj, (char*)b, 0, N*sizeof(float));
//     }
//     double fim = MPI_Wtime();
    
//     openCL.GatherResults(resultMemObj, result);
    
//     std::cout<<"Tempo de execução paralela: "<<(fim - inicio)<<std::endl;
//     double inicio_seq = MPI_Wtime();
//     for (int i = 0; i < N; i++) {
//         a[i] = 1.0f;
//         b[i] = 2.0f;
//         result2[i] = 0.0f;
//     }
    
//     for (int x = 0; x < 1000; x++) {
//         for (int i = 0; i < N; i++) {
//             result2[i] += a[i] + b[i] + 1.0f;	
//         }
//         for (int i = 0; i < N; i++) {
//             a[i] += 1.0f;
//             b[i] += 2.0f;
//         }
//     }

//     double fim_seq = MPI_Wtime();

//     std::cout<<"Tempo de execução sequencial: "<<(fim_seq - inicio_seq)<<std::endl;

//     // for (int i = 0; i < N; i++) {
//     //     if (result[i] != result2[i]) {
//     //         std::cout << "Mismatch at index " << i << ": expected " << result2[i] << " but got " << result[i] << std::endl;
//     //         break;
//     //     }
//     // }

//     openCL.FinishParallelProcessor();
//     delete[] a;
//     delete[] b;
//     delete[] result;

//     return 0;
// }



// #include "OpenCLWrapper.h"
// #include <iostream>

// int main(int argc, char **argv) {
//     // Inicializa o OpenCLWrapper
//     OpenCLWrapper openCL(argc, argv);

//     // Inicializa os dispositivos OpenCL
//     openCL.InitDevices("ALL_DEVICES", 10);  

//     // Configura o kernel
//     openCL.setKernel("kernel.cl", "vectorMulAdd");

//     int N = 2048;
//     float* a = new float[N];
//     float* b = new float[N];
//     float* c = new float[N];  // Vetor C para armazenar os resultados
//     float* result2 = new float[N];

//     // Inicializa os vetores A, B e C
//     for (int i = 0; i < N; ++i) {
//         a[i] = 0.1f ;
//         b[i] = 0.2f ;
//         c[i] = 0.0f;  // Inicialmente zero para acumulação
//     }

//     // Configura o balanceador de carga
//     openCL.setLoadBalancer(result2, N, 1);

//     // Aloca os objetos de memória
//     int aMemObj = openCL.AllocateMemoryObject(N * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, a);
//     int bMemObj = openCL.AllocateMemoryObject(N * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, b);
//     int cMemObj = openCL.AllocateMemoryObject(N * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, c);
//     //int nMemObj = openCL.AllocateMemoryObject(sizeof(int), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &N);

//     // openCL.WriteObject(nMemobj, (char*)N, 0,sizeof(int));
//     // Configura os atributos do kernel
//     openCL.setAttribute(0, aMemObj);
//     openCL.setAttribute(1, bMemObj);
//     openCL.setAttribute(2, cMemObj);
//    // openCL.setAttribute(3, nMemObj); // Passa o tamanho do vetor N como quarto argumento

//     // Executa o kernel
//     openCL.ExecuteKernel();

//     // Coleta os resultados do vetor C
//     openCL.GatherResults(cMemObj, c);

//     // Exibe os resultados
//     for (int i = 0; i < N; ++i) {
//         std::cout << "c[" << i << "] = " << c[i] << std::endl;
//     }

//     // Libera a memória
//     delete[] a;
//     delete[] b;
//     delete[] c;

//     return 0;
// }



