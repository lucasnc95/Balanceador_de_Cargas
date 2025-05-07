#include "OpenCLWrapper.h"
#include <iostream>

int main(int argc, char** argv) {
     
    OpenCLWrapper openCL(argc, argv);
    openCL.InitDevices("ALL_DEVICES", 10);  
    openCL.setKernel("kernel_teste.cl", "kernelSomaVizinhos");

    
    int tam = 6;  
    float *malha = new float[tam];  

    double	tempoInicio = MPI_Wtime();
    
	for(int i = 0; i < tam; i++)
	malha[i] = 1;
	int total_elements = 6;

    // Configuração do balanceador de carga no OpenCLWrapper
    int sub = 1  ;
    openCL.setLoadBalancer(sizeof(float), total_elements, 1, sub);

    // Alocar objetos de memória OpenCL
    int bMemObj = openCL.AllocateMemoryObject(tam * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, malha);
    int cMemObj = openCL.AllocateMemoryObject(tam * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, malha);

    float *malhaAux = new float[tam];  // Para armazenar os resultados
	int *vetArgs = new int[2];
	vetArgs[0] = bMemObj;
	vetArgs[1] = cMemObj;

	int id;
    // Definir atributos de kernel
	openCL.setAttribute(0, bMemObj);
    openCL.setAttribute(1, cMemObj);
    openCL.setSubdomainBoundary(sub, 2, vetArgs);
	openCL.setBalancingTargetID(bMemObj);
	openCL.Probing();
    for (int x = 0; x < 3; x++) {
		 if (x % 2 == 0) {
           openCL.setAttribute(0, bMemObj);
           openCL.setAttribute(1, cMemObj);
           openCL.setBalancingTargetID(bMemObj);
           openCL.setSwapBuffer(cMemObj);
		   id = bMemObj;
        } 
		else {
            openCL.setAttribute(0, cMemObj);
            openCL.setAttribute(1, bMemObj);
            openCL.setBalancingTargetID(cMemObj);
            openCL.setSwapBuffer(bMemObj);
			id = cMemObj;
        }
	
		//openCL.ExecuteKernel();        
    }

		openCL.GatherResults(id, malhaAux);
	
    for(int i = 0; i < tam; i++)
	std::cout<<"vet["<<i<<"] = "<<malhaAux[i]<<std::endl;

	double tempoFim = MPI_Wtime();
	std::cout<<"Tempo execução:"<<tempoFim-tempoInicio<<std::endl;
   
    delete[] malha;
    delete[] malhaAux;
	
    return 0;
}

