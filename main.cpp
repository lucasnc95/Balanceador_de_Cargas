#include "OpenCLWrapper.h"
#include <iostream>

int main(int argc, char **argv) {
    // Inicializa o OpenCLWrapper
    
    OpenCLWrapper openCL(argc, argv);
    //      int t= 1;
    //  while (t);

    openCL.InitDevices("ALL", 10);  
    openCL.setKernel("kernel.cl", "vectorAdd");
    const int N = 120;
    float* a = new float[N];
    float* b = new float[N];
    float* result = new float[N];
    float* result2 = new float[N];
 
    // Inicializa os vetores a e b
    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        result[i] = 0.0f; 
    }
    openCL.setLoadBalancer(result2, N, 1);
    int aMemObj = openCL.AllocateMemoryObject(N * sizeof(float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a);
    int bMemObj = openCL.AllocateMemoryObject(N * sizeof(float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b);
    int resultMemObj = openCL.AllocateMemoryObject(N * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, result);
    
    openCL.setAttribute(0, aMemObj);
    openCL.setAttribute(1, bMemObj);
    openCL.setAttribute(2, resultMemObj);
    
    long int tempo = 0;
    int intervalo = 100;
    for (int x = 0; x < 2; x++){
    //if(x % intervalo != 0){
        openCL.ExecuteKernel();
    
    // for (int i = 0; i < N; ++i) {
    //     a[i] = 1.0f+float(x);
    //     b[i] = 2.0f+float(x);
       
    // }
    // openCL.WriteObject(aMemObj, (char*)a, 0, N*sizeof(float));
    // openCL.WriteObject(bMemObj, (char*)b, 0, N*sizeof(float));
    }
    
   // else openCL.PrecisaoBalanceamento();

   // }

    
    openCL.GatherResults(resultMemObj, result2);

    for (int i = 0; i < N; ++i) {
         std::cout << "result[" << i << "] = " << result2[i] << std::endl;
         }

    tempo += openCL.GetEventTaskTicks(0,0);
    long int tempoTotal = openCL.GetEventTaskOverheadTicks(0,0);
    
    std::cout<<"Tempo de execução: "<<double(tempo)<<std::endl;
    std::cout<<"Tempo de execução total: "<<double(tempoTotal)<<std::endl;




    delete[] a;
    delete[] b;
    delete[] result;

   
    return 0;
}




