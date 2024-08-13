#include "OpenCLWrapper.h"
#include <iostream>

int main(int argc, char **argv) {
    // Inicializa o OpenCLWrapper
    
    OpenCLWrapper openCL(argc, argv);
    

    openCL.InitDevices("ALL_DEVICES", 10);  
    openCL.setKernel("kernel.cl", "vectorAdd");
    const int N = 16;
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
    openCL.setLoadBalancer(sizeof(float), N, 1, 2);
    int aMemObj = openCL.AllocateMemoryObject(N * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, a);
    int bMemObj = openCL.AllocateMemoryObject(N * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, b);
    int resultMemObj = openCL.AllocateMemoryObject(N * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, result);
    
    openCL.setAttribute(0, aMemObj);
    openCL.setAttribute(1, bMemObj);
    openCL.setAttribute(2, resultMemObj);
    openCL.setBalancingTargetID(2);
   // openCL.WriteObject(aMemObj, (char*)a, 0, N*sizeof(float));
   // openCL.WriteObject(bMemObj, (char*)b, 0, N*sizeof(float));
    openCL.Probing();
    for (int x = 0; x < 100; x++){
    
        openCL.ExecuteKernel();
    
     for (int i = 0; i < N; ++i) {
         a[i] += 1.0f+float(i);
         b[i] += 2.0f+float(i);
       
     }
     openCL.WriteObject(aMemObj, (char*)a, 0, N*sizeof(float));
     openCL.WriteObject(bMemObj, (char*)b, 0, N*sizeof(float));
    
    }
    
    
    openCL.GatherResults(resultMemObj, result);
    float resultF = 0;
    for (int i = 0; i < N; ++i) {
         std::cout << "result[" << i << "] = " << result[i]<<std::endl;
       
         }
    


    delete[] a;
    delete[] b;
    delete[] result;

   
    return 0;
}

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



