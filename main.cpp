#include "OpenCLWrapper.h"
#include <iostream>

int main(int argc, char **argv) {
    // Inicializa o OpenCLWrapper
    
    OpenCLWrapper openCL(argc, argv);
    //      int t= 1;
    //  while (t);

    openCL.InitDevices("ALL_DEVICES", 10);  
    openCL.setKernel("kernel.cl", "vectorAdd");
    const int N = 64;
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
    
    
    for (int x = 0; x < 5; x++){
    
        openCL.ExecuteKernel();
    
     for (int i = 0; i < N; ++i) {
         a[i] = 1.0f+float(5);
         b[i] = 2.0f+float(3);
       
     }
     openCL.WriteObject(aMemObj, (char*)a, 0, N*sizeof(float));
     openCL.WriteObject(bMemObj, (char*)b, 0, N*sizeof(float));
    
    }
    
   

    

    
    openCL.GatherResults(resultMemObj, result2);

    for (int i = 0; i < N; ++i) {
         std::cout << "result[" << i << "] = " << result2[i] << std::endl;
         }

    
   




    delete[] a;
    delete[] b;
    delete[] result;

   
    return 0;
}




