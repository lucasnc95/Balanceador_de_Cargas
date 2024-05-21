#include <stdio.h>
#include <iostream>
#include <mpi.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include "Balanceador.h"
#include "OpenCLWrapper.h"

using namespace std;


int main(int argc, char *argv[])
{

   
    int data[] = {1, 2, 3, 4, 5};
    int dataToKernel[] = {10, 20, 30, 40, 50}; // Dados de exemplo para DataToKernel
    const size_t N = sizeof(data) / sizeof(data[0]);
    const size_t Element_sz = sizeof(data[0]);
    const size_t div_size = 2;

    // Instanciando um objeto da classe Balanceador
    Balanceador balancer(argc, argv, static_cast<void*>(data), Element_sz, N, static_cast<void*>(dataToKernel), div_size);



    cout<<"Fim do programa..."<<endl;

return 0;

}

