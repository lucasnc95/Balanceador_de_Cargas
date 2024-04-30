#include <stdio.h>
#include <mpi.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include "Balanceador_lib.h"
#include "OpenCLWrapper.h"


int main(int argc, char *argv[])
{
	
int malha[3] = {20,20,20};
Balanceador b(argc, argv, malha);

double ***m;

b.SetMalha(m);


return 0;

}

