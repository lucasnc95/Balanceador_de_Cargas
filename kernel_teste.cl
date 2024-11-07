__kernel void kernelSomaVizinhos(
    __global float* malhaAtual,         // Malha que será atualizada
    __global float* malhaAnterior,      // Malha da iteração anterior
    __constant int* parametrosMalhaGPU) // Parâmetros da malha
{
    int dimX = parametrosMalhaGPU[2];
    int dimY = parametrosMalhaGPU[3];
    int dimZ = parametrosMalhaGPU[4];

    int globalThreadID = get_global_id(0);
    int posZ = globalThreadID / (dimY * dimX);
    int posY = (globalThreadID % (dimY * dimX)) / dimX;
    int posX = globalThreadID % dimX;

    if (posZ >= dimZ) return;

    // Índice linear da célula atual
    int index = posZ * dimY * dimX + posY * dimX + posX;

    // Valores das células vizinhas com condições de borda
    float centro = malhaAnterior[index];
    float xp = (posX < dimX - 1) ? malhaAnterior[index + 1] : centro;
    float xm = (posX > 0) ? malhaAnterior[index - 1] : centro;
    float yp = (posY < dimY - 1) ? malhaAnterior[index + dimX] : centro;
    float ym = (posY > 0) ? malhaAnterior[index - dimX] : centro;
    float zp = (posZ < dimZ - 1) ? malhaAnterior[index + dimY * dimX] : centro;
    float zm = (posZ > 0) ? malhaAnterior[index - dimY * dimX] : centro;

    // Soma dos valores das vizinhas e da célula central
    float novoValor = centro + xp + xm + yp + ym + zp + zm;

    // Armazena o valor atualizado
    malhaAtual[index] += novoValor;
}
