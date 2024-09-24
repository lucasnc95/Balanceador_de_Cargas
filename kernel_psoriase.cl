// Definições para os tipos de células
#define MALHA_TOTAL_CELULAS 5
#define CELULA_L 0    // Células T (L)
#define CELULA_M 1    // Células dendríticas (M)
#define CELULA_S 2    // Citocinas (S)
#define CELULA_W 3    // Medicamento imunológico (W)
#define CELULA_K 4    // Queratinócitos (K)

// Cada célula ocupa 5 posições na malha 
#define CELULAS_SINGLE_CELL_SIZEOF 8

// Tamanho total para armazenar todas as células (em posições)
#define CELULAS_SIZEOF 40

// Offsets para as células na malha
#define CELULAS_L_OFFSET 0    // Células T - L
#define CELULAS_M_OFFSET 8    // Células dendríticas - M
#define CELULAS_S_OFFSET 16   // Citocinas - S
#define CELULAS_W_OFFSET 24   // Medicamento imunológico - W
#define CELULAS_K_OFFSET 32   // Queratinócitos - K



#define CELULAS_POSICAO_OR_OFFSET 0  
#define CELULAS_POSICAO_XP_OFFSET 1  
#define CELULAS_POSICAO_XM_OFFSET 2  
#define CELULAS_POSICAO_YP_OFFSET 3  
#define CELULAS_POSICAO_YM_OFFSET 4  
#define CELULAS_POSICAO_ZP_OFFSET 5  
#define CELULAS_POSICAO_ZM_OFFSET 6  
#define CELULAS_NOVO_VALOR_OFFSET 7  


//Informacoes de acesso à estrutura "parametrosMalhaGPU".
#define OFFSET_COMPUTACAO		0
#define LENGTH_COMPUTACAO		1
#define MALHA_DIMENSAO_X		2
#define MALHA_DIMENSAO_Y		3
#define MALHA_DIMENSAO_Z		4
#define MALHA_DIMENSAO_POSICAO_Z	5
#define MALHA_DIMENSAO_POSICAO_Y	6
#define MALHA_DIMENSAO_POSICAO_X	7
#define MALHA_DIMENSAO_CELULAS		8
#define NUMERO_PARAMETROS_MALHA		9

// Parâmetros biológicos e fracionários
__constant float deltaT = 1e-6;
__constant float k1 = 5.0, k2 = 4.0;  // Capacidades de suporte para L e M
__constant float r1 = 0.5, r2 = 0.4;  // Taxas de crescimento para L e M
__constant float gamma1 = 0.065, gamma2 = 0.05, gamma3 = 3.0;  // Interações de L e M com S
__constant float beta1 = 0.01, beta2 = 0.01;
__constant float theta1 = 0.5, theta2 = 0.7, theta3 = 0.5;
__constant float mu1 = 0.07, mu2 = 0.02, mu3 = 0.7;
__constant float deltaX = 0.1, deltaY = 0.1, deltaZ = 0.1;

// Função de Laplace para difusão
float Laplaciano(int celulaOffset, float *celulas, int xPosicaoGlobal, int yPosicaoGlobal, int zPosicaoGlobal, __constant int *parametrosMalhaGPU) {
    return ((xPosicaoGlobal > 0 && xPosicaoGlobal < parametrosMalhaGPU[MALHA_DIMENSAO_X]-1) ? 
           (celulas[celulaOffset + CELULAS_POSICAO_XP_OFFSET] - 2 * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET] + celulas[celulaOffset + CELULAS_POSICAO_XM_OFFSET]) / (deltaX * deltaX) : 0.0f) +
           ((yPosicaoGlobal > 0 && yPosicaoGlobal < parametrosMalhaGPU[MALHA_DIMENSAO_Y]-1) ? 
           (celulas[celulaOffset + CELULAS_POSICAO_YP_OFFSET] - 2 * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET] + celulas[celulaOffset + CELULAS_POSICAO_YM_OFFSET]) / (deltaY * deltaY) : 0.0f) +
           ((zPosicaoGlobal > 0 && zPosicaoGlobal < parametrosMalhaGPU[MALHA_DIMENSAO_Z]-1) ? 
           (celulas[celulaOffset + CELULAS_POSICAO_ZP_OFFSET] - 2 * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET] + celulas[celulaOffset + CELULAS_POSICAO_ZM_OFFSET]) / (deltaZ * deltaZ) : 0.0f);
}

// Função para resolver as EDOs fracionárias de Caputo
void CalcularPontos(float *celulas, int xPosicaoGlobal, int yPosicaoGlobal, int zPosicaoGlobal, __constant int *parametrosMalhaGPU) {
    // Valores de cada célula na posição atual da malha
    float L = celulas[CELULA_L * CELULAS_SINGLE_CELL_SIZEOF + CELULAS_POSICAO_OR_OFFSET];
    float M = celulas[CELULA_M * CELULAS_SINGLE_CELL_SIZEOF + CELULAS_POSICAO_OR_OFFSET];
    float S = celulas[CELULA_S * CELULAS_SINGLE_CELL_SIZEOF + CELULAS_POSICAO_OR_OFFSET];
    float W = celulas[CELULA_W * CELULAS_SINGLE_CELL_SIZEOF + CELULAS_POSICAO_OR_OFFSET];
    float K = celulas[CELULA_K * CELULAS_SINGLE_CELL_SIZEOF + CELULAS_POSICAO_OR_OFFSET];  // Queratinócito fixo

    // Equações diferenciais para L, M, S e W com base no modelo
    float dL_dt = r1 * L * (1 - L / k1) + gamma1 * L * S / (gamma2 + S) - beta1 * L * M - mu1 * L;
    float dM_dt = r2 * M * (1 - M / k2) - beta2 * L * M - mu2 * M;
    float dS_dt = (theta1 + W) * L * S / (theta2 + S) + theta3 * S - mu3 * S;
    float dW_dt = -gamma3 * W;
    
    // Aplicar difusão às células L, M, S e W
    dL_dt += Laplaciano(CELULA_L * CELULAS_SINGLE_CELL_SIZEOF, celulas, xPosicaoGlobal, yPosicaoGlobal, zPosicaoGlobal, parametrosMalhaGPU);
    dM_dt += Laplaciano(CELULA_M * CELULAS_SINGLE_CELL_SIZEOF, celulas, xPosicaoGlobal, yPosicaoGlobal, zPosicaoGlobal, parametrosMalhaGPU);
    dS_dt += Laplaciano(CELULA_S * CELULAS_SINGLE_CELL_SIZEOF, celulas, xPosicaoGlobal, yPosicaoGlobal, zPosicaoGlobal, parametrosMalhaGPU);
    dW_dt += Laplaciano(CELULA_W * CELULAS_SINGLE_CELL_SIZEOF, celulas, xPosicaoGlobal, yPosicaoGlobal, zPosicaoGlobal, parametrosMalhaGPU);

    // Atualizar novos valores
    celulas[CELULA_L * CELULAS_SINGLE_CELL_SIZEOF + CELULAS_NOVO_VALOR_OFFSET] = L + deltaT * dL_dt;
    celulas[CELULA_M * CELULAS_SINGLE_CELL_SIZEOF + CELULAS_NOVO_VALOR_OFFSET] = M + deltaT * dM_dt;
    celulas[CELULA_S * CELULAS_SINGLE_CELL_SIZEOF + CELULAS_NOVO_VALOR_OFFSET] = S + deltaT * dS_dt;
    celulas[CELULA_W * CELULAS_SINGLE_CELL_SIZEOF + CELULAS_NOVO_VALOR_OFFSET] = W + deltaT * dW_dt;
}

// Kernel principal para processar os pontos da malha
__kernel void ProcessarPontos(__global float *malhaPrincipalAtual, __global float *malhaPrincipalAnterior, __constant int *parametrosMalhaGPU) {
    int globalThreadID = get_global_id(0);

    float celulas[CELULAS_SIZEOF];

    // Descobrir posição 3D local na malha
    int zPosicaoGlobal = globalThreadID / (parametrosMalhaGPU[MALHA_DIMENSAO_Y] * parametrosMalhaGPU[MALHA_DIMENSAO_X]);
    int yPosicaoGlobal = (globalThreadID % (parametrosMalhaGPU[MALHA_DIMENSAO_Y] * parametrosMalhaGPU[MALHA_DIMENSAO_X])) / parametrosMalhaGPU[MALHA_DIMENSAO_X];
    int xPosicaoGlobal = (globalThreadID % (parametrosMalhaGPU[MALHA_DIMENSAO_Y] * parametrosMalhaGPU[MALHA_DIMENSAO_X])) % parametrosMalhaGPU[MALHA_DIMENSAO_X];

    if (zPosicaoGlobal >= parametrosMalhaGPU[MALHA_DIMENSAO_Z]) {
        return;
    }

    // Preencher o array de células com dados da malha anterior
    int malhaIndex = zPosicaoGlobal * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Z] +
                     yPosicaoGlobal * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Y] +
                     xPosicaoGlobal * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_X];

    for (int count = 0; count < MALHA_TOTAL_CELULAS; count++) {
        // Origem da célula
        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_OR_OFFSET] = 
            malhaPrincipalAnterior[malhaIndex + (count * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS])];

        // Posições vizinhas (ZP, ZM, XP, XM, YP, YM)
        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_ZP_OFFSET] = 
            (zPosicaoGlobal + 1 < parametrosMalhaGPU[MALHA_DIMENSAO_Z]) ? 
                malhaPrincipalAnterior[malhaIndex + ((count * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS]) + parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Z])] : 0.0f;
        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_ZM_OFFSET] = 
            (zPosicaoGlobal - 1 >= 0) ? 
                malhaPrincipalAnterior[malhaIndex + ((count * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS]) - parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Z])] : 0.0f;

        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_XP_OFFSET] = 
            (xPosicaoGlobal + 1 < parametrosMalhaGPU[MALHA_DIMENSAO_X]) ? 
                malhaPrincipalAnterior[malhaIndex + ((count * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS]) + parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_X])] : 0.0f;
        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_XM_OFFSET] = 
            (xPosicaoGlobal - 1 >= 0) ? 
                malhaPrincipalAnterior[malhaIndex + ((count * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS]) - parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_X])] : 0.0f;

        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_YP_OFFSET] = 
            (yPosicaoGlobal + 1 < parametrosMalhaGPU[MALHA_DIMENSAO_Y]) ? 
                malhaPrincipalAnterior[malhaIndex + ((count * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS]) + parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Y])] : 0.0f;
        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_YM_OFFSET] = 
            (yPosicaoGlobal - 1 >= 0) ? 
                malhaPrincipalAnterior[malhaIndex + ((count * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS]) - parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Y])] : 0.0f;
    }

    // Calcular novos valores para as células
    CalcularPontos(celulas, xPosicaoGlobal, yPosicaoGlobal, zPosicaoGlobal, parametrosMalhaGPU);

    // Atualizar malha com novos valores calculados
    for (int count = 0; count < MALHA_TOTAL_CELULAS; count++) {
        malhaPrincipalAtual[malhaIndex + (count * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS])] = 
            celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_NOVO_VALOR_OFFSET];
    }
}