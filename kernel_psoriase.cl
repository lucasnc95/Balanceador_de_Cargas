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
float Laplaciano(int celulaOffset, float *celulas, int xPosicaoGlobal, int yPosicaoGlobal, int zPosicaoGlobal, __constant int *parametrosMalhaGPU)
{
    return 
        // Componente X
        (xPosicaoGlobal > 0 && xPosicaoGlobal < (parametrosMalhaGPU[MALHA_DIMENSAO_X] - 1)) 
        ? (celulas[celulaOffset + CELULAS_POSICAO_XP_OFFSET] 
           - 2 * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET] 
           + celulas[celulaOffset + CELULAS_POSICAO_XM_OFFSET]) / (deltaX * deltaX) 
        : ((((parametrosMalhaGPU[MALHA_DIMENSAO_X] - 1) - xPosicaoGlobal) / (float)(parametrosMalhaGPU[MALHA_DIMENSAO_X] - 1)) 
           * ((celulas[celulaOffset + CELULAS_POSICAO_XP_OFFSET] - celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET]) / (deltaX * deltaX)) 
           + (xPosicaoGlobal / (float)(parametrosMalhaGPU[MALHA_DIMENSAO_X] - 1)) 
           * ((celulas[celulaOffset + CELULAS_POSICAO_XM_OFFSET] - celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET]) / (deltaX * deltaX)))

        // Componente Y
        + ((yPosicaoGlobal > 0 && yPosicaoGlobal < (parametrosMalhaGPU[MALHA_DIMENSAO_Y] - 1)) 
        ? (celulas[celulaOffset + CELULAS_POSICAO_YP_OFFSET] 
           - 2 * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET] 
           + celulas[celulaOffset + CELULAS_POSICAO_YM_OFFSET]) / (deltaY * deltaY)
        : ((((parametrosMalhaGPU[MALHA_DIMENSAO_Y] - 1) - yPosicaoGlobal) / (float)(parametrosMalhaGPU[MALHA_DIMENSAO_Y] - 1)) 
           * ((celulas[celulaOffset + CELULAS_POSICAO_YP_OFFSET] - celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET]) / (deltaY * deltaY)) 
           + (yPosicaoGlobal / (float)(parametrosMalhaGPU[MALHA_DIMENSAO_Y] - 1)) 
           * ((celulas[celulaOffset + CELULAS_POSICAO_YM_OFFSET] - celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET]) / (deltaY * deltaY))))

        // Componente Z
        + ((zPosicaoGlobal > 0 && zPosicaoGlobal < (parametrosMalhaGPU[MALHA_DIMENSAO_Z] - 1)) 
        ? (celulas[celulaOffset + CELULAS_POSICAO_ZP_OFFSET] 
           - 2 * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET] 
           + celulas[celulaOffset + CELULAS_POSICAO_ZM_OFFSET]) / (deltaZ * deltaZ)
        : ((((parametrosMalhaGPU[MALHA_DIMENSAO_Z] - 1) - zPosicaoGlobal) / (float)(parametrosMalhaGPU[MALHA_DIMENSAO_Z] - 1)) 
           * ((celulas[celulaOffset + CELULAS_POSICAO_ZP_OFFSET] - celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET]) / (deltaZ * deltaZ)) 
           + (zPosicaoGlobal / (float)(parametrosMalhaGPU[MALHA_DIMENSAO_Z] - 1)) 
           * ((celulas[celulaOffset + CELULAS_POSICAO_ZM_OFFSET] - celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET]) / (deltaZ * deltaZ))));
}


float Quimiotaxia(int celulaOffset, float *celulas, int xPosicaoGlobal, int yPosicaoGlobal, int zPosicaoGlobal, __constant int *parametrosMalhaGPU)
{
    return 
        // Componente X
        ((xPosicaoGlobal > 0) 
            ? ((celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_XM_OFFSET]) > 0 
                ? -(celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_XM_OFFSET]) 
                  * celulas[celulaOffset + CELULAS_POSICAO_XM_OFFSET] / deltaX
                : -(celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_XM_OFFSET]) 
                  * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET] / deltaX)
            : 0.0f)
        + ((xPosicaoGlobal < parametrosMalhaGPU[MALHA_DIMENSAO_X] - 1) 
            ? ((celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_XP_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET]) > 0 
                ? (celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_XP_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET]) 
                  * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET] / deltaX
                : (celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_XP_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET]) 
                  * celulas[celulaOffset + CELULAS_POSICAO_XP_OFFSET] / deltaX)
            : 0.0f)) / deltaX

        // Componente Y
        + ((yPosicaoGlobal > 0) 
            ? ((celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_YM_OFFSET]) > 0 
                ? -(celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_YM_OFFSET]) 
                  * celulas[celulaOffset + CELULAS_POSICAO_YM_OFFSET] / deltaY
                : -(celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_YM_OFFSET]) 
                  * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET] / deltaY)
            : 0.0f)
        + ((yPosicaoGlobal < parametrosMalhaGPU[MALHA_DIMENSAO_Y] - 1) 
            ? ((celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_YP_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET]) > 0 
                ? (celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_YP_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET]) 
                  * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET] / deltaY
                : (celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_YP_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET]) 
                  * celulas[celulaOffset + CELULAS_POSICAO_YP_OFFSET] / deltaY)
            : 0.0f)) / deltaY

        // Componente Z
        + ((zPosicaoGlobal > 0) 
            ? ((celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_ZM_OFFSET]) > 0 
                ? -(celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_ZM_OFFSET]) 
                  * celulas[celulaOffset + CELULAS_POSICAO_ZM_OFFSET] / deltaZ
                : -(celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_ZM_OFFSET]) 
                  * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET] / deltaZ)
            : 0.0f)
        + ((zPosicaoGlobal < parametrosMalhaGPU[MALHA_DIMENSAO_Z] - 1) 
            ? ((celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_ZP_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET]) > 0 
                ? (celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_ZP_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET]) 
                  * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET] / deltaZ
                : (celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_ZP_OFFSET] - celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET]) 
                  * celulas[celulaOffset + CELULAS_POSICAO_ZP_OFFSET] / deltaZ)
            : 0.0f)) / deltaZ;
}


void CalcularPontos(float *celulas, int xPosicaoGlobal, int yPosicaoGlobal, int zPosicaoGlobal, __constant int *parametrosMalhaGPU)
{
    // Célula L (célula T) - Aplica difusão e quimiotaxia em direção a S
    float dL_dt = r1 * celulas[CELULAS_L_OFFSET + CELULAS_POSICAO_OR_OFFSET] * (1 - celulas[CELULAS_L_OFFSET + CELULAS_POSICAO_OR_OFFSET] / k1)
                  + gamma1 * celulas[CELULAS_L_OFFSET + CELULAS_POSICAO_OR_OFFSET] * celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET] 
                    / (gamma2 + celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET])
                  - beta1 * celulas[CELULAS_L_OFFSET + CELULAS_POSICAO_OR_OFFSET] * celulas[CELULAS_M_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                  - mu1 * celulas[CELULAS_L_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                  + Laplaciano(CELULAS_L_OFFSET, celulas, xPosicaoGlobal, yPosicaoGlobal, zPosicaoGlobal, parametrosMalhaGPU)
                  - Quimiotaxia(CELULAS_L_OFFSET, celulas, xPosicaoGlobal, yPosicaoGlobal, zPosicaoGlobal, parametrosMalhaGPU);

    // Célula M (célula dendrítica) - Aplica difusão e quimiotaxia em direção a S
    float dM_dt = r2 * celulas[CELULAS_M_OFFSET + CELULAS_POSICAO_OR_OFFSET] * (1 - celulas[CELULAS_M_OFFSET + CELULAS_POSICAO_OR_OFFSET] / k2)
                  - beta2 * celulas[CELULAS_L_OFFSET + CELULAS_POSICAO_OR_OFFSET] * celulas[CELULAS_M_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                  - mu2 * celulas[CELULAS_M_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                  + Laplaciano(CELULAS_M_OFFSET, celulas, xPosicaoGlobal, yPosicaoGlobal, zPosicaoGlobal, parametrosMalhaGPU)
                  - Quimiotaxia(CELULAS_M_OFFSET, celulas, xPosicaoGlobal, yPosicaoGlobal, zPosicaoGlobal, parametrosMalhaGPU);

    // Célula S (citocina) - Apenas difusão
    float dS_dt = (theta1 + celulas[CELULAS_W_OFFSET + CELULAS_POSICAO_OR_OFFSET]) * celulas[CELULAS_L_OFFSET + CELULAS_POSICAO_OR_OFFSET] 
                  * celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET] / (theta2 + celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET]) 
                  + theta3 * celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET] - mu3 * celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                  + Laplaciano(CELULAS_S_OFFSET, celulas, xPosicaoGlobal, yPosicaoGlobal, zPosicaoGlobal, parametrosMalhaGPU);

    // Célula W (medicamento imunológico) - Apenas difusão
    float dW_dt = -gamma3 * celulas[CELULAS_W_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                  + Laplaciano(CELULAS_W_OFFSET, celulas, xPosicaoGlobal, yPosicaoGlobal, zPosicaoGlobal, parametrosMalhaGPU);

    // Atualizar novos valores das células
    celulas[CELULAS_L_OFFSET + CELULAS_NOVO_VALOR_OFFSET] = max(0.0f, celulas[CELULAS_L_OFFSET + CELULAS_POSICAO_OR_OFFSET] + deltaT * dL_dt);
    celulas[CELULAS_M_OFFSET + CELULAS_NOVO_VALOR_OFFSET] = max(0.0f, celulas[CELULAS_M_OFFSET + CELULAS_POSICAO_OR_OFFSET] + deltaT * dM_dt);
    celulas[CELULAS_S_OFFSET + CELULAS_NOVO_VALOR_OFFSET] = max(0.0f, celulas[CELULAS_S_OFFSET + CELULAS_POSICAO_OR_OFFSET] + deltaT * dS_dt);
    celulas[CELULAS_W_OFFSET + CELULAS_NOVO_VALOR_OFFSET] = max(0.0f, celulas[CELULAS_W_OFFSET + CELULAS_POSICAO_OR_OFFSET] + deltaT * dW_dt);

    // O queratinócito (K) permanece no centro da malha e não é atualizado em cada passo de tempo
    celulas[CELULAS_K_OFFSET + CELULAS_NOVO_VALOR_OFFSET] = celulas[CELULAS_K_OFFSET + CELULAS_POSICAO_OR_OFFSET];
}



// Kernel principal para processar os pontos da malha
__kernel void ProcessarPontos(__global float *malhaPrincipalAtual, __global float *malhaPrincipalAnterior, __constant int *parametrosMalhaGPU)
{
    int globalThreadID = get_global_id(0);

    float celulas[CELULAS_SIZEOF];

    // Descobrir posição 3D local na malha
    int posicaoGlobalZ = (globalThreadID / (parametrosMalhaGPU[MALHA_DIMENSAO_Y] * parametrosMalhaGPU[MALHA_DIMENSAO_X]));
    int posicaoGlobalY = (globalThreadID % (parametrosMalhaGPU[MALHA_DIMENSAO_Y] * parametrosMalhaGPU[MALHA_DIMENSAO_X])) / parametrosMalhaGPU[MALHA_DIMENSAO_X];
    int posicaoGlobalX = (globalThreadID % (parametrosMalhaGPU[MALHA_DIMENSAO_Y] * parametrosMalhaGPU[MALHA_DIMENSAO_X])) % parametrosMalhaGPU[MALHA_DIMENSAO_X];

    if (posicaoGlobalZ >= parametrosMalhaGPU[MALHA_DIMENSAO_Z]) {
        return;
    }

    // Preencher células para calcular EDO's
    int malhaIndex = ((posicaoGlobalZ) * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Z]) 
                    + ((posicaoGlobalY) * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Y]) 
                    + ((posicaoGlobalX) * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_X]);

    for (int count = 0; count < MALHA_TOTAL_CELULAS; count++) {
        // Origem
        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_OR_OFFSET] = malhaPrincipalAnterior[malhaIndex + (count * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS])];

        // Vizinhança
        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_ZP_OFFSET] = ((posicaoGlobalZ + 1 < parametrosMalhaGPU[MALHA_DIMENSAO_Z])) 
        ? malhaPrincipalAnterior[malhaIndex + ((count) * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS]) + ((+1) * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Z])] : 0.0f;
        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_ZM_OFFSET] = ((posicaoGlobalZ - 1 >= 0)) 
        ? malhaPrincipalAnterior[malhaIndex + ((count) * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS]) + ((-1) * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Z])] : 0.0f;

        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_XP_OFFSET] = ((posicaoGlobalX + 1 < parametrosMalhaGPU[MALHA_DIMENSAO_X])) 
        ? malhaPrincipalAnterior[malhaIndex + ((count) * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS]) + ((+1) * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_X])] : 0.0f;
        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_XM_OFFSET] = ((posicaoGlobalX - 1 >= 0)) 
        ? malhaPrincipalAnterior[malhaIndex + ((count) * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS]) + ((-1) * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_X])] : 0.0f;

        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_YP_OFFSET] = ((posicaoGlobalY + 1 < parametrosMalhaGPU[MALHA_DIMENSAO_Y])) 
        ? malhaPrincipalAnterior[malhaIndex + ((count) * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS]) + ((+1) * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Y])] : 0.0f;
        celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_POSICAO_YM_OFFSET] = ((posicaoGlobalY - 1 >= 0)) 
        ? malhaPrincipalAnterior[malhaIndex + ((count) * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS]) + ((-1) * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Y])] : 0.0f;
    }

    // Calcular novos valores das células
    CalcularPontos(celulas, posicaoGlobalX, posicaoGlobalY, posicaoGlobalZ, parametrosMalhaGPU);

    // Atualizar malha com pontos calculados
    for (int count = 0; count < MALHA_TOTAL_CELULAS; count++) {
        malhaPrincipalAtual[malhaIndex + ((count) * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS])] = celulas[(count * CELULAS_SINGLE_CELL_SIZEOF) + CELULAS_NOVO_VALOR_OFFSET];
    }
}
