// kernelSomaVizinhos.cl
__kernel void kernelSomaVizinhos(__global float* vetor, __global float* vetor_aux) {
    int N = 20;
    int i = get_global_id(0);
    if (i < N) {
        float left  = (i > 0)     ? vetor_aux[i - 1] : 0.0f;
        float right = (i < N - 1) ? vetor_aux[i + 1] : 0.0f;
        float valor_atual = vetor_aux[i];
                
	// printf("Indice: %d, Valor atual: %f, Left: %f, Right: %f\n", i, valor_atual, left, right);
        
        // Soma dos valores dos vizinhos com o valor atual.
        vetor[i] = valor_atual + left + right;
    }
}
