__kernel void vectorAdd(__global const float *a, __global const float *b, __global float *c) {
    int id = get_global_id(0);
    const long unsigned int N = 33554432*2*2;
    if(id < N)
        c[id] += (a[id] + b[id])+1.0f;
    
}

__kernel void vectorMulAdd(__global const float* A, 
                           __global const float* B, 
                           __global float* C, 
                           const int N) {
    int idx = get_global_id(0);

    if (idx < N) 
        // Multiplica os elementos de A e B e adiciona o resultado em C
        C[idx] = A[idx] * B[idx] + C[idx];
    
}