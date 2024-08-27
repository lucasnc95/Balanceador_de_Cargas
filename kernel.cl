__kernel void vectorAdd(__global const float *a, __global const float *b, __global float *c) {
    int id = get_global_id(0);
    int N = 32768;
    if(id < N)
        c[id] = (a[id] + b[id]);
    
}

__kernel void vectorMulAdd(__global const float* A, 
                           __global const float* B, 
                           __global float* C, 
                           const int N) {
    int idx = get_global_id(0);

    if (idx < N) {
        // Multiplica os elementos de A e B e adiciona o resultado em C
        C[idx] = A[idx] * B[idx] + C[idx];
    }
}