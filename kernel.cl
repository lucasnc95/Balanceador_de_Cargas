__kernel void vector_add(__global const float *a) {
    int gid = get_global_id(0);
    a[gid] = a[gid] * gid;
}
