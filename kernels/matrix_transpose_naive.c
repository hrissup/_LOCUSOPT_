/*
 * matrix_transpose_naive.c
 * Classic cache-unfriendly matrix transpose.
 * B[j][i] = A[i][j]
 *
 * Access pattern analysis:
 *   - A[i][j]: inner loop j varies last index  -> cache-friendly  (sequential reads)
 *   - B[j][i]: inner loop j varies first index -> cache-unfriendly (strided writes)
 *
 * Recommended optimisation: loop tiling (blocking) with TILE=64
 */

#define N 512

void kernel_transpose(double A[N][N], double B[N][N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            B[j][i] = A[i][j];
        }
    }
}
