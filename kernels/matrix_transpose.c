// Sample 1

#define N 512

void kernel_transpose(double A[N][N], double B[N][N]) {
    int i,j;
    for (i=0;i<N;i++){
        for (j=0;j<N;j++){
            B[j][i] = A[i][j];
        }
    }
}
