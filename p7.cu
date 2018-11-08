/****************************************************************************************************
 * Tyler Griffith                                                                                   *
 * October 25th, 2018                                                                               *
 * Project 7: Matrix Mult on GPU                                                                    *
 * CSC-4310-01 PROF: R. Shore                                                                       *
 * Desc: Use one thread to compute each element of the solution                                     *
         matrix                                                                                     *
 * To Compile: nvcc p6.cu -o cuda_mult_v1                                                           *
 * To Run: ./cuda_mult_v1 <device #> <tile width> <matrix A file> <matrix B file> <matrix A*B file> *
 ****************************************************************************************************/
 #include <stdio.h>
 #include <stdlib.h>

int getN(char* fileName);
int* readMatrix(char* fileName);
void writeMatrix(int *x, int n, char* fileName);

__global__ 
void MatrixMultKernel(int *a, int *b, int *ab, int size){
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int prod = 0;
        if(col < size && row < size){
                for(int i=0; i<size; i++){
                        prod += a[row*size+i] * b[i*size+col];
                }
                ab[row*size+col] = prod;
        }
}

 int main(int argc, char *argv[]){
   //reading in the matrix
   int n;

   //make sure correct syntax is used
   if(argc != 6){
      printf("Error! You do not have 5 elements to your command!\n");
      printf("To multiply 2 matricies please use the following syntax:\n");
      printf("./cuda_mult_v1 <device #> <tile width> <matrix A file> <matrix B file> <matrix A*B file>\n");
      exit(1);
   }

   //variable declaration
   long dNum = strtol(argv[1], NULL, 10);
   int deviceNum = int(dNum);
   long width = strtol(argv[2], NULL, 10);
   const int tileWidth = (int)width;
   int *matrixA, *matrixB, *matrixC, *d_a, *d_b, *d_c;

   //set device
   cudaSetDevice(deviceNum);
   
   //get n
   n = getN(argv[3]);

   //file I/O
   matrixA = readMatrix(argv[3]);
   matrixB = readMatrix(argv[4]);

   //allocate and initialize result
   matrixC = new int[n*n];
   for (int i = 0; i < n*n; ++i) {
      matrixC[i] = 0;
   }

   //cuda timing
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   int size = n*n*sizeof(int);
   //allocate cuda memory and copy to global memory
   cudaMalloc((void **)&d_a, size);
   cudaMemcpy(d_a, matrixA, size, cudaMemcpyHostToDevice);
   cudaMalloc((void **)&d_b, size);
   cudaMemcpy(d_b, matrixB, size, cudaMemcpyHostToDevice);
   
   //allocate memory for result
   cudaMalloc((void **)&d_c, size);
   cudaMemset(d_c, 0, size);

   dim3 dimGrid(tileWidth, tileWidth);
   dim3 dimBlock(tileWidth>>1, tileWidth>>1);


   //start cuda timing
   cudaEventRecord(start);
   //kernel call
   MatrixMultKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
   //end cuda timing
   cudaEventRecord(stop);

   //copy answer back to CPU
   cudaMemcpy(matrixC, d_c, n*n*sizeof(int), cudaMemcpyDeviceToHost);
   
   //stop timing
   cudaEventSynchronize(stop);
   float ms = 0;
   cudaEventElapsedTime(&ms, start, stop);
   //check for error
   cudaError_t error = cudaGetLastError();
   if(error != cudaSuccess){
      //print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
   }

   //print timing
   printf("Computation completed in %fms", ms);
   writeMatrix(matrixC, n, argv[5]);

   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);
   
   return 0;
}

int getN(char* fileName){
    int n;
    FILE *inFile;
    inFile = fopen(fileName, "r");

   fscanf(inFile, "%d", &n);
   fclose(inFile);
   return n;
}

int* readMatrix(char* fileName){
   int n;
   FILE *inFile;
   inFile = fopen(fileName, "r");

   fscanf(inFile, "%d", &n);

   //allocate memory
   int *x = (int*)malloc(n*n*sizeof(int));

   //read in matrix
   for(int row=0; row<n; row++){
      for(int col=0; col<n; col++){
         fscanf(inFile, "%d", &x[row*n+col]);
      }
   }
   fclose(inFile);
   return x;
}

void writeMatrix(int *x, int n, char* fileName){
   FILE *outFile;
   outFile = fopen(fileName, "w");
   for(int row=0; row<n; row++){
      for(int col=0; col<n; col++)
         fprintf(outFile, "%d ", x[row*n+col]);
      fprintf(outFile, "\n");
   }
   fclose(outFile);
}