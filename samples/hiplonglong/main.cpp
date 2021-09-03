#include <hip/hip_runtime.h>
#include <assert.h>
#include <stdio.h>

#define CHECK(cmd)							\
    do {                                                                                              \
      hipError_t error = cmd;						\
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(1);                                                                    \
        }                                                                                          \
    } while(0)


__global__ void convert_to_long_long(double * double_val, long long *long_long_val) {

  *long_long_val = __double_as_longlong( *double_val );

}  
__global__ void convert_to_double(long long * long_long_val, double *double_val) {

  *double_val = __longlong_as_double( *long_long_val );

}  

int main(){

  double h_double = 1.0;
  long long h_long_long;

  double *d_double = NULL;
  long long *d_long_long = NULL;

  CHECK(hipMalloc(&d_double, sizeof(double)));
  CHECK(hipMalloc(&d_long_long, sizeof(long long)));

  CHECK(hipMemcpy(d_double, &h_double, sizeof(double), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_long_long, &h_long_long, sizeof(long long), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(convert_to_long_long, 1, 1, 0, 0, d_double, d_long_long);

  CHECK(hipMemcpy(&h_long_long, d_long_long, sizeof(double), hipMemcpyDeviceToHost));

  assert( h_long_long == 0x3FF0000000000000 );
  assert( h_long_long == *((long long *) &h_double) );


  hipLaunchKernelGGL(convert_to_double, 1, 1, 0, 0, d_long_long, d_double);

  CHECK(hipMemcpy(&h_double, d_double, sizeof(double), hipMemcpyDeviceToHost));

  assert( h_double == 1.0 );
  assert( h_double == *((double *) &h_long_long) );

  
  CHECK(hipFree(d_double));
  CHECK(hipFree(d_long_long));
  printf("PASSED\n");
  return 0;
}
