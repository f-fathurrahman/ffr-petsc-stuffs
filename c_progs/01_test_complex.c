#include <stdio.h>
#include <complex.h>

double complex calc_euler(double x) {
  return cexp(I*x);
}

int main() {
  double x = -1.2;
  complex z = calc_euler(x);

  printf("%f %fi\n", creal(z), cimag(z));

  printf("%f + %fi\n", creal(z), cimag(z));

  // %+f will automatically fix the sign
  printf("%f%+fi\n", creal(z), cimag(z));
  return 0;
}

