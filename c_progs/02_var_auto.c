// This needs recent gcc

#include <stdio.h>

int calc_something(int a, int b) {
  return a + 2*b;
}

double calc_something_v2(double x, double y) {
  return 1.1*x + 2;
}

int main() {

  auto c = calc_something(2, 4);
  printf("c = %d\n", c);

  // not working ??
  auto x = calc_something_v2(2.1, 2.1);
  printf("x = %g\n", x);

  return 0;
}

