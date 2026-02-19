#include <petsc.h>

int main(int argc, char** argv)
{

  int ierr;
  PetscScalar a, b, c;

  PetscInitialize(&argc, &argv, PETSC_NULLPTR, PETSC_NULLPTR);
  
  a = 1.1;
  b = 3.1;
  c = a*b;
  PetscPrintf(PETSC_COMM_WORLD, "a = %f\n", a);
  PetscPrintf(PETSC_COMM_WORLD, "b = %f\n", b);
  PetscPrintf(PETSC_COMM_WORLD, "c = %f\n", c);

  ierr = PetscFinalize();
  CHKERRQ(ierr);

  return 0;
}
