#include <petsc.h>
#include <slepceps.h>

int main(int argc, char **argv) {

//    Mat H;
//    EPS eps;
//    PetscInt N = 500, i;
//    PetscScalar h, v;
//    PetscReal xmin = -10.0, xmax = 10.0;
  SlepcInitialize(&argc, &argv, NULL, NULL);

//    h = (xmax - xmin)/(N-1);

  PetscInt Nx = 40, Ny = 40, Nz = 40;
  PetscInt N  = Nx * Ny * Nz;

  PetscReal xmin = -6.0, xmax = 6.0;
  PetscReal h = (xmax - xmin) / (Nx - 1);

  PetscScalar kin = 1.0 / (2.0 * h * h);

  Mat H;
  MatCreate(PETSC_COMM_WORLD, &H);
  MatSetSizes(H, PETSC_DECIDE, PETSC_DECIDE, N, N);
  MatSetFromOptions(H);
  MatSetUp(H);

  for (PetscInt k = 0; k < Nz; k++) {
    for (PetscInt j = 0; j < Ny; j++) {
      for (PetscInt i = 0; i < Nx; i++) {
        
        // Build the Hamiltonian row-by-row
        PetscInt row = i + Nx * (j + Ny * k);

        /* Coordinates */
        PetscReal x = xmin + i*h;
        PetscReal y = xmin + j*h;
        PetscReal z = xmin + k*h;

        /* Potential */
        PetscScalar V = 0.5 * (x*x + y*y + z*z);

        /* Diagonal */
        PetscScalar diag = 6.0 * kin + V;
        MatSetValue(H, row, row, diag, INSERT_VALUES);

        /* Neighbors */
        if (i > 0) MatSetValue(H, row, row-1, -kin, INSERT_VALUES);
        if(i < Nx-1) MatSetValue(H, row, row+1, -kin, INSERT_VALUES);
        //
        if(j > 0) MatSetValue(H, row, row-Nx, -kin, INSERT_VALUES);
        if(j < Ny-1) MatSetValue(H, row, row+Nx, -kin, INSERT_VALUES);
        //
        if(k > 0) MatSetValue(H, row, row-Nx*Ny, -kin, INSERT_VALUES);
        if(k < Nz-1) MatSetValue(H, row, row+Nx*Ny, -kin, INSERT_VALUES);
      }
    }
  }

  MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);

  /* Mark Hermitian */
  MatSetOption(H, MAT_SYMMETRIC, PETSC_TRUE);
  MatSetOption(H, MAT_HERMITIAN, PETSC_TRUE);

  EPS eps;
  EPSCreate(PETSC_COMM_WORLD, &eps);
  EPSSetOperators(eps, H, NULL);
  EPSSetProblemType(eps, EPS_HEP);
  EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
  EPSSetFromOptions(eps);
  EPSSolve(eps);

  PetscInt nconv;
  EPSGetConverged(eps, &nconv);
  PetscPrintf(PETSC_COMM_WORLD, "Converged eigenpairs: %d\n", nconv);
  
  int i;
  for(i=0; i < nconv; i++) {
    PetscScalar eig;
    EPSGetEigenvalue(eps,i,&eig,NULL);
    PetscPrintf(PETSC_COMM_WORLD, "E[%d] = %g\n", i, (double)PetscRealPart(eig));
  }

  EPSDestroy(&eps);
  MatDestroy(&H);
  SlepcFinalize();
  return 0;

}
