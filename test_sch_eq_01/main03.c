#include <petscdmda.h>
#include <slepceps.h>

int main(int argc,char **argv)
{
  SlepcInitialize(&argc,&argv,NULL,NULL);

  PetscInt i;
  PetscInt Nx = 40, Ny = 40, Nz = 40;
  PetscReal xmin = -6.0, xmax = 6.0;
  PetscReal h = (xmax - xmin) / (Nx - 1);
  PetscScalar kin = 1.0 / (2.0 * h * h);

  DM da;
  DMDACreate3d(
    PETSC_COMM_WORLD,
    DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
    DMDA_STENCIL_STAR,
    Nx, Ny, Nz,
    PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
    1,               /* dof */
    1,               /* stencil width */
    NULL, NULL, NULL,
    &da
  );
  DMSetFromOptions(da);
  DMSetUp(da);

  Mat H;
  DMCreateMatrix(da, &H);

  PetscInt xs, ys, zs, xm, ym, zm;
  DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);

  for (PetscInt k = zs; k < zs+zm; k++) {
    for (PetscInt j = ys; j < ys+ym; j++) {
      for (PetscInt i = xs; i < xs+xm; i++) {

        MatStencil row, col;
        row.i = i; row.j = j; row.k = k; row.c = 0;

        PetscReal x = xmin + i*h;
        PetscReal y = xmin + j*h;
        PetscReal z = xmin + k*h;

        PetscScalar V = 0.5 * (x*x + y*y + z*z);

        PetscScalar diag = 6.0*kin + V;
        MatSetValuesStencil(H, 1, &row, 1, &row, &diag, INSERT_VALUES);

        if (i > 0) {
          col = row; col.i--;
          PetscScalar v = -kin;
          MatSetValuesStencil(H,1,&row,1,&col,&v,INSERT_VALUES);
        }
        if (i < Nx-1) {
          col = row; col.i++;
          PetscScalar v = -kin;
          MatSetValuesStencil(H,1,&row,1,&col,&v,INSERT_VALUES);
        }

        if (j > 0) {
          col = row; col.j--;
          PetscScalar v = -kin;
          MatSetValuesStencil(H,1,&row,1,&col,&v,INSERT_VALUES);
        }
        if (j < Ny-1) {
          col = row; col.j++;
          PetscScalar v = -kin;
          MatSetValuesStencil(H,1,&row,1,&col,&v,INSERT_VALUES);
        }

        if (k > 0) {
          col = row; col.k--;
          PetscScalar v = -kin;
          MatSetValuesStencil(H,1,&row,1,&col,&v,INSERT_VALUES);
        }
        if (k < Nz-1) {
          col = row; col.k++;
          PetscScalar v = -kin;
          MatSetValuesStencil(H,1,&row,1,&col,&v,INSERT_VALUES);
        }
      }
    }
  }

  MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);

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
  EPSGetConverged(eps,&nconv);
  PetscPrintf(PETSC_COMM_WORLD,"Converged eigenpairs: %d\n",nconv);

  for (i=0; i < nconv; i++) {
    PetscScalar eig;
    EPSGetEigenvalue(eps,i,&eig,NULL);
    PetscPrintf(PETSC_COMM_WORLD,"E[%d] = %g\n",i,(double)PetscRealPart(eig));
  }

  EPSDestroy(&eps);
  MatDestroy(&H);
  SlepcFinalize();
  return 0;
}

