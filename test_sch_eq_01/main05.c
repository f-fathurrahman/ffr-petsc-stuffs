#include <petscdmda.h>
#include <slepceps.h>

int main(int argc,char **argv)
{
  SlepcInitialize(&argc,&argv,NULL,NULL);

  /* ---------------- Problem parameters ---------------- */
  PetscInt    Nx = 40, Ny = 32, Nz = 24;
  PetscReal   xmin = -6.0, xmax = 6.0;
  PetscReal   ymin = -5.0, ymax = 5.0;
  PetscReal   zmin = -4.0, zmax = 4.0;

  PetscReal   hx = (xmax - xmin) / (Nx - 1);
  PetscReal   hy = (ymax - ymin) / (Ny - 1);
  PetscReal   hz = (zmax - zmin) / (Nz - 1);

  /* 4th-order Laplacian coefficients (anisotropic) */
  PetscScalar cx0 =  5.0 / (4.0 * hx * hx);
  PetscScalar cx1 = -2.0 / (3.0 * hx * hx);
  PetscScalar cx2 =  1.0 / (24.0 * hx * hx);

  PetscScalar cy0 =  5.0 / (4.0 * hy * hy);
  PetscScalar cy1 = -2.0 / (3.0 * hy * hy);
  PetscScalar cy2 =  1.0 / (24.0 * hy * hy);

  PetscScalar cz0 =  5.0 / (4.0 * hz * hz);
  PetscScalar cz1 = -2.0 / (3.0 * hz * hz);
  PetscScalar cz2 =  1.0 / (24.0 * hz * hz);

  PetscLogDouble tA0, tA1, tS0, tS1;

  /* ---------------- DMDA ---------------- */
  DM da;
  DMDACreate3d(
    PETSC_COMM_WORLD,
    DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
    DMDA_STENCIL_STAR,
    Nx, Ny, Nz,
    PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
    1,        /* dof */
    2,        /* stencil width = 2 (4th order) */
    NULL, NULL, NULL,
    &da
  );
  DMSetUp(da);

  /* ---------------- Matrix ---------------- */
  Mat H;
  DMCreateMatrix(da, &H);

  MPI_Barrier(PETSC_COMM_WORLD);
  PetscTime(&tA0);

  PetscInt xs, ys, zs, xm, ym, zm;
  DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);

  for (PetscInt k = zs; k < zs+zm; k++) {
    for (PetscInt j = ys; j < ys+ym; j++) {
      for (PetscInt i = xs; i < xs+xm; i++) {

        MatStencil row, col;
        row.i = i; row.j = j; row.k = k; row.c = 0;

        PetscReal x = xmin + i*hx;
        PetscReal y = ymin + j*hy;
        PetscReal z = zmin + k*hz;

        /* Harmonic potential */
        PetscScalar V = 0.5 * (x*x + y*y + z*z);

        PetscScalar diag = cx0 + cy0 + cz0 + V;
        MatSetValuesStencil(H,1,&row,1,&row,&diag,INSERT_VALUES);

        /* x-direction */
        if (i > 0)    { col=row; col.i--;  MatSetValuesStencil(H,1,&row,1,&col,&cx1,INSERT_VALUES); }
        if (i < Nx-1) { col=row; col.i++;  MatSetValuesStencil(H,1,&row,1,&col,&cx1,INSERT_VALUES); }
        if (i > 1)    { col=row; col.i-=2; MatSetValuesStencil(H,1,&row,1,&col,&cx2,INSERT_VALUES); }
        if (i < Nx-2) { col=row; col.i+=2; MatSetValuesStencil(H,1,&row,1,&col,&cx2,INSERT_VALUES); }

        /* y-direction */
        if (j > 0)    { col=row; col.j--;  MatSetValuesStencil(H,1,&row,1,&col,&cy1,INSERT_VALUES); }
        if (j < Ny-1) { col=row; col.j++;  MatSetValuesStencil(H,1,&row,1,&col,&cy1,INSERT_VALUES); }
        if (j > 1)    { col=row; col.j-=2; MatSetValuesStencil(H,1,&row,1,&col,&cy2,INSERT_VALUES); }
        if (j < Ny-2) { col=row; col.j+=2; MatSetValuesStencil(H,1,&row,1,&col,&cy2,INSERT_VALUES); }

        /* z-direction */
        if (k > 0)    { col=row; col.k--;  MatSetValuesStencil(H,1,&row,1,&col,&cz1,INSERT_VALUES); }
        if (k < Nz-1) { col=row; col.k++;  MatSetValuesStencil(H,1,&row,1,&col,&cz1,INSERT_VALUES); }
        if (k > 1)    { col=row; col.k-=2; MatSetValuesStencil(H,1,&row,1,&col,&cz2,INSERT_VALUES); }
        if (k < Nz-2) { col=row; col.k+=2; MatSetValuesStencil(H,1,&row,1,&col,&cz2,INSERT_VALUES); }
      }
    }
  }

  MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);

  MPI_Barrier(PETSC_COMM_WORLD);
  PetscTime(&tA1);

  PetscPrintf(PETSC_COMM_WORLD, "Hamiltonian assembly time: %g s\n", (double)(tA1 - tA0));

  MatSetOption(H, MAT_SYMMETRIC, PETSC_TRUE);
  MatSetOption(H, MAT_HERMITIAN, PETSC_TRUE);

  /* ---------------- Eigenproblem ---------------- */
  EPS eps;
  EPSCreate(PETSC_COMM_WORLD, &eps);
  EPSSetOperators(eps, H, NULL);
  EPSSetProblemType(eps, EPS_HEP);
  EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
  EPSSetFromOptions(eps);

  MPI_Barrier(PETSC_COMM_WORLD);
  PetscTime(&tS0);
  //
  EPSSolve(eps);
  //
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscTime(&tS1);

  PetscPrintf(PETSC_COMM_WORLD, "Eigenvalue solve time: %g s\n", (double)(tS1 - tS0));

  /* ---------------- Output eigenvalues ---------------- */
  PetscInt nconv;
  EPSGetConverged(eps, &nconv);

  for (PetscInt i=0; i<nconv; i++) {
    PetscScalar kr;
    EPSGetEigenvalue(eps, i, &kr, NULL);
    PetscPrintf(PETSC_COMM_WORLD,
      "Eigenvalue %d: %.12f\n", i, (double)PetscRealPart(kr));
  }

  /* ---------------- Cleanup ---------------- */
  EPSDestroy(&eps);
  MatDestroy(&H);
  DMDestroy(&da);

  SlepcFinalize();
  return 0;
}
