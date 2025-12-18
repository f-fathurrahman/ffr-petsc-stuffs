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

  PetscReal   dx = (xmax - xmin) / (Nx - 1);
  PetscReal   dy = (ymax - ymin) / (Ny - 1);
  PetscReal   dz = (zmax - zmin) / (Nz - 1);

  PetscLogDouble tA0, tA1, tS0, tS1;

  /* ---------------- DMDA ---------------- */
  DM da;
  DMDACreate3d(PETSC_COMM_WORLD,
              DM_BOUNDARY_PERIODIC,
              DM_BOUNDARY_PERIODIC,
              DM_BOUNDARY_PERIODIC,
              DMDA_STENCIL_STAR,
              Nx, Ny, Nz,
              PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
              1, 1,
              NULL, NULL, NULL,
              &da);
  DMSetUp(da);


  /* ---------------- Matrix ---------------- */
  Mat H;
  DMCreateMatrix(da, &H);

  MPI_Barrier(PETSC_COMM_WORLD);
  PetscTime(&tA0);

  PetscInt xs, ys, zs, xm, ym, zm;
  PetscInt i, j, k;
  DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);

  for (k = zs; k < zs+zm; k++) {
    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {

        PetscInt    ncols = 0;
        MatStencil  row, col[7];
        PetscScalar val[7];

        row.i = i; row.j = j; row.k = k;

        /* Optional: periodic potential (example: zero) */
        PetscReal V = 0.0;

        /* diagonal */
        col[ncols].i = i;
        col[ncols].j = j;
        col[ncols].k = k;
        val[ncols++] =
          (1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)) + V;

        /* x */
        col[ncols].i = i-1; col[ncols].j = j; col[ncols].k = k;
        val[ncols++] = -0.5/(dx*dx);

        col[ncols].i = i+1; col[ncols].j = j; col[ncols].k = k;
        val[ncols++] = -0.5/(dx*dx);

        /* y */
        col[ncols].i = i; col[ncols].j = j-1; col[ncols].k = k;
        val[ncols++] = -0.5/(dy*dy);

        col[ncols].i = i; col[ncols].j = j+1; col[ncols].k = k;
        val[ncols++] = -0.5/(dy*dy);

        /* z */
        col[ncols].i = i; col[ncols].j = j; col[ncols].k = k-1;
        val[ncols++] = -0.5/(dz*dz);

        col[ncols].i = i; col[ncols].j = j; col[ncols].k = k+1;
        val[ncols++] = -0.5/(dz*dz);

        MatSetValuesStencil(H,1,&row,ncols,col,val,INSERT_VALUES);
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
