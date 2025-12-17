#include <petscdm.h>
#include <petscdmda.h>
#include <petscmat.h>
#include <slepceps.h>

int main(int argc, char **argv)
{
  DM            da;
  Mat           H;
  EPS           eps;
  PetscInt      Nx = 50, Ny = 50, Nz = 50;
  PetscReal     Lx = 10.0, Ly = 10.0, Lz = 10.0;
  PetscReal     dx, dy, dz;
  PetscInt      xs, ys, zs, xm, ym, zm;
  PetscInt      i, j, k;
  MatStencil    row, col[7];
  PetscScalar   val[7];
  PetscLogDouble tA0, tA1, tS0, tS1;

  SlepcInitialize(&argc, &argv, NULL, NULL);

  PetscOptionsGetInt(NULL,NULL,"-Nx",&Nx,NULL);
  PetscOptionsGetInt(NULL,NULL,"-Ny",&Ny,NULL);
  PetscOptionsGetInt(NULL,NULL,"-Nz",&Nz,NULL);

  dx = Lx / (Nx - 1);
  dy = Ly / (Ny - 1);
  dz = Lz / (Nz - 1);

  /* --- DMDA --- */
  DMDACreate3d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
               DMDA_STENCIL_STAR,
               Nx, Ny, Nz,
               PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
               1, 1,
               NULL, NULL, NULL,
               &da);
  DMSetUp(da);

  /* --- Matrix --- */
  DMCreateMatrix(da, &H);

  /* ===========================
     Hamiltonian construction
     =========================== */

  MPI_Barrier(PETSC_COMM_WORLD);
  PetscTime(&tA0);

  DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);

  for (k = zs; k < zs+zm; k++) {
    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {

        PetscInt ncols = 0;
        PetscReal x = i * dx - Lx/2.0;
        PetscReal y = j * dy - Ly/2.0;
        PetscReal z = k * dz - Lz/2.0;

        PetscReal V = 0.5 * (x*x + y*y + z*z);

        row.i = i; row.j = j; row.k = k;

        /* center */
        col[ncols].i = i;
        col[ncols].j = j;
        col[ncols].k = k;
        val[ncols++] =
            (1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)) + V;

        /* x neighbors */
        if (i > 0) {
          col[ncols].i = i-1; col[ncols].j = j; col[ncols].k = k;
          val[ncols++] = -0.5/(dx*dx);
        }
        if (i < Nx-1) {
          col[ncols].i = i+1; col[ncols].j = j; col[ncols].k = k;
          val[ncols++] = -0.5/(dx*dx);
        }

        /* y neighbors */
        if (j > 0) {
          col[ncols].i = i; col[ncols].j = j-1; col[ncols].k = k;
          val[ncols++] = -0.5/(dy*dy);
        }
        if (j < Ny-1) {
          col[ncols].i = i; col[ncols].j = j+1; col[ncols].k = k;
          val[ncols++] = -0.5/(dy*dy);
        }

        /* z neighbors */
        if (k > 0) {
          col[ncols].i = i; col[ncols].j = j; col[ncols].k = k-1;
          val[ncols++] = -0.5/(dz*dz);
        }
        if (k < Nz-1) {
          col[ncols].i = i; col[ncols].j = j; col[ncols].k = k+1;
          val[ncols++] = -0.5/(dz*dz);
        }

        MatSetValuesStencil(H,1,&row,ncols,col,val,INSERT_VALUES);
      }
    }
  }

  MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);

  MPI_Barrier(PETSC_COMM_WORLD);
  PetscTime(&tA1);

  PetscPrintf(PETSC_COMM_WORLD,
    "Hamiltonian assembly time: %g s\n", (double)(tA1 - tA0));

  /* ===========================
     Eigenvalue solve
     =========================== */

  EPSCreate(PETSC_COMM_WORLD, &eps);
  EPSSetOperators(eps, H, NULL);
  EPSSetProblemType(eps, EPS_HEP);
  EPSSetDimensions(eps, 5, PETSC_DEFAULT, PETSC_DEFAULT);
  EPSSetFromOptions(eps);

  MPI_Barrier(PETSC_COMM_WORLD);
  PetscTime(&tS0);

  EPSSolve(eps);

  MPI_Barrier(PETSC_COMM_WORLD);
  PetscTime(&tS1);

  PetscPrintf(PETSC_COMM_WORLD,
    "Eigenvalue solve time: %g s\n", (double)(tS1 - tS0));

  /* Optional: print eigenvalues */
  PetscInt nconv;
  EPSGetConverged(eps, &nconv);
  for (i = 0; i < nconv && i < 5; i++) {
    PetscScalar kr;
    EPSGetEigenvalue(eps, i, &kr, NULL);
    PetscPrintf(PETSC_COMM_WORLD,
      "Eigenvalue %d = %.8f\n", i, (double)PetscRealPart(kr));
  }

  EPSDestroy(&eps);
  MatDestroy(&H);
  DMDestroy(&da);

  SlepcFinalize();
  return 0;
}
