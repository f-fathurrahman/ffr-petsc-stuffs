#include <petscdm.h>
#include <petscdmda.h>
#include <petscmat.h>
#include <slepceps.h>

int main(int argc,char **argv)
{
  DM            da;
  Mat           H;
  EPS           eps;
  PetscInt      Nx = 32, Ny = 32, Nz = 32;
  PetscReal     Lx = 10.0, Ly = 10.0, Lz = 10.0;
  PetscReal     dx, dy, dz;
  PetscInt      xs, ys, zs, xm, ym, zm;
  PetscInt      i, j, k;

  SlepcInitialize(&argc,&argv,NULL,NULL);

  PetscOptionsGetInt(NULL,NULL,"-Nx",&Nx,NULL);
  PetscOptionsGetInt(NULL,NULL,"-Ny",&Ny,NULL);
  PetscOptionsGetInt(NULL,NULL,"-Nz",&Nz,NULL);

  dx = Lx / Nx;
  dy = Ly / Ny;
  dz = Lz / Nz;

  /* ---------------------------
     DMDA with periodic BCs
     --------------------------- */
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

  /* Create matrix compatible with DMDA */
  DMCreateMatrix(da,&H);

  /* ---------------------------
     Hamiltonian assembly
     --------------------------- */
  DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);

  for (k = zs; k < zs+zm; k++) {
    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {

        MatStencil  row, col[7];
        PetscScalar val[7];
        PetscInt    ncols = 0;

        row.i = i; row.j = j; row.k = k;

        /* Example: free particle (V = 0)
           Replace V with a periodic potential if desired */
        //PetscReal V = 0.0;
        PetscReal x = 0.5*Lx + i*dx;
        PetscReal y = 0.5*Ly + j*dy;
        PetscReal z = 0.5*Lz + k*dz;

        /* Harmonic potential */
        PetscScalar V = 0.5 * (x*x + y*y + z*z);


        /* Diagonal */
        col[ncols].i = i;
        col[ncols].j = j;
        col[ncols].k = k;
        val[ncols++] =
          (1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)) + V;

        /* x neighbors */
        col[ncols].i = i-1; col[ncols].j = j; col[ncols].k = k;
        val[ncols++] = -0.5/(dx*dx);

        col[ncols].i = i+1; col[ncols].j = j; col[ncols].k = k;
        val[ncols++] = -0.5/(dx*dx);

        /* y neighbors */
        col[ncols].i = i; col[ncols].j = j-1; col[ncols].k = k;
        val[ncols++] = -0.5/(dy*dy);

        col[ncols].i = i; col[ncols].j = j+1; col[ncols].k = k;
        val[ncols++] = -0.5/(dy*dy);

        /* z neighbors */
        col[ncols].i = i; col[ncols].j = j; col[ncols].k = k-1;
        val[ncols++] = -0.5/(dz*dz);

        col[ncols].i = i; col[ncols].j = j; col[ncols].k = k+1;
        val[ncols++] = -0.5/(dz*dz);

        MatSetValuesStencil(H,1,&row,ncols,col,val,INSERT_VALUES);
      }
    }
  }

  MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);

  /* ---------------------------
     Eigenvalue solver
     --------------------------- */
  EPSCreate(PETSC_COMM_WORLD,&eps);
  EPSSetOperators(eps,H,NULL);
  EPSSetProblemType(eps,EPS_HEP);
  EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
  //EPSSetDimensions(eps, 5, PETSC_DEFAULT,PETSC_DEFAULT);
  EPSSetFromOptions(eps);

  EPSSolve(eps);

  /* Print eigenvalues */
  PetscInt nconv;
  EPSGetConverged(eps,&nconv);

  for (i = 0; i < nconv && i < 5; i++) {
    PetscScalar kr;
    EPSGetEigenvalue(eps,i,&kr,NULL);
    PetscPrintf(PETSC_COMM_WORLD,
      "Eigenvalue %d = %.12f\n",
      i,(double)PetscRealPart(kr));
  }

  EPSDestroy(&eps);
  MatDestroy(&H);
  DMDestroy(&da);

  SlepcFinalize();
  return 0;
}
