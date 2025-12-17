#include <petscdmda.h>
#include <slepceps.h>

int main(int argc,char **argv)
{
  SlepcInitialize(&argc,&argv,NULL,NULL);

  PetscInt i;
  PetscInt Nx = 40, Ny = 40, Nz = 40;
  PetscReal xmin = -6.0, xmax = 6.0;
  PetscReal ymin = -6.0, ymax = 6.0;
  PetscReal zmin = -6.0, zmax = 6.0;

  PetscReal hx = (xmax - xmin) / (Nx - 1);
  PetscReal hy = (ymax - ymin) / (Ny - 1);
  PetscReal hz = (zmax - zmin) / (Nz - 1);

  PetscScalar cx0 =  5.0 / (4.0 * hx * hx);
  PetscScalar cx1 = -2.0 / (3.0 * hx * hx);
  PetscScalar cx2 =  1.0 / (24.0 * hx * hx);
  
  PetscScalar cy0 =  5.0 / (4.0 * hy * hy);
  PetscScalar cy1 = -2.0 / (3.0 * hy * hy);
  PetscScalar cy2 =  1.0 / (24.0 * hy * hy);
  
  PetscScalar cz0 =  5.0 / (4.0 * hz * hz);
  PetscScalar cz1 = -2.0 / (3.0 * hz * hz);
  PetscScalar cz2 =  1.0 / (24.0 * hz * hz);

  DM da;
  DMDACreate3d(
    PETSC_COMM_WORLD,
    DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
    DMDA_STENCIL_STAR,
    Nx, Ny, Nz,
    PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
    1,               /* dof */
    2,               /* stencil width */
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
        row.i=i; row.j=j; row.k=k; row.c=0;

        PetscReal x = xmin + i*hx;
        PetscReal y = ymin + j*hy;
        PetscReal z = zmin + k*hz;

        PetscScalar V = 0.5 * (x*x + y*y + z*z);

        PetscScalar diag = cx0 + cy0 + cz0 + V;
        MatSetValuesStencil(H,1,&row,1,&row,&diag,INSERT_VALUES);

        /* x-direction */
        if (i > 0)    { col=row; col.i--; MatSetValuesStencil(H,1,&row,1,&col,&cx1,INSERT_VALUES); }
        if (i < Nx-1) { col=row; col.i++; MatSetValuesStencil(H,1,&row,1,&col,&cx1,INSERT_VALUES); }
        if (i > 1)    { col=row; col.i-=2; MatSetValuesStencil(H,1,&row,1,&col,&cx2,INSERT_VALUES); }
        if (i < Nx-2) { col=row; col.i+=2; MatSetValuesStencil(H,1,&row,1,&col,&cx2,INSERT_VALUES); }

        /* y-direction */
        if (j > 0)    { col=row; col.j--; MatSetValuesStencil(H,1,&row,1,&col,&cy1,INSERT_VALUES); }
        if (j < Ny-1) { col=row; col.j++; MatSetValuesStencil(H,1,&row,1,&col,&cy1,INSERT_VALUES); }
        if (j > 1)    { col=row; col.j-=2; MatSetValuesStencil(H,1,&row,1,&col,&cy2,INSERT_VALUES); }
        if (j < Ny-2) { col=row; col.j+=2; MatSetValuesStencil(H,1,&row,1,&col,&cy2,INSERT_VALUES); }

        /* z-direction */
        if (k > 0)    { col=row; col.k--; MatSetValuesStencil(H,1,&row,1,&col,&cz1,INSERT_VALUES); }
        if (k < Nz-1) { col=row; col.k++; MatSetValuesStencil(H,1,&row,1,&col,&cz1,INSERT_VALUES); }
        if (k > 1)    { col=row; col.k-=2; MatSetValuesStencil(H,1,&row,1,&col,&cz2,INSERT_VALUES); }
        if (k < Nz-2) { col=row; col.k+=2; MatSetValuesStencil(H,1,&row,1,&col,&cz2,INSERT_VALUES); }
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

