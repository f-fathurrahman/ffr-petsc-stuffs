#include <petsc.h>

void print_petsc_scalar() {
  PetscPrintf(PETSC_COMM_WORLD,
#if defined(PETSC_USE_COMPLEX)
    "PETScScalar = complex\n"
#else
    "PETScScalar = real\n"
#endif
);
}


PetscErrorCode vec_view_DMDA2D(DM da, Vec x)
{
  PetscInt xs, ys, xm, ym;
  PetscInt mx, my;
  PetscInt i, j;
  PetscScalar **xa;
  PetscInt rstart, rend;

  /* Global grid size */
  DMDAGetInfo(da, NULL, &mx, &my, NULL,
              NULL, NULL, NULL,
              NULL, NULL, NULL, NULL, NULL, NULL);

  /* Ownership range of the Vec */
  VecGetOwnershipRange(x, &rstart, &rend);

  /* Local grid corners */
  DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);

  /* Access vector as 2D array */
  DMDAVecGetArrayRead(da, x, &xa);

  for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {

          /* Global linear index in DMDA ordering */
          PetscInt gid = j * mx + i;

          PetscPrintf(PETSC_COMM_WORLD,
              "gid=%6d  (i=%3d, j=%3d)  value=%g\n",
              (int)gid, (int)i, (int)j,
              (double)PetscRealPart(xa[j][i]));
      }
  }

  DMDAVecRestoreArrayRead(da, x, &xa);
  return 0;
}



void test_set_elements(Vec x) {
  VecSet(x, 2.011); /* x_i = 2 for all i */
  return;
}

// Set element by index. This is slow (?)
void slow_set_element(Vec x) {
  PetscInt idx;
  PetscScalar val;
  
  idx = 3; val = 5.13;
  VecSetValue(x, idx, val, INSERT_VALUES);

  idx = 2; val = 9.11;
  VecSetValue(x, idx, val, INSERT_VALUES);

  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
  return;
}


void dmda_set_element(DM da, Vec x) {
  PetscInt xs, ys, xm, ym, i, j;
  PetscScalar **xarr;

  DMDAVecGetArray(da, x, &xarr);
  DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);
  // Set the array
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
        xarr[j][i] = i + 10.0*j;
    }
  }
  // Put the array back
  DMDAVecRestoreArray(da, x, &xarr);
}


int main(int argc, char **argv) {
  
  PetscInitialize(&argc, &argv, NULL, NULL);

  DM da;
  Vec x, y;

  PetscInt mx = 2, my = 3;

  print_petsc_scalar();

  DMDACreate2d(PETSC_COMM_WORLD,
    DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
    DMDA_STENCIL_STAR,
    mx, my,
    PETSC_DECIDE, PETSC_DECIDE,
    1, 1, NULL, NULL, &da
  );

  DMSetUp(da);

  DMCreateGlobalVector(da, &x);
  DMCreateGlobalVector(da, &y);

  //test_set_elements(x);
  //slow_set_element(x);
  dmda_set_element(da, x);
  VecView(x, PETSC_VIEWER_STDOUT_WORLD);
  vec_view_DMDA2D(da, x);

  VecDestroy(&x);
  VecDestroy(&y);
  DMDestroy(&da);
  PetscFinalize();
  return 0;
}
