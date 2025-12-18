#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

int main(int argc, char **argv) {
  PetscErrorCode ierr;
  DM da;
  Vec x, b;
  PetscInt mx = 2, my = 3;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);
  if (ierr)
    return ierr;

  /* Create a 2D DMDA */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, mx, my, PETSC_DECIDE,
                      PETSC_DECIDE, 1, /* dof */
                      1,               /* stencil width */
                      NULL, NULL, &da);
  CHKERRQ(ierr);

  ierr = DMSetFromOptions(da);
  CHKERRQ(ierr);
  ierr = DMSetUp(da);
  CHKERRQ(ierr);

  /* Create vectors CONSISTENT with DMDA */
  ierr = DMCreateGlobalVector(da, &x);
  CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &b);
  CHKERRQ(ierr);

  ierr = VecSet(b, 9.9);
  CHKERRQ(ierr);
  ierr = VecSet(x, 1.1);
  CHKERRQ(ierr);

  /* Check vector type (debugging) */
  PetscPrintf(PETSC_COMM_WORLD, "Vector type:\n");
  ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD);
  CHKERRQ(ierr);

  /* Clean up */
  ierr = VecDestroy(&x);
  CHKERRQ(ierr);
  ierr = VecDestroy(&b);
  CHKERRQ(ierr);
  ierr = DMDestroy(&da);
  CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
