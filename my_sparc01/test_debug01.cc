#include <stdio.h>
#include <petsc.h>

int main(int argc, char* argv[])
{
  PetscErrorCode ierr;
  PetscInt o = 6;
  PetscInt n_x = 50, n_y = 50, n_z = 50;
  DM da;
  Vec elecDensRho;

  PetscInitialize(&argc, &argv, (char*)0, NULL);

  ierr =
  DMDACreate3d( PETSC_COMM_WORLD,
    DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
    DMDA_STENCIL_STAR,
    n_x, n_y, n_z,
    PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, o,
    0, 0, 0, &da );
  CHKERRQ(ierr);
  
  ierr = DMSetUp(da);
  CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da, &elecDensRho); // error
  CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD, "Program ended normally\n");

  ierr = PetscFinalize();
  CHKERRQ(ierr);

  return 0;
}