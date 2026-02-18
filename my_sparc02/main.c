
static char help[] = "Simulation Package for Ab-initio Real-space Calculations (SPARC) \n\
options:\n\
-name name of file\n";

#include "isddft.h"
#include "sddft.h"
#include <mpi.h>
#include <petsctime.h>

#undef __FUNCT__
#define __FUNCT__ "main"


// main: the main function
int main(int argc, char **argv) {
  int ierr;
  SDDFT_OBJ sddft;
  PetscLogDouble t1, t2, elapsed_time;
  PetscInitialize(&argc, &argv, (char *)0, help);
  SddftObjInitialize(&sddft);
  /*
   * read files
   */
  Read_parameters(&sddft);
  Read_ion(&sddft);
  Read_relax(&sddft);
  Read_pseudopotential(&sddft);
  /*
   * calculate pseudocharge cutoff
   */
  ChargDensB_cutoff(&sddft);
  /*
   * DFT calculation with appropriate boundary condition
   */

  if (sddft.BC == 1) {
    SDDFT_Nonperiodic(&sddft);
    Objects_Destroy(&sddft); // destroy variables to free memory
  } else if (sddft.BC == 2) {
    if (sddft.Nkpts == 1) {
      PetscPrintf(PETSC_COMM_WORLD, "Pass here 42 in main\n");
      SDDFT_Periodic(&sddft);
      Objects_Destroy(&sddft); // destroy variables to free memory
    } else {
      SDDFT_kPointPeriodic(&sddft);
      kPointObjects_Destroy(&sddft); // destroy variables to free memory
    }
  }

  ierr = PetscFinalize();
  CHKERRQ(ierr);

  return 0;
}
