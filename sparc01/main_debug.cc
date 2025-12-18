#include "isddft.h"
#include "sddft.h"

#include "my_functions.h"

static char help[] = "Simulation Package for Ab-initio Real-space Calculations (SPARC) \n options:\n\" -name name of file\n";

int main( int argc, char **argv )
{
  PetscErrorCode ierr;
  SDDFT_OBJ sddft;

  PetscLogDouble t1, t2, elapsed_time;
  PetscInitialize(&argc, &argv, (char *)0, help);
  my_SddftObjInitialize(&sddft);
  // Read files
  my_Read_parameters(&sddft);
  my_Read_ion(&sddft);
  my_Read_relax(&sddft);
  my_Read_pseudopotential(&sddft);
  /*
   * calculate pseudocharge cutoff
   */
  ChargDensB_cutoff(&sddft);
  /*
   * DFT calculation
   */
  //SDDFT_Nonperiodic(&sddft);
  /*
   * destroy variables to free memory
   */
  Objects_Destroy(&sddft);

  ierr = PetscFinalize();
  CHKERRQ(ierr);

  return 0;
}
