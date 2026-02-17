#include <petsc.h>
#include "isddft.h"

// Get filename from command line options using Petsc
void my_SddftObjInitialize(SDDFT_OBJ *pSddft) {
  PetscBool set;
  PetscOptionsGetString(
    PETSC_NULL, PETSC_NULL, "-name",
    pSddft->file, sizeof(pSddft->file),
  &set);
  // set is not strictly needed
  return;
}