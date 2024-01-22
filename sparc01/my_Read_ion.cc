#include <petsc.h>
#include <iostream>
#include "isddft.h"

using namespace std;

// Read_ion: Reads the .ion file for ionic positions
void my_Read_ion(SDDFT_OBJ *pSddft) {
  FILE *fConfFile;
  char ConfigFile[100] = "./";
  strcat(ConfigFile, pSddft->file);
  strcat(ConfigFile, ".ion");
  pSddft->nAtoms = 0;
  PetscInt poscnt, index = 0;
  PetscScalar x0, y0, z0;
  pSddft->noetot = 0;

  int at = 0;
  if ((fConfFile = fopen((const char *)ConfigFile, "rb")) == NULL) {
    cout << "Couldn't open config file...\n" << ConfigFile;
    cout << "Exiting for this configuration...\n";
    exit(1);
  }
  /*
   * first read total number of atoms
   */
  fscanf(fConfFile, "%d", &pSddft->nAtoms);

  /*
   * create local vectors for storing forces and atom positions
   */
  VecCreate(PETSC_COMM_SELF, &pSddft->forces);
  VecSetSizes(pSddft->forces, PETSC_DECIDE, 3 * pSddft->nAtoms);
  VecSetFromOptions(pSddft->forces);
  VecDuplicate(pSddft->forces, &pSddft->Atompos);

  pSddft->noetot = 0;

  do {
    if (fscanf(fConfFile, "%s", pSddft->atomType[at]) == EOF) // atom type (string)
      break;
    if (fscanf(fConfFile, "%d", &pSddft->noe[at]) == EOF) // number of electrons
      break;
    if (fscanf(fConfFile, "%d", &pSddft->noa[at]) == EOF) // number of atoms
      break;
    pSddft->startPos[at] = index;

    /*
     * loop over number of atoms of a particular type and read their positions
     */
    for (poscnt = 0; poscnt < pSddft->noa[at]; poscnt++) {
      pSddft->noetot += pSddft->noe[at];
      fscanf(fConfFile, "%lf", &x0);
      fscanf(fConfFile, "%lf", &y0);
      fscanf(fConfFile, "%lf", &z0);

      VecSetValues(pSddft->Atompos, 1, &index, &x0, INSERT_VALUES);
      index++;
      VecSetValues(pSddft->Atompos, 1, &index, &y0, INSERT_VALUES);
      index++;
      VecSetValues(pSddft->Atompos, 1, &index, &z0, INSERT_VALUES);
      index++;
    }
    pSddft->endPos[at] = index - 1;

    at++;
  } while (!feof(fConfFile));

  fclose(fConfFile);

  if (at != pSddft->Ntype) {
    cout << "Number of blocks: " << at << " in .ion file not same as number of atom type:" << pSddft->Ntype
         << "in .inpt file\n";
    exit(1);
  }

  PetscPrintf(PETSC_COMM_WORLD, "Total number of atoms: %d\n", pSddft->nAtoms);
  PetscPrintf(PETSC_COMM_WORLD, "Total number of electrons: %d\n", pSddft->noetot);

  for (at = 0; at < pSddft->Ntype; at++) {
    PetscPrintf(PETSC_COMM_WORLD, "Atom type %d: %s\n", at + 1, pSddft->atomType[at]);
    PetscPrintf(PETSC_COMM_WORLD, "Number of atoms of type %d: %d\n", at + 1,
                (pSddft->endPos[at] - pSddft->startPos[at] + 1) / 3);
  }

#ifdef _DEBUG
  VecView(pSddft->Atompos, PETSC_VIEWER_STDOUT_SELF);
  cout << "noetot" << pSddft->noetot << '\n';
  cout << "natoms" << pSddft->nAtoms << '\n';
#endif

  return;
}