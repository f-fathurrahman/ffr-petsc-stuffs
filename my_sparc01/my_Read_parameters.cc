#include <petsc.h>
#include <iostream>

#include "isddft.h"
#include "sddft.h"

using namespace std;

// Read_parameters: reads the .inpt file
void my_Read_parameters(SDDFT_OBJ *pSddft) {

  FILE *fConfFile;
  PetscInt p, a;
  char ConfigFile[100] = "./";
  strcat(ConfigFile, pSddft->file);
  strcat(ConfigFile, ".inpt");

  pSddft->Ntype = -1;
  pSddft->SCFNewRhoCalcCtr = 3;         // default value unless overwritten
  pSddft->RelaxCount = 0;               // incremented as NLCG progresses
  pSddft->ChebyshevCallCounter = 0;     // number of times chebyshev filtering has been called
  pSddft->MixingParameter = 0.3;        // default mixing parameter
  pSddft->MixingHistory = 7;            // default mixing history
  pSddft->RelaxFlag = 0;                // default: no relaxation
  pSddft->TOLSCF = 1e-6;                // default tolerance for SCF
  pSddft->KSPTOL = 1e-8;                // default tolerance for KSP
  pSddft->NLCGTOL = 1e-10;              // default tolerance for NLCG
  pSddft->LANCZOSTOL = 1e-6;            // default tolerance for Lanczos
  pSddft->PseudochargeRadiusTOL = 1e-8; // default tolerance for pseudocharge density radius
  pSddft->MAXITSCF = 100;               // default maximum number of SCF iterations
  pSddft->MAXIT_NLCG = 300;             // default maximum number of iterations in NLCG
  pSddft->order = 6;                    // default finite-difference order is 12
  pSddft->Beta = 1000;                  // default smearing
  pSddft->ChebDegree = 20;              // default chebychev filter polynomial degree
  pSddft->REFERENCE_CUTOFF = 0.5;       // default cutoff for reference pseudopotential
  strcpy(pSddft->XC, "LDA");            // default exchange correlation LDA Perdew-Wang

  int i;
  if ((fConfFile = fopen((const char *)ConfigFile, "rb")) == NULL) {
    cout << "Couldn't open config file " << ConfigFile << '\n';
    cout << "Exiting for this configuration...\n";
    exit(1);
  }
  char str[60];

  do {
    fscanf(fConfFile, "%s", str);
    if (strcmp(str, "FD_ORDER:") == 0) {
      fscanf(fConfFile, "%d", &pSddft->order);
      pSddft->order = (pSddft->order) / 2; // half finite difference order
    } else if (strcmp(str, "CELL:") == 0) {
      fscanf(fConfFile, "%lf", &pSddft->range_x);
      fscanf(fConfFile, "%lf", &pSddft->range_y);
      fscanf(fConfFile, "%lf", &pSddft->range_z);
    } else if (strcmp(str, "FD_GRID:") == 0) {
      fscanf(fConfFile, "%d", &pSddft->numPoints_x);
      fscanf(fConfFile, "%d", &pSddft->numPoints_y);
      fscanf(fConfFile, "%d", &pSddft->numPoints_z);
    } else if (strcmp(str, "BETA:") == 0) {
      fscanf(fConfFile, "%lf", &pSddft->Beta);
    } else if (strcmp(str, "CHEB_DEGREE:") == 0) {
      fscanf(fConfFile, "%d", &pSddft->ChebDegree);
    } else if (strcmp(str, "EXCHANGE_CORRELATION:") == 0) {
      fscanf(fConfFile, "%s", pSddft->XC);

    } else if (strcmp(str, "NSTATES:") == 0) {
      fscanf(fConfFile, "%d", &pSddft->Nstates);
      /*
       * allocate memory for storing eigenvalues
       */
      PetscMalloc(sizeof(PetscScalar) * (pSddft->Nstates), &pSddft->lambda);
    } else if (strcmp(str, "ION_RELAX:") == 0) {
      fscanf(fConfFile, "%d", &pSddft->RelaxFlag);
    } else if (strcmp(str, "SCF_RHO_CALC_COUNT:") == 0) {
      fscanf(fConfFile, "%d", &pSddft->SCFNewRhoCalcCtr);
    } else if (strcmp(str, "MIXING_PARAMETER:") == 0) {
      fscanf(fConfFile, "%lf", &pSddft->MixingParameter);
    } else if (strcmp(str, "REFERENCE_CUTOFF:") == 0) {
      fscanf(fConfFile, "%lf", &pSddft->REFERENCE_CUTOFF);
    } else if (strcmp(str, "MIXING_HISTORY:") == 0) {
      fscanf(fConfFile, "%d", &pSddft->MixingHistory);
    } else if (strcmp(str, "TOL_SCF:") == 0) {
      fscanf(fConfFile, "%lf", &pSddft->TOLSCF);
    } else if (strcmp(str, "TOL_POISSON:") == 0) {
      fscanf(fConfFile, "%lf", &pSddft->KSPTOL);
    } else if (strcmp(str, "TOL_NLCG:") == 0) {
      fscanf(fConfFile, "%lf", &pSddft->NLCGTOL);
    } else if (strcmp(str, "TOL_LANCZOS:") == 0) {
      fscanf(fConfFile, "%lf", &pSddft->LANCZOSTOL);
    } else if (strcmp(str, "TOL_PSEUDOCHARGE:") == 0) {
      fscanf(fConfFile, "%lf", &pSddft->PseudochargeRadiusTOL);
    } else if (strcmp(str, "MAXIT_SCF:") == 0) {
      fscanf(fConfFile, "%d", &pSddft->MAXITSCF);
    } else if (strcmp(str, "MAXIT_NLCG:") == 0) {
      fscanf(fConfFile, "%d", &pSddft->MAXIT_NLCG);
    } else if (strcmp(str, "NTYPE:") == 0) {
      fscanf(fConfFile, "%d", &pSddft->Ntype);
      /*
       * allocate memory
       */
      PetscMalloc(sizeof(PetscScalar) * pSddft->Ntype, &pSddft->CUTOFF);
      PetscMalloc(sizeof(PetscInt) * pSddft->Ntype, &pSddft->startPos);
      PetscMalloc(sizeof(PetscInt) * pSddft->Ntype, &pSddft->endPos);
      PetscMalloc(sizeof(PetscInt) * pSddft->Ntype, &pSddft->noa);
      PetscMalloc(sizeof(PetscInt) * pSddft->Ntype, &pSddft->noe);
      PetscMalloc(sizeof(PetscInt) * pSddft->Ntype, &pSddft->localPsd);

      pSddft->psdName = (char **)malloc(pSddft->Ntype * sizeof(char *));
      for (i = 0; i < pSddft->Ntype; i++) {
        pSddft->psdName[i] = (char *)malloc(25 + 1);
      }
      pSddft->atomType = (char **)malloc(pSddft->Ntype * sizeof(char *));
      for (i = 0; i < pSddft->Ntype; i++) {
        pSddft->atomType[i] = (char *)malloc(10 + 1);
      }
    } else if (strcmp(str, "CUTOFF:") == 0) {
      if (pSddft->Ntype <= 0) {
        cout << "Input number of types of elements first\n";
        exit(1);
      }
      for (i = 0; i < pSddft->Ntype; i++) {
        fscanf(fConfFile, "%lf", &pSddft->CUTOFF[i]);
      }
    } else if (strcmp(str, "PSEUDOPOTENTIAL_LOCAL:") == 0) {
      if (pSddft->Ntype <= 0) {
        cout << "Input number of types of elements first\n";
        exit(1);
      }
      for (i = 0; i < pSddft->Ntype; i++) {
        fscanf(fConfFile, "%d", &pSddft->localPsd[i]);
      }
    } else if (strcmp(str, "PSEUDOPOTENTIAL_FILE:") == 0) {
      if (pSddft->Ntype <= 0) {
        cout << "Input number of types of elements first\n";
        exit(1);
      }
      for (i = 0; i < pSddft->Ntype; i++) {
        fscanf(fConfFile, "%s", pSddft->psdName[i]);
      }
    }
  } while (!feof(fConfFile));

  fclose(fConfFile);

  /*
   * now that the .inpt file is read, we calculate the mesh spacing
   */
  PetscScalar delta_x = 2 * pSddft->range_x / (pSddft->numPoints_x - 1);
  PetscScalar delta_y = 2 * pSddft->range_y / (pSddft->numPoints_y - 1);
  PetscScalar delta_z = 2 * pSddft->range_z / (pSddft->numPoints_z - 1);
  /*
   * check if delta_x=delta_y=delta_z within some tolerance
   */
  if ((fabs(delta_x - delta_y) >= 1e-10) || (fabs(delta_x - delta_z) >= 1e-10) || (fabs(delta_y - delta_z) >= 1e-10)) {
    PetscPrintf(PETSC_COMM_WORLD, "EXITING: mesh spacing MUST be same in all directions \n");
    exit(0);
  } else {
    pSddft->delta = delta_x;
  }

  /*
   * finite difference coefficients of -(1/2)*laplacian operator
   */
  pSddft->coeffs[0] = 0;
  for (a = 1; a <= pSddft->order; a++)
    pSddft->coeffs[0] += ((PetscScalar)1.0 / (a * a));
  pSddft->coeffs[0] *= ((PetscScalar)3.0 / (pSddft->delta * pSddft->delta));

  for (p = 1; p <= pSddft->order; p++)
    pSddft->coeffs[p] =
        (PetscScalar)(-1 * pow(-1, p + 1) * fract(pSddft->order, p) / (p * p * pSddft->delta * pSddft->delta));

  /*
   * finite difference coefficients for the gradient operator
   */
  pSddft->coeffs_grad[0] = 0;
  for (p = 1; p <= pSddft->order; p++)
    pSddft->coeffs_grad[p] = (PetscScalar)(pow(-1, p + 1) * fract(pSddft->order, p) / (p * pSddft->delta));

  /*
   * for convinence of the user, display the parameters
   */
  PetscPrintf(PETSC_COMM_WORLD, "***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD, "                           Initialization                                  \n");
  PetscPrintf(PETSC_COMM_WORLD, "***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD, "Domain lengths:                  { %f, %f, %f } (Bohr)\n", 2 * pSddft->range_x,
              2 * pSddft->range_y, 2 * pSddft->range_z);
  PetscPrintf(PETSC_COMM_WORLD, "Number of finite-difference nodes                    %d, %d, %d\n",
              pSddft->numPoints_x, pSddft->numPoints_y, pSddft->numPoints_z);
  PetscPrintf(PETSC_COMM_WORLD, "Finite-difference order:                             %d\n", 2 * pSddft->order);
  PetscPrintf(PETSC_COMM_WORLD, "Mesh spacing:                                        %f\n", pSddft->delta);
  PetscPrintf(PETSC_COMM_WORLD, "Smearing:                                            %lf(1/Hartree)\n", pSddft->Beta);
  PetscPrintf(PETSC_COMM_WORLD, "Degree of Chebyshev polynomial for Chebyshev filter: %d\n", pSddft->ChebDegree);
  PetscPrintf(PETSC_COMM_WORLD, "Number of states of electronic occupation:           %d\n", pSddft->Nstates);
  PetscPrintf(PETSC_COMM_WORLD, "Number of types of atomic species:                   %d\n", pSddft->Ntype);
  PetscPrintf(PETSC_COMM_WORLD, "Exchange correlation functional:                     %s\n", pSddft->XC);
  PetscPrintf(PETSC_COMM_WORLD, "Geometry optimization flag:                          %d\n", pSddft->RelaxFlag);
  PetscPrintf(PETSC_COMM_WORLD, "Maximum number of SCF iterations:                    %d\n", pSddft->MAXITSCF);
  PetscPrintf(PETSC_COMM_WORLD, "SCF Tolerence:                                       %e\n", pSddft->TOLSCF);
  PetscPrintf(PETSC_COMM_WORLD, "Poisson Tolerence:                                   %e\n", pSddft->KSPTOL);
  PetscPrintf(PETSC_COMM_WORLD, "Lanczos Tolerence:                                   %e\n", pSddft->LANCZOSTOL);
  PetscPrintf(PETSC_COMM_WORLD, "Mixing parameter:                                    %f\n", pSddft->MixingParameter);
  PetscPrintf(PETSC_COMM_WORLD, "Mixing history:                                      %d\n", pSddft->MixingHistory);
  for (i = 0; i < pSddft->Ntype; i++) {
    PetscPrintf(PETSC_COMM_WORLD, "Pseudopotential file name for atom type %d is         %s\n", i + 1,
                pSddft->psdName[i]);
    PetscPrintf(PETSC_COMM_WORLD, "Choice of local pseudopotential for atom type %d is   %d\n", i + 1,
                pSddft->localPsd[i]);
  }

  return;
}