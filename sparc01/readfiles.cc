/*=============================================================================================
  | Simulation Package for Ab-initio Real-space Calculations (SPARC)
  | Copyright (C) 2016 Material Physics & Mechanics Group at Georgia Tech.
  |
  | S. Ghosh, P. Suryanarayana, SPARC: Accurate and efficient finite-difference formulation and
  | parallel implementation of Density Functional Theory. Part I: Isolated clusters, Computer
  | Physics Communications
  |
  | file name: readfiles.cc
  |
  | Description: This file contains the functions required for reading the input files
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana
  |
  | Last Modified: 1/26/2016
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "petscsys.h"
#include <iostream>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace std;
///////////////////////////////////////////////////////////////////////////////////////////////
//                              Read_parameters: reads the .inpt file                        //
///////////////////////////////////////////////////////////////////////////////////////////////
void Read_parameters(SDDFT_OBJ *pSddft) {

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

///////////////////////////////////////////////////////////////////////////////////////////////
//               fract: calculates (n!)^2/((n-k)!(n+k)!), used for calculating the           //
//                    finite-difference weights of the gradient and lplacian                 //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar fract(PetscInt n, PetscInt k) {
  int i;
  PetscScalar Nr = 1, Dr = 1, val;

  for (i = n - k + 1; i <= n; i++)
    Nr *= i;
  for (i = n + 1; i <= n + k; i++)
    Dr *= i;
  val = Nr / Dr;

  return (val);
}

///////////////////////////////////////////////////////////////////////////////////////////////
//                     Read_ion: Reads the .ion file for ionic positions                     //
///////////////////////////////////////////////////////////////////////////////////////////////
void Read_ion(SDDFT_OBJ *pSddft) {
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
///////////////////////////////////////////////////////////////////////////////////////////////
//         Read_relax: reads the .relax file for constraint on movement of atoms             //
///////////////////////////////////////////////////////////////////////////////////////////////
void Read_relax(SDDFT_OBJ *pSddft) {
  FILE *fConfFile;
  char ConfigFile[100] = "./";
  strcat(ConfigFile, pSddft->file);
  strcat(ConfigFile, ".relax");
  PetscInt poscnt, index = 0;
  PetscScalar xct, yct, zct;
  char name[10];
  int at = 0;
  PetscInt noa;

  VecDuplicate(pSddft->forces, &pSddft->mvAtmConstraint);
  VecSet(pSddft->mvAtmConstraint, 1);

  if (pSddft->RelaxFlag == 1) {

    if ((fConfFile = fopen((const char *)ConfigFile, "rb")) == NULL) {
      cout << "Couldn't open config file... " << ConfigFile << '\n';
      cout << "Exiting for this configuration...\n";
      exit(1);
    }

    do {
      if (fscanf(fConfFile, "%s", name) == EOF)
        break;
      if (fscanf(fConfFile, "%d", &noa) == EOF)
        break;
      /*
       * check if .relax file is consistent with .ion file
       */
      if (strcmp(name, pSddft->atomType[at]) != 0) {
        cout << "Incorrect atom type in .relax file. Give same type and same order as .ion file\n";
        exit(1);
      }
      for (poscnt = 0; poscnt < noa; poscnt++) {
        fscanf(fConfFile, "%lf", &xct);
        fscanf(fConfFile, "%lf", &yct);
        fscanf(fConfFile, "%lf", &zct);

        VecSetValues(pSddft->mvAtmConstraint, 1, &index, &xct, INSERT_VALUES);
        index++;
        VecSetValues(pSddft->mvAtmConstraint, 1, &index, &yct, INSERT_VALUES);
        index++;
        VecSetValues(pSddft->mvAtmConstraint, 1, &index, &zct, INSERT_VALUES);
        index++;
      }
      at++;
    } while (!feof(fConfFile));
    fclose(fConfFile);

    if (at != pSddft->Ntype) {
      cout << "Number of blocks: " << at << " in .relax file not same as number of atom type: " << pSddft->Ntype
           << " in .inpt file\n";
      exit(1);
    }
  }
  return;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//                   Read_pseudopotential: reads the pseudopotential file                     //
////////////////////////////////////////////////////////////////////////////////////////////////
void Read_pseudopotential(SDDFT_OBJ *pSddft) {

  PetscPrintf(PETSC_COMM_WORLD, "\n\n *** ENTER Read_pseudopotential\n\n");

  FILE *fRvsVFile;
  int at, i, l;
  PetscScalar value, rc;
  int lineCtr;
  PetscScalar Pi = M_PI;

  /*
   * allocate memory for pseudopotential object in the structure
   */
  pSddft->psd = (PSD_OBJ *)malloc(sizeof(PSD_OBJ) * pSddft->Ntype);
  if (pSddft->psd == NULL) {
    cout << "Memory alocation fail in pSddft->psd";
    exit(1);
  }

  for (at = 0; at < pSddft->Ntype; at++) {
    char RvsVFile[100] = "./pseudopotential/";
    strcat(RvsVFile, pSddft->psdName[at]);

    if ((fRvsVFile = fopen((const char *)RvsVFile, "rb")) == NULL) {
      cout << "Couldn't open pseudopotential file... " << RvsVFile << endl;
      cout << "Exiting for this configuration...\n";
      exit(1);
    }

    int count = 0; // number of elements in the radial grid
    char str[60];
    /*
     * check if pseudopotential file is consistent with input atom type
     */
    fscanf(fRvsVFile, "%s", str);
    if (strcmp(str, pSddft->atomType[at]) != 0) {
      cout << "pseudopotential file does not match with input atom type" << at << endl;
      exit(1);
    }

    /*
     * read the pseudopotential file (must be in Troulier-Martins format)
     */
    do {
      fgets(str, 60, fRvsVFile);
    } while (strcmp(str, " Radial grid follows\n"));
    do {
      fscanf(fRvsVFile, "%s", str);
      count++;
    } while (strcmp(str, "Pseudopotential") != 0);
    count = count - 1;
    fclose(fRvsVFile);
    pSddft->psd[at].size = count;

    pSddft->psd[at].rc_s = 0.0;
    pSddft->psd[at].rc_p = 0.0;
    pSddft->psd[at].rc_d = 0.0;
    pSddft->psd[at].rc_f = 0.0;

    /*
     * allocate memory for arrays storing radial grid, pseudopotentials and
     * pseudowavefunctions
     */
    pSddft->psd[at].RadialGrid = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].RadialGrid == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].RadialGrid" << endl;
      exit(1);
    }

    pSddft->psd[at].Vs = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Vs == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Vs" << endl;
      exit(1);
    }

    pSddft->psd[at].Vp = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Vp == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Vp" << endl;
      exit(1);
    }

    pSddft->psd[at].Vd = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Vd == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Vd" << endl;
      exit(1);
    }

    pSddft->psd[at].Vf = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Vf == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Vf" << endl;
      exit(1);
    }

    pSddft->psd[at].Us = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Us == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Us" << endl;
      exit(1);
    }

    pSddft->psd[at].Up = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Up == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Up" << endl;
      exit(1);
    }

    pSddft->psd[at].Ud = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Ud == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Ud" << endl;
      exit(1);
    }

    pSddft->psd[at].Uf = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Uf == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Uf" << endl;
      exit(1);
    }

    pSddft->psd[at].uu = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].uu == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].uu" << endl;
      exit(1);
    }

    pSddft->psd[at].Vloc = (PetscScalar *)malloc(sizeof(PetscScalar) * count);
    if (pSddft->psd[at].Vloc == NULL) {
      cout << "Memory alocation fail in pSddft->psd[at].Vloc" << endl;
      exit(1);
    }

    /*
     * open file again and read the pseudopotentials and pseudo wave functions
     */
    if ((fRvsVFile = fopen((const char *)RvsVFile, "rb")) == NULL) {
      cout << "Couldn't open pseudopotential file... " << RvsVFile << endl;
      cout << "Exiting for this configuration...\n";
      exit(1);
    }
    do {
      fgets(str, 60, fRvsVFile);
    } while (strcmp(str, " Radial grid follows\n"));

    for (i = 0; i < count; i++) {
      fscanf(fRvsVFile, "%lf", &value);
      pSddft->psd[at].RadialGrid[i] = value;
    }

    lineCtr = 0;
    while (strcmp(str, " Pseudopotential follows (l on next line)\n")) {
      fgets(str, 60, fRvsVFile);
      lineCtr++;
    }
    /*
     * read pseudopotential
     */
    while (strcmp(str, " Pseudopotential follows (l on next line)\n") == 0) {
      fscanf(fRvsVFile, "%d", &l);
      if (l == 0) // s orbital
      {
        pSddft->psd[at].lmax = 0;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = 0.5 * value / (pSddft->psd[at].RadialGrid[i]);
          pSddft->psd[at].Vs[i] = value;

          if (pSddft->localPsd[at] == 0)
            pSddft->psd[at].Vloc[i] = value;
        }
      }
      if (l == 1) // p orbital
      {
        pSddft->psd[at].lmax = 1;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = 0.5 * value / (pSddft->psd[at].RadialGrid[i]);
          pSddft->psd[at].Vp[i] = value;

          if (pSddft->localPsd[at] == 1)
            pSddft->psd[at].Vloc[i] = value;
        }
      }
      if (l == 2) // d orbital
      {
        pSddft->psd[at].lmax = 2;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = 0.5 * value / (pSddft->psd[at].RadialGrid[i]);
          pSddft->psd[at].Vd[i] = value;

          if (pSddft->localPsd[at] == 2)
            pSddft->psd[at].Vloc[i] = value;
        }
      }
      if (l == 3) // f orbital
      {
        pSddft->psd[at].lmax = 3;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = 0.5 * value / (pSddft->psd[at].RadialGrid[i]);
          pSddft->psd[at].Vf[i] = value;

          if (pSddft->localPsd[at] == 3)
            pSddft->psd[at].Vloc[i] = value;
        }
      }

      for (i = 0; i < lineCtr; i++)
        fgets(str, 60, fRvsVFile);
    }
    /*
     * read until valence charge block is found
     */
    while (strcmp(str, " Valence charge follows\n")) {
      fgets(str, 60, fRvsVFile);
    }
    /*
     * read valence charge
     */
    while (strcmp(str, " Valence charge follows\n") == 0) {
      for (i = 0; i < count; i++) {
        fscanf(fRvsVFile, "%lf", &value);
        value = value / (4 * Pi * pSddft->psd[at].RadialGrid[i] * pSddft->psd[at].RadialGrid[i]);
        pSddft->psd[at].uu[i] = value;
      }

      for (i = 0; i < lineCtr; i++)
        fgets(str, 60, fRvsVFile);
    }
    /*
     * read pseudowavefunction
     */
    while (strcmp(str, " Pseudo-wave-function follows (l, zelect, rc)\n") == 0) {
      fscanf(fRvsVFile, "%d", &l);
      fscanf(fRvsVFile, "%lf", &value);
      fscanf(fRvsVFile, "%lf", &rc);
      if (l == 0) // s orbital
      {
        pSddft->psd[at].rc_s = rc;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = value / pSddft->psd[at].RadialGrid[i];
          pSddft->psd[at].Us[i] = value;
        }
      }
      if (l == 1) // p orbital
      {
        pSddft->psd[at].rc_p = rc;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = value / pSddft->psd[at].RadialGrid[i];
          pSddft->psd[at].Up[i] = value;
        }
      }
      if (l == 2) // d orbital
      {
        pSddft->psd[at].rc_d = rc;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = value / pSddft->psd[at].RadialGrid[i];
          pSddft->psd[at].Ud[i] = value;
        }
      }
      if (l == 3) // f orbital
      {
        pSddft->psd[at].rc_f = rc;
        for (i = 0; i < count; i++) {
          fscanf(fRvsVFile, "%lf", &value);
          value = value / pSddft->psd[at].RadialGrid[i];
          pSddft->psd[at].Uf[i] = value;
        }
      }

      for (i = 0; i < lineCtr; i++) {
        if (feof(fRvsVFile))
          break;
        fgets(str, 60, fRvsVFile);
      }
    }
    fclose(fRvsVFile);
  }

  PetscPrintf(PETSC_COMM_WORLD, "\n\n *** EXIT Read_pseudopotential\n\n");

  return;
}
