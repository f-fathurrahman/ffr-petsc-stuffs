/*=============================================================================================
  | Description: This file contains the functions required for atomic relaxation
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "isddft.h"
///////////////////////////////////////////////////////////////////////////////////////////////
//           FormFunction_relaxAtoms: performs one electronic structure minimization for     //
//                                      fixed atomic positions                               //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode FormFunction_relaxAtoms(SDDFT_OBJ *pSddft) {
  PetscPrintf(PETSC_COMM_WORLD, " ***  Start of DFT calculation for fixed atomic positions ***\n");

  /*
   * calculate pseudocharge density
   */
  ChargDensB_VecInit(pSddft);

  /*
   * update the nonlocal pseudopotential
   */
  if (pSddft->RelaxCount != 0) {
    PetscPrintf(PETSC_COMM_WORLD, "Estimating Nonzeros in Hamiltonian : \n");
    EstimateNonZerosNonlocalPseudopot(pSddft);
    PetscPrintf(PETSC_COMM_WORLD, "making Nonlocal Pseudopotential matrix \n");
    LaplacianNonlocalPseudopotential_MatInit(pSddft);
    Wavefunctions_MatMatMultSymbolic(pSddft);
  }

  /*
   * calculate energy correction
   */
  ChargDensB_TM_VecInit(pSddft);
  CorrectionEnergy_Calc(pSddft);
  /*
   * calculate electron density using Self Consistent Field iteration
   */
  SelfConsistentField(pSddft);
  /*
   * calculate force correction
   */
  Calculate_forceCorrection(pSddft);
  /*
   * calculate local component of forces
   */
  Calculate_force(pSddft);
  /*
   * calculate nonlocal component of forces
   */
  Force_Nonlocal(pSddft, &pSddft->XOrb);
  /*
   * symmetrysize forces
   */
  Symmetrysize_force(pSddft);
  /*
   * initialize vectors to zero
   */
  Set_VecZero(pSddft);
  /*
   * destroy Hamiltonian
   */
  MatDestroy(&pSddft->HamiltonianOpr);
  /*
   * display forces on atoms and atomic poistions
   */
  Display_force(pSddft);
  Display_Atompos(pSddft);
  pSddft->RelaxCount++;
  PetscPrintf(PETSC_COMM_WORLD, "***   End of DFT calculation for fixed atomic positions  ***\n");

  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//    NLCG_relaxAtoms: Nonlinear Conjugate Gradient method for ground state minimization     //
//             Reference: An Introduction to the Conjugate Gradient Method Without           //
//                         Agonizing Pain, Jonathan Richard Shewchuk                         //
///////////////////////////////////////////////////////////////////////////////////////////////
void NLCG_relaxAtoms(SDDFT_OBJ *pSddft) {
  PetscInt i = 0, j, k = 0, imax = pSddft->MAXIT_NLCG, jmax = 6, n = 30, jsum = 0;
  PetscScalar deltaNew, deltad, deltaOld, deltaMid, tol1, tol2 = 1e-10, sigma0 = 0.5, alpha, etaPrev, eta, beta;
  Vec r, d, s, y, F;
  int inCtr;

  tol1 = pSddft->NLCGTOL * 3.0 * pSddft->nAtoms;

  VecDuplicate(pSddft->Atompos, &F);
  VecDuplicate(pSddft->Atompos, &r);
  VecDuplicate(pSddft->Atompos, &d);
  VecDuplicate(pSddft->Atompos, &s);
  VecDuplicate(pSddft->Atompos, &y);

  /*
   * update electron density, energy and forces
   */
  FormFunction_relaxAtoms(pSddft);

  VecCopy(pSddft->forces, r);
  inCtr = 0;
  VecCopy(r, s);
  VecCopy(s, d);
  VecDot(r, d, &deltaNew);

  while ((i < imax) && (deltaNew > tol1)) {
    PetscPrintf(PETSC_COMM_WORLD, "---------------------------------------\n");
    PetscPrintf(PETSC_COMM_WORLD, " \n (Outer) Relaxation step: %d \n", i + 1);
    PetscPrintf(PETSC_COMM_WORLD, "-------------------------------------- \n");

    PetscPrintf(PETSC_COMM_WORLD, "deltaNew FORCE: %0.16lf \t \n", deltaNew);
    j = 0;
    VecDot(d, d, &deltad);
    alpha = -sigma0;

    /*
     * perturb atomic positions
     */
    VecWAXPY(y, sigma0, d, pSddft->Atompos);
    VecCopy(y, pSddft->Atompos);

    /*
     * update electron density, energy and forces
     */
    FormFunction_relaxAtoms(pSddft);
    VecCopy(pSddft->forces, F);

    /*
     * replace back the original atomic positions
     */
    VecWAXPY(pSddft->Atompos, -sigma0, d, y);
    VecDot(F, d, &etaPrev);
    etaPrev = -etaPrev;

    /*
     * line search
     */
    do {
      PetscPrintf(PETSC_COMM_WORLD, "---------------------------------------\n");
      PetscPrintf(PETSC_COMM_WORLD, "\n  (Inner) Relaxation step: %d \n", j + 1);
      PetscPrintf(PETSC_COMM_WORLD, "---------------------------------------\n");
      if (inCtr == 0) {
        VecDot(r, d, &eta);
        eta = -eta;
      } else {
        /*
         * update electron density, energy and forces
         */
        FormFunction_relaxAtoms(pSddft);
        VecCopy(pSddft->forces, F);
        VecDot(F, d, &eta);
        eta = -eta;
      }

      alpha = alpha * (eta / (etaPrev - eta));
      /*
       * perturb atomic positions
       */
      VecAXPY(pSddft->Atompos, alpha, d);
      etaPrev = eta;
      j++;
      inCtr++;
      PetscPrintf(PETSC_COMM_WORLD, "************************************** \n");

    } while ((j < jmax) && (alpha * alpha * deltad > tol2));
    jsum = jsum + j;

    /*
     * update electron density, energy and forces
     */
    FormFunction_relaxAtoms(pSddft);
    VecCopy(pSddft->forces, r);
    inCtr = 0;

    deltaOld = deltaNew;

    VecDot(r, s, &deltaMid);

    VecCopy(r, s);

    VecDot(r, s, &deltaNew);
    beta = (deltaNew - deltaMid) / deltaOld;
    k++;

    if ((k == n) || (beta <= 0)) {
      VecCopy(s, d);
      k = 0;
    } else {
      VecAYPX(d, beta, s);
    }
    i++;
    PetscPrintf(PETSC_COMM_WORLD, "************************************** \n");
  }

  PetscPrintf(PETSC_COMM_WORLD, "Total number of inner iterations:%d \n", jsum);
  PetscPrintf(PETSC_COMM_WORLD, "Total number of outer iterations:%d \n", i);

  VecDestroy(&F);
  VecDestroy(&r);
  VecDestroy(&d);
  VecDestroy(&s);
  VecDestroy(&y);

  return;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                          Display_Atompos: prints the atomic positions                      //
///////////////////////////////////////////////////////////////////////////////////////////////
void Display_Atompos(SDDFT_OBJ *pSddft) {
  PetscScalar *pAtompos;
  PetscInt poscnt, Index = 0;

  VecGetArray(pSddft->Atompos, &pAtompos);
  PetscPrintf(PETSC_COMM_WORLD, "Atomic positions (Bohr) \n");
  for (poscnt = 0; poscnt < pSddft->nAtoms; poscnt++) {
    PetscPrintf(PETSC_COMM_WORLD, "%9.9f \t %9.9f \t %9.9f \n", pAtompos[Index], pAtompos[Index + 1],
                pAtompos[Index + 2]);
    Index = Index + 3;
  }
  PetscPrintf(PETSC_COMM_WORLD, "\n");

  VecRestoreArray(pSddft->Atompos, &pAtompos);
  return;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//                   Display_Relax: print the constraints on atomic relaxations              //
///////////////////////////////////////////////////////////////////////////////////////////////
void Display_Relax(SDDFT_OBJ *pSddft) {
  PetscScalar *pmvAtmConstraint;
  PetscInt poscnt, Index = 0;

  VecGetArray(pSddft->mvAtmConstraint, &pmvAtmConstraint);
  PetscPrintf(PETSC_COMM_WORLD, "Atomic relaxation flag \n");
  for (poscnt = 0; poscnt < pSddft->nAtoms; poscnt++) {
    PetscPrintf(PETSC_COMM_WORLD, "%f\t %f\t %f\n", pmvAtmConstraint[Index], pmvAtmConstraint[Index + 1],
                pmvAtmConstraint[Index + 2]);
    Index = Index + 3;
  }
  PetscPrintf(PETSC_COMM_WORLD, "\n");

  PetscPrintf(PETSC_COMM_WORLD, "***************************************************************************\n\n");
  VecRestoreArray(pSddft->mvAtmConstraint, &pmvAtmConstraint);
  return;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//     SDDFT_Nonperiodic: function for nonperiodic Density Functional theory calculation     //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode SDDFT_Nonperiodic(SDDFT_OBJ *pSddft) {

  int ierr;
  PetscLogDouble t1, t2, elapsed_time;
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscPrintf(PETSC_COMM_WORLD, "\n\n **** ENTER SDDFT_Nonperiodic *** \n\n");

  /*
   * create the data structures required for DFT calculation
   */
  ierr = Objects_Create(pSddft);
  CHKERRQ(ierr);
  
  //PetscPrintf(PETSC_COMM_WORLD, "BREAK HERE by ffr");
  //exit(0);

  Laplace_matInit(pSddft);
  Gradient_matInit(pSddft);

  /*
   * calculate initial guess electron density
   */
  SuperpositionAtomicCharge_VecInit(pSddft);
  VecCopy(pSddft->SuperposAtRho, pSddft->elecDensRho);

  /*
   * create nonlocal pseudopotential operator
   */
  EstimateNonZerosNonlocalPseudopot(pSddft);
  LaplacianNonlocalPseudopotential_MatInit(pSddft);
  Wavefunctions_MatInit(pSddft);

  if (pSddft->RelaxFlag == 1) {
    /*
     * perform atomic relaxation
     */
    NLCG_relaxAtoms(pSddft);
  } else {
    /*
     * DFT calculation with fixed atomic positions
     */

    Display_Atompos(pSddft);
    Display_Relax(pSddft);

    /*
     * calculate pseudocharge density
     */
    ChargDensB_VecInit(pSddft);

    /*
     * calculate energy correction
     */
    ChargDensB_TM_VecInit(pSddft);
    CorrectionEnergy_Calc(pSddft);

    /*
     * calculate electron density using Self Consistent Field iteration
     */
    SelfConsistentField(pSddft);

    /*
     * calculate force correction
     */
    Calculate_forceCorrection(pSddft);
    /*
     * calculate local component of forces
     */
    Calculate_force(pSddft);
    /*
     * calculate nonlocal component of forces
     */
    Force_Nonlocal(pSddft, &pSddft->XOrb);
    /*
     * symmetrysize forces
     */
    Symmetrysize_force(pSddft);

    /*
     * display forces on atoms and atomic poistions
     */
    Display_force(pSddft);
    Set_VecZero(pSddft);
    MatDestroy(&pSddft->HamiltonianOpr);
  }

  return 0;
}
