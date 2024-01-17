/*=============================================================================================
  | Simulation Package for Ab-initio Real-space Calculations (SPARC) 
  | Copyright (C) 2016 Material Physics & Mechanics Group at Georgia Tech.
  |
  | S. Ghosh, P. Suryanarayana, SPARC: Accurate and efficient finite-difference formulation and
  | parallel implementation of Density Functional Theory. Part I: Isolated clusters, Computer
  | Physics Communications
  | S. Ghosh, P. Suryanarayana, SPARC: Accurate and efficient finite-difference formulation and
  | parallel implementation of Density Functional Theory. Part II: Periodic systems, Computer
  | Physics Communications  
  |
  | file name: poisson.cc          
  |
  | Description: This file contains the functions required for solving the Poisson equation
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana, Deepa Phanish
  |
  | Last Modified: 2/17/2016   
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "petscsys.h"
#include "math.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
///////////////////////////////////////////////////////////////////////////////////////////////
//    SolvePoisson: calculates the electrostatic potential by solving the Poisson equation   //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode SolvePoisson(SDDFT_OBJ* pSddft)
{
  KSPConvergedReason reason;
  PetscReal norm;
  PetscScalar shift;
   
  VecWAXPY(pSddft->twopiRhoPB,1,pSddft->elecDensRho,pSddft->chrgDensB);
  if(pSddft->BC==1)// nonperiodic
    {
      MultipoleExpansion_Phi(pSddft,&pSddft->twopiRhoPB);
      VecScale(pSddft->twopiRhoPB, 2*M_PI); // scale rhs by 2pi       
      VecAXPY(pSddft->twopiRhoPB,1.0,pSddft->PoissonRHSAdd); // add charge correction to rhs
    }
  else if(pSddft->BC==2) // 3D periodic
    {
      VecScale(pSddft->twopiRhoPB, 2*M_PI); // scale rhs by 2pi 
      /*
       * remove nullspace
       */
      MatNullSpaceRemove(pSddft->nullspace,pSddft->twopiRhoPB); 
    }  
  KSPSetInitialGuessNonzero(pSddft->ksp,PETSC_TRUE);
  KSPSolve(pSddft->ksp,pSddft->twopiRhoPB,pSddft->potentialPhi);
  KSPGetConvergedReason(pSddft->ksp,&reason);

  if(reason!=2)
    {
      PetscPrintf(PETSC_COMM_WORLD,"\nKSP not converged reason: %d\n",reason);
      VecZeroEntries(pSddft->potentialPhi);
      KSPSolve(pSddft->ksp,pSddft->twopiRhoPB,pSddft->potentialPhi);
      KSPGetConvergedReason(pSddft->ksp,&reason);
    }
  assert(reason==2);  

  /*
   * shift to enforce integral of electrostatic potential=0 for periodic systems
   */   
  if(pSddft->BC==2)
    {
      VecSum(pSddft->potentialPhi,&shift);
      shift = shift/(pSddft->numPoints_x*pSddft->numPoints_y*pSddft->numPoints_z);
      VecShift(pSddft->potentialPhi,-shift); 
    }  

  
#ifdef _DEBUG
  PetscPrintf(PETSC_COMM_WORLD,"KSP Converged reason: %d\n",reason);  
  KSPGetIterationNumber(pSddft->ksp, &its);
  PetscPrintf(PETSC_COMM_WORLD,"Poisson number of iterations: %d\n",its);
  KSPView(pSddft->ksp,PETSC_VIEWER_STDOUT_WORLD);
#endif
 
  return 0;
}

