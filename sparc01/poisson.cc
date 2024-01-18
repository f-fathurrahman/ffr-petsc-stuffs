/*=============================================================================================
  | file name: poisson.cc          
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

#ifdef _DEBUG
  int its;
#endif

  VecWAXPY(pSddft->twopiRhoPB,1,pSddft->elecDensRho,pSddft->chrgDensB);
  
  MultipoleExpansion_Phi(pSddft,&pSddft->twopiRhoPB);
  
  VecScale(pSddft->twopiRhoPB, 2*M_PI); // scale rhs by 2pi    
  VecAXPY(pSddft->twopiRhoPB,1.0,pSddft->PoissonRHSAdd); // add charge correction to rhs
  
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

#ifdef _DEBUG
  PetscPrintf(PETSC_COMM_WORLD,"KSP Converged reason: %d\n",reason);  
  KSPGetIterationNumber(pSddft->ksp, &its);
  PetscPrintf(PETSC_COMM_WORLD,"Poisson number of iterations: %d\n",its);
  KSPView(pSddft->ksp,PETSC_VIEWER_STDOUT_WORLD);
#endif
 
  return 0;
}

