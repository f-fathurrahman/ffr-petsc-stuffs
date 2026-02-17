/*=============================================================================================
  | Simulation Package for Ab-initio Real-space Calculations (SPARC) 
  | Copyright (C) 2016 Material Physics & Mechanics Group at Georgia Tech.
  |
  | S. Ghosh, P. Suryanarayana, SPARC: Accurate and efficient finite-difference formulation and
  | parallel implementation of Density Functional Theory. Part I: Isolated clusters, Computer
  | Physics Communications
  |
  | file name: energy.cc          
  |
  | Description: This file contains the functions required for calculation of energy
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana, Deepa Phanish
  |
  | Last Modified: 1/26/2016   
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "petscsys.h"
///////////////////////////////////////////////////////////////////////////////////////////////
//                  SystemEnergy_Calc: calculates total energy per atom                      //
///////////////////////////////////////////////////////////////////////////////////////////////
void SystemEnergy_Calc(SDDFT_OBJ* pSddft) 
{
  PetscScalar Eatom,Eband=0.0,E1,E2,E3,delVol,Cst = 27.211384523,delta;
  PetscScalar Entropy=0.0,gn;
  PetscInt i;
  PetscScalar Etemp;

  delta = pSddft->delta;
  delVol = pow(delta,3);
  
  Exc_Calc_CA(pSddft);  // exchange correlation energy
 
  /*
   * calculate Band structure energy
   */
  for(i=0; i<pSddft->Nstates; i++)
    {
      Eband = Eband + 2.0*smearing_FermiDirac(pSddft->Beta,pSddft->lambda[i],pSddft->lambda_f)*pSddft->lambda[i];    
    }
 
  pSddft->Eband = Eband;

  VecDot(pSddft->elecDensRho,pSddft->potentialPhi,&E2) ;
  E2 = 0.5*E2*delVol;

  VecDot(pSddft->chrgDensB,pSddft->potentialPhi,&E1);
  E1 = 0.5*E1*delVol;
  
  VecDot(pSddft->PoissonRHSAdd,pSddft->potentialPhi,&Etemp);
  Etemp = 0.5*Etemp*delVol;
   
  /*
   * calculate entropy
   */
  for(i=0; i<pSddft->Nstates; i++)
    {
      gn = smearing_FermiDirac(pSddft->Beta,pSddft->lambda[i],pSddft->lambda_f);
      if(fabs(1.0-gn)>1e-14 && gn>1e-14)
  {     
    Entropy = Entropy + (gn*log(gn) + (1.0-gn)*log(1.0-gn));
  }
    }
  Entropy = (2.0/pSddft->Beta)*Entropy; 
  pSddft->Entropy = Entropy;
     
  VecDot(pSddft->elecDensRho,pSddft->Vxc,&E3);
  E3 = E3*delVol;
  /*
   * calculate energy per atom
   */

  Eatom = (Eband+E1-E2-E3+pSddft->Exc+pSddft->Eself+pSddft->Ecorrection + Entropy)/pSddft->nAtoms;
  Eatom*=Cst;
  pSddft->Eatom = Eatom; 
 
#ifdef _DEBUG
  VecSum(pSddft->elecDensRho,&E1);
  E1*=delVol;
  PetscPrintf(PETSC_COMM_WORLD,"integral rho : %f\n", E1);
  PetscPrintf(PETSC_COMM_WORLD,"Eband (Hartree) : %.16f\n", Eband);
  PetscPrintf(PETSC_COMM_WORLD,"E1 (Hartree) : %.16f\n", E1);
  PetscPrintf(PETSC_COMM_WORLD,"Exc+E2+E3 (Hartree) : %.16f\n", ExcPE2PE3);
#endif
  
  return;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//                    CorrectionEnergy_Calc: Calculate correction in energy                   //
////////////////////////////////////////////////////////////////////////////////////////////////
void CorrectionEnergy_Calc(SDDFT_OBJ* pSddft) 
{

  PetscScalar delta;
  KSPConvergedReason reason;
  delta = pSddft->delta;
    
  VecWAXPY(pSddft->twopiBTMmBPS,-1,pSddft->chrgDensB,pSddft->chrgDensB_TM);
   
  MultipoleExpansion_Phi(pSddft,&pSddft->twopiBTMmBPS);
   
  VecScale(pSddft->twopiBTMmBPS, 2*M_PI);
  VecAXPY(pSddft->twopiBTMmBPS,1.0,pSddft->PoissonRHSAdd); // add to rhs of the poisson equation

  KSPSolve(pSddft->ksp,pSddft->twopiBTMmBPS,pSddft->Phi_c);
  KSPGetConvergedReason(pSddft->ksp,&reason);

  if(reason!=2)
    {
      printf("\nKSP not converged reason: %d\n",reason);
      VecZeroEntries(pSddft->Phi_c);
      KSPSolve(pSddft->ksp,pSddft->twopiBTMmBPS,pSddft->Phi_c);
      KSPGetConvergedReason(pSddft->ksp,&reason);
    } 
  assert(reason==2);   

#ifdef _DEBUG
  PetscPrintf(PETSC_COMM_WORLD,"KSP Converged reason: %d\n",reason);  
  KSPGetIterationNumber(pSddft->ksp, &its);
  PetscPrintf(PETSC_COMM_WORLD,"Poisson number of iterations: %d\n",its);
  KSPView(pSddft->ksp,PETSC_VIEWER_STDOUT_WORLD);
#endif
  /*
   * Calculate energy correction term   
   */
  VecZeroEntries(pSddft->twopiBTMmBPS);
  VecWAXPY(pSddft->twopiBTMmBPS,1,pSddft->chrgDensB,pSddft->chrgDensB_TM);
  VecDot(pSddft->twopiBTMmBPS,pSddft->Phi_c,&pSddft->Ecorrection);
  pSddft->Ecorrection = 0.5*pSddft->Ecorrection*delta*delta*delta; 
  pSddft->Ecorrection = pSddft->Ecorrection - pSddft->Eself + pSddft->Eself_TM; 
  PetscPrintf(PETSC_COMM_WORLD,"Correction in energy: %8.12f (Hartree) \n", pSddft->Ecorrection);
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n\n");
  
}

///////////////////////////////////////////////////////////////////////////////////////////////
//                                  Display_Energy: print energy                             //
///////////////////////////////////////////////////////////////////////////////////////////////
void Display_Energy(SDDFT_OBJ* pSddft)
{    
  PetscScalar Cst = 1.0/27.211384523;
  
  
  /*
   * display different components of energy   
   */
  PetscPrintf(PETSC_COMM_WORLD,"Energy correction:     %lf (Hartree/atom)\n", pSddft->Ecorrection*Cst/pSddft->nAtoms);
  PetscPrintf(PETSC_COMM_WORLD,"Band structure energy: %lf (Hartree/atom)\n", pSddft->Eband*Cst/pSddft->nAtoms);  
  PetscPrintf(PETSC_COMM_WORLD,"Exchange correlation:  %lf (Hartree/atom)\n", pSddft->Exc*Cst/pSddft->nAtoms);
  PetscPrintf(PETSC_COMM_WORLD,"Entropy*kb*T:          %lf (Hartree/atom)\n", -pSddft->Entropy*Cst/pSddft->nAtoms);
   
  /*
   * display total energy
   */
  PetscPrintf(PETSC_COMM_WORLD,"Free energy:  %.12lf (Hartree/atom)\n",Cst*pSddft->Eatom);    
  PetscPrintf(PETSC_COMM_WORLD,"Total free energy of system: %.12lf (Hartree)\n",Cst*pSddft->Eatom*pSddft->nAtoms);
  
}
