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
  | file name: energy.cc          
  |
  | Description: This file contains the functions required for calculation of energy
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana, Deepa Phanish
  |
  | Last Modified: 2/16/2016   
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
///////////////////////////////////////////////////////////////////////////////////////////////
//     kPointSystemEnergy_Calc: calculates total energy per atom for k point calculations    //
///////////////////////////////////////////////////////////////////////////////////////////////
void kPointSystemEnergy_Calc(SDDFT_OBJ* pSddft) 
{ 
  PetscScalar Eatom,Eband=0.0,E1,E2,E3,delVol,Cst = 27.211384523,delta;
  PetscScalar Entropy=0.0,gnk;
  PetscInt i,k;
  PetscScalar Etemp;
  delta = pSddft->delta;
  delVol = pow(delta,3);    
  Exc_Calc_CA(pSddft); // exchange correlation energy
 
  /*
   * calculate Band structure energy
   */
  for(k=0; k<pSddft->Nkpts_sym;k++)
    for(i=0; i<pSddft->Nstates; i++)
      {
	Eband = Eband + (2.0/pSddft->Nkpts)*pSddft->kptWts[k]*smearing_FermiDirac(pSddft->Beta,pSddft->lambdakpt[k][i],pSddft->lambda_f)*pSddft->lambdakpt[k][i];    
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
  for(k=0;k<pSddft->Nkpts_sym;k++)
    for(i=0;i<pSddft->Nstates; i++)
      {
	gnk = smearing_FermiDirac(pSddft->Beta,pSddft->lambdakpt[k][i],pSddft->lambda_f);     
	if(fabs(1.0-gnk)>1e-14 && gnk>1e-14)
	  {	 
	    Entropy = Entropy + pSddft->kptWts[k]*(gnk*log(gnk) + (1.0-gnk)*log(1.0-gnk));
	  }
      }
  Entropy = (2.0/(pSddft->Nkpts*pSddft->Beta))*Entropy; 
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
  PetscScalar intb,intb_TM;

  VecSum(pSddft->chrgDensB,&intb);
  VecSum(pSddft->chrgDensB_TM,&intb_TM);
  intb_TM = intb/intb_TM;
  VecScale(pSddft->chrgDensB_TM,intb_TM);
  VecSum(pSddft->chrgDensB,&intb_TM);
  intb_TM = intb_TM*delta*delta*delta;
  VecWAXPY(pSddft->twopiBTMmBPS,-1,pSddft->chrgDensB,pSddft->chrgDensB_TM);

  /*
   * Multipole expansion for non-periodic boundary condition
   */
  if(pSddft->BC==1)
    {      
      MultipoleExpansion_Phi(pSddft,&pSddft->twopiBTMmBPS); 
      VecScale(pSddft->twopiBTMmBPS, 2*M_PI);
      VecAXPY(pSddft->twopiBTMmBPS,1.0,pSddft->PoissonRHSAdd); // add to rhs of the poisson equation
    }else if(pSddft->BC==2) // periodic case
    {
      VecScale(pSddft->twopiBTMmBPS, 2*M_PI);
    } 
  /*
   * solving the poission equation
   */
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
   * shift by a constant for periodic boundary conditions
   */
  if(pSddft->BC==2)
    {
      ConstShift_Calc(pSddft);
      VecShift(pSddft->Phi_c,pSddft->ConstShift);
    }

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
  PetscPrintf(PETSC_COMM_WORLD,"Energy correction:     %.12lf (Hartree/atom)\n", pSddft->Ecorrection*Cst/pSddft->nAtoms);
  PetscPrintf(PETSC_COMM_WORLD,"Band structure energy: %.12lf (Hartree/atom)\n", pSddft->Eband*Cst/pSddft->nAtoms);  
  PetscPrintf(PETSC_COMM_WORLD,"Exchange correlation:  %.12lf (Hartree/atom)\n", pSddft->Exc*Cst/pSddft->nAtoms);
  PetscPrintf(PETSC_COMM_WORLD,"Entropy*kb*T:          %.12lf (Hartree/atom)\n", -pSddft->Entropy*Cst/pSddft->nAtoms);
   
  /*
   * display total energy
   */
  PetscPrintf(PETSC_COMM_WORLD,"Free energy:  %.12lf (Hartree/atom)\n",Cst*pSddft->Eatom);    
  PetscPrintf(PETSC_COMM_WORLD,"Total free energy of system: %.12lf (Hartree)\n",Cst*pSddft->Eatom*pSddft->nAtoms);
  
}

///////////////////////////////////////////////////////////////////////////////////////////////
//                              ConstShift_Calc: constant shift in                           //
//                       electrostatic potential for periodic boundary conditions            //
///////////////////////////////////////////////////////////////////////////////////////////////
void ConstShift_Calc(SDDFT_OBJ* pSddft) 
{
 
  PetscScalar tableR[MAX_TABLE_SIZE],tableVps[MAX_TABLE_SIZE],r,rmax,rcut=pSddft->REFERENCE_CUTOFF,delta=pSddft->delta;
  PetscInt  i=0,poscnt,index=0,l;
  PetscInt tablesize;  
  PetscScalar *pAtompos; 
  PetscScalar *YD=NULL;
  PetscScalar Dtemp;  
  PetscInt I,J,K;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscScalar x,y,z,x0,y0,z0,X0,Y0,Z0,noetot=pSddft->noetot,Vps,Vps_TM,cutoffr;
  PetscScalar Vc0[1];
  PetscInt pos[1];pos[0]=0;PetscInt offset;
  int at,count;
  PetscInt PP,QQ,RR;
  PetscInt Imax_x,Imin_x,Imax_y,Imin_y,Imax_z,Imin_z;
  PetscScalar val;
  Vec wts;

  /*
   * Difference in pseudopotentials at the leftmost corner node is calculated.
   * This is the first entry of a DMDA vector.
   */
  I=0;J=0;K=0;
  VecGetArray(pSddft->Atompos,&pAtompos);
  for(at=0;at<pSddft->Ntype;at++)
    {
      cutoffr=pSddft->CUTOFF[at];
      offset = (PetscInt)ceil(cutoffr/delta + 0.5);
	 
      tableR[0]=0.0; tableVps[0]=pSddft->psd[at].Vloc[0]; 
      count=1;
      do{
	tableR[count] = pSddft->psd[at].RadialGrid[count-1];
	tableVps[count] = pSddft->psd[at].Vloc[count-1]; 
	count++;

      }while(tableR[count-1] <= pSddft->CUTOFF[at]+4.0);
      rmax = tableR[count-1];
	 
      int start,end;
      start = (int)floor(pSddft->startPos[at]/3);
      end = (int)floor(pSddft->endPos[at]/3);

      PetscMalloc(sizeof(PetscScalar)*count,&YD);
      getYD_gen(tableR,tableVps,YD,count); 

      for(poscnt=start;poscnt<=end;poscnt++)
	{        
	  X0 = pAtompos[index++];
	  Y0 = pAtompos[index++];
	  Z0 = pAtompos[index++];	

	  Imax_x=0; Imin_x=0; Imax_y=0; Imin_y=0; Imax_z=0; Imin_z=0;
     
	  Imax_x= ceil(cutoffr/R_x);      
      
	  Imin_x= -ceil(cutoffr/R_x);       
     
	  Imax_y= ceil(cutoffr/R_y);
	
	  Imin_y= -ceil(cutoffr/R_y);
	
	  Imax_z= ceil(cutoffr/R_z);
	
	  Imin_z= -ceil(cutoffr/R_z);

	  for(PP=Imin_x;PP<=Imax_x;PP++)
	    for(QQ=Imin_y;QQ<=Imax_y;QQ++)
	      for(RR=Imin_z;RR<=Imax_z;RR++)
		{ 	    
		  x0 = X0 + PP*2.0*R_x;
		  y0 = Y0 + QQ*2.0*R_y;
		  z0 = Z0 + RR*2.0*R_z;

		  x = delta*I - R_x ;
		  y = delta*J - R_y ;
		  z = delta*K - R_z ;   
		  r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
       
		  if(r == tableR[0])
		    {
		      Vps = Vps+tableVps[0];
		    }
		  else if(r > rmax)
		    {
		      Vps = Vps-pSddft->noe[at]/r;
		    }
		  else
		    {
		      ispline_gen(tableR,tableVps,count,&r,&val,&Dtemp,1,YD);
		      Vps=Vps+val;
		    }	 
		  Vps_TM=Vps_TM+PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);

		}
	}
    }  
  VecDuplicate(pSddft->elecDensRho,&wts);   
  VecZeroEntries(wts);
  VecSetValue(wts,0,1.0,INSERT_VALUES);
  VecAssemblyBegin(wts);
  VecAssemblyEnd(wts);   
  VecDot(pSddft->Phi_c,wts,&Vc0[0]);   
  VecDestroy(&wts);   
  pSddft->ConstShift = -1.0*(Vc0[0] - (Vps_TM - Vps));      
}

