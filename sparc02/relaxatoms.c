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
  | file name: relaxatoms.cc          
  |
  | Description: This file contains the functions required for atomic relaxation
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana, Deepa Phanish
  |
  | Last Modified: 2/29/2016   
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "isddft.h"
///////////////////////////////////////////////////////////////////////////////////////////////
//           FormFunction_relaxAtoms: performs one electronic structure minimization for     //
//                                      fixed atomic positions                               //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode FormFunction_relaxAtoms(SDDFT_OBJ* pSddft)
{
 
  PetscPrintf(PETSC_COMM_WORLD," ***  Start of DFT calculation for fixed atomic positions (NONPERIODIC)***\n");
 
  /*
   * calculate pseudocharge density
   */
  ChargDensB_VecInit(pSddft);
  
  /*
   * update the nonlocal pseudopotential
   */
  if(pSddft->RelaxCount !=0) // not the first calculation
    {
      PetscPrintf(PETSC_COMM_WORLD,"Estimating Nonzeros in Hamiltonian : \n");   
      EstimateNonZerosNonlocalPseudopot(pSddft);   
      PetscPrintf(PETSC_COMM_WORLD,"making Nonlocal Pseudopotential matrix \n");     
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
  Force_Nonlocal(pSddft,&pSddft->XOrb);
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
  PetscPrintf(PETSC_COMM_WORLD,"***   End of DFT calculation for fixed atomic positions  (NON-PERIODIC)***\n");   
   
  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//   GammaPoint_FormFunction_relaxAtoms: performs one electronic structure minimization for  //
//                         fixed atomic positions (gamma point formulation)                  //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode GammaPoint_FormFunction_relaxAtoms(SDDFT_OBJ* pSddft)
{    
  PetscPrintf(PETSC_COMM_WORLD," ***  Start of DFT calculation for fixed atomic positions (GAMMA POINT PERIODIC)***\n");

  /*
   * calculate pseudocharge density
   */ 
  PeriodicChargDensB_VecInit(pSddft);
  /*
   * update the nonlocal pseudopotential
   */
  if(pSddft->RelaxCount !=0) 
    {
      PetscPrintf(PETSC_COMM_WORLD,"Estimating Nonzeros in Hamiltonian : \n");   
      PeriodicEstimateNonZerosNonlocalPseudopot(pSddft);   
      PetscPrintf(PETSC_COMM_WORLD,"making Nonlocal Pseudopotential matrix \n");     
      PeriodicLaplacianNonlocalPseudopotential_MatInit(pSddft);        
      Wavefunctions_MatMatMultSymbolic(pSddft); 
    }
   
  /*
   * calculate energy correction
   */
  PeriodicChargDensB_TM_VecInit(pSddft);
  CorrectionEnergy_Calc(pSddft);

  /*
   * calculate electron density using Self Consistent Field iteration
   */
  SelfConsistentField(pSddft);
  /*
   * calculate force correction
   */
  PeriodicCalculate_forceCorrection(pSddft);
  /*
   * calculate local component of forces
   */
  PeriodicCalculate_force(pSddft);
  /*
   * calculate nonlocal component of forces
   */
  PeriodicForce_Nonlocal(pSddft,&pSddft->XOrb);
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
  PetscPrintf(PETSC_COMM_WORLD,"***   End of DFT calculation for fixed atomic positions  (GAMMA POINT PERIODIC)***\n");  

  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//   kPoint_FormFunction_relaxAtoms: performs one electronic structure minimization for  //
//                         fixed atomic positions (k-point formulation)                  //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode kPoint_FormFunction_relaxAtoms(SDDFT_OBJ* pSddft)
{
       
  PetscPrintf(PETSC_COMM_WORLD," ***  Start of DFT calculation for fixed atomic positions (k-point PERIODIC)***\n");

  /*
   * calculate pseudocharge density
   */
  PeriodicChargDensB_VecInit(pSddft);
  /*
   * update the nonlocal pseudopotential
   */
  if(pSddft->RelaxCount !=0)
    {
      PetscPrintf(PETSC_COMM_WORLD,"Estimating Nonzeros in Hamiltonian : \n");   
      PeriodicEstimateNonZerosNonlocalPseudopot(pSddft);   
      PetscPrintf(PETSC_COMM_WORLD,"making Nonlocal Pseudopotential matrix \n");     
      kPointHamiltonian_MatCreate(pSddft);     
    }

  /*
   * calculate energy correction
   */
  PeriodicChargDensB_TM_VecInit(pSddft);
  CorrectionEnergy_Calc(pSddft);
  /*
   * calculate electron density using Self Consistent Field iteration
   */
  kPointSelfConsistentField(pSddft);
  /*
   * calculate force correction
   */
  PeriodicCalculate_forceCorrection(pSddft);
  /*
   * calculate local component of forces
   */
  PeriodicCalculate_force(pSddft);
  /*
   * calculate nonlocal component of forces
   */
  kPointPeriodicForce_Nonlocal(pSddft);  
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
  MatDestroy(&pSddft->HamiltonianOpr1); // destroying real hamiltonian
  MatDestroy(&pSddft->HamiltonianOpr2); // destroying imaginary hamiltonian
   
  /*
   * display forces on atoms and atomic poistions
   */
  Display_force(pSddft);
  Display_Atompos(pSddft);    
  pSddft->RelaxCount++;    
  PetscPrintf(PETSC_COMM_WORLD,"***   End of DFT calculation for fixed atomic positions  (k POINT PERIODIC)***\n");   
   

  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//    Solve_ElectronicStructureProblem: call respective  electronic structure minimization   //
///////////////////////////////////////////////////////////////////////////////////////////////
void Solve_ElectronicStructureProblem(SDDFT_OBJ* pSddft)
{
  if(pSddft->BC==1)
    {
      /*
       * Non-periodic code
       */
      FormFunction_relaxAtoms(pSddft);

    }else if (pSddft->BC==2)
    {
      if(pSddft->Nkpts==1)
	{
	  /*
	   * Periodic Periodic Gamma Point code
	   */
	  GammaPoint_FormFunction_relaxAtoms(pSddft);
	}else
	{
	  /*
	   * Periodic k Point code
	   */
	  kPoint_FormFunction_relaxAtoms(pSddft);
	}
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////
//    NLCG_relaxAtoms: Nonlinear Conjugate Gradient method for ground state minimization     //
//             Reference: An Introduction to the Conjugate Gradient Method Without           //
//                         Agonizing Pain, Jonathan Richard Shewchuk                         //
///////////////////////////////////////////////////////////////////////////////////////////////
void NLCG_relaxAtoms(SDDFT_OBJ* pSddft)
{
  PetscInt  i=0,j,k=0,imax=pSddft->MAXIT_NLCG,jmax=6, n=30, jsum=0;
  PetscScalar deltaNew,deltad,deltaOld,deltaMid,tol1,tol2=1e-10,sigma0=0.5,alpha,etaPrev,eta,beta;
  Vec r,d,s,y,F;  
  int inCtr;

  tol1 = pSddft->NLCGTOL*3*pSddft->nAtoms;
  
  VecDuplicate(pSddft->Atompos,&F);
  VecDuplicate(pSddft->Atompos,&r);
  VecDuplicate(pSddft->Atompos,&d);
  VecDuplicate(pSddft->Atompos,&s);
  VecDuplicate(pSddft->Atompos,&y);
  /*
   * update electron density, energy and forces
   */  
  Solve_ElectronicStructureProblem(pSddft);
  
  VecCopy(pSddft->forces,r); inCtr=0;   
  VecCopy(r,s);
  VecCopy(s,d);
  VecDot(r,d,&deltaNew);  
 
  while((i<imax) && (deltaNew >tol1))
    {
      PetscPrintf(PETSC_COMM_WORLD,"---------------------------------------\n");
      PetscPrintf(PETSC_COMM_WORLD," \n (Outer) Relaxation step: %d \n", i+1); 
      PetscPrintf(PETSC_COMM_WORLD,"-------------------------------------- \n");

      PetscPrintf(PETSC_COMM_WORLD,"deltaNew FORCE: %0.16lf \t \n",deltaNew);
      j=0;
      VecDot(d,d,&deltad);
      alpha = -sigma0;
      /*
       * perturb atomic positions
       */
      VecWAXPY(y,sigma0,d,pSddft->Atompos);

      /*
       * if an atom has moved out of the simulation domain, map it back.
       * this is only applicable for periodic boundary conditions
       */
      Periodic_MapAtoms(pSddft);   
      VecCopy(y,pSddft->Atompos);
      /*
       * update electron density, energy and forces
       */  
      Solve_ElectronicStructureProblem(pSddft); 
      VecCopy(pSddft->forces,F);    
      /*
       * replace back the original atomic positions
       */   
      VecWAXPY(pSddft->Atompos,-sigma0,d,y);       
      /*
       * if an atom has moved out of the simulation domain, map it back.
       * this is only applicable for periodic boundary conditions
       */
      Periodic_MapAtoms(pSddft); 
      VecDot(F,d,&etaPrev); etaPrev = -etaPrev;

      /*
       * line search
       */
      do
	{      
	  PetscPrintf(PETSC_COMM_WORLD,"---------------------------------------\n");
	  PetscPrintf(PETSC_COMM_WORLD,"\n  (Inner) Relaxation step: %d \n", j+1);
	  PetscPrintf(PETSC_COMM_WORLD,"---------------------------------------\n");
      
	  if(inCtr==0)
	    {
	      VecDot(r,d,&eta); eta = -eta;
	    }
	  else{
	    /*
	     * update electron density, energy and forces
	     */
	    Solve_ElectronicStructureProblem(pSddft);
	    VecCopy(pSddft->forces,F);       
	    VecDot(F,d,&eta); eta = -eta;
	  }
      
	  alpha = alpha*(eta/(etaPrev-eta));      
	  /*
	   * perturb atomic positions
	   */
	  VecAXPY(pSddft->Atompos,alpha,d);    
     
	  /*
	   * if an atom has moved out of the simulation domain, map it back.
	   * this is only applicable for periodic boundary conditions
	   */
	  Periodic_MapAtoms(pSddft);   
      
	  etaPrev = eta;
	  j++; inCtr++;
	  PetscPrintf(PETSC_COMM_WORLD,"************************************** \n"); 

	}while((j<jmax) && (alpha*alpha*deltad>tol2));
      jsum = jsum+j;
      /*
       * update electron density, energy and forces
       */
      Solve_ElectronicStructureProblem(pSddft); 
      VecCopy(pSddft->forces,r); inCtr=0;
        
      deltaOld = deltaNew;

      VecDot(r,s,&deltaMid);

      VecCopy(r,s);
 
      VecDot(r,s,&deltaNew);
      beta = (deltaNew-deltaMid)/deltaOld;
      k++;
    
      if((k==n) || (beta<=0))
	{
	  VecCopy(s,d);
	  k=0;
	}
      else
	{
	  VecAYPX(d,beta,s);
	}
      i++;
      PetscPrintf(PETSC_COMM_WORLD,"************************************** \n"); 
    }

  PetscPrintf(PETSC_COMM_WORLD,"Total number of inner iterations:%d \n",jsum);
  PetscPrintf(PETSC_COMM_WORLD,"Total number of outer iterations:%d \n",i);
 
          
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
void Display_Atompos(SDDFT_OBJ* pSddft)
{    
  PetscScalar *pAtompos;
  PetscInt poscnt,Index=0;

  VecGetArray(pSddft->Atompos,&pAtompos); 
  PetscPrintf(PETSC_COMM_WORLD,"Atomic positions (Bohr) \n");  
  for(poscnt=0;poscnt<pSddft->nAtoms;poscnt++)
    {
      PetscPrintf(PETSC_COMM_WORLD,"%9.9f \t %9.9f \t %9.9f \n",pAtompos[Index],pAtompos[Index+1],pAtompos[Index+2]);
      Index = Index+3;
    }
  PetscPrintf(PETSC_COMM_WORLD,"\n");

  VecRestoreArray(pSddft->Atompos,&pAtompos);
  return;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//                   Display_Relax: print the constraints on atomic relaxations              //
///////////////////////////////////////////////////////////////////////////////////////////////
void Display_Relax(SDDFT_OBJ* pSddft)
{    
  PetscScalar *pmvAtmConstraint;
  PetscInt poscnt,Index=0;
    
  VecGetArray(pSddft->mvAtmConstraint,&pmvAtmConstraint); 
  PetscPrintf(PETSC_COMM_WORLD,"Atomic relaxation flag \n"); 
  for(poscnt=0;poscnt<pSddft->nAtoms;poscnt++)
    {
      PetscPrintf(PETSC_COMM_WORLD,"%f\t %f\t %f\n",pmvAtmConstraint[Index],pmvAtmConstraint[Index+1],pmvAtmConstraint[Index+2]);
      Index = Index+3;
    }
  PetscPrintf(PETSC_COMM_WORLD,"\n");

  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n\n");
  VecRestoreArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);
  return;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//     SDDFT_Nonperiodic: function for nonperiodic Density Functional theory calculation     //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode SDDFT_Nonperiodic(SDDFT_OBJ* pSddft)
{

  int ierr; 
  PetscLogDouble t1,t2,elapsed_time;    
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  
  /*
   * create the data structures required for DFT calculation
   */
  ierr = Objects_Create(pSddft); CHKERRQ(ierr);     
  Laplace_matInit(pSddft); 
  Gradient_matInit(pSddft);

  /*
   * calculate initial guess electron density
   */
  SuperpositionAtomicCharge_VecInit(pSddft);
  VecCopy(pSddft->SuperposAtRho,pSddft->elecDensRho); 

  /*
   * create nonlocal pseudopotential operator
   */
  EstimateNonZerosNonlocalPseudopot(pSddft);   
  LaplacianNonlocalPseudopotential_MatInit(pSddft);
  Wavefunctions_MatInit(pSddft);
   
  if(pSddft->RelaxFlag==1)
    {
      /*
       * perform atomic relaxation
       */
      NLCG_relaxAtoms(pSddft);
    }
  else    
    { 
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
      Force_Nonlocal(pSddft,&pSddft->XOrb);
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
///////////////////////////////////////////////////////////////////////////////////////////////
//     SDDFT_Periodic: function for nonperiodic Density Functional theory calculation        //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode SDDFT_Periodic(SDDFT_OBJ* pSddft)
{

  int ierr; 
  PetscLogDouble t1,t2,elapsed_time;    
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  
  /*
   * create the data structures required for DFT calculation
   */
  ierr = Objects_Create(pSddft); CHKERRQ(ierr);     
  Laplace_matInit(pSddft); 
  Gradient_matInit(pSddft);
  /*
   * calculate initial guess electron density
   */
  PeriodicSuperpositionAtomicCharge_VecInit(pSddft);
  VecCopy(pSddft->SuperposAtRho,pSddft->elecDensRho); 

  /*
   * create nonlocal pseudopotential operator
   */
  PeriodicEstimateNonZerosNonlocalPseudopot(pSddft);      
  PeriodicLaplacianNonlocalPseudopotential_MatInit(pSddft);
  Wavefunctions_MatInit(pSddft);  
   
  if(pSddft->RelaxFlag==1)
    {
      /*
       * perform atomic relaxation
       */
      NLCG_relaxAtoms(pSddft);
    }
  else 
    { 
      /*
       * DFT calculation with fixed atomic positions
       */
      Display_Atompos(pSddft);
      Display_Relax(pSddft);
      /*
       * calculate pseudocharge density
       */
      PeriodicChargDensB_VecInit(pSddft);
      /*
       * calculate energy correction
       */
      PeriodicChargDensB_TM_VecInit(pSddft);
      CorrectionEnergy_Calc(pSddft);  
      /*
       * calculate electron density using Self Consistent Field iteration
       */
      SelfConsistentField(pSddft);
      /*
       * calculate force correction
       */
      PeriodicCalculate_forceCorrection(pSddft); 
      /*
       * calculate local component of forces
       */
      PeriodicCalculate_force(pSddft); 

      /*
       * calculate nonlocal component of forces
       */
      PeriodicForce_Nonlocal(pSddft,&pSddft->XOrb); 
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
///////////////////////////////////////////////////////////////////////////////////////////////
//     SDDFT_kPointPeriodic: function for nonperiodic Density Functional theory calculation  //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode SDDFT_kPointPeriodic(SDDFT_OBJ* pSddft)
{

  int ierr; 
  PetscLogDouble t1,t2,elapsed_time;    
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  /*
   * create the data structures required for DFT calculation
   */
  ierr = Objects_Create(pSddft); CHKERRQ(ierr);     
  Laplace_matInit(pSddft); 
  Gradient_matInit(pSddft);
  /*
   * calculate initial guess electron density
   */
  PeriodicSuperpositionAtomicCharge_VecInit(pSddft);
  VecCopy(pSddft->SuperposAtRho,pSddft->elecDensRho);   

  /*
   * create nonlocal pseudopotential operator
   */
  PeriodicEstimateNonZerosNonlocalPseudopot(pSddft);   
  kPointHamiltonian_MatCreate(pSddft);
  kPointWavefunctions_MatInit(pSddft);
  
  if(pSddft->RelaxFlag==1)
    {
      /*
       * perform atomic relaxation
       */
      NLCG_relaxAtoms(pSddft);
    }
  else
    { 
      /*
       * DFT calculation with fixed atomic positions
       */
      Display_Atompos(pSddft);
      Display_Relax(pSddft);
      /*
       * calculate pseudocharge density
       */
      PeriodicChargDensB_VecInit(pSddft);
      /*
       * calculate energy correction
       */
      PeriodicChargDensB_TM_VecInit(pSddft);
      CorrectionEnergy_Calc(pSddft);  
         
      /*
       * calculate electron density using Self Consistent Field iteration
       */
      kPointSelfConsistentField(pSddft);
      /*
       * calculate force correction
       */
      PeriodicCalculate_forceCorrection(pSddft); 
      /*
       * calculate local component of forces
       */
      PeriodicCalculate_force(pSddft); 
      /*
       * calculate nonlocal component of forces
       */    
      kPointPeriodicForce_Nonlocal(pSddft);
      /*
       * symmetrysize forces 
       */
      Symmetrysize_force(pSddft);      
      
      Display_force(pSddft); 
       /*
	* initialize vectors to zero 
	*/
      Set_VecZero(pSddft); 
      /*
       * destroy Hamiltonian 
       */
      MatDestroy(&pSddft->HamiltonianOpr1); // destroying real hamiltonian
      MatDestroy(&pSddft->HamiltonianOpr2); // destroying imaginary hamiltonian	
    }
      
  return 0;   
} 
///////////////////////////////////////////////////////////////////////////////////////////////
//     Periodic_MapAtoms: Mapping outside atoms back into the domain using periodic mapping  //
///////////////////////////////////////////////////////////////////////////////////////////////
void Periodic_MapAtoms(SDDFT_OBJ* pSddft)
{

  PetscScalar *pAtompos;
  int start,end;int at,poscnt;
  PetscInt index=0;
  PetscScalar Rx=pSddft->range_x;
  PetscScalar Ry=pSddft->range_y;
  PetscScalar Rz=pSddft->range_z;
  
  /*
   * we assume that the an atom can only move to a periodic cell which is an immediate neighbor 
   * of the simulation domain
   */
  if(pSddft->BC==2)
    {
      VecGetArray(pSddft->Atompos,&pAtompos); 
      for(at=0;at<pSddft->Ntype;at++)
	{ 
	  start = (int)floor(pSddft->startPos[at]/3);
	  end = (int)floor(pSddft->endPos[at]/3);

	  for(poscnt=start;poscnt<=end;poscnt++)
	    { 
	      if(pAtompos[index]<-Rx)
		pAtompos[index]=pAtompos[index]+2.0*Rx;
	      else if(pAtompos[index]>Rx)
		pAtompos[index]=pAtompos[index]-2.0*Rx;

	      index++;

	      if(pAtompos[index]<-Ry)
		pAtompos[index]=pAtompos[index]+2.0*Ry;
	      else if(pAtompos[index]>Ry)
		pAtompos[index]=pAtompos[index]-2.0*Ry;

	      index++;

	      if(pAtompos[index]<-Rz)
		pAtompos[index]=pAtompos[index]+2.0*Rz;
	      else if(pAtompos[index]>Rz)
		pAtompos[index]=pAtompos[index]-2.0*Rz;

	      index++;
	    }
	}  
      VecRestoreArray(pSddft->Atompos,&pAtompos);
    }
}
 

