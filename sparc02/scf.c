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
  | file name: scf.cc          
  |
  | Description: This file contains the functions required for self consistent field iteration
  | and subspace projection
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana
  |
  | Last Modified: 2/18/2016   
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "petscsys.h"
#include <cmath>

#include "cblas.h"
#include <lapacke.h>
//#include "mkl_lapacke.h"
//#include "mkl.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace std;
///////////////////////////////////////////////////////////////////////////////////////////////
//                        SelfConsistentField: Self Consistent Field iteration               // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar SelfConsistentField(SDDFT_OBJ* pSddft)
{
  
  int rank;
  int count=0,mixCtr,SCFcount=1;
  PetscScalar EigenMin,EigenMax,lambda_cutoff=0.0;
  PetscInt Nstates = pSddft->Nstates;   
  PetscErrorCode ierr;
  PetscInt *idxn;
  Vec tempVec,Veff_temp;   
  PetscScalar Cst = 1.0/27.211384523;

  VecDuplicate(pSddft->elecDensRho,&Veff_temp);
  VecDuplicate(pSddft->elecDensRho,&tempVec);
  PetscScalar error=1.0,norm1,norm2;    
  Mat Msub,Hsub;
  PetscScalar *arrHsub,*arrMsub;
  int i;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);     
 
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD,"                          Self Consistent Field                            \n");
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");

  /*
   * set up dense matrices for subspace eigen problem
   */
  MatCreate(PETSC_COMM_SELF,&Hsub);
  MatSetSizes(Hsub,PETSC_DECIDE,PETSC_DECIDE,Nstates,Nstates);
  MatSetType(Hsub,MATSEQDENSE);
  MatSetFromOptions(Hsub);
  MatSetUp(Hsub);
  MatDuplicate(Hsub,MAT_DO_NOT_COPY_VALUES,&Msub);   
  
  /* 
   * solve poission equation and form effective potential
   */
  SolvePoisson(pSddft); // electrostatic potential is calculated
  Vxc_Calc_CA(pSddft); // exchange correlation potential is calculated
  VecWAXPY(pSddft->Veff,1.0,pSddft->Vxc,pSddft->potentialPhi); // effective potential 
	
  VecCopy(pSddft->Veff,pSddft->xk); // this is required for mixing
     
  while(error>pSddft->TOLSCF && count <= (pSddft->MAXITSCF-pSddft->SCFNewRhoCalcCtr))
    {        
      
      /*
       * add effective potential to Hamiltonian. Need to be subtracted out at the end of SCF
       */
      MatDiagonalSet(pSddft->HamiltonianOpr,pSddft->Veff,ADD_VALUES);
        
      /*
       * find smallest and largest eigenvalues of Hamiltonian using lanczos algorithm
       */
      Lanczos(pSddft,&EigenMin,&EigenMax);             
      ChebyshevFiltering(pSddft,pSddft->ChebDegree,lambda_cutoff,EigenMax,EigenMin);
     
      /*
       * calculate projected Hamiltonian and Overlap matrix 
       */
      ProjectMatrices(pSddft,&pSddft->YOrb,&Hsub,&Msub);
        
      /*
       * solve the generalized eigenvalue problem for eigenvalues and eigenvectors
       */
      MatDenseGetArray(Hsub,&arrHsub);
      MatDenseGetArray(Msub,&arrMsub);	
      LAPACKE_dsygv(LAPACK_COL_MAJOR,1,'V','U',Nstates,arrHsub,Nstates,arrMsub,Nstates,pSddft->lambda);
          	 
      MatDenseRestoreArray(Hsub,&arrHsub); // this has the eigenvectors
      MatDenseRestoreArray(Msub,&arrMsub);
	 	
      /*
       * subspace rotation
       */
      RotatePsi(pSddft,&pSddft->YOrb,&Hsub,&pSddft->XOrb);
	 
      /*
       * calculate fermi energy
       */
      pSddft->lambda_f = findRootBrent(pSddft,EigenMin,EigenMax,1.0e-12);   
      lambda_cutoff = pSddft->lambda_f+0.1;              
      VecCopy(pSddft->Veff,Veff_temp);           
      /*
       * subtract the effective potential from the Hamiltonian that was previously added     
       */
      VecScale(pSddft->Veff,-1.0);
      MatDiagonalSet(pSddft->HamiltonianOpr,pSddft->Veff,ADD_VALUES);
      VecScale(pSddft->Veff,-1.0); 
            
      /*
       * update electron density
       */
      if(count > pSddft->SCFNewRhoCalcCtr)
        {
	  PetscPrintf(PETSC_COMM_WORLD,"Iteration number: %d \n",SCFcount);
	  CalculateDensity(pSddft,&pSddft->XOrb); // electron density is updated         	 
	  SolvePoisson(pSddft); 
	  Vxc_Calc_CA(pSddft); 
	  VecWAXPY(pSddft->Veff,1.0,pSddft->Vxc,pSddft->potentialPhi);
           
	  VecWAXPY(tempVec,-1.0,Veff_temp,pSddft->Veff);
           
	  VecNorm(tempVec,NORM_2,&norm1);
	  VecNorm(pSddft->Veff,NORM_2,&norm2);           
	  SystemEnergy_Calc(pSddft);
	  PetscPrintf(PETSC_COMM_WORLD,"Free Energy: %0.12lf (Ha/atom) \n",pSddft->Eatom*Cst);
	  mixCtr = count-pSddft->SCFNewRhoCalcCtr;           
	  error = mix(pSddft,mixCtr,pSddft->Veff);
	  error = norm1/(norm2+1e-16);
	  PetscPrintf(PETSC_COMM_WORLD,"Error = %.12lf \n",error);
	  SCFcount++;    
	  PetscPrintf(PETSC_COMM_WORLD,"===========================================\n");
        } 	       
      count++;   
     
    }     
    
  pSddft->SCFNewRhoCalcCtr = -1;  
  PetscPrintf(PETSC_COMM_WORLD,"Convergence of Self Consistent Field iterations achieved! \n");
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n\n");
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD,"                             Energy                                   \n");
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD,"Fermi energy: %.12lf (Hartree)\n",pSddft->lambda_f);
  Display_Energy(pSddft);
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n\n");
    
  VecDestroy(&tempVec);
  VecDestroy(&Veff_temp);
  MatDestroy(&Hsub);
  MatDestroy(&Msub);
    
  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                       ProjectMatrices: Projection of matrix onto subspace                 // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode ProjectMatrices(SDDFT_OBJ* pSddft, Mat* Psi,Mat *Hsub,Mat* Msub)
{

  /*
   * Psi is the matrix used for projecting
   * Hsub is the subspace Hamiltonian
   * Msub is the subspace overlap matrix
   */

  PetscInt Nstates=pSddft->Nstates;   
  int M,N,K;
  double alpha,beta;
  PetscInt rowStart,rowEnd;
  int i,rank,ierr;
  IS irow,icol;
  PetscScalar *arrSubMat,*arrPsiSeq,*arrHPsiSeq;
  alpha=1.0; beta=0.0;

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank); 
  MatGetOwnershipRange(*Psi,&rowStart,&rowEnd);
  ISCreateStride(MPI_COMM_SELF,rowEnd-rowStart,rowStart,1,&irow);
  ISCreateStride(MPI_COMM_SELF,Nstates,0,1,&icol); 
 
  MatDenseGetArray(*Psi,&arrPsiSeq);
  MatDenseGetArray(*Msub,&arrSubMat);
  
  M=Nstates; N=Nstates; K=rowEnd-rowStart;
  /*
   * compute subspace overlap matrix first
   */

  /*
   * multiplying sequential parts of matrices
   */
  cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,M,N,K,alpha,arrPsiSeq,K,arrPsiSeq,K,beta,arrSubMat,M);
   
  /*
   * sum entries from all processors
   */
  ierr =  MPI_Allreduce(MPI_IN_PLACE,arrSubMat,Nstates*Nstates,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); CHKERRQ(ierr); 
 
  /*
   * assemble subspace overlap matrix
   */
  MatDenseRestoreArray(*Msub,&arrSubMat);
  MatAssemblyBegin(*Msub,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*Msub,MAT_FINAL_ASSEMBLY);   
  
  /*
   * compute subspace Hamiltonian
   */

  /*
   * first multiply the Hamiltonian and orbitals 
   */
  MatMatMultNumeric(pSddft->HamiltonianOpr,*Psi,pSddft->YOrbNew);
  MatDenseGetArray(*Hsub,&arrSubMat);
  MatDenseGetArray(pSddft->YOrbNew,&arrHPsiSeq);
  
  /*
   * multiplying sequential parts of matrices
   */
  cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,M,N,K,alpha,arrPsiSeq,K,arrHPsiSeq,K,beta,arrSubMat,M);
  
  /*
   * sum entries from all processors
   */
  ierr =  MPI_Allreduce(MPI_IN_PLACE,arrSubMat,Nstates*Nstates,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
  
  /*
   * assemble matrices
   */
  MatDenseRestoreArray(*Hsub,&arrSubMat);
  MatAssemblyBegin(*Hsub,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*Hsub,MAT_FINAL_ASSEMBLY);
  
  /*
   * restore wavefunctions
   */
  MatDenseRestoreArray(*Psi,&arrPsiSeq);
  MatDenseRestoreArray(pSddft->YOrbNew,&arrHPsiSeq);

  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//       kPointSelfConsistentField: Self Consistent Field iteration for k point sampling     // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar kPointSelfConsistentField(SDDFT_OBJ* pSddft)
{
  
  int rank;
  int count=0,mixCtr,SCFcount=1;
  PetscScalar EigenMin,EigenMax,lambda_cutoff=0.0;
  PetscInt Nstates = pSddft->Nstates;   
  PetscErrorCode ierr;
  PetscInt *idxn;
  Vec tempVec,Veff_temp;
  int k=0,nk1,nk2,nk3;
  PetscScalar k1,k2,k3;
  PetscScalar R_x=pSddft->range_x;
  PetscScalar R_y=pSddft->range_y;
  PetscScalar R_z=pSddft->range_z;
  PetscScalar Cst = 1.0/27.211384523;
  
  VecDuplicate(pSddft->elecDensRho,&Veff_temp);
  VecDuplicate(pSddft->elecDensRho,&tempVec);
  PetscScalar error=1.0,norm1,norm2;
 
  Mat Msub1,Hsub1; Mat Msub2,Hsub2;
  PetscScalar *arrHsub,*arrMsub;// for storing the values in an array (column wise) and passing to lapack functions
  int i;

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD,"                          Self Consistent Field                            \n");
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");

  Vec kVec;
  VecDuplicate(pSddft->elecDensRho,&kVec);
  PetscScalar kval;
  
  /*
   * set up dense matrices for subspace eigen problem
   */
  MatCreate(PETSC_COMM_SELF,&Hsub1);
  MatSetSizes(Hsub1,PETSC_DECIDE,PETSC_DECIDE,Nstates,Nstates);
  MatSetType(Hsub1,MATSEQDENSE);
  MatSetFromOptions(Hsub1);
  MatSetUp(Hsub1);
 
  MatDuplicate(Hsub1,MAT_DO_NOT_COPY_VALUES,&Hsub2);
  MatDuplicate(Hsub1,MAT_DO_NOT_COPY_VALUES,&Msub1);
  MatDuplicate(Hsub1,MAT_DO_NOT_COPY_VALUES,&Msub2);  
  /* 
   * solve poission equation and form effective potential
   */
  SolvePoisson(pSddft);// electrostatic potential is calculated
  Vxc_Calc_CA(pSddft); // exchange correlation potential is calculated 
  VecWAXPY(pSddft->Veff,1.0,pSddft->Vxc,pSddft->potentialPhi);// effective potential 
  VecCopy(pSddft->Veff,pSddft->xk);  
  k=0;
   
  while(error>pSddft->TOLSCF && count <= (pSddft->MAXITSCF-pSddft->SCFNewRhoCalcCtr))
    {  
      k=0;
      /*
       * loop over k points
       */
      for(nk1=1;nk1<=pSddft->Kx;nk1++)
	for(nk2=1;nk2<=pSddft->Ky;nk2++)
	  for(nk3=1;nk3<=ceil(pSddft->Kz/2.0);nk3++)
	    {
	      /*
	       * calculate k points using Monkhorst pack grid
	       */
	      k1=((2.0*nk1-pSddft->Kx-1.0)/(2.0*pSddft->Kx))*(M_PI/R_x);
	      k2=((2.0*nk2-pSddft->Ky-1.0)/(2.0*pSddft->Ky))*(M_PI/R_y);
	      k3=((2.0*nk3-pSddft->Kz-1.0)/(2.0*pSddft->Kz))*(M_PI/R_z);
	    	    
	      kval=(k1*k1+k2*k2+k3*k3)/2.0;	     
	      VecSet(kVec,kval);
	   
	      /*
	       * form Hamiltonians(real and imaginary parts)
	       */	    
	      kPointHamiltonian_MatInit(pSddft,k1,k2,k3);	   
	      MatDiagonalSet(pSddft->HamiltonianOpr1,pSddft->Veff,ADD_VALUES);

	      /*
	       * set real part of Hamiltonian for k1,k2,k3
	       */
	      MatDiagonalSet(pSddft->HamiltonianOpr1,kVec,ADD_VALUES);
	  
	      /*
	       * do symbolic multiplication for the first SCF iteration
	       */
	      if(count==0)
		{	     
		  MatMatMultSymbolic(pSddft->HamiltonianOpr1,pSddft->XOrb1[k],PETSC_DEFAULT,&pSddft->YOrb1[k]); 
		  MatMatMultNumeric(pSddft->HamiltonianOpr1,pSddft->XOrb1[k],pSddft->YOrb1[k]);
  
		  MatMatMultSymbolic(pSddft->HamiltonianOpr2,pSddft->XOrb2[k],PETSC_DEFAULT,&pSddft->ZOrb1);
		  MatMatMultNumeric(pSddft->HamiltonianOpr2,pSddft->XOrb2[k],pSddft->ZOrb1);
 
		  MatMatMultSymbolic(pSddft->HamiltonianOpr1,pSddft->XOrb2[k],PETSC_DEFAULT,&pSddft->YOrb2[k]);
		  MatMatMultNumeric(pSddft->HamiltonianOpr1,pSddft->XOrb2[k],pSddft->YOrb2[k]);

		  MatMatMultSymbolic(pSddft->HamiltonianOpr2,pSddft->XOrb1[k],PETSC_DEFAULT,&pSddft->ZOrb2);
		  MatMatMultNumeric(pSddft->HamiltonianOpr2,pSddft->XOrb1[k],pSddft->ZOrb2);

		  MatMatMultSymbolic(pSddft->HamiltonianOpr1,pSddft->YOrb1[k],PETSC_DEFAULT,&pSddft->YOrbNew1);
		  MatMatMultNumeric(pSddft->HamiltonianOpr1,pSddft->YOrb1[k],pSddft->YOrbNew1);

		  MatMatMultSymbolic(pSddft->HamiltonianOpr2,pSddft->YOrb2[k],PETSC_DEFAULT,&pSddft->ZOrbNew1);
		  MatMatMultNumeric(pSddft->HamiltonianOpr2,pSddft->YOrb2[k],pSddft->ZOrbNew1);

		  MatMatMultSymbolic(pSddft->HamiltonianOpr1,pSddft->YOrb2[k],PETSC_DEFAULT,&pSddft->YOrbNew2);
		  MatMatMultNumeric(pSddft->HamiltonianOpr1,pSddft->YOrb2[k],pSddft->YOrbNew2);

		  MatMatMultSymbolic(pSddft->HamiltonianOpr2,pSddft->YOrb1[k],PETSC_DEFAULT,&pSddft->ZOrbNew2);
		  MatMatMultNumeric(pSddft->HamiltonianOpr2,pSddft->YOrb1[k],pSddft->ZOrbNew2); 
		}
  
	      kPointLanczos(pSddft,&EigenMin,&EigenMax);
              
	      if(count==0)
		lambda_cutoff = 0.5*(EigenMax-EigenMin);
	
	      /* 
	       * Perform Chebyshev filtering on existing wavefunctions
	       */	
	      kPointChebyshevFiltering(pSddft,pSddft->ChebDegree,lambda_cutoff,EigenMax,EigenMin,k);
      
	      /*
	       * calculate projected Hamiltonian and Overlap matrix 
	       */
	      kPointProjectMatrices(pSddft,&pSddft->YOrb1[k],&pSddft->YOrb2[k],&Hsub1,&Hsub2,&Msub1,&Msub2);
        
	      /*
	       * solve the generalized eigenvalue problem for eigenvalues and eigenvectors
	       */
	      SolveGeneralizedEigen(pSddft,&Hsub1,&Hsub2,&Msub1,&Msub2,k);	

	      /*
	       * subspace rotation
	       */
	      kPointRotatePsi(pSddft,&pSddft->YOrb1[k],&pSddft->YOrb2[k],&Hsub1,&Hsub2,&pSddft->XOrb1[k],&pSddft->XOrb2[k]);
	
	      k++;
	      VecScale(kVec,-1.0);
	      MatDiagonalSet(pSddft->HamiltonianOpr1,kVec,ADD_VALUES);
	    } 
	 
      /*
       * calculate fermi energy
       */
      pSddft->lambda_f = kPointfindRootBrent(pSddft,EigenMin,EigenMax,1.0e-12);   
      lambda_cutoff = pSddft->lambda_f+0.1; 	        
      VecCopy(pSddft->Veff,Veff_temp);
           
      /*
       * subtract the effective potential from the Hamiltonian that was previously added     
       */
      VecScale(pSddft->Veff,-1.0);
      MatDiagonalSet(pSddft->HamiltonianOpr1,pSddft->Veff,ADD_VALUES);
      VecScale(pSddft->Veff,-1.0); 
     
      /*
       * update electron density
       */
      if(count > pSddft->SCFNewRhoCalcCtr)
        {	  
	  PetscPrintf(PETSC_COMM_WORLD,"Iteration number: %d \n",SCFcount);
	  /*
	   * initialize electron density to zero first and then loop over k points
	   */
	  VecZeroEntries(pSddft->elecDensRho); 
	  for(k=0;k<pSddft->Nkpts_sym;k++)
	    {	     
	      kPointCalculateDensity(pSddft,&pSddft->XOrb1[k],&pSddft->XOrb2[k],k); 
	    }                     
	  /*
	   * solve poission equation and form effective potential  
	   */
	  SolvePoisson(pSddft);
	  Vxc_Calc_CA(pSddft); 
	  VecWAXPY(pSddft->Veff,1.0,pSddft->Vxc,pSddft->potentialPhi);
           
	  VecWAXPY(tempVec,-1.0,Veff_temp,pSddft->Veff);           
	  VecNorm(tempVec,NORM_2,&norm1);
	  VecNorm(pSddft->Veff,NORM_2,&norm2); 
           
	  /*
	   * calculate energy     
	   */
	  kPointSystemEnergy_Calc(pSddft); 
	  PetscPrintf(PETSC_COMM_WORLD,"Free Energy: %0.12lf (Ha/atom) \n",pSddft->Eatom*Cst);

	  mixCtr = count-pSddft->SCFNewRhoCalcCtr;           
	  error = mix(pSddft,mixCtr,pSddft->Veff);
	  error = norm1/(norm2+1e-16);
	  PetscPrintf(PETSC_COMM_WORLD,"Error = %.12lf \n",error);	 
	  SCFcount++;   
	  PetscPrintf(PETSC_COMM_WORLD,"===========================================\n");
        }      	       
      count++; 
    }   

  pSddft->SCFNewRhoCalcCtr = -1;  
  PetscPrintf(PETSC_COMM_WORLD,"Convergence of Self Consistent Field iterations achieved! \n");
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n\n");
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD,"                             Energy                                   \n");
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD,"Fermi energy: %.12lf (Hartree)\n",pSddft->lambda_f);
  Display_Energy(pSddft);
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n\n");
  
  VecDestroy(&tempVec);
  VecDestroy(&Veff_temp);  
  MatDestroy(&Hsub1);
  MatDestroy(&Msub1);
  MatDestroy(&Hsub2);
  MatDestroy(&Msub2);  
  
  VecDestroy(&kVec);
  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                 kPointProjectMatrices: Projection of matrix onto subspace                 // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode kPointProjectMatrices(SDDFT_OBJ* pSddft, Mat* U1,Mat* U2,Mat* Hsub1,Mat* Hsub2,Mat* Msub1,Mat* Msub2)
{
  /*
   * Psi is the matrix used for projecting
   * Hsub is the subspace Hamiltonian
   * Msub is the subspace overlap matrix
   */
  PetscInt Nstates=pSddft->Nstates;   
  int M,N,K;
  double alpha,beta;
  PetscInt rowStart,rowEnd;
  int i,rank,ierr;
  IS irow,icol;
  PetscScalar *arrSubMat1,*arrUSeq1,*arrHUSeq1;
  PetscScalar *arrSubMat2,*arrUSeq2,*arrHUSeq2;
  Mat tempMat; 

  MatDuplicate(*Hsub1,MAT_SHARE_NONZERO_PATTERN,&tempMat);
  alpha=1.0; beta=0.0;

  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MatGetOwnershipRange(*U1,&rowStart,&rowEnd);   
  ISCreateStride(MPI_COMM_SELF,rowEnd-rowStart,rowStart,1,&irow);
  ISCreateStride(MPI_COMM_SELF,Nstates,0,1,&icol); 
  M=Nstates; N=Nstates; K=rowEnd-rowStart; 
  
  /*
   * compute subspace overlap matrix first
   */
   
  /*
   * get local array
   */
  MatDenseGetArray(*U1,&arrUSeq1);
  MatDenseGetArray(*U2,&arrUSeq2);

  /*
   * compute real part 
   */
  MatDenseGetArray(*Msub1,&arrSubMat1);
  MatDenseGetArray(tempMat,&arrSubMat2);
   
  /*
   * multiplying sequential parts of matrices
   */
  cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,M,N,K,alpha,arrUSeq1,K,arrUSeq1,K,beta,arrSubMat1,M);   
  cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,M,N,K,alpha,arrUSeq2,K,arrUSeq2,K,beta,arrSubMat2,M);   

  /*
   * sum entries from all processors
   */
  ierr =  MPI_Allreduce(MPI_IN_PLACE,arrSubMat1,Nstates*Nstates,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); CHKERRQ(ierr); 
  ierr =  MPI_Allreduce(MPI_IN_PLACE,arrSubMat2,Nstates*Nstates,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); CHKERRQ(ierr); 
    
  MatDenseRestoreArray(*Msub1,&arrSubMat1);
  MatAssemblyBegin(*Msub1,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*Msub1,MAT_FINAL_ASSEMBLY);

  MatDenseRestoreArray(tempMat,&arrSubMat2);
  MatAssemblyBegin(tempMat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(tempMat,MAT_FINAL_ASSEMBLY);

  MatAXPY(*Msub1,1.0,tempMat,SAME_NONZERO_PATTERN);   

  /*
   * compute imaginary part 
   */
  MatDenseGetArray(*Msub2,&arrSubMat2);
  MatDenseGetArray(tempMat,&arrSubMat1);
   
  /*
   * multiplying sequential parts of matrices
   */
  cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,M,N,K,alpha,arrUSeq1,K,arrUSeq2,K,beta,arrSubMat2,M);   
  cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,M,N,K,alpha,arrUSeq2,K,arrUSeq1,K,beta,arrSubMat1,M);  
    
  /*
   * sum entries from all processors
   */
  ierr =  MPI_Allreduce(MPI_IN_PLACE,arrSubMat2,Nstates*Nstates,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); CHKERRQ(ierr); 
  ierr =  MPI_Allreduce(MPI_IN_PLACE,arrSubMat1,Nstates*Nstates,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); CHKERRQ(ierr); 
    
  MatDenseRestoreArray(*Msub2,&arrSubMat2);
  MatAssemblyBegin(*Msub2,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*Msub2,MAT_FINAL_ASSEMBLY);

  MatDenseRestoreArray(tempMat,&arrSubMat1);
  MatAssemblyBegin(tempMat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(tempMat,MAT_FINAL_ASSEMBLY);

  MatAXPY(*Msub2,-1.0,tempMat,SAME_NONZERO_PATTERN);   
   
  /*
   * compute subspace hamiltonian
   */
  MatMatMultNumeric(pSddft->HamiltonianOpr1,*U1,pSddft->YOrbNew1);
  MatMatMultNumeric(pSddft->HamiltonianOpr2,*U2,pSddft->ZOrbNew1);

  MatMatMultNumeric(pSddft->HamiltonianOpr1,*U2,pSddft->YOrbNew2);
  MatMatMultNumeric(pSddft->HamiltonianOpr2,*U1,pSddft->ZOrbNew2);

  MatAXPY(pSddft->YOrbNew1,-1.0,pSddft->ZOrbNew1,SAME_NONZERO_PATTERN); 
  MatAXPY(pSddft->YOrbNew2,1.0,pSddft->ZOrbNew2,SAME_NONZERO_PATTERN);

  MatDenseGetArray(pSddft->YOrbNew1,&arrHUSeq1); 
  MatDenseGetArray(pSddft->YOrbNew2,&arrHUSeq2); 

  /*
   * compute real part of subspace hamiltonian
   */
  MatDenseGetArray(*Hsub1,&arrSubMat1);
  MatDenseGetArray(tempMat,&arrSubMat2);
  /*
   * multiplying sequential parts of matrices
   */   
  cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,M,N,K,alpha,arrUSeq1,K,arrHUSeq1,K,beta,arrSubMat1,M); 
  cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,M,N,K,alpha,arrUSeq2,K,arrHUSeq2,K,beta,arrSubMat2,M); 

  /*
   * sum entries from all processors
   */
  ierr =  MPI_Allreduce(MPI_IN_PLACE,arrSubMat1,Nstates*Nstates,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
  ierr =  MPI_Allreduce(MPI_IN_PLACE,arrSubMat2,Nstates*Nstates,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
    
  MatDenseRestoreArray(*Hsub1,&arrSubMat1);
  MatDenseRestoreArray(tempMat,&arrSubMat2);

   
  /*
   * assemble real part of subspace hamiltonian
   */
  MatAssemblyBegin(*Hsub1,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*Hsub1,MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(tempMat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(tempMat,MAT_FINAL_ASSEMBLY);

  MatAXPY(*Hsub1,1.0,tempMat,SAME_NONZERO_PATTERN);   
   
  /*
   * imaginary part of subspace hamiltonian
   */
  MatDenseGetArray(*Hsub2,&arrSubMat2);
  MatDenseGetArray(tempMat,&arrSubMat1);
     
  /*
   * Multiplying sequential parts of matrices
   */
  cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,M,N,K,alpha,arrUSeq1,K,arrHUSeq2,K,beta,arrSubMat2,M); 
  cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,M,N,K,alpha,arrUSeq2,K,arrHUSeq1,K,beta,arrSubMat1,M); 
     
  /*
   * sum entries from all processors
   */
  ierr =  MPI_Allreduce(MPI_IN_PLACE,arrSubMat1,Nstates*Nstates,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
  ierr =  MPI_Allreduce(MPI_IN_PLACE,arrSubMat2,Nstates*Nstates,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);

  MatDenseRestoreArray(*Hsub2,&arrSubMat2);
  MatDenseRestoreArray(tempMat,&arrSubMat1);

  MatAssemblyBegin(*Hsub2,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*Hsub2,MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(tempMat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(tempMat,MAT_FINAL_ASSEMBLY);

  MatAXPY(*Hsub2,-1.0,tempMat,SAME_NONZERO_PATTERN);   
  /*
   * restore orbitals
   */
  MatDenseRestoreArray(*U1,&arrUSeq1);
  MatDenseRestoreArray(*U2,&arrUSeq2);

  MatDenseRestoreArray(pSddft->YOrbNew1,&arrHUSeq1);
  MatDenseRestoreArray(pSddft->YOrbNew2,&arrHUSeq2);

  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//    SolveGeneralizedEigen: solve subspace eigenvalue problem for k point sampling method   // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode SolveGeneralizedEigen(SDDFT_OBJ* pSddft,Mat* Hsub1, Mat* Hsub2, Mat* Msub1, Mat* Msub2,int k)
{
  PetscErrorCode ierr;
  PetscScalar *arrReal;
  PetscScalar *arrImag;
  PetscInt Nstates=pSddft->Nstates;
  double *w;
  lapack_complex_double *arrHsub,*arrMsub;
  int i;
  lapack_int info;
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  PetscMalloc(sizeof(double)*(pSddft->Nstates),&w);
  PetscMalloc(sizeof(lapack_complex_double)*(pSddft->Nstates*pSddft->Nstates),&arrHsub);
  PetscMalloc(sizeof(lapack_complex_double)*(pSddft->Nstates*pSddft->Nstates),&arrMsub);  
 
  /*
   * form real and complex matrices
   */ 
  MatDenseGetArray(*Msub1,&arrReal);
  MatDenseGetArray(*Msub2,&arrImag);   
  for(i=0;i<Nstates*Nstates;i++)
    {
      arrMsub[i].real = (double)arrReal[i];
      arrMsub[i].imag = (double)arrImag[i];     
    }
  MatDenseRestoreArray(*Msub1,&arrReal);
  MatDenseRestoreArray(*Msub2,&arrImag);
  
  MatDenseGetArray(*Hsub1,&arrReal);
  MatDenseGetArray(*Hsub2,&arrImag);   
  for(i=0;i<Nstates*Nstates;i++)
    {
      arrHsub[i].real = (double)arrReal[i];
      arrHsub[i].imag = (double)arrImag[i];  
      
    }	 
  info=  LAPACKE_zhegv(LAPACK_COL_MAJOR,1,'V','U',Nstates,arrHsub,Nstates,arrMsub,Nstates,w);
  if(info !=0)
    {
      printf("Error in eigenvalue problem! rank=%d, value returned from LAPACKE_zhegv =%d \n",rank,info);
    }

  for(i=0;i<Nstates;i++)
    {	     
      pSddft->lambdakpt[k][i]=w[i];	    
    }

  /*
   * copy the eigenvectors back to Hsub
   */
  for(i=0;i<Nstates*Nstates;i++)
    {
      arrReal[i] = arrHsub[i].real;
      arrImag[i] = arrHsub[i].imag;       
    }
  MatDenseRestoreArray(*Hsub1,&arrReal);
  MatDenseRestoreArray(*Hsub2,&arrImag);

  /*
   * free memory
   */
  PetscFree(arrHsub);
  PetscFree(arrMsub);
  PetscFree(w);

  return ierr;

}
