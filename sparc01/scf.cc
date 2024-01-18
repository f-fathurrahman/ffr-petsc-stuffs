/*=============================================================================================
  | Description: This file contains the functions required for self consistent field iteration
  | and subspace projection
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "petscsys.h"
#include <cmath>

#include <cblas.h>
#include <lapacke.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

///////////////////////////////////////////////////////////////////////////////////////////////
//                        SelfConsistentField: Self Consistent Field iteration               // 
///////////////////////////////////////////////////////////////////////////////////////////////
using namespace std;
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
        
      MPI_Comm_rank(PETSC_COMM_WORLD,&rank);     

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
 
