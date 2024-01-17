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
  | file name: mixing.cc          
  |
  | Description: This file contains the function required for potential mixing for the 
  | accleration of SCF iterations
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana, Deepa Phanish
  |
  | Last Modified: 1/26/2016   
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "isddft.h"
#include <lapacke.h>

//#include "mkl_lapacke.h"
//#include "mkl.h"


///////////////////////////////////////////////////////////////////////////////////////////////
//                                       mix: Anderson mixing                                //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar mix(SDDFT_OBJ* pSddft, PetscInt its, Vec fk)
{
  static PetscInt icnt=-1; 
  PetscScalar PREVITWT=1.0-pSddft->MixingParameter;
  int MAX_ITS_ANDERSON=pSddft->MixingHistory;

  PetscInt j,*idx;
  PetscMalloc(sizeof(int)*(MAX_ITS_ANDERSON),&idx);

  PetscScalar alpha=-1.0,beta=1-(!(!its)*PREVITWT);
  
  double *array,*arrMat,*arrVec;
  PetscScalar *val,*val1,norm,norm1;
  
  Mat FktFk;
  Vec Fktfk;
  Vec temp;
  Vec xk=pSddft->xk; 
  int rank;
  int SYSTEM_SIZE=1;   
  
  if(its>=MAX_ITS_ANDERSON) 
    SYSTEM_SIZE=MAX_ITS_ANDERSON;
  else
    SYSTEM_SIZE=its-1;
 
  PetscMalloc(SYSTEM_SIZE*SYSTEM_SIZE*sizeof(PetscScalar),&val);
  PetscMalloc(SYSTEM_SIZE*sizeof(PetscScalar),&val1);
  
  MatCreateSeqDense(PETSC_COMM_SELF,SYSTEM_SIZE,SYSTEM_SIZE,NULL,&FktFk);

  VecCreate(PETSC_COMM_SELF,&Fktfk);
  VecSetSizes(Fktfk,PETSC_DECIDE,SYSTEM_SIZE);
  VecSetFromOptions(Fktfk);
  VecDuplicate(Fktfk,&temp);

  VecAXPY(fk,alpha,xk);
  if(icnt >= 0)
    {
      VecWAXPY(pSddft->Xk[icnt],alpha,pSddft->xkprev,xk); 
      VecWAXPY(pSddft->Fk[icnt],alpha,pSddft->fkprev,fk);
      VecWAXPY(pSddft->XpbF[icnt],beta,pSddft->Fk[icnt],pSddft->Xk[icnt]);
    }
  
  VecCopy(xk,pSddft->xkprev);
  VecCopy(fk,pSddft->fkprev);    
  VecAXPY(xk,beta,fk); 
      
  if(its > 1)
    {
      for(j=0;j<SYSTEM_SIZE;j++)
	{        
	  VecMTDot(pSddft->Fk[j],SYSTEM_SIZE,pSddft->Fk,&val[j*SYSTEM_SIZE]);
	  idx[j]=j;
	  VecDot(pSddft->Fk[j],fk,&val1[j]);
	}
  
      MatSetValues(FktFk,SYSTEM_SIZE,idx,SYSTEM_SIZE,idx,val,INSERT_VALUES);
      MatAssemblyBegin(FktFk,MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(FktFk,MAT_FINAL_ASSEMBLY);
      VecSetValues(Fktfk,SYSTEM_SIZE,idx,val1,INSERT_VALUES); 
      VecAssemblyBegin(Fktfk);
      VecAssemblyEnd(Fktfk);       
      VecGetArray(temp,&array);
      VecGetArray(Fktfk,&arrVec);
      MatDenseGetArray(FktFk,&arrMat);
    
      LAPACKE_dgelsd(LAPACK_COL_MAJOR,SYSTEM_SIZE,SYSTEM_SIZE,1,arrMat,SYSTEM_SIZE,arrVec,MAX_ITS_ANDERSON,array,-1.0,&rank);
    
      for(j=0;j<SYSTEM_SIZE;j++)
	{     
	  VecAXPY(xk,-arrVec[j],pSddft->XpbF[j]);	 
	} 
      VecRestoreArray(temp,&array);
      VecRestoreArray(Fktfk,&arrVec);
      MatDenseRestoreArray(FktFk,&arrMat);
    }
  VecCopy(xk,fk); 
  
  VecNorm(pSddft->xkprev, NORM_2, &norm1);
  VecWAXPY(pSddft->tempVec,alpha,pSddft->xkprev,xk);
  VecNorm(pSddft->tempVec, NORM_2, &norm);
  
  icnt++;
  if(icnt==MAX_ITS_ANDERSON)
    icnt=0;
  MatDestroy(&FktFk);  
  VecDestroy(&Fktfk);
  VecDestroy(&temp);  
  return (norm/(norm1+1e-16));
   
}


