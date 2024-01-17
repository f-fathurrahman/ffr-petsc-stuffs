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
  | file name: initObjs.cc          
  |
  | Description: This file contains the functions required for initializing variables
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana, Deepa Phanish
  |
  | Last Modified: 2/25/2016   
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "petscsys.h"
#include <petsctime.h>
//#include "mkl_lapacke.h"
//#include "mkl.h"

//#include <iostream>
//using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
///////////////////////////////////////////////////////////////////////////////////////////////
//                       SddftObjInitialize: reads the input filename                        //
///////////////////////////////////////////////////////////////////////////////////////////////  
void SddftObjInitialize(SDDFT_OBJ* pSddft)
{
  PetscOptionsGetString(PETSC_NULL,"-name",pSddft->file,sizeof(pSddft->file),PETSC_NULL);
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                         Objects_Create: Creates the PETSc objects                         //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode Objects_Create(SDDFT_OBJ* pSddft)
{
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscInt o = pSddft->order; 
  int MAX_ITS_ANDERSON = pSddft->MixingHistory;
  PetscInt xcor,ycor,zcor,lxdim,lydim,lzdim;
  int i;
  Mat A;
  PetscMPIInt comm_size;
  MPI_Comm_size(PETSC_COMM_WORLD,&comm_size);

  if(pSddft->BC==1)
    { 
      /*
       * Non periodic boundary condition
       */
      if(comm_size==1)
	{
	  DMDACreate3d(PETSC_COMM_SELF,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_STAR,n_x,n_y,n_z,
		       PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,o,0,0,0,&pSddft->da);

	  DMDACreate3d(PETSC_COMM_SELF,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_STAR,n_x,n_y,n_z,
		       PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,o,0,0,0,&pSddft->da_grad);
	}
      else{
	DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_STAR,n_x,n_y,n_z,
		     PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,o,0,0,0,&pSddft->da);

	DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_STAR,n_x,n_y,n_z,
		     PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,o,0,0,0,&pSddft->da_grad);

      }

    }else if(pSddft->BC==2)
    { 
      /*
       * Periodic boundary condition
       */      
      if(comm_size==1)
	{
	  DMDACreate3d(PETSC_COMM_SELF,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,n_x,n_y,n_z,
		       PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,o,0,0,0,&pSddft->da);

	  DMDACreate3d(PETSC_COMM_SELF,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,n_x,n_y,n_z,
		       PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,o,0,0,0,&pSddft->da_grad);
	}else{

	DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,n_x,n_y,n_z,
		     PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,o,0,0,0,&pSddft->da);

	DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,n_x,n_y,n_z,
		     PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,o,0,0,0,&pSddft->da_grad);
      } 
      MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&pSddft->nullspace);
    }

  DMCreateGlobalVector(pSddft->da,&pSddft->elecDensRho);  
  VecDuplicate(pSddft->elecDensRho,&pSddft->SuperposAtRho);  
  VecDuplicate(pSddft->elecDensRho,&pSddft->chrgDensB);  
  VecDuplicate(pSddft->elecDensRho,&pSddft->chrgDensB_TM);
  VecDuplicate(pSddft->elecDensRho,&pSddft->potentialPhi); 
  VecDuplicate(pSddft->elecDensRho,&pSddft->Phi_c);
  VecDuplicate(pSddft->elecDensRho,&pSddft->twopiRhoPB); 
  VecDuplicate(pSddft->elecDensRho,&pSddft->twopiBTMmBPS);
  VecDuplicate(pSddft->elecDensRho,&pSddft->Veff); 
  VecDuplicate(pSddft->elecDensRho,&pSddft->bjVj);  
  VecDuplicate(pSddft->elecDensRho,&pSddft->bjVj_TM);  
  VecDuplicate(pSddft->elecDensRho,&pSddft->Vxc);    
  VecDuplicate(pSddft->elecDensRho,&pSddft->tempVec);
  VecDuplicate(pSddft->elecDensRho,&pSddft->PoissonRHSAdd);

  VecDuplicate(pSddft->elecDensRho,&pSddft->xkprev);
  VecDuplicate(pSddft->elecDensRho,&pSddft->xk);
  VecDuplicate(pSddft->elecDensRho,&pSddft->fkprev);

  PetscMalloc(sizeof(Vec)*(MAX_ITS_ANDERSON),&pSddft->Xk);
  PetscMalloc(sizeof(Vec)*(MAX_ITS_ANDERSON),&pSddft->Fk);
  PetscMalloc(sizeof(Vec)*(MAX_ITS_ANDERSON),&pSddft->XpbF);

  for(i=0; i<MAX_ITS_ANDERSON; i++)
    {
      VecDuplicate(pSddft->elecDensRho,&pSddft->Xk[i]);
      VecDuplicate(pSddft->elecDensRho,&pSddft->Fk[i]);
      VecDuplicate(pSddft->elecDensRho,&pSddft->XpbF[i]);    
    }
 
 
  if(comm_size == 1 ) 
    {
      DMCreateMatrix(pSddft->da,&pSddft->laplaceOpr);
      DMSetMatType(pSddft->da,MATSEQSBAIJ);
    }
  else 
    {
      DMCreateMatrix(pSddft->da,&pSddft->laplaceOpr);
      DMSetMatType(pSddft->da,MATMPISBAIJ);
    }
  A=pSddft->laplaceOpr;
  KSPCreate(PETSC_COMM_WORLD,&pSddft->ksp);  
  KSPSetType(pSddft->ksp, KSP_TYPE);
  KSPSetOperators(pSddft->ksp,A,A);
  KSPSetTolerances(pSddft->ksp,pSddft->KSPTOL,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
  KSPSetFromOptions(pSddft->ksp);

  /*
   * Creating gradient operators
   */
  if(comm_size == 1 ) 
    {
      DMCreateMatrix(pSddft->da_grad,&pSddft->gradient_x);
      DMCreateMatrix(pSddft->da_grad,&pSddft->gradient_y);
      DMCreateMatrix(pSddft->da_grad,&pSddft->gradient_z);
      DMSetMatType(pSddft->da_grad,MATSEQBAIJ);
    }
  else 
    {
      DMCreateMatrix(pSddft->da_grad,&pSddft->gradient_x);
      DMCreateMatrix(pSddft->da_grad,&pSddft->gradient_y);
      DMCreateMatrix(pSddft->da_grad,&pSddft->gradient_z); 
      DMSetMatType(pSddft->da_grad,MATMPIBAIJ);
    }  
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
  PetscMalloc(sizeof(PetscInt)*(lzdim*lydim*lxdim),&pSddft->nnzDArray);
  PetscMalloc(sizeof(PetscInt)*(lzdim*lydim*lxdim),&pSddft->nnzODArray);
    
  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                         Laplace_matInit: Initializes the -(1/2)*Laplacian operator        //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode Laplace_matInit(SDDFT_OBJ* pSddft)
{
  PetscInt i,j,k,l,colidx,gxdim,gydim,gzdim,xcor,ycor,zcor,lxdim,lydim,lzdim;
  MatStencil row;
  MatStencil* col;
  PetscScalar* val;
  Mat A = pSddft->laplaceOpr;
  PetscInt o = pSddft->order;
#if _DEBUG
  PetscTruth flg;
#endif
  
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
  
  MatScale(pSddft->laplaceOpr,0.0);

  PetscMalloc(sizeof(MatStencil)*(o*6+1),&col);
  PetscMalloc(sizeof(PetscScalar)*(o*6+1),&val);

  for(k=zcor; k<zcor+lzdim; k++)
    for(j=ycor; j<ycor+lydim; j++)
      for(i=xcor; i<xcor+lxdim; i++)
	{
	  row.k = k; row.j = j, row.i = i;   
	  colidx=0; 
	  col[colidx].i=i ;col[colidx].j=j ;col[colidx].k=k ;
	  val[colidx++]=pSddft->coeffs[0]  ;
	  for(l=1;l<=o;l++)
	    {
	      col[colidx].i=i ;col[colidx].j=j ;col[colidx].k=k-l ;
	      val[colidx++]=pSddft->coeffs[l];
	      col[colidx].i=i ;col[colidx].j=j ;col[colidx].k=k+l ;
	      val[colidx++]=pSddft->coeffs[l];
	      col[colidx].i=i ;col[colidx].j=j-l ;col[colidx].k=k ;
	      val[colidx++]=pSddft->coeffs[l];
	      col[colidx].i=i ;col[colidx].j=j+l ;col[colidx].k=k ;
	      val[colidx++]=pSddft->coeffs[l];
	      col[colidx].i=i-l ;col[colidx].j=j ;col[colidx].k=k ;
	      val[colidx++]=pSddft->coeffs[l];
	      col[colidx].i=i+l ;col[colidx].j=j ;col[colidx].k=k ;
	      val[colidx++]=pSddft->coeffs[l];
	    }    
	  MatSetValuesStencil(A,1,&row,6*o+1,col,val,ADD_VALUES);
	}
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
   
#if _DEBUG
  MatIsSymmetric(A, 1.e-15, &flg);
  assert(flg);
#endif  
  
#ifdef _DEBUG
  MatView(A, PETSC_VIEWER_STDOUT_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"Is symmetric: %d\n",flg);  
  PetscPrintf(PETSC_COMM_WORLD,"Istart: %d\n",Istart);
  PetscPrintf(PETSC_COMM_WORLD,"Iend: %d\n",Iend);
#endif 
 
  return 0;  
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                         Gradient_matInit: Initializes the Gradient operators              //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode Gradient_matInit(SDDFT_OBJ* pSddft)
{
  
  PetscInt i,j,k,l,colidx,gxdim,gydim,gzdim,xcor,ycor,zcor,lxdim,lydim,lzdim;
  MatStencil row;
  MatStencil* col_x;
  MatStencil* col_y;
  MatStencil* col_z;
  PetscScalar* val;
  PetscInt o = pSddft->order;

  DMDAGetInfo(pSddft->da_grad,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da_grad,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);

  MatScale(pSddft->gradient_x,0.0);
  MatScale(pSddft->gradient_y,0.0);
  MatScale(pSddft->gradient_z,0.0);
  
  PetscMalloc(sizeof(MatStencil)*(o*2+1),&col_x);
  PetscMalloc(sizeof(MatStencil)*(o*2+1),&col_y);
  PetscMalloc(sizeof(MatStencil)*(o*2+1),&col_z);
  PetscMalloc(sizeof(PetscScalar)*(o*2+1),&val);

  
  for(k=zcor; k<zcor+lzdim; k++)
    for(j=ycor; j<ycor+lydim; j++)
      for(i=xcor; i<xcor+lxdim; i++)
	{
	  row.k = k; row.j = j, row.i = i;    
	  colidx=0; 
    
	  col_x[colidx].i=i ;col_x[colidx].j=j ;col_x[colidx].k=k ;
	  col_y[colidx].i=i ;col_y[colidx].j=j ;col_y[colidx].k=k ;
	  col_z[colidx].i=i ;col_z[colidx].j=j ;col_z[colidx].k=k ;
	  val[colidx++]=pSddft->coeffs_grad[0];
    
	  for(l=1;l<=o;l++)
	    {
	      col_x[colidx].i=i-l; col_x[colidx].j=j;col_x[colidx].k=k;
	      col_y[colidx].i=i; col_y[colidx].j=j-l;col_y[colidx].k=k;
	      col_z[colidx].i=i; col_z[colidx].j=j; col_z[colidx].k=k-l;
	      val[colidx++]=-pSddft->coeffs_grad[l];

	      col_x[colidx].i=i+l; col_x[colidx].j=j;col_x[colidx].k=k;
	      col_y[colidx].i=i; col_y[colidx].j=j+l;col_y[colidx].k=k;
	      col_z[colidx].i=i; col_z[colidx].j=j; col_z[colidx].k=k+l;
	      val[colidx++]=pSddft->coeffs_grad[l];	
	    }
   
	  MatSetValuesStencil(pSddft->gradient_x,1,&row,2*o+1,col_x,val,ADD_VALUES);
	  MatSetValuesStencil(pSddft->gradient_y,1,&row,2*o+1,col_y,val,ADD_VALUES);
	  MatSetValuesStencil(pSddft->gradient_z,1,&row,2*o+1,col_z,val,ADD_VALUES);
	}


  MatAssemblyBegin(pSddft->gradient_x, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pSddft->gradient_x, MAT_FINAL_ASSEMBLY);

  MatAssemblyBegin(pSddft->gradient_y, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pSddft->gradient_y, MAT_FINAL_ASSEMBLY);

  MatAssemblyBegin(pSddft->gradient_z, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pSddft->gradient_z, MAT_FINAL_ASSEMBLY);

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//                   ChargDensB_cutoff: calculate pseudocharge density cutoff                //
///////////////////////////////////////////////////////////////////////////////////////////////
void ChargDensB_cutoff(SDDFT_OBJ* pSddft)
{ 
  PetscScalar delta = pSddft->delta;
  PetscScalar ***pVpsArray, ***pBJArray;
  PetscReal rmax, MaxRadius=15.0;
  PetscScalar tableVps[MAX_TABLE_SIZE], *YD=NULL;
  PetscScalar tableR[MAX_TABLE_SIZE];
  PetscScalar x, y, z, coeffs[MAX_ORDER+1], Dtemp, Bint, val;
  PetscReal error;
  PetscReal Rcut;
  PetscReal r;
  PetscInt i,j,k,at,p,a,lloc;
  PetscInt Npts = ceil(MaxRadius/delta), Ncube;  
  PetscInt o = pSddft->order;
  int count;
  for(p=0;p<=o;p++)
    {
      coeffs[p] = pSddft->coeffs[p]/(2*M_PI);
    }  

  for(at=0;at<pSddft->Ntype;at++)
    { 
      /*
       * initial cutoff radius
       */
      lloc = pSddft->localPsd[at];
      if(lloc==0)
      	Rcut= pSddft->psd[at].rc_s;
      if(lloc==1)
      	Rcut= pSddft->psd[at].rc_p;
      if(lloc==2)
      	Rcut= pSddft->psd[at].rc_d;
      if(lloc==3)
	Rcut= pSddft->psd[at].rc_f;
	   
      tableR[0]=0.0; tableVps[0]=pSddft->psd[at].Vloc[0]; 
      count=1;
      do{
	tableR[count] = pSddft->psd[at].RadialGrid[count-1];
	tableVps[count] = pSddft->psd[at].Vloc[count-1]; 
	count++;

      }while( PetscRealPart(tableR[count-1]) <= 15.0);
      rmax = tableR[count-1];

      /*
       * derivatives of the spline fit to the pseudopotential
       */
      PetscMalloc(sizeof(PetscScalar)*count,&YD);
      getYD_gen(tableR,tableVps,YD,count); 
   
      PetscMalloc(sizeof(PetscScalar**)*(Npts+2*o),&pVpsArray);	       
      if(pVpsArray == NULL)
	{
	  printf("Memory allocation fail\n");
	  exit(1);
	}
      for(i=0;i<(Npts+2*o);i++)
	{
	  PetscMalloc(sizeof(PetscScalar*)*(Npts+2*o),&pVpsArray[i]);
	 
	  if(pVpsArray[i] == NULL)
	    {
	      printf("Memory allocation fail\n");
	      exit(1);
	    }
	  for(j=0;j<(Npts+2*o);j++)
	    {
	      PetscMalloc(sizeof(PetscScalar)*(Npts+2*o),&pVpsArray[i][j]);
	      if(pVpsArray[i][j] == NULL)
		{
		  //cout<<"Memory allocation fail";
		  exit(1);
		} 
	    }      
	}
	   
      /*
       * we assume that the atom is situated on the origin and go over the grid
       */
      PetscMalloc(sizeof(PetscScalar**)*(Npts),&pBJArray);	       
      if(pBJArray == NULL)
	{
	  //cout<<"Memory allocation fail";
	  exit(1);
	}
      for(i = 0;i<(Npts);i++)
	{
	  PetscMalloc(sizeof(PetscScalar*)*(Npts),&pBJArray[i]);	 
	  if(pBJArray[i] == NULL)
	    {
	      //cout<<"Memory allocation fail";
	      exit(1);
	    }
	  for(j=0;j<(Npts);j++)
	    {
	      PetscMalloc(sizeof(PetscScalar)*(Npts),&pBJArray[i][j]);
	      if(pBJArray[i][j] == NULL)
		{
		  //cout<<"Memory allocation fail";
		  exit(1);
		} 
	    }      
	}
	   	   
      /*
       * calculate pseudocharge density from the pseudopotential
       */
      for(k=0;k<(Npts+2*o);k++)
	for(j=0;j<(Npts+2*o);j++)
	  for(i=0;i<(Npts+2*o);i++)
	    {	
	      x=(i-o)*delta;
	      y=(j-o)*delta;
	      z=(k-o)*delta;
	      r=sqrt(x*x+y*y+z*z);
		       
	      if(r == tableR[0])
		{
		  pVpsArray[k][j][i] = tableVps[0];
		}
	      else if(r > rmax)
		{
		  pVpsArray[k][j][i] = -pSddft->noe[at]/r;
		}
	      else
		{
		  ispline_gen(tableR,tableVps,count,&r,&pVpsArray[k][j][i],&Dtemp,1,YD);
		}
	    }
	   
      /*
       * calculate pseudocharge density from the pseudopotential
       */
      for(k=o;k<Npts+o;k++)
	for(j=o;j<Npts+o;j++)
	  for(i=o;i<Npts+o;i++)
	    {
	      pBJArray[k-o][j-o][i-o]=0.0;
	      pBJArray[k-o][j-o][i-o]=pVpsArray[k][j][i]*coeffs[0];
	      for(a=1;a<=o;a++)
		{
		  pBJArray[k-o][j-o][i-o]+=(pVpsArray[k][j][i-a] + pVpsArray[k][j][i+a] + pVpsArray[k][j-a][i] +        
					    pVpsArray[k][j+a][i] + pVpsArray[k-a][j][i] + pVpsArray[k+a][j][i])*coeffs[a];		  
		}
	    }
      /*
       * now go over different radii by incrementing radius by 1 bohr, unless relative
       * error in charge is below tolerence
       */
      do{
	Ncube=ceil(Rcut/delta); // initial number to start with
	Bint=0.0;
	for(k=0;k<Ncube;k++)
	  for(j=0;j<Ncube;j++)
	    for(i=0;i<Ncube;i++)
	      {
		val=pBJArray[k][j][i];
		if(k==0)
		  val=0.5*val;
		if(k==Ncube-1)
		  val=0.5*val;
		if(j==0)
		  val=0.5*val;
		if(j==Ncube-1)
		  val=0.5*val;
		if(i==0)
		  val=0.5*val;
		if(i==Ncube-1)
		  val=0.5*val;
		    
		Bint+= val;
	      }
	Bint=delta*delta*delta*Bint*8.0; 
	error = fabs(Bint+pSddft->noe[at])/fabs(pSddft->noe[at]); 
	Rcut=Rcut+1.0;
      }while(Rcut<=15.0 && error>pSddft->PseudochargeRadiusTOL);
              
      /*
       * store cutoff radius
       */
      pSddft->CUTOFF[at]=Ncube*delta;
      PetscPrintf(PETSC_COMM_WORLD,"atom type=%d, error in nuclear charge density = %0.16lf,\n            pseudocharge radius: %lf (Bohr)\n",at+1,error,pSddft->CUTOFF[at]);
	     
      for(i=0;i<Npts+2*o;i++)
	{
	  for(j=0;j<Npts+2*o;j++)
	    {
	      PetscFree(pVpsArray[i][j]);
	    }
	  PetscFree(pVpsArray[i]);
	}
      PetscFree(pVpsArray);

      for(i=0;i<Npts;i++)
	{
	  for(j=0;j<Npts;j++)
	    {
	      PetscFree(pBJArray[i][j]);
	    }
	  PetscFree(pBJArray[i]);
	}
      PetscFree(pBJArray);
      PetscFree(YD);     

    } 
  return;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//             Wavefunctions_MatInit: Initialize matrices for storing wavefunctions          //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode Wavefunctions_MatInit(SDDFT_OBJ* pSddft)
{

  PetscInt Nstates = pSddft->Nstates;
  PetscInt xm,ym,zm;
  //PetscInt start,end;
 
  PetscErrorCode ierr;
  //PetscInt Npts = pSddft->numPoints_x * pSddft->numPoints_y * pSddft->numPoints_z;

  DMDAGetCorners(pSddft->da,0,0,0,&zm,&ym,&xm);
  MatCreate(PetscObjectComm((PetscObject)pSddft->da),&pSddft->XOrb);
      
  MatSetSizes(pSddft->XOrb,xm*ym*zm,PETSC_DETERMINE,PETSC_DETERMINE,Nstates);
  MatSetType(pSddft->XOrb,MATDENSE);
  MatSetFromOptions(pSddft->XOrb);      
  MatSetUp(pSddft->XOrb); 
  MatZeroEntries(pSddft->XOrb);
  MatAssemblyBegin(pSddft->XOrb,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pSddft->XOrb,MAT_FINAL_ASSEMBLY);
           
  PetscRandom rctx;
  
  PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
  PetscRandomSetType(rctx,PETSCRAND);
  MatSetRandom(pSddft->XOrb,rctx);
  PetscRandomDestroy(&rctx);
       
  MatAssemblyBegin(pSddft->XOrb,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pSddft->XOrb,MAT_FINAL_ASSEMBLY);
              
  ierr = MatDuplicate(pSddft->XOrb,MAT_SHARE_NONZERO_PATTERN,&pSddft->YOrb);
  CHKERRQ(ierr);
      
  ierr = MatDuplicate(pSddft->XOrb,MAT_SHARE_NONZERO_PATTERN,&pSddft->YOrbNew);
  CHKERRQ(ierr);
     
  PetscLogDouble t1,t2; //,elapsed_time;
  PetscTime(&t1);
     
  MatMatMultSymbolic(pSddft->HamiltonianOpr,pSddft->XOrb,PETSC_DEFAULT,&pSddft->YOrb);
  MatMatMultNumeric(pSddft->HamiltonianOpr,pSddft->XOrb,pSddft->YOrb);

  MatMatMultSymbolic(pSddft->HamiltonianOpr,pSddft->XOrb,PETSC_DEFAULT,&pSddft->YOrbNew);
  MatMatMultNumeric(pSddft->HamiltonianOpr,pSddft->XOrb,pSddft->YOrbNew);
 
  PetscTime(&t2);
  //elapsed_time = t2-t1;
       
  MatMatMultNumeric(pSddft->HamiltonianOpr,pSddft->XOrb,pSddft->YOrb);  
     
  return 0;  
}
///////////////////////////////////////////////////////////////////////////////////////////////
//       kPointWavefunctions_MatInit: Initialize matrices for storing k point wavefunctions  //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode kPointWavefunctions_MatInit(SDDFT_OBJ* pSddft)
{

  PetscInt Nstates = pSddft->Nstates;
  PetscInt Nkpts_sym=pSddft->Nkpts_sym;
  PetscInt xm,ym,zm;
  //PetscInt start,end;
  int k;

  PetscErrorCode ierr;
  //PetscInt Npts=pSddft->numPoints_x*pSddft->numPoints_y*pSddft->numPoints_z;
      
  DMDAGetCorners(pSddft->da,0,0,0,&zm,&ym,&xm);
          
  PetscMalloc(sizeof(Mat)*(pSddft->Nkpts_sym),&pSddft->XOrb1);     
  PetscMalloc(sizeof(Mat)*(pSddft->Nkpts_sym),&pSddft->XOrb2);      
  PetscMalloc(sizeof(Mat)*(pSddft->Nkpts_sym),&pSddft->YOrb1);      
  PetscMalloc(sizeof(Mat)*(pSddft->Nkpts_sym),&pSddft->YOrb2);
     
  MatCreate(PetscObjectComm((PetscObject)pSddft->da),&pSddft->XOrb1[0]);
     
  MatSetSizes(pSddft->XOrb1[0],xm*ym*zm,PETSC_DETERMINE,PETSC_DETERMINE,Nstates);
      
  MatSetType(pSddft->XOrb1[0],MATDENSE);

  MatSetFromOptions(pSddft->XOrb1[0]);

  MatSetUp(pSddft->XOrb1[0]); 
  MatZeroEntries(pSddft->XOrb1[0]);
  MatAssemblyBegin(pSddft->XOrb1[0],MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pSddft->XOrb1[0],MAT_FINAL_ASSEMBLY);

  ierr = MatDuplicate(pSddft->XOrb1[0],MAT_SHARE_NONZERO_PATTERN,&pSddft->XOrb2[0]);
  CHKERRQ(ierr);
      
  PetscRandom rctx;
  PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
  PetscRandomSetType(rctx,PETSCRAND);

  MatSetRandom(pSddft->XOrb1[0],rctx);
  MatSetRandom(pSddft->XOrb2[0],rctx);
      
  MatAssemblyBegin(pSddft->XOrb1[0],MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pSddft->XOrb1[0],MAT_FINAL_ASSEMBLY);

  MatAssemblyBegin(pSddft->XOrb2[0],MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pSddft->XOrb2[0],MAT_FINAL_ASSEMBLY);
      
  for(k=1;k<Nkpts_sym;k++)
    {      
      ierr = MatDuplicate(pSddft->XOrb1[0],MAT_SHARE_NONZERO_PATTERN,&pSddft->XOrb1[k]);
      CHKERRQ(ierr);       
      ierr = MatDuplicate(pSddft->XOrb2[0],MAT_SHARE_NONZERO_PATTERN,&pSddft->XOrb2[k]);
      CHKERRQ(ierr);

      MatSetRandom(pSddft->XOrb1[k],rctx);
      MatSetRandom(pSddft->XOrb2[k],rctx);
	
      MatAssemblyBegin(pSddft->XOrb1[k],MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(pSddft->XOrb1[k],MAT_FINAL_ASSEMBLY);

      MatAssemblyBegin(pSddft->XOrb2[k],MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(pSddft->XOrb2[k],MAT_FINAL_ASSEMBLY);       
     
    }

      
  PetscRandomDestroy(&rctx);       
       
  for(k=0;k<Nkpts_sym;k++)
    {
      ierr = MatDuplicate(pSddft->XOrb1[k],MAT_SHARE_NONZERO_PATTERN,&pSddft->YOrb1[k]);
      CHKERRQ(ierr);
      ierr = MatDuplicate(pSddft->XOrb2[k],MAT_SHARE_NONZERO_PATTERN,&pSddft->YOrb2[k]);
      CHKERRQ(ierr);       
    }
      
  ierr = MatDuplicate(pSddft->XOrb1[0],MAT_SHARE_NONZERO_PATTERN,&pSddft->YOrbNew1);
  CHKERRQ(ierr);
  ierr = MatDuplicate(pSddft->XOrb2[0],MAT_SHARE_NONZERO_PATTERN,&pSddft->YOrbNew2);
  CHKERRQ(ierr);

  ierr = MatDuplicate(pSddft->XOrb2[0],MAT_SHARE_NONZERO_PATTERN,&pSddft->ZOrb1);
  CHKERRQ(ierr);
  ierr = MatDuplicate(pSddft->XOrb1[0],MAT_SHARE_NONZERO_PATTERN,&pSddft->ZOrb2);
  CHKERRQ(ierr); 

  ierr = MatDuplicate(pSddft->XOrb2[0],MAT_SHARE_NONZERO_PATTERN,&pSddft->ZOrbNew1);
  CHKERRQ(ierr);
  ierr = MatDuplicate(pSddft->XOrb1[0],MAT_SHARE_NONZERO_PATTERN,&pSddft->ZOrbNew2);
  CHKERRQ(ierr);
           
  return 0;  
}
 
///////////////////////////////////////////////////////////////////////////////////////////////
//  Wavefunctions_MatMatMultSymbolic: Symbolic matrix matrix multiplication for the orbitals //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode Wavefunctions_MatMatMultSymbolic(SDDFT_OBJ* pSddft)
{

  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"Starting Destroying previous YOrbs \n");
  MatDestroy(&pSddft->YOrb);
  MatDestroy(&pSddft->YOrbNew);
 
  ierr = MatDuplicate(pSddft->XOrb,MAT_SHARE_NONZERO_PATTERN,&pSddft->YOrb);
  CHKERRQ(ierr);
 
  ierr = MatDuplicate(pSddft->XOrb,MAT_SHARE_NONZERO_PATTERN,&pSddft->YOrbNew);
  CHKERRQ(ierr);
 
  MatMatMultSymbolic(pSddft->HamiltonianOpr,pSddft->XOrb,PETSC_DEFAULT,&pSddft->YOrb);
  MatMatMultNumeric(pSddft->HamiltonianOpr,pSddft->XOrb,pSddft->YOrb);
	
  MatMatMultSymbolic(pSddft->HamiltonianOpr,pSddft->XOrb,PETSC_DEFAULT,&pSddft->YOrbNew);
  MatMatMultNumeric(pSddft->HamiltonianOpr,pSddft->XOrb,pSddft->YOrbNew);
  
  return 0;

}
///////////////////////////////////////////////////////////////////////////////////////////////
//                         kpointWeight: Weights associated with kpoints                     //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar kpointWeight(PetscScalar kx,PetscScalar ky,PetscScalar kz)
{
  /*
   * Appropriate weights for k-point sampling
   * we find the weight of the z-direction k point. If the kz has 0, then the weight is 1.0
   * else the weight is 2.0
   */
  PetscScalar w;
  if(fabs(kz)>0)
    {
      w=2.0;
    }
  else
    {
      w=1.0; 
    }
  return w;

}
///////////////////////////////////////////////////////////////////////////////////////////////
//         kpointWeight_Init: initialize weights associated with kpoints                     //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar kpointWeight_Init(SDDFT_OBJ* pSddft)
{
  /*
   * Initialize the weights of k points
   * by going over all the k points from the Monkhorst Pack grid
   */
  int k,nk1,nk2,nk3;
  PetscScalar k1,k2,k3;
  PetscScalar R_x=pSddft->range_x;
  PetscScalar R_y=pSddft->range_y;
  PetscScalar R_z=pSddft->range_z;
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
	  pSddft->kptWts[k]= kpointWeight(k1,k2,k3);
	  k++;
	}
  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                               Objects_Destroy: Destroys PETSc objects                     //
///////////////////////////////////////////////////////////////////////////////////////////////
void Objects_Destroy(SDDFT_OBJ* pSddft)
{
  
  int at,i;
  int MAX_ITS_ANDERSON = pSddft->MixingHistory;

  /*
   * destroy vectors
   */
  VecDestroy(&pSddft->elecDensRho); 
  VecDestroy(&pSddft->SuperposAtRho); 
  VecDestroy(&pSddft->chrgDensB);
  VecDestroy(&pSddft->chrgDensB_TM);
  VecDestroy(&pSddft->potentialPhi);    
  VecDestroy(&pSddft->twopiRhoPB);
  VecDestroy(&pSddft->bjVj);  
  VecDestroy(&pSddft->bjVj_TM); 
  VecDestroy(&pSddft->Vxc);
  VecDestroy(&pSddft->forces);  
  VecDestroy(&pSddft->Atompos);    
  VecDestroy(&pSddft->mvAtmConstraint);    
  VecDestroy(&pSddft->Veff);
  VecDestroy(&pSddft->Phi_c);
  VecDestroy(&pSddft->twopiBTMmBPS);
  VecDestroy(&pSddft->PoissonRHSAdd);
  VecDestroy(&pSddft->tempVec);
  VecDestroy(&pSddft->xkprev);
  VecDestroy(&pSddft->xk);
  VecDestroy(&pSddft->fkprev);
  
  for(i=0; i<MAX_ITS_ANDERSON; i++)
    {
      VecDestroy(&pSddft->Xk[i]);
      VecDestroy(&pSddft->Fk[i]);
      VecDestroy(&pSddft->XpbF[i]);    
    }

  /*
   * destroy matrices
   */
  MatDestroy(&pSddft->laplaceOpr);
  MatDestroy(&pSddft->gradient_x);
  MatDestroy(&pSddft->gradient_y);
  MatDestroy(&pSddft->gradient_z);
  MatDestroy(&pSddft->XOrb);
  MatDestroy(&pSddft->YOrb);
  MatDestroy(&pSddft->YOrbNew);

  /*
   * destroy ksp
   */
  KSPDestroy(&pSddft->ksp);

  /*
   * free memory occupied by pseudopotentials
   */
  for(at=0;at<pSddft->Ntype;at++)
    {
      free(pSddft->psd[at].Vloc);
      free(pSddft->psd[at].Vs);
      free(pSddft->psd[at].Vp);
      free(pSddft->psd[at].Vd);
      free(pSddft->psd[at].Vf);
      free(pSddft->psd[at].Us);
      free(pSddft->psd[at].Up);
      free(pSddft->psd[at].Ud);
      free(pSddft->psd[at].Uf);
      free(pSddft->psd[at].uu);
      free(pSddft->psd[at].RadialGrid);
    }
  free(pSddft->psd);

  /*
   * free memory occupied by arrays
   */
  PetscFree(pSddft->nnzDArray);
  PetscFree(pSddft->nnzODArray);
  PetscFree(pSddft->pForces_corr);
  PetscFree(pSddft->lambda);
  PetscFree(pSddft->CUTOFF);
  PetscFree(pSddft->startPos);
  PetscFree(pSddft->endPos);
  PetscFree(pSddft->localPsd);
  
  return;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                        Set_VecZero: Initializes vectors with zero entries                 //
///////////////////////////////////////////////////////////////////////////////////////////////
void Set_VecZero(SDDFT_OBJ* pSddft)
{
   
  VecZeroEntries(pSddft->chrgDensB);
  VecZeroEntries(pSddft->chrgDensB_TM);
  VecZeroEntries(pSddft->potentialPhi); 
  VecZeroEntries(pSddft->Phi_c);
  VecZeroEntries(pSddft->twopiRhoPB);
  VecZeroEntries(pSddft->twopiBTMmBPS);
  VecZeroEntries(pSddft->bjVj);
  VecZeroEntries(pSddft->bjVj_TM);
  VecZeroEntries(pSddft->Veff);
  VecZeroEntries(pSddft->Vxc);
  VecZeroEntries(pSddft->PoissonRHSAdd);
  VecZeroEntries(pSddft->tempVec);
  
  return;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//                         kPointObjects_Destroy: Destroys PETSc objects                     //
///////////////////////////////////////////////////////////////////////////////////////////////
void kPointObjects_Destroy(SDDFT_OBJ* pSddft)
{
  
  int at,i;
  int MAX_ITS_ANDERSON = pSddft->MixingHistory;
  int Nkpts=pSddft->Nkpts_sym;

  /*
   * destroy vectors
   */
  VecDestroy(&pSddft->elecDensRho); 
  VecDestroy(&pSddft->SuperposAtRho); 
  VecDestroy(&pSddft->chrgDensB);
  VecDestroy(&pSddft->chrgDensB_TM);
  VecDestroy(&pSddft->potentialPhi);    
  VecDestroy(&pSddft->twopiRhoPB);
  VecDestroy(&pSddft->bjVj);  
  VecDestroy(&pSddft->bjVj_TM); 
  VecDestroy(&pSddft->Vxc);
  VecDestroy(&pSddft->forces);  
  VecDestroy(&pSddft->Atompos);    
  VecDestroy(&pSddft->mvAtmConstraint);    
  VecDestroy(&pSddft->Veff);
  VecDestroy(&pSddft->Phi_c);
  VecDestroy(&pSddft->twopiBTMmBPS);
  VecDestroy(&pSddft->PoissonRHSAdd);
  VecDestroy(&pSddft->tempVec);
  VecDestroy(&pSddft->xkprev);
  VecDestroy(&pSddft->xk);
  VecDestroy(&pSddft->fkprev);
  
  for(i=0; i<MAX_ITS_ANDERSON; i++)
    {
      VecDestroy(&pSddft->Xk[i]);
      VecDestroy(&pSddft->Fk[i]);
      VecDestroy(&pSddft->XpbF[i]);    
    }

  /*
   * destroy matrices
   */
  MatDestroy(&pSddft->laplaceOpr);
  MatDestroy(&pSddft->gradient_x);
  MatDestroy(&pSddft->gradient_y);
  MatDestroy(&pSddft->gradient_z);

  MatDestroy(&pSddft->YOrbNew1);
  MatDestroy(&pSddft->YOrbNew2);
  MatDestroy(&pSddft->ZOrb1);
  MatDestroy(&pSddft->ZOrb2);
  MatDestroy(&pSddft->ZOrbNew1);
  MatDestroy(&pSddft->ZOrbNew2);

  for(i=0;i<Nkpts;i++)
    {
      MatDestroy(&pSddft->XOrb1[i]);
      MatDestroy(&pSddft->XOrb2[i]);
      MatDestroy(&pSddft->YOrb1[i]);
      MatDestroy(&pSddft->YOrb2[i]);
    }

  /*
   * destroy ksp
   */
  KSPDestroy(&pSddft->ksp);

  /*
   * free memory occupied by pseudopotentials
   */
  for(at=0;at<pSddft->Ntype;at++)
    {
      free(pSddft->psd[at].Vloc);
      free(pSddft->psd[at].Vs);
      free(pSddft->psd[at].Vp);
      free(pSddft->psd[at].Vd);
      free(pSddft->psd[at].Vf);
      free(pSddft->psd[at].Us);
      free(pSddft->psd[at].Up);
      free(pSddft->psd[at].Ud);
      free(pSddft->psd[at].Uf);
      free(pSddft->psd[at].uu);
      free(pSddft->psd[at].RadialGrid);
    }
  free(pSddft->psd);

  for(i=0;i<Nkpts;i++)
    {
      PetscFree(pSddft->lambdakpt[i]);      
    }
  PetscFree(pSddft->lambdakpt);      
  PetscFree(pSddft->kptWts);

  /*
   * free memory occupied by arrays
   */
  PetscFree(pSddft->nnzDArray);
  PetscFree(pSddft->nnzODArray);
  PetscFree(pSddft->pForces_corr);
  PetscFree(pSddft->CUTOFF);
  PetscFree(pSddft->startPos);
  PetscFree(pSddft->endPos);
  PetscFree(pSddft->localPsd);
  
  return;
}
