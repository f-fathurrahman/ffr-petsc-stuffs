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
  | file name: multipole.cc          
  |
  | Description: This file contains the functions required for calculation of the correction 
  | term using multipole expansion
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana
  |
  | Last Modified: 1/26/2016   
  |-------------------------------------------------------------------------------------------*/

#include "sddft.h"
#include "petscsys.h"
#include "math.h"
#include<iostream>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
///////////////////////////////////////////////////////////////////////////////////////////////
//     MultipoleExpansion_Phi: multipole expansion to find the correct boundary condition    //
//                                for the electrostatic potential                            //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode MultipoleExpansion_Phi(SDDFT_OBJ* pSddft,Vec* RhopBvec)
{
  PetscScalar **Q1,**Q2; // multipole moments
  PetscScalar ***pRhopBArray;
  PetscScalar ***pBArray;
  PetscScalar ***pRHSVec;
  PetscScalar ***pPhiArray=NULL;
  PetscScalar Phi_l;
  PetscErrorCode ierr;

  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0;
  PetscInt XS[6],XL[6],YS[6],YL[6],ZS[6],ZL[6];
  PetscInt Xstart,Ystart,Zstart,Xend,Yend,Zend,nXPhi,nYPhi,nZPhi;
  PetscScalar coeffs[MAX_ORDER+1];

  int flag=0;
  int i,j,k,I,J,K,poscnt;
  PetscScalar SpHarmoniclm,SpHarmoniclmm;
  PetscInt o = pSddft->order,a;
  PetscScalar delta=pSddft->delta;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;

  int l,lmax,m,p; lmax=6;
  
  PetscScalar Rcut=2.0*sqrt(R_x*R_x+ R_y*R_y + R_z*R_z)+10.0;
  PetscScalar x,y,z,r;
  for(p=0;p<=o;p++)
    {
      coeffs[p] = pSddft->coeffs[p]; // coefficients of -(1/2)*laplacian
    }  

  PetscMPIInt comm_size,rank;
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
 
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim); 
  
  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&Q1);
  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&Q2);
  for(l=0;l<=lmax;l++)
    {
      PetscMalloc(sizeof(PetscScalar)*(l+1),&Q1[l]);
      PetscMalloc(sizeof(PetscScalar)*(l+1),&Q2[l]);
    }

  for(l=0;l<=lmax;l++)
    for(m=0;m<(l+1);m++)
      {
        Q1[l][m]=0.0;
        Q2[l][m]=0.0;
      }
 
  DMDAVecGetArray(pSddft->da,*RhopBvec,&pRhopBArray);  
 
  /*
   *  each processor goes over its local nodes
   */
  for(k=zcor; k<zcor+lzdim; k++)
    for(j=ycor; j<ycor+lydim; j++)
      for(i=xcor; i<xcor+lxdim; i++)
	{
	  x = delta*i - R_x ;
	  y = delta*j - R_y ;
	  z = delta*k - R_z ;   
	  r = sqrt((x*x)+(y*y)+(z*z));	

	  /*
	   * calculate multipole moments
	   */
	  for(l=0;l<=lmax;l++)
	    for(m=0;m<(l+1);m++)
	      {
		SpHarmoniclm=SphericalHarmonic(x,y,z,l,m,Rcut);	    
		Q1[l][m] += (pRhopBArray[k][j][i]*pow(r,1.0*l)*SpHarmoniclm);	     
		if(m!=0)
		  {
		    SpHarmoniclmm=SphericalHarmonic(x,y,z,l,-m,Rcut);
		    Q2[l][m] += (pRhopBArray[k][j][i]*pow(r,1.0*l)*SpHarmoniclmm);
		  }			     
	      }
	}
  DMDAVecRestoreArray(pSddft->da,*RhopBvec,&pRhopBArray);  

  /*
   * sum contribution of multipole moments from all processes
   */
  for(l=0;l<=lmax;l++)
    {
      for(m=0;m<(l+1);m++)
	{
	  ierr =  MPI_Allreduce(MPI_IN_PLACE,&Q1[l][m],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
	  Q1[l][m] = delta*delta*delta*Q1[l][m];
	  Q2[l][0]=Q1[l][0]; 
	  if(m!=0)
	    {
	      ierr =  MPI_Allreduce(MPI_IN_PLACE,&Q2[l][m],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
	      Q2[l][m] = delta*delta*delta*Q2[l][m]; 	      
	    }	  
	}
     
    }

  /*
   * compute the charge correction
   */

  /*
   * define regions of the domain which contributes to the charge correction. i.e. 0 to 0+o and
   * nx-o nx in each directions.
   */
  XS[0]=0; XL[0]=0+o; YS[0]=0; YL[0]=n_y-1; ZS[0]=0; ZL[0]=n_z-1;
  XS[1]=n_x-o-1; XL[1]=n_x-1; YS[1]=0; YL[1]=n_y-1; ZS[1]=0; ZL[1]=n_z-1;  
  XS[2]=0; XL[2]=n_x-1; YS[2]=0; YL[2]=0+o; ZS[2]=0; ZL[2]=n_z-1;
  XS[3]=0; XL[3]=n_x-1; YS[3]=n_y-o-1; YL[3]=n_y-1; ZS[3]=0; ZL[3]=n_z-1;
  XS[4]=0; XL[4]=n_x-1; YS[4]=0; YL[4]=n_y-1; ZS[4]=0; ZL[4]=0+o;
  XS[5]=0; XL[5]=n_x-1; YS[5]=0; YL[5]=n_y-1; ZS[5]=n_z-o-1; ZL[5]=n_z-1;
 
  VecZeroEntries(pSddft->PoissonRHSAdd);
  DMDAVecGetArray(pSddft->da,pSddft->PoissonRHSAdd,&pRHSVec);  

  /*
   * loop over the defined regions
   */
  for(poscnt=0;poscnt<6;poscnt++)
    {

      xstart=-1; ystart=-1; zstart=-1; xend=-1; yend=-1; zend=-1; overlap=0;
      
      xs = XS[poscnt]; xl = XL[poscnt];
      ys = YS[poscnt]; yl = YL[poscnt];
      zs = ZS[poscnt]; zl = ZL[poscnt];

      /* 
       * find overlap of the region with processor domain
       */       
      if(xs >= xcor && xs <= xcor+lxdim-1)
	xstart = xs;
      else if(xcor >= xs && xcor <= xl)
	xstart = xcor;
	  
      if(xl>=xcor && xl<=xcor+lxdim-1)
	xend = xl;
      else if(xcor+lxdim-1>=xs && xcor+lxdim-1<=xl)
	xend  = xcor+lxdim-1;
	  
      if(ys >= ycor && ys <= ycor+lydim-1)
	ystart = ys;
      else if(ycor >= ys && ycor <= yl)
	ystart = ycor;
	
      if(yl>=ycor && yl<=ycor+lydim-1)
	yend = yl;
      else if(ycor+lydim-1>=ys && ycor+lydim-1<=yl)
	yend = ycor+lydim-1;
   
      if(zs >= zcor && zs <= zcor+lzdim-1)
	zstart = zs;
      else if(zcor >= zs && zcor <= zl)
	zstart = zcor;
	  
      if(zl>=zcor &&zl<=zcor+lzdim-1)
	zend  = zl;
      else if(zcor+lzdim-1>=zs && zcor+lzdim-1<=zl)
	zend = zcor+lzdim-1;
	     
      if((xstart!=-1)&&(xend!=-1)&&(ystart!=-1)&&(yend!=-1)&&(zstart!=-1)&&(zend!=-1))
	overlap =1;   
       
      if(overlap)
	{	  
	  /*
	   * if poscnt==0, then we need to calculate phi using multipole expansion at points 
	   * inside [xstart-o,-1,ystart,yend,zstart,zend]
	   * if poscnt==1, then we need to calculate phi using multipole expansion at points 
	   * inside [n_x,xend+o,ystart,yend,zstart,zend]
	   * if poscnt==2, then we need to calculate phi using multipole expansion at points 
	   * inside [xstart,xend,ystart-o,-1,zstart,zend]
	   * if poscnt==3, then we need to calculate phi using multipole expansion at points 
	   * inside [xstart,xend,n_y,yend+o,zstart,zend]
	   * if poscnt==4, then we need to calculate phi using multipole expansion at points
	   * inside [xstart,xend,ystart,yend,zstart-o,-1]
	   * if poscnt==5, then we need to calculate phi using multipole expansion at points
	   * inside [xstart,xend,ystart,yend,n_z,zend+o]
	   */

	  Xstart = xstart; Ystart=ystart; Zstart=zstart; Xend=xend; Yend=yend; Zend=zend;

	  if(poscnt==0)
	    {
	      Xstart=xstart-o; Xend=-1;
	    }
	  else if(poscnt==1)
	    {
	      Xstart=n_x; Xend=xend+o;
	    }
	  else if(poscnt==2)
	    {
	      Ystart=ystart-o; Yend=-1;
	    }
	  else if(poscnt==3)
	    {
	      Ystart=n_y; Yend=yend+o;
	    }
	  else if(poscnt==4)
	    {
	      Zstart=zstart-o; Zend=-1;
	    }
	  else if(poscnt==5)
	    {
	      Zstart=n_z; Zend=zend+o;
	    }
	  
	  nZPhi=Zend-Zstart+1;
	  nYPhi=Yend-Ystart+1;
	  nXPhi=Xend-Xstart+1;
	 
	  PetscMalloc(sizeof(PetscScalar**)*nZPhi,&pPhiArray);	       
	  if(pPhiArray == NULL)
	    {
	      cout<<"Memory allocation fail";
	      exit(1);
	    }
	  for(i = 0; i < nZPhi; i++)
	    {
	      PetscMalloc(sizeof(PetscScalar*)*nYPhi,&pPhiArray[i]);		 
	      if(pPhiArray[i] == NULL)
		{
		  cout<<"Memory allocation fail";
		  exit(1);
		}	    
	      for(j=0;j<nYPhi;j++)
		{
		  PetscMalloc(sizeof(PetscScalar)*nXPhi,&pPhiArray[i][j]);		    
		  if(pPhiArray[i][j] == NULL)
		    {
		      cout<<"Memory allocation fail";
		      exit(1);
		    } 
		}      
	    }
	  
	  /*
	   * calculate the electrostatic potential using multipole expansion at points inside
	   * the overlap region
	   */
	  for(k=0;k<nZPhi;k++)
	    for(j=0;j<nYPhi;j++)
	      for(i=0;i<nXPhi;i++)
		{
		  I=Xstart+i;
		  J=Ystart+j;
		  K=Zstart+k;
		  pPhiArray[k][j][i]=0.0;
		  
		  x = delta*I - R_x ;
		  y = delta*J - R_y ;
		  z = delta*K - R_z ;   
		  r = sqrt((x*x)+(y*y)+(z*z));
		   
		  for(l=0;l<=lmax;l++)
		    {	Phi_l = 0.0;		      
		      SpHarmoniclm=SphericalHarmonic(x,y,z,l,0,Rcut);
		      Phi_l += SpHarmoniclm*Q1[l][0];    
		      for(m=1;m<(l+1);m++)
			{
			  SpHarmoniclm=SphericalHarmonic(x,y,z,l,m,Rcut);
			  SpHarmoniclmm=SphericalHarmonic(x,y,z,l,-m,Rcut);
			 		 
			  Phi_l += SpHarmoniclm*Q1[l][m] + SpHarmoniclmm*Q2[l][m];
			}
		      pPhiArray[k][j][i] += ((4.0*M_PI)/(2.0*l+1.0))*(1.0/pow(r,1.0*l+1.0))*Phi_l;
		    }		 
		}    	  
	  	    
	  /*
	   * calculate the correction
	   */
	  for(k=zstart;k<=zend;k++)
	    for(j=ystart;j<=yend;j++)
	      for(i=xstart;i<=xend;i++)
		{
		  for(a=1;a<=o;a++)
		    {
		      if(poscnt==0)
			{
			  if((i-a)<0) // this means node is outside left boundary
			    {			      
			      pRHSVec[k][j][i] += -coeffs[a]*pPhiArray[k-Zstart][j-Ystart][i-Xstart-a];			  
			    }
			}
		      else if(poscnt==1)
			{
			  if((i+a)>=n_x) // this means node is outside right boundary 
			    {			     
			      pRHSVec[k][j][i] += -coeffs[a]*pPhiArray[k-Zstart][j-Ystart][i-Xstart+a];			   
			    }
			}		   
		      else if(poscnt==2)
			{
			  if((j-a)<0) // this means node is outside bottom boundary
			    {			      
			      pRHSVec[k][j][i] += -coeffs[a]*pPhiArray[k-Zstart][j-Ystart-a][i-Xstart];	     			  
			    }
			}		   
		      else if(poscnt==3)
			{
			  if((j+a)>=n_y) // this means node is outside top boundary
			    {			      
			      pRHSVec[k][j][i] += -coeffs[a]*pPhiArray[k-Zstart][j-Ystart+a][i-Xstart];			 
			    }
			}		   
		      else if(poscnt==4)
			{
			  if((k-a)<0) // this means node is outside front boundary
			    {			     
			      pRHSVec[k][j][i] += -coeffs[a]*pPhiArray[k-Zstart-a][j-Ystart][i-Xstart];			  
			    }
			}		   
		      else if(poscnt==5)
			{
			  if((k+a)>=n_z) // this means node is outside back boundary
			    {
			      pRHSVec[k][j][i] += -coeffs[a]*pPhiArray[k-Zstart+a][j-Ystart][i-Xstart];			   
			    }
			}  

		    } 
		}

	  for(i = 0; i < nZPhi; i++)
	    {
	      for(j=0;j<nYPhi;j++)
		{
		  PetscFree(pPhiArray[i][j]);                    
		}
	      PetscFree(pPhiArray[i]);
	    }  
	  PetscFree(pPhiArray);	  	   
	} 
     	
    }
  
  DMDAVecRestoreArray(pSddft->da,pSddft->PoissonRHSAdd,&pRHSVec); 
   
  for(l=0;l<lmax;l++)
    {
      PetscFree(Q1[l]);
      PetscFree(Q2[l]);
    }
  PetscFree(Q1);
  PetscFree(Q2);

  if(flag==1)
    {
      printf("WARNING: In Multipole expansion, charge at boundary has not decayed. Might need to rerun calculation with a larger domain size if you see this warning in the last SCF iteration.\n");
    }

  return ierr;       
}

///////////////////////////////////////////////////////////////////////////////////////////////
//                          CalculateDipoleMoment: Dipole moment calculation.                //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode CalculateDipoleMoment(SDDFT_OBJ* pSddft)
{

  PetscScalar delta=pSddft->delta;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar x,y,z,r;
  PetscScalar DMx=0.0,DMy=0.0,DMz=0.0;
  PetscScalar ***pRhoArray;
  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscErrorCode ierr;

  int i,j,k;
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim); 
  DMDAVecGetArray(pSddft->da,pSddft->elecDensRho,&pRhoArray);
  
  for(k=zcor; k<zcor+lzdim; k++)
    for(j=ycor; j<ycor+lydim; j++)
      for(i=xcor; i<xcor+lxdim; i++)
	{
	  x = delta*i - R_x ;
	  y = delta*j - R_y ;
	  z = delta*k - R_z ;   
   
	  DMx += x*pRhoArray[k][j][i];
	  DMy += y*pRhoArray[k][j][i];
	  DMz += z*pRhoArray[k][j][i];
     
	}

  

  DMDAVecRestoreArray(pSddft->da,pSddft->elecDensRho,&pRhoArray); 

  ierr =  MPI_Allreduce(MPI_IN_PLACE,&DMx,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
  ierr =  MPI_Allreduce(MPI_IN_PLACE,&DMy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
  ierr =  MPI_Allreduce(MPI_IN_PLACE,&DMz,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);

  DMx=delta*delta*delta*DMx;
  DMy=delta*delta*delta*DMy;
  DMz=delta*delta*delta*DMz;

      
  PetscPrintf(PETSC_COMM_WORLD,"Dipole moment (due to only electron density) in Debye: DMx=%0.16lf, DMy=%0.16lf, DMz=%0.16lf \n",2.54746*DMx,2.54746*DMy,2.54746*DMz);

  return 0;
}
