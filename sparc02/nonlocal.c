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
  | file name: nonlocal.cc          
  |
  | Description: This file contains the functions required for calculation of nonlocal
  | pseudopotential part of the Hamiltonian matrix and spherical harmonics required for the
  | calculation of nonlocal projectors
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana
  |
  | Last Modified: 2/18/2016   
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "petscsys.h"
#include "math.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
///////////////////////////////////////////////////////////////////////////////////////////////
//  LaplacianNonlocalPseudopotential_MatInit: creates -(1/2)laplacian+Vnonloc operator       //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode LaplacianNonlocalPseudopotential_MatInit(SDDFT_OBJ* pSddft)
{
  PetscScalar *pAtompos;
  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0,colidx;
  PetscScalar x0, y0, z0,cutoffr,max1,max2,max;
  int start,end,lmax,lloc;  
  PetscScalar tableR[MAX_TABLE_SIZE],tableUlDeltaV[4][MAX_TABLE_SIZE];
  PetscScalar tableU[4][MAX_TABLE_SIZE];
 
  PetscScalar pUlDeltaVl1,pUlDeltaVl2,r1,r2;  
  PetscScalar *Dexact=NULL;
  PetscScalar **YDUlDeltaV;
  PetscScalar **YDU;
  PetscErrorCode ierr;
  PetscScalar SpHarmonic,UlDelVl,Ul,DUlDelVl,DUl;
  PetscScalar Rcut;
  PetscScalar dr=1e-3;
  int OrbitalSize;
  PetscInt  i=0,j,k,xi,yj,zk,offset,I,J,K,II,JJ,KK,nzVps,nyVps,nxVps,nzVpsloc,nyVpsloc,nxVpsloc,i0,j0,k0,poscnt,index=0,at,ii,jj,kk,l,m;
  PetscScalar delta=pSddft->delta,x,y,z,r,xx,yy,zz,rmax,rtemp;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  MatStencil row;
  PetscScalar* val;  
  int count;
  PetscMPIInt comm_size;    
  PetscInt s;PetscScalar temp;

  max=0;
  MPI_Comm_size(PETSC_COMM_WORLD,&comm_size);
 
  PetscInt xm,ym,zm,starts[3],dims[3];
  ISLocalToGlobalMapping ltog;
 
  AO aodmda;
  DMDAGetAO(pSddft->da,&aodmda); 
  PetscInt LIrow,*LIcol;
        
  DMDAGetCorners(pSddft->da,0,0,0,&xm,&ym,&zm);
    
  DMGetLocalToGlobalMapping(pSddft->da,&ltog);

  /*
   * create the distributed matrix with the same communication pattern as the laplacian operator
   */
  MatCreate(PetscObjectComm((PetscObject)pSddft->da),&pSddft->HamiltonianOpr);   
  MatSetSizes(pSddft->HamiltonianOpr,xm*ym*zm,xm*ym*zm,n_x*n_y*n_z,n_x*n_y*n_z); 
    
  if(comm_size == 1 ) 
    MatSetType(pSddft->HamiltonianOpr,MATSEQAIJ);
  else
    MatSetType(pSddft->HamiltonianOpr,MATMPIAIJ); 
       
  MatSetFromOptions(pSddft->HamiltonianOpr);
  MatSetLocalToGlobalMapping(pSddft->HamiltonianOpr,ltog,ltog);

  DMDAGetGhostCorners(pSddft->da,&starts[0],&starts[1],&starts[2],&dims[0],&dims[1],&dims[2]);
  MatSetStencil(pSddft->HamiltonianOpr,3,dims,starts,1);
  MatSetUp(pSddft->HamiltonianOpr);        
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);

  /*
   * set preallocation here
   */

  if(comm_size == 1 ) 
    MatSeqAIJSetPreallocation(pSddft->HamiltonianOpr,0,pSddft->nnzDArray);
  else
    MatMPIAIJSetPreallocation(pSddft->HamiltonianOpr,0,pSddft->nnzDArray,0,pSddft->nnzODArray); 

#ifdef _DEBUG
  printf("Gxdim:%d,Gydim:%d,Gzdim:%d\n",gxdim,gydim,gzdim); 
  printf("xcor:%d,ycor:%d,zcor:%d\n",xcor,ycor,zcor); 
  printf("lxdim:%d,lydim:%d,lzdim:%d\n",lxdim,lydim,lzdim); 
#endif  

  /*
   * put entries of -(1/2)laplacian first
   */

  MatStencil Laprow;
  MatStencil* Lapcol;
  PetscScalar* Lapval;
  PetscInt o = pSddft->order;

  PetscMalloc(sizeof(MatStencil)*(o*6+1),&Lapcol);
  PetscMalloc(sizeof(PetscScalar)*(o*6+1),&Lapval);

  for(k=zcor; k<zcor+lzdim; k++)
    for(j=ycor; j<ycor+lydim; j++)
      for(i=xcor; i<xcor+lxdim; i++)
	{
	  Laprow.k = k; Laprow.j = j, Laprow.i = i;   
	  colidx=0; 
	  Lapcol[colidx].i=i ;Lapcol[colidx].j=j ;Lapcol[colidx].k=k ;
	  Lapval[colidx++]=pSddft->coeffs[0]  ;
	  for(l=1;l<=o;l++)
	    {
	      Lapcol[colidx].i=i ;Lapcol[colidx].j=j ;Lapcol[colidx].k=k-l ;
	      Lapval[colidx++]=pSddft->coeffs[l];
	      Lapcol[colidx].i=i ;Lapcol[colidx].j=j ;Lapcol[colidx].k=k+l ;
	      Lapval[colidx++]=pSddft->coeffs[l];
	      Lapcol[colidx].i=i ;Lapcol[colidx].j=j-l ;Lapcol[colidx].k=k ;
	      Lapval[colidx++]=pSddft->coeffs[l];
	      Lapcol[colidx].i=i ;Lapcol[colidx].j=j+l ;Lapcol[colidx].k=k ;
	      Lapval[colidx++]=pSddft->coeffs[l];
	      Lapcol[colidx].i=i-l ;Lapcol[colidx].j=j ;Lapcol[colidx].k=k ;
	      Lapval[colidx++]=pSddft->coeffs[l];
	      Lapcol[colidx].i=i+l ;Lapcol[colidx].j=j ;Lapcol[colidx].k=k ;
	      Lapval[colidx++]=pSddft->coeffs[l];
	    }
	  MatSetValuesStencil(pSddft->HamiltonianOpr,1,&Laprow,6*o+1,Lapcol,Lapval,INSERT_VALUES);
	}
  ierr = MatAssemblyBegin(pSddft->HamiltonianOpr,MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pSddft->HamiltonianOpr,MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);

  PetscFree(Lapcol);
  PetscFree(Lapval);
   
  /*
   * put entries of nonlocal pseudopotential
   */
  VecGetArray(pSddft->Atompos,&pAtompos);           
  for(at=0;at<pSddft->Ntype;at++)
    {                          
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];      
      start = floor(pSddft->startPos[at]/3);
      end = floor(pSddft->endPos[at]/3);
      /*
       * if the atom is hydrogen, then we omit the calculation of nonlocal pseudopotential 
       */
      if(lmax==0) 
	{
	  index = index + 3*(end-start+1); 
	}
      if(lmax!=0) 
	{
	  max1=  pSddft->psd[at].rc_s > pSddft->psd[at].rc_p ? pSddft->psd[at].rc_s:pSddft->psd[at].rc_p;
	  max2=  pSddft->psd[at].rc_d > pSddft->psd[at].rc_f ? pSddft->psd[at].rc_d:pSddft->psd[at].rc_f;
	  cutoffr =  max1>max2 ? max1:max2;                
	  offset = ceil(cutoffr/delta);
          tableR[0]=0.0;            
	  for(l=0;l<=lmax;l++)
            {
	      /*
	       * since the pseudopotential table read from the file does not have a value 
	       * at r=0, for s orbital, we assume the nonlocal projector and the 
	       * pseudowavefunction value at r=0 to be same as that calculated from the first
	       * entry read from the file. However for p,d and f orbitals, we assume the values
	       * at r=0 to be 0.
	       */
	      if(l==0)
		{	  
		  tableUlDeltaV[0][0]=pSddft->psd[at].Us[0]*(pSddft->psd[at].Vs[0]-pSddft->psd[at].Vloc[0]);
		  tableU[0][0]=pSddft->psd[at].Us[0]; 		      
		}
	      if(l==1)
		{
		  tableUlDeltaV[1][0]=0.0; 
		  tableU[1][0]=0.0;         	     
		}
	      if(l==2)
		{
		  tableUlDeltaV[2][0]=0.0; 
		  tableU[2][0]=0.0;        	      
		}
	      if(l==3)
		{
		  tableUlDeltaV[3][0]=0.0; 
		  tableU[3][0]=0.0;            	  
		}
            }
	  count=1;
	  do{
	    tableR[count] = pSddft->psd[at].RadialGrid[count-1];
	    for(l=0;l<=lmax;l++)
	      {
		if(l==0)
		  {
		    tableUlDeltaV[0][count]=pSddft->psd[at].Us[count-1]*(pSddft->psd[at].Vs[count-1]-pSddft->psd[at].Vloc[count-1]); 
		    tableU[0][count]=pSddft->psd[at].Us[count-1];  		      
		  }
		if(l==1)
		  {
		    tableUlDeltaV[1][count]=pSddft->psd[at].Up[count-1]*(pSddft->psd[at].Vp[count-1]-pSddft->psd[at].Vloc[count-1]); 		      
		    tableU[1][count]=pSddft->psd[at].Up[count-1];    	       
		  }
		if(l==2)
		  {

		    tableUlDeltaV[2][count]=pSddft->psd[at].Ud[count-1]*(pSddft->psd[at].Vd[count-1]-pSddft->psd[at].Vloc[count-1]); 		      
		    tableU[2][count]=pSddft->psd[at].Ud[count-1];        	       
		  }   
		if(l==3)
		  {
		    tableUlDeltaV[3][count]=pSddft->psd[at].Uf[count-1]*(pSddft->psd[at].Vf[count-1]-pSddft->psd[at].Vloc[count-1]); 	      
		    tableU[3][count]=pSddft->psd[at].Uf[count-1];                      
		  }  
	      }
	    count++;
	  }while(tableR[count-1] <= 1.732*pSddft->CUTOFF[at]+4.0); 
	  rmax = tableR[count-1];            
                     	                            
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDUlDeltaV);
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDU);
           
	  for(l=0;l<=lmax;l++)
            {
              PetscMalloc(sizeof(PetscScalar)*count,&YDUlDeltaV[l]);
              PetscMalloc(sizeof(PetscScalar)*count,&YDU[l]);  
	      /*
	       * derivatives of the spline fit to the pseudopotentials and pseudowavefunctions
	       */
              getYD_gen(tableR,tableUlDeltaV[l],YDUlDeltaV[l],count);
              getYD_gen(tableR,tableU[l],YDU[l],count);             
            }            	  
	 
	  /*
	   * Computation of denominator term
	   */
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&Dexact);	        
	  if(Dexact == NULL)
	    {
	      printf("memory allocation fail");
	      exit(1);
	    }
	  for(l=0;l<=lmax;l++) 
	    {			  
	      Dexact[l]= 0; rtemp=dr;
	      if(l!=lloc)
		{
		  if(l==0)
		    Rcut= pSddft->psd[at].rc_s;
		  if(l==1)
		    Rcut= pSddft->psd[at].rc_p;
		  if(l==2)
		    Rcut= pSddft->psd[at].rc_d;
		  if(l==3)
		    Rcut= pSddft->psd[at].rc_f;				    
		 
		  while(rtemp<=(Rcut+delta))
		    {				      
		      ispline_gen(tableR,tableUlDeltaV[l],count,&rtemp,&UlDelVl,&DUlDelVl,1,YDUlDeltaV[l]);
		      ispline_gen(tableR,tableU[l],count,&rtemp,&Ul,&DUl,1,YDU[l]);
												
		      Dexact[l]+= UlDelVl*Ul*rtemp*rtemp*dr;
		      rtemp=rtemp+dr;					
		    }
		  Dexact[l]= (Dexact[l] - 0.5*UlDelVl*Ul*(rtemp-dr)*(rtemp-dr)*dr)/(delta*delta*delta); 
			    		  
		}
	    }	 
	  
	  /*
	   * loop over every atom of a given type
	   */
	  for(poscnt=start;poscnt<=end;poscnt++)
	    { 
	      xstart=-1; ystart=-1; zstart=-1; xend=-1; yend=-1; zend=-1; overlap=0;	     
	      x0 = pAtompos[index++];
	      y0 = pAtompos[index++];
	      z0 = pAtompos[index++];
	          
	      xi = (int)((x0 + R_x)/delta + 0.5); 
	      yj = (int)((y0 + R_y)/delta + 0.5);
	      zk = (int)((z0 + R_z)/delta + 0.5);
	  
	      assert ((xi-offset >= 0)&&(xi+offset<n_x));
	      assert ((yj-offset >= 0)&&(yj+offset<n_y));
	      assert ((zk-offset >= 0)&&(zk+offset<n_z));  
	      
	      xs = xi-offset; xl = xi+offset;
	      ys = yj-offset; yl = yj+offset;
	      zs = zk-offset; zl = zk+offset;
	     
	      /*
	       * find if domain of influence of pseudocharge overlaps with the domain stored
	       * by processor  
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
	        
		  nzVps = zl-zs+1;
		  nyVps = yl-ys+1;
		  nxVps = xl-xs+1;
		
		  nzVpsloc = zend-zstart+1;
		  nyVpsloc = yend-ystart+1;
		  nxVpsloc = xend-xstart+1;
				  		              
		  PetscMalloc(sizeof(PetscScalar)*(nzVps*nyVps*nxVps),&val);
		  PetscMalloc(sizeof(PetscInt)*(nzVps*nyVps*nxVps),&LIcol);
						  
		  i0 = xs;
		  j0 = ys;
		  k0 = zs;
		  /*
		   * while evaluating the nonlocal pseudopotential entries, each processor
		   * enters matrix entries in local rows. local rows commensurate with the
		   * overlapping nodes. For every local row, we go over all the nodes in the
		   * pseudopotential cutoff region and calculate the matrix entries
		   */	  

		  for(k=zstart;k<=zend;k++)
		    for(j=ystart;j<=yend;j++)
		      for(i=xstart;i<=xend;i++)
			{		 
			 
			  x = delta*i - R_x ;
			  y = delta*j - R_y ;
			  z = delta*k - R_z ;   
			  r1 = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));			  
			  colidx=0; 			
			  LIrow = k*n_x*n_y + j*n_x + i; 
			  AOApplicationToPetsc(aodmda,1,&LIrow);			
			 
			  for(kk=zs;kk<=zl;kk++)
			    for(jj=ys;jj<=yl;jj++)
			      for(ii=xs;ii<=xl;ii++)
				{			  
				  xx = delta*ii - R_x ;
				  yy = delta*jj - R_y ;
				  zz = delta*kk - R_z ;   
				  r2 = sqrt((xx-x0)*(xx-x0)+(yy-y0)*(yy-y0)+(zz-z0)*(zz-z0));		                
				  val[colidx] = 0;	                               		 
				  for(l=0;l<=lmax;l++)
				    {
				      if(l!=lloc)
					{					 
					  if(r1 == tableR[0])
					    {			      
					      pUlDeltaVl1 = tableUlDeltaV[l][0];	   
					    }
					  else
					    {  
					      ispline_gen(tableR,tableUlDeltaV[l],count,&r1,&pUlDeltaVl1,&DUlDelVl,1,YDUlDeltaV[l]);			                          
					    }
					 
					  if(r2 == tableR[0])
					    {			      
					      pUlDeltaVl2 = tableUlDeltaV[l][0];				   
					    }
					  else
					    {	  
					      ispline_gen(tableR,tableUlDeltaV[l],count,&r2,&pUlDeltaVl2,&DUlDelVl,1,YDUlDeltaV[l]);			                          
					    }			
					  for(m=-l;m<=l;m++)
					    {				     
					      SpHarmonic=SphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut);	
					      val[colidx] += (pUlDeltaVl1*SphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut)*pUlDeltaVl2*SphericalHarmonic(xx-x0,yy-y0,zz-z0,l,m,Rcut))/Dexact[l]; 
					    }					   
					}
				    }  
				  LIcol[colidx] = kk*n_x*n_y + jj*n_x + ii; 
				  /*
				   * only insert values whose magnitude is >=1e-16
				   */
				  if(fabs(val[colidx]) >= 1e-16)
				    {                        
				      colidx++;		
				    }
				} 
			  if(colidx>0)
			    {
			      AOApplicationToPetsc(aodmda,colidx,LIcol);			
			      ierr=MatSetValues(pSddft->HamiltonianOpr,1,&LIrow,colidx,LIcol,val,ADD_VALUES);CHKERRQ(ierr);
			    }			                   
			} 	
		  PetscFree(val);
		  PetscFree(LIcol);				
		}   
	    }		  
	  for(l=0;l<=lmax;l++)
	    {
	      PetscFree(YDUlDeltaV[l]);
	      PetscFree(YDU[l]);	    
	    }
	  PetscFree(YDUlDeltaV);
	  PetscFree(YDU);	  
	  PetscFree(Dexact);
	    	
	}  
    }

  /*
   * assemble matrix
   */

  ierr = MatAssemblyBegin(pSddft->HamiltonianOpr,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pSddft->HamiltonianOpr,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
             
  VecRestoreArray(pSddft->Atompos,&pAtompos);  
  return ierr;
}
///////////////////////////////////////////////////////////////////////////////////////////////
// PeriodicLaplacianNonlocalPseudopotential_MatInit: creates -(1/2)laplacian+Vnonloc operator//
//                  for periodic boundary conditions (without k point sampling)              //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode PeriodicLaplacianNonlocalPseudopotential_MatInit(SDDFT_OBJ* pSddft)
{
  PetscScalar *pAtompos;
  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0,colidx;
  PetscScalar x0, y0, z0,X0,Y0,Z0,cutoffr,max1,max2,max;
  int start,end,lmax,lloc;  
  PetscScalar tableR[MAX_TABLE_SIZE],tableUlDeltaV[4][MAX_TABLE_SIZE];
  PetscScalar tableU[4][MAX_TABLE_SIZE];

  PetscScalar pUlDeltaVl1,pUlDeltaVl2,r1,r2;
  PetscScalar *Dexact=NULL;
  PetscScalar **YDUlDeltaV;
  PetscScalar **YDU;
  PetscErrorCode ierr;
  PetscScalar SpHarmonic,UlDelVl,Ul,DUlDelVl,DUl;

  PetscScalar Rcut;
  PetscScalar dr=1e-3;
  int OrbitalSize;
  PetscInt  i=0,j,k,xi,yj,zk,offset,I,J,K,II,JJ,KK,nzVps,nyVps,nxVps,nzVpsloc,nyVpsloc,nxVpsloc,i0,j0,k0,poscnt,index=0,at,ii,jj,kk,l,m;
  PetscScalar delta=pSddft->delta,x,y,z,r,xx,yy,zz,rmax,rtemp,XX,YY,ZZ;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;

  PetscInt PP,QQ,RR;
  PetscInt Imax_x,Imin_x,Imax_y,Imin_y,Imax_z,Imin_z;
  MatStencil row;
  PetscScalar* val;  
  int count;
  PetscMPIInt comm_size;
    
  PetscInt s;PetscScalar temp;
  max=0;
  MPI_Comm_size(PETSC_COMM_WORLD,&comm_size);
 
  PetscInt xm,ym,zm,starts[3],dims[3];
  ISLocalToGlobalMapping ltog;
  
  AO aodmda;
  DMDAGetAO(pSddft->da,&aodmda); 
  PetscInt LIrow,*LIcol;
        
  DMDAGetCorners(pSddft->da,0,0,0,&xm,&ym,&zm);
    
  DMGetLocalToGlobalMapping(pSddft->da,&ltog);

  /*
   * create the distributed matrix with the same numbering as the laplacian operator
   */
  MatCreate(PetscObjectComm((PetscObject)pSddft->da),&pSddft->HamiltonianOpr); 
          
  MatSetSizes(pSddft->HamiltonianOpr,xm*ym*zm,xm*ym*zm,n_x*n_y*n_z,n_x*n_y*n_z);
    
  if(comm_size == 1 ) 
    MatSetType(pSddft->HamiltonianOpr,MATSEQAIJ);
  else
    MatSetType(pSddft->HamiltonianOpr,MATMPIAIJ);
   
  MatSetFromOptions(pSddft->HamiltonianOpr);
  MatSetLocalToGlobalMapping(pSddft->HamiltonianOpr,ltog,ltog);

  DMDAGetGhostCorners(pSddft->da,&starts[0],&starts[1],&starts[2],&dims[0],&dims[1],&dims[2]);
  MatSetStencil(pSddft->HamiltonianOpr,3,dims,starts,1);
  MatSetUp(pSddft->HamiltonianOpr);        
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);

  /*
   * set preallocation here
   */
  if(comm_size == 1 ) 
    MatSeqAIJSetPreallocation(pSddft->HamiltonianOpr,0,pSddft->nnzDArray);
  else
    MatMPIAIJSetPreallocation(pSddft->HamiltonianOpr,0,pSddft->nnzDArray,0,pSddft->nnzODArray);  

  MatSetOption(pSddft->HamiltonianOpr, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); 

#ifdef _DEBUG
  printf("Gxdim:%d,Gydim:%d,Gzdim:%d\n",gxdim,gydim,gzdim); 
  printf("xcor:%d,ycor:%d,zcor:%d\n",xcor,ycor,zcor); 
  printf("lxdim:%d,lydim:%d,lzdim:%d\n",lxdim,lydim,lzdim); 
#endif  

  /*
   * put entries of -(1/2)laplacian first
   */
  MatStencil Laprow;
  MatStencil* Lapcol;
  PetscScalar* Lapval;
  PetscInt o = pSddft->order;

  PetscMalloc(sizeof(MatStencil)*(o*6+1),&Lapcol);
  PetscMalloc(sizeof(PetscScalar)*(o*6+1),&Lapval);

  for(k=zcor; k<zcor+lzdim; k++)
    for(j=ycor; j<ycor+lydim; j++)
      for(i=xcor; i<xcor+lxdim; i++)
	{
	  Laprow.k = k; Laprow.j = j, Laprow.i = i;
   
	  colidx=0; 
	  Lapcol[colidx].i=i ;Lapcol[colidx].j=j ;Lapcol[colidx].k=k ;
	  Lapval[colidx++]=pSddft->coeffs[0]  ;
	  for(l=1;l<=o;l++)
	    {
	      Lapcol[colidx].i=i ;Lapcol[colidx].j=j ;Lapcol[colidx].k=k-l ;
	      Lapval[colidx++]=pSddft->coeffs[l];
	      Lapcol[colidx].i=i ;Lapcol[colidx].j=j ;Lapcol[colidx].k=k+l ;
	      Lapval[colidx++]=pSddft->coeffs[l];
	      Lapcol[colidx].i=i ;Lapcol[colidx].j=j-l ;Lapcol[colidx].k=k ;
	      Lapval[colidx++]=pSddft->coeffs[l];
	      Lapcol[colidx].i=i ;Lapcol[colidx].j=j+l ;Lapcol[colidx].k=k ;
	      Lapval[colidx++]=pSddft->coeffs[l];
	      Lapcol[colidx].i=i-l ;Lapcol[colidx].j=j ;Lapcol[colidx].k=k ;
	      Lapval[colidx++]=pSddft->coeffs[l];
	      Lapcol[colidx].i=i+l ;Lapcol[colidx].j=j ;Lapcol[colidx].k=k ;
	      Lapval[colidx++]=pSddft->coeffs[l];
	    }
	  MatSetValuesStencil(pSddft->HamiltonianOpr,1,&Laprow,6*o+1,Lapcol,Lapval,ADD_VALUES);
	}

  ierr = MatAssemblyBegin(pSddft->HamiltonianOpr,MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pSddft->HamiltonianOpr,MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);

  PetscFree(Lapcol);
  PetscFree(Lapval);
  /*
   * put entries of nonlocal pseudopotential
   */
  VecGetArray(pSddft->Atompos,&pAtompos);           
  for(at=0;at<pSddft->Ntype;at++)
    {                          
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];
            
      start = floor(pSddft->startPos[at]/3);
      end = floor(pSddft->endPos[at]/3);
      /*
       * if the atom is hydrogen, then we omit the calculation of nonlocal pseudopotential 
       */
      if(lmax==0)
	{
	  index = index + 3*(end-start+1);
	}
      if(lmax!=0) 
	{
	  max1=  pSddft->psd[at].rc_s > pSddft->psd[at].rc_p ? pSddft->psd[at].rc_s:pSddft->psd[at].rc_p;
	  max2=  pSddft->psd[at].rc_d > pSddft->psd[at].rc_f ? pSddft->psd[at].rc_d:pSddft->psd[at].rc_f;
	  cutoffr =  max1>max2 ? max1:max2;               
	  offset = ceil(cutoffr/delta);
	  tableR[0]=0.0;            
	  for(l=0;l<=lmax;l++)
            {
	      
	      /*
	       * since the pseudopotential table read from the file does not have a value 
	       * at r=0, for s orbital, we assume the nonlocal projector and the 
	       * pseudowavefunction value at r=0 to be same as that calculated from the first
	       * entry read from the file. However for p,d and f orbitals, we assume the values
	       * at r=0 to be 0.
	       */
	      if(l==0)
		{
		  tableUlDeltaV[0][0]=pSddft->psd[at].Us[0]*(pSddft->psd[at].Vs[0]-pSddft->psd[at].Vloc[0]);
		  tableU[0][0]=pSddft->psd[at].Us[0]; 		      
		}
	      if(l==1)
		{
		  tableUlDeltaV[1][0]=0.0; 
		  tableU[1][0]=0.0;         	     
		}
	      if(l==2)
		{
		  tableUlDeltaV[2][0]=0.0; 
		  tableU[2][0]=0.0;        	      
		}
	      if(l==3)
		{
		  tableUlDeltaV[3][0]=0.0; 
		  tableU[3][0]=0.0;            	  
		}
            }
	  count=1;
	  do{
	    tableR[count] = pSddft->psd[at].RadialGrid[count-1];
	    for(l=0;l<=lmax;l++)
	      {
		if(l==0)
		  {
		    tableUlDeltaV[0][count]=pSddft->psd[at].Us[count-1]*(pSddft->psd[at].Vs[count-1]-pSddft->psd[at].Vloc[count-1]); 
		    tableU[0][count]=pSddft->psd[at].Us[count-1]; 		      
		  }
		if(l==1)
		  {
		    tableUlDeltaV[1][count]=pSddft->psd[at].Up[count-1]*(pSddft->psd[at].Vp[count-1]-pSddft->psd[at].Vloc[count-1]); 		      
		    tableU[1][count]=pSddft->psd[at].Up[count-1];  
		  }
		if(l==2)
		  {
		    tableUlDeltaV[2][count]=pSddft->psd[at].Ud[count-1]*(pSddft->psd[at].Vd[count-1]-pSddft->psd[at].Vloc[count-1]);     
		    tableU[2][count]=pSddft->psd[at].Ud[count-1];   	       
		  }   
		if(l==3)
		  {
		    tableUlDeltaV[3][count]=pSddft->psd[at].Uf[count-1]*(pSddft->psd[at].Vf[count-1]-pSddft->psd[at].Vloc[count-1]); 	      
		    tableU[3][count]=pSddft->psd[at].Uf[count-1];                   
		  }  
	      }
	    count++;
	  }while(tableR[count-1] <= 1.732*pSddft->CUTOFF[at]+4.0);
	  rmax = tableR[count-1];               
	     
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDUlDeltaV);
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDU);
           
	  for(l=0;l<=lmax;l++)
            {
              PetscMalloc(sizeof(PetscScalar)*count,&YDUlDeltaV[l]);
              PetscMalloc(sizeof(PetscScalar)*count,&YDU[l]);
	      /*
	       * derivatives of the spline fit to the pseudopotentials and pseudowavefunctions
	       */
              getYD_gen(tableR,tableUlDeltaV[l],YDUlDeltaV[l],count); 
              getYD_gen(tableR,tableU[l],YDU[l],count);           
            }         
	  /*
	   * Computation of denominator term
	   */	   		    	
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&Dexact);	        
	  if(Dexact == NULL)
	    {
	      printf("memory allocation fail");
	      exit(1);
	    }
	  for(l=0;l<=lmax;l++) 
	    {			  
	      Dexact[l]= 0; rtemp=dr;
	      if(l!=lloc)
		{
		  if(l==0)
		    Rcut= pSddft->psd[at].rc_s;
		  if(l==1)
		    Rcut= pSddft->psd[at].rc_p;
		  if(l==2)
		    Rcut= pSddft->psd[at].rc_d;
		  if(l==3)
		    Rcut= pSddft->psd[at].rc_f;
		
		  while(rtemp<=(Rcut+delta))
		    {				      
		      ispline_gen(tableR,tableUlDeltaV[l],count,&rtemp,&UlDelVl,&DUlDelVl,1,YDUlDeltaV[l]);
		      ispline_gen(tableR,tableU[l],count,&rtemp,&Ul,&DUl,1,YDU[l]);
												
		      Dexact[l]+= UlDelVl*Ul*rtemp*rtemp*dr;
		      rtemp=rtemp+dr;					
		    }
		  Dexact[l]= (Dexact[l] - 0.5*UlDelVl*Ul*(rtemp-dr)*(rtemp-dr)*dr)/(delta*delta*delta);     		  
		}
	    }
	
	  /*
	   * loop over every atom of a given type
	   */
	  for(poscnt=start;poscnt<=end;poscnt++)
	    { 	      
	      X0 = pAtompos[index++];
	      Y0 = pAtompos[index++];
	      Z0 = pAtompos[index++];

	      xi = roundf((X0 + R_x)/delta); 
	      yj = roundf((Y0 + R_y)/delta);
	      zk = roundf((Z0 + R_z)/delta);
	  
	      Imax_x=0; Imin_x=0; Imax_y=0; Imin_y=0; Imax_z=0; Imin_z=0;
	      
	      Imax_x= ceil(cutoffr/R_x);
	      Imin_x= -ceil(cutoffr/R_x);
	      Imax_y= ceil(cutoffr/R_y);	
	      Imin_y= -ceil(cutoffr/R_y);	
	      Imax_z= ceil(cutoffr/R_z);	
	      Imin_z= -ceil(cutoffr/R_z);

	      /*
	       * loop over periodic images
	       */
	      for(PP=Imin_x;PP<=Imax_x;PP++)
		for(QQ=Imin_y;QQ<=Imax_y;QQ++)
		  for(RR=Imin_z;RR<=Imax_z;RR++)
		    {
		      xstart=-1000; ystart=-1000; zstart=-1000; xend=-1000; yend=-1000; zend=-1000; overlap=0;
		     
		      x0 = X0 + PP*2.0*R_x;
		      y0 = Y0 + QQ*2.0*R_y;
		      z0 = Z0 + RR*2.0*R_z;
		    
		      xi = roundf((x0 + R_x)/delta); 
		      yj = roundf((y0 + R_y)/delta);
		      zk = roundf((z0 + R_z)/delta);
		      
		      xs = xi-offset; xl = xi+offset;
		      ys = yj-offset; yl = yj+offset;
		      zs = zk-offset; zl = zk+offset;	     
		      
		      /*
		       * find if domain of influence of pseudocharge overlaps with the domain stored
		       * by processor  
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
	     
		      if((xstart!=-1000)&&(xend!=-1000)&&(ystart!=-1000)&&(yend!=-1000)&&(zstart!=-1000)&&(zend!=-1000))
			overlap =1;  
	    
		      if(overlap)
			{			        
			  nzVps = zl-zs+1;
			  nyVps = yl-ys+1;
			  nxVps = xl-xs+1;
		
			  nzVpsloc = zend-zstart+1;
			  nyVpsloc = yend-ystart+1;
			  nxVpsloc = xend-xstart+1;
	
			  PetscMalloc(sizeof(PetscScalar)*(nzVps*nyVps*nxVps),&val);
			  PetscMalloc(sizeof(PetscInt)*(nzVps*nyVps*nxVps),&LIcol);
						  
			  i0 = xs;
			  j0 = ys;
			  k0 = zs;	
			  /*
			   * while evaluating the nonlocal pseudopotential entries, each processor
			   * enters matrix entries in local rows. local rows commensurate with the
			   * overlapping nodes. For every local row, we go over all the nodes in the
			   * pseudopotential cutoff region and calculate the matrix entries
			   */	
		
			  for(k=zstart;k<=zend;k++)
			    for(j=ystart;j<=yend;j++)
			      for(i=xstart;i<=xend;i++)
				{		 

				  x = delta*i - R_x ;
				  y = delta*j - R_y ;
				  z = delta*k - R_z ;   
				  r1 = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));		  
		  
				  colidx=0; 	
			
				  LIrow = k*n_x*n_y + j*n_x + i; 
				  AOApplicationToPetsc(aodmda,1,&LIrow);
		
				  for(kk=zs;kk<=zl;kk++)
				    for(jj=ys;jj<=yl;jj++)
				      for(ii=xs;ii<=xl;ii++)
					{			  
					  xx = delta*ii - R_x ;
					  yy = delta*jj - R_y ;
					  zz = delta*kk - R_z ;
					  r2 = sqrt((xx-x0)*(xx-x0)+(yy-y0)*(yy-y0)+(zz-z0)*(zz-z0));
		                
					  XX = xx - PP*2.0*R_x;
					  YY = yy - QQ*2.0*R_y;
					  ZZ = zz - RR*2.0*R_z;	

					  val[colidx] = 0;	                               		 
					  for(l=0;l<=lmax;l++)
					    {
					      if(l!=lloc)
						{
						  if(r1 == tableR[0])
						    {			      
						      pUlDeltaVl1 = tableUlDeltaV[l][0];					   
						    }
						  else if(r1>=rmax)
						    {
						      pUlDeltaVl1=0.0;
						    }
						  else
						    {	
						      ispline_gen(tableR,tableUlDeltaV[l],count,&r1,&pUlDeltaVl1,&DUlDelVl,1,YDUlDeltaV[l]);	                          
						    }

						  if(r2 == tableR[0])
						    {			      
						      pUlDeltaVl2 = tableUlDeltaV[l][0];				   
						    }
						  else if(r2>=rmax)
						    {
						      pUlDeltaVl2 = 0.0;
						    }
						  else
						    {
						      ispline_gen(tableR,tableUlDeltaV[l],count,&r2,&pUlDeltaVl2,&DUlDelVl,1,YDUlDeltaV[l]);	                          
						    }
			
						  for(m=-l;m<=l;m++)
						    {				     
						      SpHarmonic=SphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut);	
						      val[colidx] += (pUlDeltaVl1*SphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut)*pUlDeltaVl2*SphericalHarmonic(xx-x0,yy-y0,zz-z0,l,m,Rcut))/Dexact[l]; 
						    }					   
						}
					    }  
				 
					  /*
					   * map points outside the computational domain to points inside computational domain 
					   */
					  if(roundf((XX+R_x)/delta) < 0 ) 
					    XX=XX+2.0*R_x;

					  if(roundf((XX+R_x)/delta) >= n_x)
					    XX=XX-2.0*R_x;
				
					  if(roundf((YY+R_y)/delta) < 0)
					    YY=YY+2.0*R_y;

					  if(roundf((YY+R_y)/delta) >= n_y)
					    YY=YY-2.0*R_y;

					  if(roundf((ZZ+R_z)/delta) < 0)
					    ZZ=ZZ+2.0*R_z;

					  if(roundf((ZZ+R_z)/delta) >= n_z)
					    ZZ=ZZ-2.0*R_z;				
				 
					  /*
					   * linear index of column position if only 1 proc was employed 
					   */
					  LIcol[colidx] = roundf((ZZ+R_z)/delta)*n_x*n_y + roundf((YY+R_y)/delta)*n_x + roundf((XX+R_x)/delta); 

					  /*
					   * only insert values whose magnitude is >=1e-16
					   */
					  if(fabs(val[colidx]) >= 1e-16) 
					    {                        
					      colidx++;		
					    }
					} 
				  if(colidx>0)
				    {
				      AOApplicationToPetsc(aodmda,colidx,LIcol);			
				      ierr=MatSetValues(pSddft->HamiltonianOpr,1,&LIrow,colidx,LIcol,val,ADD_VALUES);CHKERRQ(ierr);
				    }
			                   
				}	
			  PetscFree(val);
			  PetscFree(LIcol);
				
			} 	    
		    } 
	    }	    

	  for(l=0;l<=lmax;l++)
	    {
	      PetscFree(YDUlDeltaV[l]);
	      PetscFree(YDU[l]);	    
	    }
	  PetscFree(YDUlDeltaV);
	  PetscFree(YDU);	       
	  
	  PetscFree(Dexact);
	    	
	}   
    }

  /*
   * assemble matrix
   */
  ierr = MatAssemblyBegin(pSddft->HamiltonianOpr,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pSddft->HamiltonianOpr,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        
  VecRestoreArray(pSddft->Atompos,&pAtompos);  
   
  return ierr;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//          SphericalHarmonic: returns the real spherical harmonic for some given l,m        // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar SphericalHarmonic(PetscScalar x,PetscScalar y,PetscScalar z,int l,int m,PetscScalar rc)
{
  /*
   * only l=0,1,2,3,4,5,6 implemented for now
   */

  
  /*
   * l=0
   */
  PetscScalar C00 = 0.282094791773878; // 0.5*sqrt(1/pi)
  /*
   * l=1
   */
  PetscScalar C1m1 = 0.488602511902920; // sqrt(3/(4*pi))
  PetscScalar C10 = 0.488602511902920; // sqrt(3/(4*pi))
  PetscScalar C1p1 = 0.488602511902920; // sqrt(3/(4*pi))
  /*
   * l=2
   */
  PetscScalar C2m2 = 1.092548430592079; // 0.5*sqrt(15/pi)
  PetscScalar C2m1 = 1.092548430592079; // 0.5*sqrt(15/pi)  
  PetscScalar C20 =  0.315391565252520; // 0.25*sqrt(5/pi)
  PetscScalar C2p1 = 1.092548430592079; // 0.5*sqrt(15/pi)  
  PetscScalar C2p2 =  0.546274215296040; // 0.25*sqrt(15/pi)
  /*  
   * l=3
   */
  PetscScalar C3m3 =  0.590043589926644; // 0.25*sqrt(35/(2*pi))   
  PetscScalar C3m2 = 2.890611442640554; // 0.5*sqrt(105/(pi))
  PetscScalar C3m1 = 0.457045799464466; // 0.25*sqrt(21/(2*pi))
  PetscScalar C30 =  0.373176332590115; // 0.25*sqrt(7/pi)
  PetscScalar C3p1 =  0.457045799464466; // 0.25*sqrt(21/(2*pi))
  PetscScalar C3p2 = 1.445305721320277; //  0.25*sqrt(105/(pi))
  PetscScalar C3p3 = 0.590043589926644; //  0.25*sqrt(35/(2*pi))

  PetscScalar pi=M_PI;
  PetscScalar sin_theta,cos_theta,sin_phi,cos_phi,phi;
  PetscScalar p;                   
  PetscScalar r = sqrt(x*x + y*y + z*z);                    
  PetscScalar SH=0.0;		  
  if(r<=rc)
    {
      if(l==0)
	SH = C00;

      if(r!=0.0)
	{
	  if(l==0)
	    SH = C00;
		   
	  else if(l==1)
	    {
	      if(m==-1)
		SH = C1m1*(y/r);
	      else if(m==0)
		SH = C10*(z/r);
	      else if(m==1)
		SH = C1p1*(x/r);
	      else{
		printf("incorrect l: %d,m: %d\n",l,m);  
		exit(1);
	      }       
	    }
	  else if(l==2)
	    {
	      if(m==-2)
		SH = C2m2*(x*y)/(r*r);
	      else if(m==-1)
		SH = C2m1*(y*z)/(r*r);
	      else if(m==0)
		{
		  SH = C20*(-x*x - y*y + 2.0*z*z)/(r*r);		 
		}
	      else if(m==1)
		SH = C2p1*(z*x)/(r*r);
	      else if(m==2)
		SH = C2p2*(x*x - y*y)/(r*r);
	      else{
		printf("incorrect l: %d,m: %d\n",l,m);  
		exit(1);
	      }		                     
	    }
	  else if(l==3)
	    {
	      if(m==-3)
		SH = C3m3*(3*x*x - y*y)*y/(r*r*r);
	      else if(m==-2)
		SH = C3m2*(x*y*z)/(r*r*r);
	      else if(m==-1)
		SH = C3m1*y*(4*z*z - x*x - y*y)/(r*r*r);
	      else if(m==0)
		SH = C30*z*(2*z*z-3*x*x-3*y*y)/(r*r*r);
	      else if(m==1)
		SH = C3p1*x*(4*z*z - x*x - y*y)/(r*r*r);
	      else if(m==2)
		SH = C3p2*z*(x*x - y*y)/(r*r*r);
	      else if(m==3)
		SH = C3p3*x*(x*x-3*y*y)/(r*r*r);
	      else{
		printf("incorrect l: %d,m: %d\n",l,m);  
		exit(1);
	      }		   
	    }
	  else if(l==4)
	    {
	      if(m==-4)
		SH=(3.0/4.0)*sqrt(35.0/pi)*(x*y*(x*x-y*y))/(r*r*r*r);
	      else if(m==-3)
		SH=(3.0/4.0)*sqrt(35.0/(2.0*pi))*(3.0*x*x-y*y)*y*z/(r*r*r*r);
	      else if(m==-2)
		SH=(3.0/4.0)*sqrt(5.0/pi)*x*y*(7.0*z*z-r*r)/(r*r*r*r);
	      else if(m==-1)
		SH=(3.0/4.0)*sqrt(5.0/(2.0*pi))*y*z*(7.0*z*z-3.0*r*r)/(r*r*r*r);
	      else if(m==0)
		SH=(3.0/16.0)*sqrt(1.0/pi)*(35.0*z*z*z*z-30.0*z*z*r*r+3.0*r*r*r*r)/(r*r*r*r);
	      else if(m==1)
		SH=(3.0/4.0)*sqrt(5.0/(2.0*pi))*x*z*(7.0*z*z-3.0*r*r)/(r*r*r*r);
	      else if(m==2)
		SH=(3.0/8.0)*sqrt(5.0/(pi))*(x*x-y*y)*(7.0*z*z-r*r)/(r*r*r*r);
	      else if(m==3)
		SH=(3.0/4.0)*sqrt(35.0/(2.0*pi))*(x*x-3.0*y*y)*x*z/(r*r*r*r);
	      else if(m==4)
		SH=(3.0/16.0)*sqrt(35.0/pi)*(x*x*(x*x-3.0*y*y) - y*y*(3.0*x*x-y*y))/(r*r*r*r);
	      else{
		printf("incorrect l: %d,m: %d\n",l,m);  
		exit(1);
	      }
	    }
	  else if(l==5)
	    {
	      p = sqrt(x*x+y*y);			 			   
	      if(m==-5)
		SH = (3.0*sqrt(2.0)/32.0)*(8.0*x*x*x*x*y-4.0*x*x*y*y*y + 4.0*pow(y,5)-3.0*y*p*p*p*p)/(r*r*r*r*r);
	      else if(m==-4)
		SH = (3.0/16.0)*sqrt(385.0/pi)*(4.0*x*x*x*y - 4.0*x*y*y*y)*z/(r*r*r*r*r);
	      else if(m==-3)
		SH = (sqrt(2.0*385.0/pi)/32.0)*(3.0*y*p*p - 4.0*y*y*y)*(9*z*z-r*r)/(r*r*r*r*r);
	      else if(m==-2)
		SH = (1.0/8.0)*sqrt(1155.0/pi)*2.0*x*y*(3.0*z*z*z-z*r*r)/(r*r*r*r*r);
	      else if(m==-1)
		SH = (1.0/16.0)*sqrt(165.0/pi)*y*(21.0*z*z*z*z - 14.0*r*r*z*z+r*r*r*r)/(r*r*r*r*r);
	      else if(m==0)
		SH = (1.0/16.0)*sqrt(11.0/pi)*(63.0*z*z*z*z*z -70.0*z*z*z*r*r + 15.0*z*r*r*r*r)/(r*r*r*r*r);
	      else if(m==1)
		SH = (1.0/16.0)*sqrt(165.0/pi)*x*(21.0*z*z*z*z - 14.0*r*r*z*z+r*r*r*r)/(r*r*r*r*r);
	      else if(m==2)
		SH = (1.0/8.0)*sqrt(1155.0/pi)*(x*x-y*y)*(3.0*z*z*z - r*r*z)/(r*r*r*r*r);
	      else if(m==3)
		SH = (sqrt(2.0*385.0/pi)/32.0)*(4.0*x*x*x-3.0*p*p*x)*(9.0*z*z-r*r)/(r*r*r*r*r);
	      else if(m==4)
		SH = (3.0/16.0)*sqrt(385.0/pi)*(4.0*(x*x*x*x-y*y*y*y)-3.0*p*p*p*p)*z/(r*r*r*r*r);
	      else if(m==5)
		SH = (3.0*sqrt(2.0)/32.0)*sqrt(77.0/pi)*(4.0*x*x*x*x*x + 8.0*x*y*y*y*y -4.0*x*x*x*y*y -3.0*x*p*p*p*p)/(r*r*r*r*r);
	      else{
		printf("incorrect l: %d,m: %d\n",l,m);  
		exit(1);
	      }			 
	    }
	  else if(l==6)
	    {
	      p = sqrt(x*x+y*y);		  		 
	      if(m==-6)
		SH = (sqrt(2.0*3003.0/pi)/64.0)*(12.0*pow(x,5)*y+12.0*x*pow(y,5) - 8.0*x*x*x*y*y*y-6.0*x*y*pow(p,4))/(r*r*r*r*r*r);
	      else if(m==-5)
		SH = (3.0/32.0)*sqrt(2.0*1001.0/pi)*(8.0*pow(x,4)*y - 4.0*x*x*y*y*y + 4.0*pow(y,5) -3.0*y*pow(p,4))*z/(r*r*r*r*r*r);
	      else if(m==-4)
		SH = (3.0/32.0)*sqrt(91.0/pi)*(4.0*x*x*x*y -4.0*x*y*y*y)*(11.0*z*z-r*r)/(r*r*r*r*r*r);
	      else if(m==-3)
		SH = (sqrt(2.0*1365.0/pi)/32.0)*(-4.0*y*y*y + 3.0*y*p*p)*(11.0*z*z*z - 3.0*z*r*r)/(r*r*r*r*r*r);
	      else if(m==-2)
		SH = (sqrt(2.0*1365/pi)/64.0)*(2.0*x*y)*(33.0*pow(z,4)-18.0*z*z*r*r + pow(r,4))/(r*r*r*r*r*r);
	      else if(m==-1)
		SH = (sqrt(273.0/pi)/16.0)*y*(33.0*pow(z,5)-30.0*z*z*z*r*r +5.0*z*pow(r,4))/(r*r*r*r*r*r);
	      else if(m==0)
		SH = (sqrt(13.0/pi)/32.0)*(231.0*pow(z,6)-315*pow(z,4)*r*r + 105.0*z*z*pow(r,4) -5.0*pow(r,6))/(r*r*r*r*r*r);
	      else if(m==1)
		SH = (sqrt(273.0/pi)/16.0)*x*(33.0*pow(z,5)-30.0*z*z*z*r*r +5*z*pow(r,4))/(r*r*r*r*r*r);
	      else if(m==2)
		SH = (sqrt(2.0*1365/pi)/64.0)*(x*x-y*y)*(33.0*pow(z,4) - 18.0*z*z*r*r + pow(r,4))/(r*r*r*r*r*r);
	      else if(m==3)
		SH = (sqrt(2.0*1365.0/pi)/32.0)*(4.0*x*x*x -3.0*x*p*p)*(11.0*z*z*z - 3.0*z*r*r)/(r*r*r*r*r*r);
	      else if(m==4)
		SH = (3.0/32.0)*sqrt(91.0/pi)*(4.0*pow(x,4)+4.0*pow(y,4) -3.0*pow(p,4))*(11.0*z*z -r*r)/(r*r*r*r*r*r);
	      else if(m==5)
		SH = (3.0/32.0)*sqrt(2.0*1001.0/pi)*(4.0*pow(x,5) + 8.0*x*pow(y,4)-4.0*x*x*x*y*y-3.0*x*pow(p,4))*z/(r*r*r*r*r*r);
	      else if(m==6)
		SH = (sqrt(2.0*3003.0/pi)/64.0)*(4.0*pow(x,6)-4.0*pow(y,6) +12.0*x*x*pow(y,4)-12.0*pow(x,4)*y*y + 3.0*y*y*pow(p,4)-3.0*x*x*pow(p,4))/(r*r*r*r*r*r);
	      else{
		printf("incorrect l: %d,m: %d\n",l,m);  
		exit(1);
	      }			 
	    } 
	  else
	    {
	      printf("Only l=0,1,2,3,4,5,6 supported. Input l:%d\n",l);
	      exit(1);
	    }
		   
	}
    } 
  return SH;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//  SphericalHarmonic_Derivatives: returns the real spherical harmonic and their derivatives //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar SphericalHarmonic_Derivatives(PetscScalar x,PetscScalar y,PetscScalar z,int l,int m,PetscScalar rc, PetscScalar *Dx,PetscScalar *Dy,PetscScalar *Dz)
{

  /*
   * only l=0,1,2,3 implemented for now
   */
  PetscScalar r = sqrt(x*x + y*y + z*z);                         
  PetscScalar val=0.0;
  PetscScalar c1,c2,c3,c4,der1=0.0,der2=0.0,der3=0.0;
	   
  if(r<=rc)
    {
      if(r!=0.0)
	{
	  if(l==0)
	    {
	      val = 1.0/(2.0*sqrt(M_PI));
	      der1=0.0;
	      der2=0.0;
	      der3=0.0;
	    }
	  else if(l==1)
	    {
	      c1=sqrt(3.0/M_PI);
	      if(m==-1)
		{
		  val = (c1*y)/(2.0*r);
		  der1 = -(c1*x*y)/(2.0*r*r*r);
		  der2 = (c1*(r*r - y*y))/(2.0*r*r*r);
		  der3 = -(c1*y*z)/(2.0*r*r*r);
		}
	      else if(m==0)
		{
		  val = (c1*z)/(2.0*r);
		  der1 = -(c1*x*z)/(2.0*r*r*r);
		  der2 = -(c1*y*z)/(2.0*r*r*r);
		  der3 = (c1*(r*r - z*z))/(2.0*r*r*r);
		}
	      else if(m==1)
		{
		  val = (c1*x)/(2.0*r);
		  der1 = (c1*(r*r - x*x))/(2.0*r*r*r);
		  der2 = -(c1*x*y)/(2.0*r*r*r);
		  der3 = -(c1*x*z)/(2.0*r*r*r);
		}
	    }
	  else if(l==2)
	    {
	      c1=sqrt(15.0/M_PI);
	      c2=sqrt(5.0/M_PI);
  
	      if(m==-2)
		{
		  val = (c1*x*y)/(2.0*r*r);
		  der1 = (c1*(r*r - 2*x*x)*y)/(2.0*r*r*r*r);
		  der2 = (c1*x*(r*r - 2*y*y))/(2.0*r*r*r*r);
		  der3 = -((c1*x*y*z)/(r*r*r*r));
		}
	      else if(m==-1)
		{
		  val = (c1*y*z)/(2.0*r*r);
		  der1 = -((c1*x*y*z)/(r*r*r*r));
		  der2 = (c1*(r*r - 2*y*y)*z)/(2.0*r*r*r*r);
		  der3 = (c1*y*(r*r - 2*z*z))/(2.0*r*r*r*r);
		}
	      else if(m==0)
		{
		  val = (c2*(-1.0 + (3.0*z*z)/(r*r)))/4.0;		  
		  der1 = (-3.0*c2*x*z*z)/(2.0*r*r*r*r);
		  der2 = (-3.0*c2*y*z*z)/(2.0*r*r*r*r);	      
		  der3 = (3.0*c2*z*(r*r - z*z))/(2.0*r*r*r*r);
		}
	      else if(m==1)
		{
		  val = (c1*x*z)/(2.0*r*r);
		  der1 = (c1*(r*r - 2.0*x*x)*z)/(2.0*r*r*r*r);
		  der2 = -((c1*x*y*z)/(r*r*r*r));
		  der3 = (c1*x*(r*r - 2.0*z*z))/(2.0*r*r*r*r);
		}
	      else if(m==2)
		{
		  val = (c1*(x*x - y*y))/(4.0*r*r);
		  der1 = (c1*x*(r*r - x*x + y*y))/(2.0*r*r*r*r);
		  der2 = -(c1*y*(r*r + x*x - y*y))/(2.0*r*r*r*r);
		  der3 = -(c1*(x*x - y*y)*z)/(2.0*r*r*r*r);
		}
	    }
	  else if(l==3)
	    {
	      c1=sqrt(35/(2*M_PI));
	      c2=sqrt(105/M_PI);
	      c3=sqrt(21/(2*M_PI));
	      c4=sqrt(7/M_PI);
  
	      if(m==-3)
		{
		  val = (c1*(3*x*x*y - y*y*y))/(4.0*r*r*r);
		  der1 = (3*c1*x*y*(2*r*r - 3*x*x + y*y))/(4.0*r*r*r*r*r);
		  der2 = (3*c1*(-3*x*x*y*y + y*y*y*y + r*r*(x*x - y*y)))/(4.0*r*r*r*r*r);
		  der3 = (-3*c1*(3*x*x*y - y*y*y)*z)/(4.0*r*r*r*r*r);
		}
	      else if(m==-2)
		{
		  val = (c2*x*y*z)/(2*r*r*r);
		  der1 = (c2*(r*r - 3*x*x)*y*z)/(2.0*r*r*r*r*r);
		  der2 = (c2*x*(r*r - 3*y*y)*z)/(2.0*r*r*r*r*r);
		  der3 = (c2*x*y*(r*r - 3*z*z))/(2.0*r*r*r*r*r);
		}
	      else if(m==-1)
		{
		  val = -(c3*y*(r*r - 5*z*z))/(4.0*r*r*r*r*r);
		  der1 = (c3*x*y*(r*r - 15*z*z))/(4.0*r*r*r*r*r);
		  der2 = -(c3*(r*r*r*r + 15*y*y*z*z - r*r*(y*y + 5*z*z)))/(4.0*r*r*r*r*r);
		  der3 = (c3*y*z*(11*r*r - 15*z*z))/(4.0*r*r*r*r*r);
		}
	      else if(m==0)
		{
		  val = (c4*(-3*r*r*z + 5*z*z*z))/(4.0*r*r*r);
		  der1 = (3*c4*x*z*(r*r - 5*z*z))/(4.0*r*r*r*r*r);
		  der2 = (3*c4*y*z*(r*r - 5*z*z))/(4.0*r*r*r*r*r);
		  der3 = (-3*c4*(r*r*r*r - 6*r*r*z*z + 5*z*z*z*z))/(4.0*r*r*r*r*r);
		}
	      else if(m==1)
		{
		  val = -(c3*x*(r*r - 5*z*z))/(4.0*r*r*r);
		  der1 = -(c3*(r*r*r*r + 15*x*x*z*z - r*r*(x*x + 5*z*z)))/(4.0*r*r*r*r*r);
		  der2 = (c3*x*y*(r*r - 15*z*z))/(4.0*r*r*r*r*r);
		  der3 = (c3*x*z*(11*r*r - 15*z*z))/(4.0*r*r*r*r*r);
		}
	      else if(m==2)
		{
		  val = (c2*(x*x - y*y)*z)/(4.0*r*r*r);
		  der1 = (c2*x*(2*r*r - 3*x*x + 3*y*y)*z)/(4.0*r*r*r*r*r);
		  der2 = -(c2*y*(2*r*r + 3*x*x - 3*y*y)*z)/(4.0*r*r*r*r*r);
		  der3 = (c2*(x*x - y*y)*(r*r - 3*z*z))/(4.0*r*r*r*r*r);
		}
	      else if(m==3)
		{
		  val = (c1*(x*x*x - 3*x*y*y))/(4.0*r*r*r);
		  der1 = (3*c1*(-x*x*x*x + 3*x*x*y*y + r*r*(x*x - y*y)))/(4.0*r*r*r*r*r);
		  der2 = (-3*c1*x*y*(2*r*r + x*x - 3*y*y))/(4.0*r*r*r*r*r);
		  der3 = (-3*c1*(x*x*x - 3*x*y*y)*z)/(4.0*r*r*r*r*r);
		}

	    }
	  else{
	    printf("Only l=0,1,2,3,supported. Input l:%d\n",l);
	    exit(1);
	  }
	}
    } 
  *Dx = der1; *Dy = der2; *Dz = der3;
  return val;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                ComplexSphericalHarmonic: returns the complex spherical harmonics          //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar ComplexSphericalHarmonic(PetscScalar x,PetscScalar y,PetscScalar z,int l,int m,PetscScalar rc, PetscScalar* SHreal,PetscScalar* SHimag)
{
  if(m<0)
    {
      *SHreal = (1.0/sqrt(2.0))*SphericalHarmonic(x,y,z,l,abs(m),rc);
      *SHimag = -(1.0/sqrt(2.0))*SphericalHarmonic(x,y,z,l,-abs(m),rc);
    }
  else if(m==0)
    {
      *SHreal = SphericalHarmonic(x,y,z,l,0,rc);
      *SHimag = 0.0;
    }
  else if(m>0)
    {
      *SHreal = (pow(-1.0,m)/sqrt(2.0))*SphericalHarmonic(x,y,z,l,abs(m),rc);
      *SHimag = (pow(-1.0,m)/sqrt(2.0))*SphericalHarmonic(x,y,z,l,-abs(m),rc);
    }
  else{
    printf("incorrect m=%d in complex spherical harmonic \n",m);
  }

  return 0;

}
///////////////////////////////////////////////////////////////////////////////////////////////
//    EstimateNonZerosNonlocalPseudopot: Estimating the number of nonzeros for the           //
//                         nonlocal pseudopotential sparse matrix                            //
///////////////////////////////////////////////////////////////////////////////////////////////
void EstimateNonZerosNonlocalPseudopot(SDDFT_OBJ* pSddft)
{

  PetscScalar *pAtompos;
      
  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0,ctr,index=0;
  PetscScalar x0, y0, z0,cutoffr,max1,max2,max;
    
  PetscInt  i=0,j,k,xi,yj,zk,offset,I,J,K,nzVps,nyVps,nxVps,nzVpsloc,nyVpsloc,nxVpsloc,poscnt,at;
  PetscScalar delta=pSddft->delta,x,y,z,r,rmax;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt o = pSddft->order;
  
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
  
  VecGetArray(pSddft->Atompos,&pAtompos);      
  
  int start,end; 
  PetscInt ***pDiagArray=NULL;
  PetscInt ***pOffDiagArray=NULL;

  PetscMalloc(sizeof(PetscInt**)*lzdim,&pDiagArray);
  PetscMalloc(sizeof(PetscInt**)*lzdim,&pOffDiagArray);
  if(pDiagArray == NULL || pOffDiagArray == NULL)
    {
      printf("memory allocation fail");
      exit(1);
    }
  for(i=0;i<lzdim;i++)
    {
      PetscMalloc(sizeof(PetscInt*)*lydim,&pDiagArray[i]);
      PetscMalloc(sizeof(PetscInt*)*lydim,&pOffDiagArray[i]);
      if(pDiagArray[i] == NULL || pOffDiagArray[i] == NULL)
	{
	  printf("memory allocation fail");
	  exit(1);
	}	    
      for(j=0;j<lydim;j++)
	{
	  PetscMalloc(sizeof(PetscInt)*lxdim,&pDiagArray[i][j]);
	  PetscMalloc(sizeof(PetscInt)*lxdim,&pOffDiagArray[i][j]);	    
	  if(pDiagArray[i][j] == NULL || pOffDiagArray[i][j] == NULL)
	    {
	      printf("memory allocation fail");
	      exit(1);
	    } 
	}      
    }

  /*
   * set number of nonzeroes due to laplacian entries 
   */

  for(k=0;k<lzdim;k++)
    for(j=0;j<lydim;j++)
      for(i=0;i<lxdim;i++)
	{
	  pDiagArray[k][j][i]= 6*o+1;
	  pOffDiagArray[k][j][i]= 6*o+1;
 	      
	}	  
  
  /*
   * set number of nonzeroes due to nonlocal pseudopotential entries 
   */
  for(at=0;at<pSddft->Ntype;at++)
    {              
      max1=  pSddft->psd[at].rc_s > pSddft->psd[at].rc_p ? pSddft->psd[at].rc_s:pSddft->psd[at].rc_p;
      max2=  pSddft->psd[at].rc_d > pSddft->psd[at].rc_f ? pSddft->psd[at].rc_d:pSddft->psd[at].rc_f;
      max =  max1>max2 ? max1:max2;
      cutoffr=max;
      offset = ceil(cutoffr/delta);
		
      start = floor(pSddft->startPos[at]/3);
      end = floor(pSddft->endPos[at]/3);
			
      for(poscnt=start;poscnt<=end;poscnt++)
	{ 
	  xstart=-1; ystart=-1; zstart=-1; xend=-1; yend=-1; zend=-1; overlap=0;
	 
	  x0 = pAtompos[index++];
	  y0 = pAtompos[index++];
	  z0 = pAtompos[index++];
	  
	  xi = (int)((x0 + R_x)/delta + 0.5); 
	  yj = (int)((y0 + R_y)/delta + 0.5);
	  zk = (int)((z0 + R_z)/delta + 0.5);
				  
	  assert ((xi-offset >= 0)&&(xi+offset<n_x));
	  assert ((yj-offset >= 0)&&(yj+offset<n_y));
	  assert ((zk-offset >= 0)&&(zk+offset<n_z));  
				  
	  xs = xi-offset; xl = xi+offset;
	  ys = yj-offset; yl = yj+offset;
	  zs = zk-offset; zl = zk+offset;
		 
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
	      
	      nzVps = zl-zs+1;
	      nyVps = yl-ys+1;
	      nxVps = xl-xs+1;
	     
	      nzVpsloc = zend-zstart+1;
	      nyVpsloc = yend-ystart+1;
	      nxVpsloc = xend-xstart+1;					    
	          		                                                                                  
	      for(k=zstart;k<=zend;k++)
		for(j=ystart;j<=yend;j++)
		  for(i=xstart;i<=xend;i++)
		    {
		      K = k-zcor;
		      J = j-ycor;
		      I = i-xcor;   
                                         
		      pDiagArray[K][J][I] =  pDiagArray[K][J][I] + nxVpsloc*nyVpsloc*nzVpsloc;
		      pOffDiagArray[K][J][I] =  pOffDiagArray[K][J][I] + (nxVps*nyVps*nzVps - nxVpsloc*nyVpsloc*nzVpsloc);

		    }    					
	    }
	}
    
    }
  VecRestoreArray(pSddft->Atompos,&pAtompos); 	               

  /*
   * convert 3D array to 1D array
   */
  ctr=0;
  for(k=0;k<lzdim;k++)
    for(j=0;j<lydim;j++)
      for(i=0;i<lxdim;i++)
	{
	  pSddft->nnzDArray[ctr] = pDiagArray[k][j][i];
	  pSddft->nnzODArray[ctr] = pOffDiagArray[k][j][i];                                         
	  ctr++;      
	}

  /*
   * free memory    
   */
  for(i=0;i<lzdim;i++)
    {
      for(j=0;j<lydim;j++)
	{
	  PetscFree(pDiagArray[i][j]);  
	  PetscFree(pOffDiagArray[i][j]);                  
	}
      PetscFree(pDiagArray[i]);
      PetscFree(pOffDiagArray[i]);
    }  
  PetscFree(pDiagArray);
  PetscFree(pOffDiagArray);        
                
		
  return;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//  PeriodicEstimateNonZerosNonlocalPseudopot: Estimating the number of nonzeros for the     //
//                         nonlocal pseudopotential sparse matrix                            //
///////////////////////////////////////////////////////////////////////////////////////////////
void PeriodicEstimateNonZerosNonlocalPseudopot(SDDFT_OBJ* pSddft)
{

  PetscScalar *pAtompos;
      
  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0,ctr,index=0;
  PetscScalar x0, y0, z0,X0,Y0,Z0,cutoffr,max1,max2,max;
    
  PetscInt  i=0,j,k,xi,yj,zk,offset,I,J,K,nzVps,nyVps,nxVps,nzVpsloc,nyVpsloc,nxVpsloc,poscnt,at;
  PetscScalar delta=pSddft->delta,x,y,z,r,rmax;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt o = pSddft->order;
  
  PetscInt PP,QQ,RR;
  PetscInt Imax_x,Imin_x,Imax_y,Imin_y,Imax_z,Imin_z;

  PetscMPIInt comm_size;
  MPI_Comm_size(PETSC_COMM_WORLD,&comm_size);

  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
  
  VecGetArray(pSddft->Atompos,&pAtompos);           
  
  int start,end;
  PetscInt ***pDiagArray=NULL;
  PetscInt ***pOffDiagArray=NULL;

  PetscMalloc(sizeof(PetscInt**)*lzdim,&pDiagArray);
  PetscMalloc(sizeof(PetscInt**)*lzdim,&pOffDiagArray);
  if(pDiagArray == NULL || pOffDiagArray == NULL)
    {
      printf("memory allocation fail");
      exit(1);
    }
  for(i=0;i<lzdim;i++)
    {
      PetscMalloc(sizeof(PetscInt*)*lydim,&pDiagArray[i]);
      PetscMalloc(sizeof(PetscInt*)*lydim,&pOffDiagArray[i]);
      if(pDiagArray[i] == NULL || pOffDiagArray[i] == NULL)
	{
	  printf("memory allocation fail");
	  exit(1);
	}	    
      for(j=0;j<lydim;j++)
	{
	  PetscMalloc(sizeof(PetscInt)*lxdim,&pDiagArray[i][j]);
	  PetscMalloc(sizeof(PetscInt)*lxdim,&pOffDiagArray[i][j]);	    
	  if(pDiagArray[i][j] == NULL || pOffDiagArray[i][j] == NULL)
	    {
	      printf("memory allocation fail");
	      exit(1);
	    } 
	}      
    }

  /*
   * set number of nonzeroes due to laplacian entries 
   */
  for(k=0;k<lzdim;k++)
    for(j=0;j<lydim;j++)
      for(i=0;i<lxdim;i++)
	{
	  pDiagArray[k][j][i]= 6*o+1;
	  pOffDiagArray[k][j][i]= 6*o+1; 	      
	}	  
  /*
   * set number of nonzeroes due to nonlocal pseudopotential entries 
   */
  for(at=0;at<pSddft->Ntype;at++)
    {              
      max1=  pSddft->psd[at].rc_s > pSddft->psd[at].rc_p ? pSddft->psd[at].rc_s:pSddft->psd[at].rc_p;
      max2=  pSddft->psd[at].rc_d > pSddft->psd[at].rc_f ? pSddft->psd[at].rc_d:pSddft->psd[at].rc_f;
      max =  max1>max2 ? max1:max2;           
      cutoffr=max;          
      offset = ceil(cutoffr/delta);
		
      start = floor(pSddft->startPos[at]/3);
      end = floor(pSddft->endPos[at]/3);
			
      for(poscnt=start;poscnt<=end;poscnt++)
	{ 		  
	  X0 = pAtompos[index++];
	  Y0 = pAtompos[index++];
	  Z0 = pAtompos[index++];
		  
	  xi = roundf((X0 + R_x)/delta); 
	  yj = roundf((Y0 + R_y)/delta);
	  zk = roundf((Z0 + R_z)/delta);
	  
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
		  xstart=-1000; ystart=-1000; zstart=-1000; xend=-1000; yend=-1000; zend=-1000; overlap=0;
		    
		  x0 = X0 + PP*2.0*R_z;
		  y0 = Y0 + QQ*2.0*R_y;
		  z0 = Z0 + RR*2.0*R_x;
		    
		  xi = roundf((x0 + R_x)/delta); 
		  yj = roundf((y0 + R_y)/delta);
		  zk = roundf((z0 + R_z)/delta);		      

		  xs = xi-offset; xl = xi+offset;
		  ys = yj-offset; yl = yj+offset;
		  zs = zk-offset; zl = zk+offset;				  
			
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
			 
		  if((xstart!=-1000)&&(xend!=-1000)&&(ystart!=-1000)&&(yend!=-1000)&&(zstart!=-1000)&&(zend!=-1000))
		    overlap =1;  
			  
		  if(overlap)
		    {
		      nzVps = zl-zs+1;
		      nyVps = yl-ys+1;
		      nxVps = xl-xs+1;
			
		      nzVpsloc = zend-zstart+1;
		      nyVpsloc = yend-ystart+1;
		      nxVpsloc = xend-xstart+1;					    
          		                                                                                  
		      for(k=zstart;k<=zend;k++)
			for(j=ystart;j<=yend;j++)
			  for(i=xstart;i<=xend;i++)
			    {
			      K = k-zcor;
			      J = j-ycor;
			      I = i-xcor;                                           
			      pDiagArray[K][J][I] =  pDiagArray[K][J][I] + nxVpsloc*nyVpsloc*nzVpsloc;
			      pOffDiagArray[K][J][I] =  pOffDiagArray[K][J][I] + (nxVps*nyVps*nzVps - nxVpsloc*nyVpsloc*nzVpsloc);	       
			    }                            
				
		    }
		}
	}
    
    }
  VecRestoreArray(pSddft->Atompos,&pAtompos); 	               
                
  /*
   * convert 3D array to 1D array
   */ 
  ctr=0;
  for(k=0;k<lzdim;k++)
    for(j=0;j<lydim;j++)
      for(i=0;i<lxdim;i++)
	{
	  pSddft->nnzDArray[ctr] = pDiagArray[k][j][i];
	  pSddft->nnzODArray[ctr] = pOffDiagArray[k][j][i];
		     		     
	  if(pSddft->nnzDArray[ctr]+pSddft->nnzODArray[ctr] > n_x*n_y*n_z )
	    {
	      pSddft->nnzDArray[ctr]= n_x*n_y*n_z/comm_size;
	      pSddft->nnzODArray[ctr]= (comm_size-1)*n_x*n_y*n_z/comm_size;			 
	    }
	  ctr++;		    
	}                
		  
  /*
   * free memory    
   */
  for(i=0;i<lzdim;i++)
    {
      for(j=0;j<lydim;j++)
	{
	  PetscFree(pDiagArray[i][j]);  
	  PetscFree(pOffDiagArray[i][j]);                  
	}
      PetscFree(pDiagArray[i]);
      PetscFree(pOffDiagArray[i]);
    }  
  PetscFree(pDiagArray);
  PetscFree(pOffDiagArray);    		
  return;
}

