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
  | file name: kPointHamiltonian.cc          
  |
  | Description: This file contains the functions required for calculation of nonlocal
  | pseudopotential part of the Hamiltonian matrix for k-point sampling code
  | 
  | Authors: Swarnava Ghosh, Phanish Suryanarayana
  |
  | Last Modified: 2/24/2016   
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "petscsys.h"
#include "math.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
///////////////////////////////////////////////////////////////////////////////////////////////
//            kPointHamiltonian_MatCreate: creates -(1/2)laplacian+Vnonloc operator          //
//                                 but does not store the entries                            //  
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode kPointHamiltonian_MatCreate(SDDFT_OBJ* pSddft)
{
 
  /*
   * create gamma point hamiltonian first to get the exact non-zero positions of the hamiltonian
   */ 
  PeriodicLaplacianNonlocalPseudopotential_MatInit(pSddft); 
  /*
   * duplicate the gamma point Hamiltonian to create the real Hamiltonian matrix for k-point
   */
  MatDuplicate(pSddft->HamiltonianOpr,MAT_SHARE_NONZERO_PATTERN,&pSddft->HamiltonianOpr1);
  /* 
   * duplicate the gamma point Hamiltonian to create the imaginary Hamiltonian matrix for k-point
   */
  MatDuplicate(pSddft->HamiltonianOpr,MAT_SHARE_NONZERO_PATTERN,&pSddft->HamiltonianOpr2);
  /*
   * destroy the created gamma point Hamiltonian
   */
  MatDestroy(&pSddft->HamiltonianOpr); 
  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//  LaplacianNonlocalPseudopotential_MatInit: creates -(1/2)laplacian+Vnonloc operator       //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode kPointHamiltonian_MatInit(SDDFT_OBJ* pSddft,PetscScalar k1,PetscScalar k2,PetscScalar k3)
{
  /*
   * for a given k-point (k1,k2,k3), put the entries of HamiltonianOpr1 and HamiltonianOpr2
   *  1) Real: HamiltonianOpr1 = -0.5*Laplacian operator + Real part of Nonlocal operator
   *  2) Imag: HamiltonianOpr2 = -k.Gradient operator + Imaginary part of Nonlocal operator
   */

  /*
   * First set entries to zero
   */  
  MatScale(pSddft->HamiltonianOpr1,0.0);
  MatScale(pSddft->HamiltonianOpr2,0.0);
  
  /*
   * declare variables
   */
  PetscScalar *pAtompos;
  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0,colidx,colidx_real,colidx_imag;
  PetscScalar x0, y0, z0,X0,Y0,Z0,cutoffr,max1,max2,max,X,Y,Z,XX,YY,ZZ;
  PetscInt xm,ym,zm,starts[3],dims[3];
  int start,end,lmax,lloc;  
  PetscScalar tableR[MAX_TABLE_SIZE],tableUlDeltaV[4][MAX_TABLE_SIZE];
  PetscScalar tableU[4][MAX_TABLE_SIZE];

  PetscScalar pUlDeltaVl1,pUlDeltaVl2,r1,r2;
  PetscScalar *Dexact=NULL;  
  PetscScalar **YDUlDeltaV;
  PetscScalar **YDU;
  PetscErrorCode ierr;
  PetscScalar SHreal_row,SHimag_row,SHreal_col,SHimag_col,UlDelVl,Ul,DUlDelVl,DUl;

  PetscScalar PlmJ_row,PlmJ_col,QlmJ_row,QlmJ_col,Tlmk_row,Tlmk_col,Slmk_row,Slmk_col;
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

  PetscInt PP,QQ,RR;
  PetscInt Imax_x,Imin_x,Imax_y,Imin_y,Imax_z,Imin_z;
  MatStencil row;
  PetscScalar *val_real,*val_imag;  
  int count;
  PetscMPIInt comm_size;
    
  PetscInt s;PetscScalar temp;
  max=0;
  MPI_Comm_size(PETSC_COMM_WORLD,&comm_size);
  ISLocalToGlobalMapping ltog;
 
  AO aodmda;
  DMDAGetAO(pSddft->da,&aodmda); 
  PetscInt LIrow,*LIcol_real,*LIcol_imag;
        
  DMDAGetCorners(pSddft->da,0,0,0,&xm,&ym,&zm);
   
  DMGetLocalToGlobalMapping(pSddft->da,&ltog);

  MatSetFromOptions(pSddft->HamiltonianOpr1);
  MatSetLocalToGlobalMapping(pSddft->HamiltonianOpr1,ltog,ltog);

  DMDAGetGhostCorners(pSddft->da,&starts[0],&starts[1],&starts[2],&dims[0],&dims[1],&dims[2]);
  MatSetStencil(pSddft->HamiltonianOpr1,3,dims,starts,1);
  MatSetUp(pSddft->HamiltonianOpr1);        

  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
   
  /*
   * put entries of -(1/2)laplacian in HamiltonianOpr1
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
	  MatSetValuesStencil(pSddft->HamiltonianOpr1,1,&Laprow,6*o+1,Lapcol,Lapval,ADD_VALUES);
	}

  ierr = MatAssemblyBegin(pSddft->HamiltonianOpr1,MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pSddft->HamiltonianOpr1,MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);

  PetscFree(Lapcol);
  PetscFree(Lapval);  
 
  /*
   * insert -k.Gradient operator in HamiltonianOpr2
   */
    
  MatAXPY(pSddft->HamiltonianOpr2,-k1,pSddft->gradient_x,SUBSET_NONZERO_PATTERN);
  MatAXPY(pSddft->HamiltonianOpr2,-k2,pSddft->gradient_y,SUBSET_NONZERO_PATTERN);
  MatAXPY(pSddft->HamiltonianOpr2,-k3,pSddft->gradient_z,SUBSET_NONZERO_PATTERN);
      
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
		  Dexact[l]= (Dexact[l] - 0.5*UlDelVl*Ul*(rtemp-dr)*(rtemp-dr)*dr)/(delta*delta*delta); // weight of last point is 0.5
			    		  
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
			              
			  PetscMalloc(sizeof(PetscScalar)*(nzVps*nyVps*nxVps),&val_real);
			  PetscMalloc(sizeof(PetscScalar)*(nzVps*nyVps*nxVps),&val_imag);
			  PetscMalloc(sizeof(PetscInt)*(nzVps*nyVps*nxVps),&LIcol_real);
			  PetscMalloc(sizeof(PetscInt)*(nzVps*nyVps*nxVps),&LIcol_imag);
						  
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

				  X = x - PP*2.0*R_x;
				  Y = y - QQ*2.0*R_y;
				  Z = z - RR*2.0*R_z;

				  colidx_real=0;
				  colidx_imag=0;
			
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

					  /*
					   * find the coordinates of the unshifted nodes
					   */
					  XX = xx - PP*2.0*R_x;
					  YY = yy - QQ*2.0*R_y;
					  ZZ = zz - RR*2.0*R_z;				
		                
					  val_real[colidx_real] = 0;
					  val_imag[colidx_imag] = 0;

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
						      ComplexSphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut,&SHreal_row,&SHimag_row);
						      ComplexSphericalHarmonic(xx-x0,yy-y0,zz-z0,l,m,Rcut,&SHreal_col,&SHimag_col);

						      PlmJ_row = pUlDeltaVl1*SHreal_row;
						      QlmJ_row = pUlDeltaVl1*SHimag_row;
				     
						      PlmJ_col = pUlDeltaVl2*SHreal_col;
						      QlmJ_col = pUlDeltaVl2*SHimag_col;

						      Tlmk_row = cos(k1*X+k2*Y+k3*Z)*PlmJ_row + sin(k1*X+k2*Y+k3*Z)*QlmJ_row;
						      Slmk_row = cos(k1*X+k2*Y+k3*Z)*QlmJ_row - sin(k1*X+k2*Y+k3*Z)*PlmJ_row;
				     
						      Tlmk_col = cos(k1*XX+k2*YY+k3*ZZ)*PlmJ_col + sin(k1*XX+k2*YY+k3*ZZ)*QlmJ_col;
						      Slmk_col = cos(k1*XX+k2*YY+k3*ZZ)*QlmJ_col - sin(k1*XX+k2*YY+k3*ZZ)*PlmJ_col;
			     
						      val_real[colidx_real] += (Tlmk_row*Tlmk_col + Slmk_row*Slmk_col)/Dexact[l];
						      val_imag[colidx_imag] += (Slmk_row*Tlmk_col - Tlmk_row*Slmk_col)/Dexact[l];
						    }					   
						}
					    }				 
				 
					  /*
					   * map nodes from outside the computational domain to inside the computational domain 
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
					  LIcol_real[colidx_real] = roundf((ZZ+R_z)/delta)*n_x*n_y + roundf((YY+R_y)/delta)*n_x + roundf((XX+R_x)/delta); 
					  LIcol_imag[colidx_imag] = roundf((ZZ+R_z)/delta)*n_x*n_y + roundf((YY+R_y)/delta)*n_x + roundf((XX+R_x)/delta);
				    
					  /*
					   * only values which are greater than 1e-16 are taken
					   */
					  if(fabs(val_real[colidx_real]) >= 1e-16) 
					    {
					      colidx_real++;		
					    }
					  if(fabs(val_imag[colidx_imag]) >= 1e-16) 
					    {                        
					      colidx_imag++;		
					    }
				
					} 
				  if(colidx_real>0)
				    {
				      AOApplicationToPetsc(aodmda,colidx_real,LIcol_real);
				      ierr=MatSetValues(pSddft->HamiltonianOpr1,1,&LIrow,colidx_real,LIcol_real,val_real,ADD_VALUES);CHKERRQ(ierr);		
				    }
				  if(colidx_imag>0)
				    {
				      AOApplicationToPetsc(aodmda,colidx_imag,LIcol_imag);			
				      ierr=MatSetValues(pSddft->HamiltonianOpr2,1,&LIrow,colidx_imag,LIcol_imag,val_imag,ADD_VALUES);CHKERRQ(ierr);
				    }
			                   
				} 	
			  PetscFree(val_real);
			  PetscFree(LIcol_real);
			  PetscFree(val_imag);
			  PetscFree(LIcol_imag);				
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
  ierr = MatAssemblyBegin(pSddft->HamiltonianOpr1,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pSddft->HamiltonianOpr1,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(pSddft->HamiltonianOpr2,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pSddft->HamiltonianOpr2,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
            
       
  VecRestoreArray(pSddft->Atompos,&pAtompos);  
  return ierr;

}
