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
  | file name: forces.cc          
  |
  | Description: This file contains the functions required for calculation of local component of
  | force, force correction and nonlocal component of force 
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana
  |
  | Last Modified: 2/29/2016   
  |---------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "petscsys.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
///////////////////////////////////////////////////////////////////////////////////////////////
//                      Calculate_force: local component of force on atoms                   // 
///////////////////////////////////////////////////////////////////////////////////////////////
void Calculate_force(SDDFT_OBJ* pSddft)
{ 
  PetscScalar ***PotPhiArrGlbIdx;
  PetscScalar *pAtompos;
  PetscScalar *pForces;
  PetscScalar *pmvAtmConstraint;
  PetscMPIInt comm_size,rank;
  PetscScalar *YD=NULL;

  PetscScalar ForceProcComponentz,ForceProcComponenty,ForceProcComponentx;
  Mat ForceProcMatz,ForceProcMaty,ForceProcMatx;
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  
  MatCreate(PETSC_COMM_WORLD,&ForceProcMatz);
  MatSetSizes(ForceProcMatz,PETSC_DECIDE,PETSC_DECIDE,comm_size,pSddft->nAtoms);
  MatSetType(ForceProcMatz,MATMPIDENSE);
  MatSetUp(ForceProcMatz);
  
  MatCreate(PETSC_COMM_WORLD,&ForceProcMaty);
  MatSetSizes(ForceProcMaty,PETSC_DECIDE,PETSC_DECIDE,comm_size,pSddft->nAtoms);
  MatSetType(ForceProcMaty,MATMPIDENSE);
  MatSetUp(ForceProcMaty);
  
  MatCreate(PETSC_COMM_WORLD,&ForceProcMatx);
  MatSetSizes(ForceProcMatx,PETSC_DECIDE,PETSC_DECIDE,comm_size,pSddft->nAtoms);
  MatSetType(ForceProcMatx,MATMPIDENSE);
  MatSetUp(ForceProcMatx);

  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0;
  PetscInt Xstart,Ystart,Zstart,Xend,Yend,Zend,XBJstart,YBJstart,ZBJstart,XBJend,YBJend,ZBJend;
  PetscScalar x0, y0, z0,coeffs[MAX_ORDER+1],coeffs_grad[MAX_ORDER+1];
  PetscScalar GradBJx,GradBJy,GradBJz,PhimVpsJ,FORCE;
    
  PetscScalar tableR[MAX_TABLE_SIZE],tableVps[MAX_TABLE_SIZE],Bval,noetot=pSddft->noetot;
  PetscScalar ***pVpsArray=NULL;
  PetscScalar ***pBArray=NULL;
  PetscScalar Dtemp;
      
  PetscInt  i=0,j,k,xi,yj,zk,offset,tablesize,l,I,J,K,p,nzVps,nyVps,nxVps,nzBJ,nyBJ,nxBJ,nzGradBJ,nyGradBJ,nxGradBJ,  i0,j0,k0,a,poscnt,index=0,index_force=0,index_mvatm=0,count,at;
  PetscScalar delta=pSddft->delta,x,y,z,r,cutoffr,rmax;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt o = pSddft->order;
  Vec forcex;Vec forcey;Vec forcez; 
  
  VecCreate(PETSC_COMM_WORLD,&forcex);
  VecSetSizes(forcex,PETSC_DECIDE,comm_size);
  VecSetFromOptions(forcex);
   
  VecCreate(PETSC_COMM_WORLD,&forcey);
  VecSetSizes(forcey,PETSC_DECIDE,comm_size);
  VecSetFromOptions(forcey);
   
  VecCreate(PETSC_COMM_WORLD,&forcez);
  VecSetSizes(forcez,PETSC_DECIDE,comm_size);
  VecSetFromOptions(forcez);   

  VecZeroEntries(forcex);
  VecZeroEntries(forcey);
  VecZeroEntries(forcez);
  
  for(p=0;p<=o;p++)
    {
      coeffs[p] = pSddft->coeffs[p]/(2*M_PI);
      coeffs_grad[p] = pSddft->coeffs_grad[p];
    }  
 
  VecGetArray(pSddft->forces,&pForces); 
  VecGetArray(pSddft->mvAtmConstraint,&pmvAtmConstraint); 

  DMDAVecGetArray(pSddft->da,pSddft->potentialPhi,&PotPhiArrGlbIdx);    
     
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
  VecGetArray(pSddft->Atompos,&pAtompos); 
  /*
   * loop over different types of atoms 
   */
  for(at=0;at<pSddft->Ntype;at++)
    {   
      cutoffr=pSddft->CUTOFF[at]+o*delta;
      offset = (PetscInt)ceil(cutoffr/delta + 0.5);
      /*
       * since the pseudopotential table read from the file does not have a value ar r=0,
       * we assume the pseudopotential value at r=0 to be same as the first entry read from
       * the file. 
       */
      tableR[0]=0.0; tableVps[0]=pSddft->psd[at].Vloc[0]; 
      count=1;
      do{
	tableR[count] = pSddft->psd[at].RadialGrid[count-1];
	tableVps[count] = pSddft->psd[at].Vloc[count-1]; 
	count++;

      }while(tableR[count-1] <= 1.732*cutoffr); 
      rmax = tableR[count-1];
      int start,end;
      start = (int)floor(pSddft->startPos[at]/3);
      end = (int)floor(pSddft->endPos[at]/3);
         
      /*
       * derivatives of the spline fit to the pseudopotential
       */
      PetscMalloc(sizeof(PetscScalar)*count,&YD);
      getYD_gen(tableR,tableVps,YD,count);  
   
      /*
       * loop over every atom of a given type
       */
      for(poscnt=start;poscnt<=end;poscnt++)
	{ 
	  x0 = pAtompos[index++];
	  y0 = pAtompos[index++];
	  z0 = pAtompos[index++];

	  if(pmvAtmConstraint[index_mvatm]==1 || pmvAtmConstraint[index_mvatm+1]==1 || pmvAtmConstraint[index_mvatm+2]==1)
	    {      
	      xstart=-1; ystart=-1; zstart=-1; xend=-1; yend=-1; zend=-1; overlap=0;
    
	      xi = (int)((x0 + R_x)/delta + 0.5); 
	      yj = (int)((y0 + R_y)/delta + 0.5);
	      zk = (int)((z0 + R_z)/delta + 0.5);
  
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
	          /*
		   * find the starting and ending indices of finite difference nodes in each 
		   * direction for calculation of pseudopotential and pseudocharge density 
		   */
            
		  Xstart = xstart+o; Xend = xend-o;  
		  Ystart = ystart+o; Yend = yend-o;
		  Zstart = zstart+o; Zend = zend-o;
     
		  XBJstart = xstart; XBJend = xend;
		  YBJstart = ystart; YBJend = yend;
		  ZBJstart = zstart; ZBJend = zend;
        
		  if(xstart == xcor)
		    {
		      Xstart = xstart; XBJstart = xstart-o;
		    }
		  if(xend == xcor+lxdim-1)
		    {
		      Xend = xend; XBJend = xend+o;
		    }
        
		  if(ystart == ycor)
		    {
		      Ystart = ystart; YBJstart = ystart-o;
		    }
		  if(yend == ycor+lydim-1)
		    {
		      Yend = yend; YBJend = yend+o;
		    }
 
		  if(zstart == zcor)
		    {
		      Zstart = zstart; ZBJstart = zstart-o;
		    }
		  if(zend == zcor+lzdim-1)
		    {
		      Zend = zend; ZBJend = zend+o;
		    }

		  nzBJ = ZBJend-ZBJstart+1;
		  nyBJ = YBJend-YBJstart+1;
		  nxBJ = XBJend-XBJstart+1;
        
		  nzVps = nzBJ+o*2;
		  nyVps = nyBJ+o*2;
		  nxVps = nxBJ+o*2;

		  nzGradBJ = Zend-Zstart+1;
		  nyGradBJ = Yend-Ystart+1;
		  nxGradBJ = Xend-Xstart+1;
                		  
		  PetscMalloc(sizeof(PetscScalar**)*nzVps,&pVpsArray);
       
		  if(pVpsArray == NULL)
		    {
		      printf("Memory alocation fail");
		      exit(1);
		    }
		  for(i = 0; i < nzVps; i++)
		    {
		      PetscMalloc(sizeof(PetscScalar*)*nyVps,&pVpsArray[i]);
         
		      if(pVpsArray[i] == NULL)
			{
			  printf("Memory alocation fail");
			  exit(1);
			}
    
		      for(j=0;j<nyVps;j++)
			{
			  PetscMalloc(sizeof(PetscScalar)*nxVps,&pVpsArray[i][j]);
            
			  if(pVpsArray[i][j] == NULL)
			    {
			      printf("Memory alocation fail");
			      exit(1);
			    } 
			}      
		    }
  		  
		  PetscMalloc(sizeof(PetscScalar**)*nzBJ,&pBArray);
       
		  if(pBArray == NULL)
		    {
		      printf("Memory alocation fail");
		      exit(1);
		    }
		  for(i = 0; i < nzBJ; i++)
		    {
		      PetscMalloc(sizeof(PetscScalar*)*nyBJ,&pBArray[i]);
         
		      if(pBArray[i] == NULL)
			{
			  printf("Memory alocation fail");
			  exit(1);
			}
    
		      for(j=0;j<nyBJ;j++)
			{
			  PetscMalloc(sizeof(PetscScalar)*nxBJ,&pBArray[i][j]);
            
			  if(pBArray[i][j] == NULL)
			    {
			      printf("Memory alocation fail");
			      exit(1);
			    } 
			}      
		    }

		  i0 = XBJstart-o;
		  j0 = YBJstart-o;
		  k0 = ZBJstart-o;
		  /* 
		   * evaluate the pseudopotential at nodes in the overlap region + 
		   * finite-difference order in each direction
		   */     
		  for(k=0;k<nzVps;k++) 
		    for(j=0;j<nyVps;j++)  
		      for(i=0;i<nxVps;i++) 
			{   
			  I = i+i0;
			  J = j+j0;
			  K = k+k0;
    
			  x = delta*I - R_x ;
			  y = delta*J - R_y ;
			  z = delta*K - R_z ;   
			  r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
         
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
		   * calculate pseudocharge density at nodes inside the overlap region from
		   * previously calculated pseudopotential values using finite difference stencil  
		   */
		  for(k=ZBJstart;k<=ZBJend;k++)
		    for(j=YBJstart;j<=YBJend;j++)
		      for(i=XBJstart;i<=XBJend;i++)
			{
			  Bval=0;         
			  I=i-i0;
			  J=j-j0;
			  K=k-k0;
                                   
			  Bval=pVpsArray[K][J][I]*coeffs[0];
			  for(a=1;a<=o;a++)
			    {
			      Bval+=(pVpsArray[K][J][I-a] + pVpsArray[K][J][I+a] + pVpsArray[K][J-a][I] +        
				     pVpsArray[K][J+a][I] + pVpsArray[K-a][J][I] + pVpsArray[K+a][J][I])*coeffs[a];
          
			    }
			  pBArray[k-ZBJstart][j-YBJstart][i-XBJstart] = Bval;           
			} 

		  ForceProcComponentz = 0.0;
		  ForceProcComponenty = 0.0;
		  ForceProcComponentx = 0.0;

		  /*
		   * calculate contribution of force	      
		   */
		  for(k=Zstart;k<=Zend;k++) 
		    for(j=Ystart;j<=Yend;j++)  
		      for(i=Xstart;i<=Xend;i++) 
			{  
			  GradBJx=0.0;GradBJy=0.0;GradBJz=0.0;        
			  I=i-i0;
			  J=j-j0;
			  K=k-k0;    
      
			  for(a=1;a<=o;a++)
			    {
			      if(pmvAtmConstraint[index_mvatm+2]==1)
				GradBJz += (pBArray[K-o+a][J-o][I-o] - pBArray[K-o-a][J-o][I-o])*coeffs_grad[a];
			      if(pmvAtmConstraint[index_mvatm+1]==1)
				GradBJy += (pBArray[K-o][J-o+a][I-o] - pBArray[K-o][J-o-a][I-o])*coeffs_grad[a];
			      if(pmvAtmConstraint[index_mvatm]==1)
				GradBJx += (pBArray[K-o][J-o][I-o+a] - pBArray[K-o][J-o][I-o-a])*coeffs_grad[a];            
             
			    }
			  PhimVpsJ = PotPhiArrGlbIdx[k][j][i]-pVpsArray[K][J][I];

			  ForceProcComponentz+=GradBJz*PhimVpsJ;
			  ForceProcComponenty+=GradBJy*PhimVpsJ;
			  ForceProcComponentx+=GradBJx*PhimVpsJ; 

			}         
		  MPI_Comm_rank(MPI_COMM_WORLD,&rank);	 
		  MatSetValue(ForceProcMatz,rank,poscnt,ForceProcComponentz,INSERT_VALUES);	
		  MatSetValue(ForceProcMaty,rank,poscnt,ForceProcComponenty,INSERT_VALUES);
		  MatSetValue(ForceProcMatx,rank,poscnt,ForceProcComponentx,INSERT_VALUES);

		  for(i = 0; i < nzVps; i++)
		    {
		      for(j=0;j<nyVps;j++)
			{
			  PetscFree(pVpsArray[i][j]);                    
			}
		      PetscFree(pVpsArray[i]);
		    }  
		  PetscFree(pVpsArray);   

		  for(i = 0; i < nzBJ; i++)
		    {
		      for(j=0;j<nyBJ;j++)
			{
			  PetscFree(pBArray[i][j]);                    
			}
		      PetscFree(pBArray[i]);
		    }  
		  PetscFree(pBArray); 
       
		}    
	    }
	  index_mvatm = index_mvatm+3;
	}     
      PetscFree(YD); 
    }


  DMDAVecRestoreArray(pSddft->da,pSddft->potentialPhi,&PotPhiArrGlbIdx);
  VecRestoreArray(pSddft->Atompos,&pAtompos);

  MatAssemblyBegin(ForceProcMatx,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ForceProcMatx,MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ForceProcMaty,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ForceProcMaty,MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ForceProcMatz,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ForceProcMatz,MAT_FINAL_ASSEMBLY);	

  /*
   * sum contribution of forces from all processors
   */
  for(poscnt=0;poscnt<pSddft->nAtoms;poscnt++)
    {
      MatGetColumnVector(ForceProcMatx,forcex,poscnt);
      MatGetColumnVector(ForceProcMaty,forcey,poscnt);
      MatGetColumnVector(ForceProcMatz,forcez,poscnt);		
      VecSum(forcex,&FORCE);pForces[index_force++] = delta*delta*delta*FORCE;
      VecSum(forcey,&FORCE);pForces[index_force++] = delta*delta*delta*FORCE;
      VecSum(forcez,&FORCE);pForces[index_force++] = delta*delta*delta*FORCE;		
    }

  VecDestroy(&forcex);
  VecDestroy(&forcey);
  VecDestroy(&forcez);  

  MatDestroy(&ForceProcMatx);
  MatDestroy(&ForceProcMaty);
  MatDestroy(&ForceProcMatz);
  /*
   * adding force correction to forces 
   */   
  for(poscnt=0;poscnt<3*pSddft->nAtoms;poscnt++)
    {
      pForces[poscnt] = pForces[poscnt]+pSddft->pForces_corr[poscnt];         
      VecSetValues(pSddft->forces,1,&poscnt,&pForces[poscnt],INSERT_VALUES);     
        
    }
    
  VecRestoreArray(pSddft->forces,&pForces);
  VecRestoreArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);
  return;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                      Calculate_forceCorrection: force correction on atoms                 //
///////////////////////////////////////////////////////////////////////////////////////////////
void Calculate_forceCorrection(SDDFT_OBJ* pSddft)
{
 
  PetscScalar ***PotPhicArrGlbIdx;
  PetscScalar ***BlcArrGlbIdx;
  PetscScalar ***BlcArrGlbIdx_TM;
  PetscScalar *pmvAtmConstraint;
  PetscMPIInt comm_size,rank;
  PetscScalar *YD=NULL;
  PetscScalar Dtemp;
 
  PetscScalar *pAtompos;  
  PetscScalar ForceProcComponentz,ForceProcComponenty,ForceProcComponentx;
  Mat ForceProcMatz,ForceProcMaty,ForceProcMatx;
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  
  MatCreate(PETSC_COMM_WORLD,&ForceProcMatz);
  MatSetSizes(ForceProcMatz,PETSC_DECIDE,PETSC_DECIDE,comm_size,pSddft->nAtoms);
  MatSetType(ForceProcMatz,MATMPIDENSE);
  MatSetUp(ForceProcMatz);
  
  MatCreate(PETSC_COMM_WORLD,&ForceProcMaty);
  MatSetSizes(ForceProcMaty,PETSC_DECIDE,PETSC_DECIDE,comm_size,pSddft->nAtoms);
  MatSetType(ForceProcMaty,MATMPIDENSE);
  MatSetUp(ForceProcMaty);
  
  MatCreate(PETSC_COMM_WORLD,&ForceProcMatx);
  MatSetSizes(ForceProcMatx,PETSC_DECIDE,PETSC_DECIDE,comm_size,pSddft->nAtoms);
  MatSetType(ForceProcMatx,MATMPIDENSE);
  MatSetUp(ForceProcMatx);


  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0;
  PetscInt Xstart,Ystart,Zstart,Xend,Yend,Zend,XBJstart,YBJstart,ZBJstart,XBJend,YBJend,ZBJend;
  PetscScalar x0, y0, z0,coeffs[MAX_ORDER+1],coeffs_grad[MAX_ORDER+1];
  PetscScalar GradBJx,GradBJy,GradBJz,GradBJx_TM,GradBJy_TM,GradBJz_TM,PhicmVtmJ,PhicpVpsJ,BtmpBps,FORCE_corr;
  PetscScalar GradVJx,GradVJy,GradVJz,GradVJx_TM,GradVJy_TM,GradVJz_TM; //
  
  PetscScalar tableR[MAX_TABLE_SIZE],tableVps[MAX_TABLE_SIZE],Bval,Bval_TM,noetot=pSddft->noetot;
  PetscScalar ***pVpsArray=NULL;
  PetscScalar ***pBArray=NULL;
  PetscScalar ***pVpsArray_TM=NULL;
  PetscScalar ***pBArray_TM=NULL;
      
  PetscInt  i=0,j,k,xi,yj,zk,offset,tablesize,tablesize_TM,l,I,J,K,p,nzVps,nyVps,nxVps,nzBJ,nyBJ,nxBJ,  i0,j0,k0,a,poscnt,index=0,index_force=0,index_mvatm=0,at,count;
  PetscScalar delta=pSddft->delta,x,y,z,r,cutoffr,rmax,rmax_TM,rcut=pSddft->REFERENCE_CUTOFF;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt o = pSddft->order;
  Vec forcex_corr;Vec forcey_corr;Vec forcez_corr; 

  VecCreate(PETSC_COMM_WORLD,&forcex_corr);
  VecSetSizes(forcex_corr,PETSC_DECIDE,comm_size);
  VecSetFromOptions(forcex_corr);
   
  VecCreate(PETSC_COMM_WORLD,&forcey_corr);
  VecSetSizes(forcey_corr,PETSC_DECIDE,comm_size);
  VecSetFromOptions(forcey_corr);
   
  VecCreate(PETSC_COMM_WORLD,&forcez_corr);
  VecSetSizes(forcez_corr,PETSC_DECIDE,comm_size);
  VecSetFromOptions(forcez_corr);   

  VecZeroEntries(forcex_corr);
  VecZeroEntries(forcey_corr);
  VecZeroEntries(forcez_corr);
  
  for(p=0;p<=o;p++)
    {
      coeffs[p] = pSddft->coeffs[p]/(2*M_PI);
      coeffs_grad[p] = pSddft->coeffs_grad[p];
    }  

  PetscMalloc(sizeof(PetscScalar)*3*pSddft->nAtoms, &pSddft->pForces_corr);
  if(pSddft->pForces_corr == NULL)
    {
      printf("Memory alocation fail in pForces_corr");
      exit(1);
    }

  DMDAVecGetArray(pSddft->da,pSddft->Phi_c,&PotPhicArrGlbIdx); 
  DMDAVecGetArray(pSddft->da,pSddft->chrgDensB,&BlcArrGlbIdx);  
  DMDAVecGetArray(pSddft->da,pSddft->chrgDensB_TM,&BlcArrGlbIdx_TM); 
        
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
  
  VecGetArray(pSddft->Atompos,&pAtompos);   
  VecGetArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);     

  for(at=0;at<pSddft->Ntype;at++)
    {   
      cutoffr=pSddft->CUTOFF[at]+o*delta;
      offset = (PetscInt)ceil(cutoffr/delta + 0.5);

      tableR[0]=0.0; tableVps[0]=pSddft->psd[at].Vloc[0];
      count=1;
      do{
	tableR[count] = pSddft->psd[at].RadialGrid[count-1];
	tableVps[count] = pSddft->psd[at].Vloc[count-1];
	count++;

      }while(tableR[count-1] <= 1.732*cutoffr); 
      rmax = tableR[count-1];
      int start,end;
      start = (int)floor(pSddft->startPos[at]/3);
      end = (int)floor(pSddft->endPos[at]/3);
      /*
       * derivatives of the spline fit to the pseudopotential
       */
      PetscMalloc(sizeof(PetscScalar)*count,&YD);
      getYD_gen(tableR,tableVps,YD,count); 
      /*
       * loop over every atom of a given type
       */
      for(poscnt=start;poscnt<=end;poscnt++)
	{ 

	  x0 = pAtompos[index++];
	  y0 = pAtompos[index++];
	  z0 = pAtompos[index++];
      
	  if(pmvAtmConstraint[index_mvatm]==1 || pmvAtmConstraint[index_mvatm+1]==1 || pmvAtmConstraint[index_mvatm+2]==1)
	    {

	      xstart=-1; ystart=-1; zstart=-1; xend=-1; yend=-1; zend=-1; overlap=0;

      
	      xi = (int)((x0 + R_x)/delta + 0.5); 
	      yj = (int)((y0 + R_y)/delta + 0.5);
	      zk = (int)((z0 + R_z)/delta + 0.5);
     
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
		  /*
		   * find the starting and ending indices of finite difference nodes in each 
		   * direction for calculation of pseudopotential and pseudocharge density 
		   */
          
		  Xstart = xstart+o; Xend = xend-o;  
		  Ystart = ystart+o; Yend = yend-o;
		  Zstart = zstart+o; Zend = zend-o;

		  XBJstart = xstart; XBJend = xend;
		  YBJstart = ystart; YBJend = yend;
		  ZBJstart = zstart; ZBJend = zend;        
		  if(xstart == xcor)
		    {
		      Xstart = xstart; XBJstart = xstart-o;
		    }
		  if(xend == xcor+lxdim-1)
		    {
		      Xend = xend; XBJend = xend+o;
		    }        
		  if(ystart == ycor)
		    {
		      Ystart = ystart; YBJstart = ystart-o;
		    }
		  if(yend == ycor+lydim-1)
		    {
		      Yend = yend; YBJend = yend+o;
		    }
 
		  if(zstart == zcor)
		    {
		      Zstart = zstart; ZBJstart = zstart-o;
		    }
		  if(zend == zcor+lzdim-1)
		    {
		      Zend = zend; ZBJend = zend+o;
		    }

		  nzBJ = ZBJend-ZBJstart+1;
		  nyBJ = YBJend-YBJstart+1;
		  nxBJ = XBJend-XBJstart+1;
        
		  nzVps = nzBJ+o*2;
		  nyVps = nyBJ+o*2;
		  nxVps = nxBJ+o*2;
       
		  PetscMalloc(sizeof(PetscScalar**)*nzVps,&pVpsArray);       
		  if(pVpsArray == NULL)
		    {
		      printf("Memory alocation fail");
		      exit(1);
		    }
		  for(i = 0; i < nzVps; i++)
		    {
		      PetscMalloc(sizeof(PetscScalar*)*nyVps,&pVpsArray[i]);         
		      if(pVpsArray[i] == NULL)
			{
			  printf("Memory alocation fail");
			  exit(1);
			}
    
		      for(j=0;j<nyVps;j++)
			{
			  PetscMalloc(sizeof(PetscScalar)*nxVps,&pVpsArray[i][j]);
            
			  if(pVpsArray[i][j] == NULL)
			    {
			      printf("Memory alocation fail");
			      exit(1);
			    } 
			}      
		    }
  
		  PetscMalloc(sizeof(PetscScalar**)*nzBJ,&pBArray);       
		  if(pBArray == NULL)
		    {
		      printf("Memory alocation fail");
		      exit(1);
		    }
		  for(i = 0; i < nzBJ; i++)
		    {
		      PetscMalloc(sizeof(PetscScalar*)*nyBJ,&pBArray[i]);         
		      if(pBArray[i] == NULL)
			{
			  printf("Memory alocation fail");
			  exit(1);
			}
    
		      for(j=0;j<nyBJ;j++)
			{
			  PetscMalloc(sizeof(PetscScalar)*nxBJ,&pBArray[i][j]);            
			  if(pBArray[i][j] == NULL)
			    {
			      printf("Memory alocation fail");
			      exit(1);
			    } 
			}      
		    }
      
		  PetscMalloc(sizeof(PetscScalar**)*nzVps,&pVpsArray_TM);       
		  if(pVpsArray_TM == NULL)
		    {
		      printf("Memory alocation fail");
		      exit(1);
		    }
		  for(i = 0; i < nzVps; i++)
		    {
		      PetscMalloc(sizeof(PetscScalar*)*nyVps,&pVpsArray_TM[i]);
         
		      if(pVpsArray_TM[i] == NULL)
			{
			  printf("Memory alocation fail");
			  exit(1);
			}
    
		      for(j=0;j<nyVps;j++)
			{
			  PetscMalloc(sizeof(PetscScalar)*nxVps,&pVpsArray_TM[i][j]);
            
			  if(pVpsArray_TM[i][j] == NULL)
			    {
			      printf("Memory alocation fail");
			      exit(1);
			    } 
			}      
		    }     

		  PetscMalloc(sizeof(PetscScalar**)*nzBJ,&pBArray_TM);       
		  if(pBArray_TM == NULL)
		    {
		      printf("Memory alocation fail");
		      exit(1);
		    }
		  for(i = 0; i < nzBJ; i++)
		    {
		      PetscMalloc(sizeof(PetscScalar*)*nyBJ,&pBArray_TM[i]);
         
		      if(pBArray_TM[i] == NULL)
			{
			  printf("Memory alocation fail");
			  exit(1);
			}
    
		      for(j=0;j<nyBJ;j++)
			{
			  PetscMalloc(sizeof(PetscScalar)*nxBJ,&pBArray_TM[i][j]);
            
			  if(pBArray_TM[i][j] == NULL)
			    {
			      printf("Memory alocation fail");
			      exit(1);
			    } 
			}      
		    }
               
		  i0 = XBJstart-o;
		  j0 = YBJstart-o;
		  k0 = ZBJstart-o;
      
		  /* 
		   * evaluate the pseudopotential at nodes in the overlap region + 
		   * finite-difference order in each direction
		   */
		  for(k=0;k<nzVps;k++) 
		    for(j=0;j<nyVps;j++)  
		      for(i=0;i<nxVps;i++) 
			{   
			  I = i+i0;
			  J = j+j0;
			  K = k+k0;
    
			  x = delta*I - R_x ;
			  y = delta*J - R_y ;
			  z = delta*K - R_z ;   
			  r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
        
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
			  pVpsArray_TM[k][j][i] = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);          
			}
 
		  /*
		   * calculate pseudocharge density and contribution to self energy at nodes 
		   * inside the overlap region from previously calculated pseudopotential values
		   * using finite difference stencil  
		   */      
		  for(k=ZBJstart;k<=ZBJend;k++)
		    for(j=YBJstart;j<=YBJend;j++)
		      for(i=XBJstart;i<=XBJend;i++)
			{
			  Bval=0;Bval_TM=0;         
			  I=i-i0;
			  J=j-j0;
			  K=k-k0;
                                   
			  Bval=pVpsArray[K][J][I]*coeffs[0];
			  Bval_TM=pVpsArray_TM[K][J][I]*coeffs[0];
			  for(a=1;a<=o;a++)
			    {
			      Bval+=(pVpsArray[K][J][I-a] + pVpsArray[K][J][I+a] + pVpsArray[K][J-a][I] +        
				     pVpsArray[K][J+a][I] + pVpsArray[K-a][J][I] + pVpsArray[K+a][J][I])*coeffs[a];

			      Bval_TM+=(pVpsArray_TM[K][J][I-a] + pVpsArray_TM[K][J][I+a] + pVpsArray_TM[K][J-a][I] +        
					pVpsArray_TM[K][J+a][I] + pVpsArray_TM[K-a][J][I] + pVpsArray_TM[K+a][J][I])*coeffs[a];          
			    }                    
			  pBArray[k-ZBJstart][j-YBJstart][i-XBJstart] = Bval; 
			  pBArray_TM[k-ZBJstart][j-YBJstart][i-XBJstart] = Bval_TM; 

#ifdef _DEBUG
			  printf("Bval:%.16f\n", Bval);
			  printf("Bval_TM:%.16f\n", Bval_TM);
#endif
          
			} 

		  ForceProcComponentz = 0.0;
		  ForceProcComponenty = 0.0;
		  ForceProcComponentx = 0.0;

        
		  /*
		   * calculate contribution of force	      
		   */
		  for(k=Zstart;k<=Zend;k++) 
		    for(j=Ystart;j<=Yend;j++)  
		      for(i=Xstart;i<=Xend;i++) 
			{  
			  GradBJx=0;GradBJy=0;GradBJz=0; 
			  GradBJx_TM=0;GradBJy_TM=0;GradBJz_TM=0; 
			  GradVJx=0,GradVJy=0,GradVJz=0;
			  GradVJx_TM=0,GradVJy_TM=0,GradVJz_TM=0;

			  I=i-i0;
			  J=j-j0;
			  K=k-k0;    
      
			  for(a=1;a<=o;a++)
			    {

			      if(pmvAtmConstraint[index_mvatm+2]==1)
				{
		
				  GradBJz += (pBArray[K-o+a][J-o][I-o] - pBArray[K-o-a][J-o][I-o])*coeffs_grad[a];
				  GradBJz_TM += (pBArray_TM[K-o+a][J-o][I-o] - pBArray_TM[K-o-a][J-o][I-o])*coeffs_grad[a];
				  GradVJz += (pVpsArray[K+a][J][I] - pVpsArray[K-a][J][I])*coeffs_grad[a];
				  GradVJz_TM += (pVpsArray_TM[K+a][J][I] - pVpsArray_TM[K-a][J][I])*coeffs_grad[a];
				}

			      if(pmvAtmConstraint[index_mvatm+1]==1)
				{

				  GradBJy += (pBArray[K-o][J-o+a][I-o] - pBArray[K-o][J-o-a][I-o])*coeffs_grad[a];
				  GradBJy_TM += (pBArray_TM[K-o][J-o+a][I-o] - pBArray_TM[K-o][J-o-a][I-o])*coeffs_grad[a];
				  GradVJy += (pVpsArray[K][J+a][I] - pVpsArray[K][J-a][I])*coeffs_grad[a];      
				  GradVJy_TM += (pVpsArray_TM[K][J+a][I] - pVpsArray_TM[K][J-a][I])*coeffs_grad[a];
				}
  
			      if(pmvAtmConstraint[index_mvatm]==1)
				{
				  GradBJx += (pBArray[K-o][J-o][I-o+a] - pBArray[K-o][J-o][I-o-a])*coeffs_grad[a];                 
				  GradBJx_TM += (pBArray_TM[K-o][J-o][I-o+a] - pBArray_TM[K-o][J-o][I-o-a])*coeffs_grad[a];     
				  GradVJx += (pVpsArray[K][J][I+a] - pVpsArray[K][J][I-a])*coeffs_grad[a];                     
				  GradVJx_TM += (pVpsArray_TM[K][J][I+a] - pVpsArray_TM[K][J][I-a])*coeffs_grad[a];
				}             
			    }

			  PhicmVtmJ= PotPhicArrGlbIdx[k][j][i]- pVpsArray_TM[K][J][I];
			  PhicpVpsJ= PotPhicArrGlbIdx[k][j][i]+ pVpsArray[K][J][I];
			  BtmpBps  = BlcArrGlbIdx_TM[k][j][i] + BlcArrGlbIdx[k][j][i];

			  ForceProcComponentz+= -GradBJz_TM*PhicmVtmJ-GradBJz*PhicpVpsJ-(GradVJz_TM-GradVJz)*BtmpBps
			    -GradVJz*pVpsArray[K][J][I]+GradVJz_TM*pVpsArray_TM[K][J][I];

			  ForceProcComponenty+= -GradBJy_TM*PhicmVtmJ-GradBJy*PhicpVpsJ-(GradVJy_TM-GradVJy)*BtmpBps
			    -GradVJy*pVpsArray[K][J][I]+GradVJy_TM*pVpsArray_TM[K][J][I];

			  ForceProcComponentx+= -GradBJx_TM*PhicmVtmJ-GradBJx*PhicpVpsJ-(GradVJx_TM-GradVJx)*BtmpBps
			    -GradVJx*pVpsArray[K][J][I]+GradVJx_TM*pVpsArray_TM[K][J][I];  
          
			}
       
		  MPI_Comm_rank(MPI_COMM_WORLD,&rank);	 
		  MatSetValue(ForceProcMatz,rank,poscnt,ForceProcComponentz,INSERT_VALUES);	
		  MatSetValue(ForceProcMaty,rank,poscnt,ForceProcComponenty,INSERT_VALUES);
		  MatSetValue(ForceProcMatx,rank,poscnt,ForceProcComponentx,INSERT_VALUES);

		  for(i = 0; i < nzVps; i++)
		    {
		      for(j=0;j<nyVps;j++)
			{
			  PetscFree(pVpsArray[i][j]);                    
			}
		      PetscFree(pVpsArray[i]);
		    }  
		  PetscFree(pVpsArray);   

		  for(i = 0; i < nzBJ; i++)
		    {
		      for(j=0;j<nyBJ;j++)
			{
			  PetscFree(pBArray[i][j]);                    
			}
		      PetscFree(pBArray[i]);
		    }  
		  PetscFree(pBArray); 
       
		} 
   
	    }
	  index_mvatm = index_mvatm+3;
	}  
      PetscFree(YD); 
    }

  DMDAVecRestoreArray(pSddft->da,pSddft->Phi_c,&PotPhicArrGlbIdx);   
  DMDAVecRestoreArray(pSddft->da,pSddft->chrgDensB,&BlcArrGlbIdx);   
  DMDAVecRestoreArray(pSddft->da,pSddft->chrgDensB_TM,&BlcArrGlbIdx_TM); 
  VecRestoreArray(pSddft->Atompos,&pAtompos);  
  VecRestoreArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);

  MatAssemblyBegin(ForceProcMatx,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ForceProcMatx,MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ForceProcMaty,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ForceProcMaty,MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ForceProcMatz,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ForceProcMatz,MAT_FINAL_ASSEMBLY);	
  /*
   * sum contribution of forces from all processors
   */
  for(poscnt=0;poscnt<pSddft->nAtoms;poscnt++)
    {
      MatGetColumnVector(ForceProcMatx,forcex_corr,poscnt);
      MatGetColumnVector(ForceProcMaty,forcey_corr,poscnt);
      MatGetColumnVector(ForceProcMatz,forcez_corr,poscnt);
		
      VecSum(forcex_corr,&FORCE_corr);pSddft->pForces_corr[index_force++] = -0.5*delta*delta*delta*FORCE_corr;
      VecSum(forcey_corr,&FORCE_corr);pSddft->pForces_corr[index_force++] = -0.5*delta*delta*delta*FORCE_corr;
      VecSum(forcez_corr,&FORCE_corr);pSddft->pForces_corr[index_force++] = -0.5*delta*delta*delta*FORCE_corr;
    }

  VecDestroy(&forcex_corr);
  VecDestroy(&forcey_corr);
  VecDestroy(&forcez_corr);  

  MatDestroy(&ForceProcMatx);
  MatDestroy(&ForceProcMaty);
  MatDestroy(&ForceProcMatz);
      
  return;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//                      Force_Nonlocal: nonlocal component of force on atoms                 // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode Force_Nonlocal(SDDFT_OBJ* pSddft,Mat* Psi)
{
  PetscScalar *pAtompos;  
  PetscScalar *pmvAtmConstraint;

  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0,colidx;
  PetscScalar x0, y0, z0,cutoffr,max1,max2,max=0;
  int start,end,lmax,lloc,p,a;  
  PetscScalar tableR[MAX_TABLE_SIZE],tableUlDeltaV[4][MAX_TABLE_SIZE],coeffs_grad[MAX_ORDER+1],tableU[4][MAX_TABLE_SIZE];
      
  
  PetscScalar pUlDeltaVl;
  PetscInt rowStart,rowEnd;
  PetscScalar gi;
  PetscScalar *arrPsiSeq,*arrGradPsiSeq;
 
  PetscScalar ****W=NULL;
  PetscScalar ****T=NULL;

  PetscScalar *Dexact=NULL,Dtemp;      
  PetscScalar **YDUlDeltaV=NULL;
  PetscScalar **YDU=NULL;
  PetscErrorCode ierr;
  PetscScalar SpHarmonic,UlDelVl,Ul,fJ_x,fJ_y,fJ_z;

  PetscScalar Rcut;
  PetscScalar dr=1e-3;

  int OrbitalSize,rank;

  PetscInt  i=0,j,k,xi,yj,zk,offset,I,J,K,II,JJ,KK,nzVps,nyVps,nxVps,nzVpsloc,nyVpsloc,nxVpsloc,i0,j0,k0,poscnt,index=0,at,ii,jj,kk,l,m,index_mvatm=0;
  PetscInt o = pSddft->order;
  PetscScalar delta=pSddft->delta,x,y,z,r,rmax,rtemp;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
 
  int count,coord;
  PetscMPIInt comm_size;
  PetscInt s;PetscScalar temp;
      
  PetscInt xm,ym,zm,starts[3],dims[3];
  ISLocalToGlobalMapping ltog;
  
  AO aodmda1;
  ierr = DMDAGetAO(pSddft->da,&aodmda1);
  PetscInt LIrow;
    
  PetscMalloc(sizeof(PetscScalar***)*pSddft->nAtoms,&W);
  PetscMalloc(sizeof(PetscScalar***)*pSddft->nAtoms,&T);
  if(T == NULL || W == NULL)
    {
      printf("memory allocation failed in T or W for forces");
      exit(1);
    }
  for(i=0;i<pSddft->nAtoms;i++)
    {      
      PetscMalloc(sizeof(PetscScalar**)*pSddft->Nstates,&W[i]);
      PetscMalloc(sizeof(PetscScalar**)*pSddft->Nstates,&T[i]);
      if(T[i] == NULL || W[i] == NULL)
	{
	  printf("memory allocation fail");
	  exit(1);
	}
    }
      
  MatGetOwnershipRange(*Psi,&rowStart,&rowEnd); 
      
  DMDAGetCorners(pSddft->da,0,0,0,&xm,&ym,&zm);
  DMGetLocalToGlobalMapping(pSddft->da,&ltog);
      
  DMDAGetGhostCorners(pSddft->da,&starts[0],&starts[1],&starts[2],&dims[0],&dims[1],&dims[2]);

  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
   
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
 
  PetscScalar OrbScalingFactor;
  PetscReal norms[pSddft->Nstates];
  MatGetColumnNorms(*Psi,NORM_2,norms);

  /*
   * scaling factor for normalization of orbitals
   */
  OrbScalingFactor = 1.0/(sqrt(delta*delta*delta)*norms[3]); 
 
  MatDestroy(&pSddft->YOrbNew);
  ierr = MatDuplicate(*Psi,MAT_SHARE_NONZERO_PATTERN,&pSddft->YOrbNew);
  CHKERRQ(ierr);
 
  VecGetArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);          
  VecGetArray(pSddft->Atompos,&pAtompos);        


  /*
   * Calculate x-component of forces
   */
  index_mvatm=0;index=0;  
  /*
   * Calculate gradient of orbitals in x-direction
   */
  MatMatMultSymbolic(pSddft->gradient_x,*Psi,PETSC_DEFAULT,&pSddft->YOrbNew);
  MatMatMultNumeric(pSddft->gradient_x,*Psi,pSddft->YOrbNew);   
  
  /*
   * loop over different types of atoms 
   */
  for(at=0;at<pSddft->Ntype;at++)
    {                          
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];	            
	   
      start = floor(pSddft->startPos[at]/3);
      end = floor(pSddft->endPos[at]/3);

      /*
       * if the atom is hydrogen, then we omit the calculation of non-local forces
       */
      if(lmax==0) 
	{
	  index = index + 3*(end-start+1); 
	  index_mvatm = index_mvatm + 3*(end-start+1);
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
	  }while(tableR[count-1] <= 1.732*pSddft->CUTOFF[at]+10.0); 
	  rmax = tableR[count-1];
                          
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDUlDeltaV);
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDU);           
	  for(l=0;l<=lmax;l++)
	    {
	      PetscMalloc(sizeof(PetscScalar)*count,&YDUlDeltaV[l]);
	      PetscMalloc(sizeof(PetscScalar)*count,&YDU[l]);  
	      /*
	       * derivatives of the spline fit to the projectors an pseudowavefunctions
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
		      ispline_gen(tableR,tableUlDeltaV[l],count,&rtemp,&UlDelVl,&Dtemp,1,YDUlDeltaV[l]);
		      ispline_gen(tableR,tableU[l],count,&rtemp,&Ul,&Dtemp,1,YDU[l]);
											
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
	     
	      x0 = pAtompos[index++];
	      y0 = pAtompos[index++];
	      z0 = pAtompos[index++];
	      if(pmvAtmConstraint[index_mvatm]==1 || pmvAtmConstraint[index_mvatm+1]==1 || pmvAtmConstraint[index_mvatm+2]==1)
		{  
		  xstart=-1; ystart=-1; zstart=-1; xend=-1; yend=-1; zend=-1; overlap=0;
		 
		  xi = (int)((x0 + R_x)/delta + 0.5); 
		  yj = (int)((y0 + R_y)/delta + 0.5);
		  zk = (int)((z0 + R_z)/delta + 0.5);
		  
		  xs = xi-offset; xl = xi+offset;
		  ys = yj-offset; yl = yj+offset;
		  zs = zk-offset; zl = zk+offset;
     
		  /*
		   * find if domain of influence of pseudopotential overlaps with the domain 
		   * stored by processor  
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
		 	 
		  for(i=0;i<pSddft->Nstates;i++)
		    {				    
		      PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&W[poscnt][i]);
		      PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&T[poscnt][i]);
		      if( W[poscnt][i] == NULL || T[poscnt][i] == NULL )
			{
			  printf("W[][] || T"); exit(1);
			}
			     			    
		      for(l=0;l<=lmax;l++)
			{			
			  PetscMalloc(sizeof(PetscScalar)*(2*l+1),&W[poscnt][i][l]);
			  PetscMalloc(sizeof(PetscScalar)*(2*l+1),&T[poscnt][i][l]);
			  if( W[poscnt][i][l] == NULL || T[poscnt][i][l] == NULL )
			    {
			      printf("W[][][] T[][][]"); exit(1);
			    }		
			  for(m=-l;m<=l;m++)
			    {	     
			      W[poscnt][i][l][m+l]=0.0;
			      T[poscnt][i][l][m+l]=0.0;				      
			    }			
			}
		    }
		    	    	       
		  if(overlap)			 
		    {

		      i0 = xs-o;
		      j0 = ys-o;
		      k0 = zs-o;    							
		      	
		      /*
		       * Get local pointer (column wise) to the wavefunctions and gradient of 
		       * wavefunctions  
		       */
		      MatDenseGetArray(*Psi,&arrPsiSeq);	    
		      MatDenseGetArray(pSddft->YOrbNew,&arrGradPsiSeq);
		     
		      /* 
		       * evaluate the the quantities that contribute to nonlocal force at 
		       * nodes in the overlap region
		       */
		      for(k=zstart;k<=zend;k++)
			for(j=ystart;j<=yend;j++)
			  for(i=xstart;i<=xend;i++)
			    {
			     
			      I=i-i0;
			      J=j-j0;
			      K=k-k0; 
			     
			      x = delta*i-R_x;
			      y = delta*j-R_y;
			      z = delta*k-R_z;
			      r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
			      
			      LIrow = k*n_x*n_y + j*n_x + i; 
			      AOApplicationToPetsc(aodmda1,1,&LIrow);
			      for(l=0;l<=lmax;l++)
				{
				  if(l!=lloc)
				    { 				      
				      if(r == tableR[0])
					{
					  pUlDeltaVl = tableUlDeltaV[l][0];	
					}else{
					ispline_gen(tableR,tableUlDeltaV[l],count,&r,&pUlDeltaVl,&Dtemp,1,YDUlDeltaV[l]);
				      }						 
				      if(l==0)
					Rcut= pSddft->psd[at].rc_s;
				      if(l==1)
					Rcut= pSddft->psd[at].rc_p;
				      if(l==2)
					Rcut= pSddft->psd[at].rc_d;
				      if(l==3)
					Rcut= pSddft->psd[at].rc_f;		      

				      for(m=-l;m<=l;m++)
					{	
					  SpHarmonic=SphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut+1.0);
					  for(ii=0;ii<pSddft->Nstates;ii++)
					    {
					      OrbScalingFactor = 1.0/(sqrt(delta*delta*delta)*norms[ii]);    
					      if(pmvAtmConstraint[index_mvatm]==1)
						W[poscnt][ii][l][m+l] += OrbScalingFactor*(arrGradPsiSeq[LIrow-rowStart + (rowEnd-rowStart)*ii])*pUlDeltaVl*SpHarmonic;	    
					      T[poscnt][ii][l][m+l] += OrbScalingFactor*(arrPsiSeq[LIrow-rowStart + (rowEnd-rowStart)*ii])*pUlDeltaVl*SpHarmonic/Dexact[l];		     
					    }						
					}
				    }
				}
			    }			    
		      MatDenseRestoreArray(*Psi,&arrPsiSeq); 
		      MatDenseRestoreArray(pSddft->YOrbNew,&arrGradPsiSeq);			     
			     
		    }
		}
	      index_mvatm = index_mvatm+3;
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
   * sum the contributions across all the processors and compute the forces
   */     
  for(at=0;at<pSddft->Ntype;at++)
    {
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];
      if(lmax!=0)
	{	 
	  start = (int)floor(pSddft->startPos[at]/3);
	  end = (int)floor(pSddft->endPos[at]/3);
	  for(poscnt=start;poscnt<=end;poscnt++)
	    {	      
	      fJ_x=0.0;
	      for(i=0;i<pSddft->Nstates;i++)
		{  gi = smearing_FermiDirac(pSddft->Beta,pSddft->lambda[i],pSddft->lambda_f);
		  for(l=0;l<=lmax;l++)
		    {
		      if(l!=lloc)
			{
			  for(m=-l;m<=l;m++)
			    {			      
			      ierr =  MPI_Allreduce(MPI_IN_PLACE,&T[poscnt][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
			      ierr =  MPI_Allreduce(MPI_IN_PLACE,&W[poscnt][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);			     
			      fJ_x += gi*W[poscnt][i][l][m+l]*T[poscnt][i][l][m+l];			     
			    }
			}
		    }		  
		}	     	      
	      fJ_x = -4.0*delta*delta*delta*fJ_x;	     
	      VecSetValue(pSddft->forces,3*poscnt,fJ_x,ADD_VALUES);	      
      	    }
	}
    }
  /*
   * end of x-component force calculation 
   */


  /*
   * Calculate y-component of forces
   */

  /* NOTE: we do no need to calculate T since it has already been calculated while 
   * calculating the x-component of forces. Hence we do not need to calculate the exact 
   * denominator. Also, memory for W has already been created, we dont need to allocate again,
   * just need to set it to zero.
   */

  index_mvatm=0;index=0;
  /*
   * Calculate gradient of orbitals in x-direction
   */
  MatMatMultNumeric(pSddft->gradient_y,*Psi,pSddft->YOrbNew);	
  
  /*
   * loop over different types of atoms 
   */
  for(at=0;at<pSddft->Ntype;at++)
    {                          
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];
      start = floor(pSddft->startPos[at]/3);
      end = floor(pSddft->endPos[at]/3);
      /*
       * if the atom is hydrogen, then we omit the calculation of non-local forces
       */
      if(lmax==0)
	{
	  index = index + 3*(end-start+1); 
	  index_mvatm = index_mvatm + 3*(end-start+1);
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
		}
	      if(l==1)
		{
		  tableUlDeltaV[1][0]=0.0; 			     
		}
	      if(l==2)
		{
		  tableUlDeltaV[2][0]=0.0; 				      
		}
	      if(l==3)
		{
		  tableUlDeltaV[3][0]=0.0; 		   	  
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
		  }
		if(l==1)
		  {
		    tableUlDeltaV[1][count]=pSddft->psd[at].Up[count-1]*(pSddft->psd[at].Vp[count-1]-pSddft->psd[at].Vloc[count-1]); 		         	       
		  }
		if(l==2)
		  {
		    tableUlDeltaV[2][count]=pSddft->psd[at].Ud[count-1]*(pSddft->psd[at].Vd[count-1]-pSddft->psd[at].Vloc[count-1]); 		   	       
		  }   
		if(l==3)
		  {
		    tableUlDeltaV[3][count]=pSddft->psd[at].Uf[count-1]*(pSddft->psd[at].Vf[count-1]-pSddft->psd[at].Vloc[count-1]); 		            
		  }  
	      }
	    count++;
	  }while(tableR[count-1] <= 1.732*pSddft->CUTOFF[at]+10.0); 
	  rmax = tableR[count-1];                             
	                                 
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDUlDeltaV);	  
	  for(l=0;l<=lmax;l++)
	    {
	      /*
	       * derivatives of the spline fit to the projectors
	       */
	      PetscMalloc(sizeof(PetscScalar)*count,&YDUlDeltaV[l]);		         
	      getYD_gen(tableR,tableUlDeltaV[l],YDUlDeltaV[l],count);            
	    } 	  
		 
	  /*
	   * loop over every atom of a given type
	   */
	  for(poscnt=start;poscnt<=end;poscnt++)
	    { 		  
	      x0 = pAtompos[index++];
	      y0 = pAtompos[index++];
	      z0 = pAtompos[index++];
	      if(pmvAtmConstraint[index_mvatm+1]==1)
		{  
		  xstart=-1; ystart=-1; zstart=-1; xend=-1; yend=-1; zend=-1; overlap=0;
		 
		  xi = (int)((x0 + R_x)/delta + 0.5); 
		  yj = (int)((y0 + R_y)/delta + 0.5);
		  zk = (int)((z0 + R_z)/delta + 0.5);
		 
		  xs = xi-offset; xl = xi+offset;
		  ys = yj-offset; yl = yj+offset;
		  zs = zk-offset; zl = zk+offset;
		  
		  /*
		   * find if domain of influence of pseudopotential overlaps with the domain
		   * stored by processor  
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
		    
		
		  for(i=0;i<pSddft->Nstates;i++)
		    {    			    
		      for(l=0;l<=lmax;l++)
			{ 
			  for(m=-l;m<=l;m++)
			    {	     
			      W[poscnt][i][l][m+l]=0.0;						      
			    }			
			}
		    }
		    	    	       
		  if(overlap)			 
		    {
		      i0 = xs-o;
		      j0 = ys-o;
		      k0 = zs-o;
				     		
		      /*
		       * Get local pointer (column wise) to the gradient of wavefunctions  
		       */				   		    
		      MatDenseGetArray(pSddft->YOrbNew,&arrGradPsiSeq); 
			    
		      for(k=zstart;k<=zend;k++)
			for(j=ystart;j<=yend;j++)
			  for(i=xstart;i<=xend;i++)
			    {			     
			      I=i-i0;
			      J=j-j0;
			      K=k-k0; 
			      
			      x = delta*i-R_x;
			      y = delta*j-R_y;
			      z = delta*k-R_z;
			      r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
			      
			      LIrow = k*n_x*n_y + j*n_x + i; 

			      AOApplicationToPetsc(aodmda1,1,&LIrow);
			      for(l=0;l<=lmax;l++)
				{
				  if(l!=lloc)
				    {				     
				      if(r == tableR[0])
					{
					  pUlDeltaVl = tableUlDeltaV[l][0];	
					}else{
					ispline_gen(tableR,tableUlDeltaV[l],count,&r,&pUlDeltaVl,&Dtemp,1,YDUlDeltaV[l]);
				      }						 
				      if(l==0)
					Rcut= pSddft->psd[at].rc_s;
				      if(l==1)
					Rcut= pSddft->psd[at].rc_p;
				      if(l==2)
					Rcut= pSddft->psd[at].rc_d;
				      if(l==3)
					Rcut= pSddft->psd[at].rc_f;
				      for(m=-l;m<=l;m++)
					{	
					  SpHarmonic=SphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut+1.0);
					  for(ii=0;ii<pSddft->Nstates;ii++)
					    {
					      OrbScalingFactor = 1.0/(sqrt(delta*delta*delta)*norms[ii]);
					      W[poscnt][ii][l][m+l] += OrbScalingFactor*(arrGradPsiSeq[LIrow-rowStart + (rowEnd-rowStart)*ii])*pUlDeltaVl*SpHarmonic;		     
					    }						
					}
				    }
				}
			    }   
			    
		      MatDenseRestoreArray(pSddft->YOrbNew,&arrGradPsiSeq);		     

		    }
		}
	      index_mvatm = index_mvatm+3;
	    }	    
	  
	  for(l=0;l<=lmax;l++)
	    {
	      PetscFree(YDUlDeltaV[l]);	       	    
	    }
	  PetscFree(YDUlDeltaV); 
	    
	}
    }
    
  /*
   * sum the contributions across all the processors and compute the forces
   */
  for(at=0;at<pSddft->Ntype;at++)
    {
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];
      if(lmax!=0)
	{	 
	  start = (int)floor(pSddft->startPos[at]/3);
	  end = (int)floor(pSddft->endPos[at]/3);
	  for(poscnt=start;poscnt<=end;poscnt++)
	    {	      
	      fJ_y=0.0;
	      for(i=0;i<pSddft->Nstates;i++)
		{  gi = smearing_FermiDirac(pSddft->Beta,pSddft->lambda[i],pSddft->lambda_f);	
		  for(l=0;l<=lmax;l++)
		    {
		      if(l!=lloc)
			{
			  for(m=-l;m<=l;m++)
			    {			     
			      ierr =  MPI_Allreduce(MPI_IN_PLACE,&W[poscnt][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);			     
			      fJ_y += gi*W[poscnt][i][l][m+l]*T[poscnt][i][l][m+l];			     
			    }
			}
		    }		  
		}
	      	      
	      fJ_y = -4.0*delta*delta*delta*fJ_y;	     
	      VecSetValue(pSddft->forces,3*poscnt+1,fJ_y,ADD_VALUES);	      
      	    }
	}
    }
  /*
   * end of y-component force calculation 
   */


  /*
   * Calculate z-component of forces
   */

  /* we do no need to calculate T since it has already been calculated while calculating the 
   * x-component of forces. Hence we do not need to calculate the exact denominator. Also,
   * memory for W has already been created, we dont need to allocate again, just need to set it
   * to zero
   */

  index_mvatm=0;index=0;  
  /*
   * Calculate gradient of orbitals in z-direction
   */
  MatMatMultNumeric(pSddft->gradient_z,*Psi,pSddft->YOrbNew);	
 
  /*
   * loop over different types of atoms 
   */
  for(at=0;at<pSddft->Ntype;at++)
    {                          
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];	 
	   
      start = floor(pSddft->startPos[at]/3);
      end = floor(pSddft->endPos[at]/3);

      /*
       * if the atom is hydrogen, then we omit the calculation of non-local forces
       */
      if(lmax==0)
	{
	  index = index + 3*(end-start+1); 
	  index_mvatm = index_mvatm + 3*(end-start+1);
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
		}
	      if(l==1)
		{
		  tableUlDeltaV[1][0]=0.0; 		     
		}
	      if(l==2)
		{
		  tableUlDeltaV[2][0]=0.0; 		      
		}
	      if(l==3)
		{
		  tableUlDeltaV[3][0]=0.0; 	   	  
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
		  }
		if(l==1)
		  {
		    tableUlDeltaV[1][count]=pSddft->psd[at].Up[count-1]*(pSddft->psd[at].Vp[count-1]-pSddft->psd[at].Vloc[count-1]); 	         	       
		  }
		if(l==2)
		  {
		    tableUlDeltaV[2][count]=pSddft->psd[at].Ud[count-1]*(pSddft->psd[at].Vd[count-1]-pSddft->psd[at].Vloc[count-1]); 		   	       
		  }   
		if(l==3)
		  {
		    tableUlDeltaV[3][count]=pSddft->psd[at].Uf[count-1]*(pSddft->psd[at].Vf[count-1]-pSddft->psd[at].Vloc[count-1]); 		            
		  }  
	      }
	    count++;
	  }while(tableR[count-1] <= 1.732*pSddft->CUTOFF[at]+10.0); 
	  rmax = tableR[count-1];
                                          
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDUlDeltaV);	  
	  for(l=0;l<=lmax;l++)
	    {
	      /*
	       * derivatives of the spline fit to the projectors
	       */
	      PetscMalloc(sizeof(PetscScalar)*count,&YDUlDeltaV[l]);		         
	      getYD_gen(tableR,tableUlDeltaV[l],YDUlDeltaV[l],count);             
	    } 	  	
	 
	  /*
	   * loop over every atom of a given type
	   */
	  for(poscnt=start;poscnt<=end;poscnt++)
	    { 	      
	      x0 = pAtompos[index++];
	      y0 = pAtompos[index++];
	      z0 = pAtompos[index++];
	      if(pmvAtmConstraint[index_mvatm+2]==1)
		{  
		  xstart=-1; ystart=-1; zstart=-1; xend=-1; yend=-1; zend=-1; overlap=0;
		  
		  xi = (int)((x0 + R_x)/delta + 0.5); 
		  yj = (int)((y0 + R_y)/delta + 0.5);
		  zk = (int)((z0 + R_z)/delta + 0.5);
		  
		  /*
		   * find if domain of influence of pseudopotential overlaps with the domain
		   * stored by processor  
		   */
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
		  
		  for(i=0;i<pSddft->Nstates;i++)
		    {    			    
		      for(l=0;l<=lmax;l++)
			{ 
			  for(m=-l;m<=l;m++)
			    {	     
			      W[poscnt][i][l][m+l]=0.0;						      
			    }			
			}
		    }
		    	    	       
		  if(overlap)			 
		    {

		      i0 = xs-o;
		      j0 = ys-o;
		      k0 = zs-o;
		
		      /*
		       * Get local pointer (column wise) to the gradient of wavefunctions  
		       */
		      MatDenseGetArray(pSddft->YOrbNew,&arrGradPsiSeq); 
		
		      /* 
		       * evaluate the the quantities that contribute to nonlocal force at 
		       * nodes in the overlap region
		       */
		      for(k=zstart;k<=zend;k++)
			for(j=ystart;j<=yend;j++)
			  for(i=xstart;i<=xend;i++)
			    {
			     
			      I=i-i0;
			      J=j-j0;
			      K=k-k0; 
			      
			      x = delta*i-R_x;
			      y = delta*j-R_y;
			      z = delta*k-R_z;
			      r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
			     
			      LIrow = k*n_x*n_y + j*n_x + i; 
			      AOApplicationToPetsc(aodmda1,1,&LIrow);
			      for(l=0;l<=lmax;l++)
				{
				  if(l!=lloc)
				    { 				      
				      if(r == tableR[0])
					{
					  pUlDeltaVl = tableUlDeltaV[l][0];	
					}else{
					ispline_gen(tableR,tableUlDeltaV[l],count,&r,&pUlDeltaVl,&Dtemp,1,YDUlDeltaV[l]);
				      }						 
				      if(l==0)
					Rcut= pSddft->psd[at].rc_s;
				      if(l==1)
					Rcut= pSddft->psd[at].rc_p;
				      if(l==2)
					Rcut= pSddft->psd[at].rc_d;
				      if(l==3)
					Rcut= pSddft->psd[at].rc_f;
				      for(m=-l;m<=l;m++)
					{	
					  SpHarmonic=SphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut+1.0);
					  for(ii=0;ii<pSddft->Nstates;ii++)
					    {
					      OrbScalingFactor = 1.0/(sqrt(delta*delta*delta)*norms[ii]);   
					      W[poscnt][ii][l][m+l] += OrbScalingFactor*(arrGradPsiSeq[LIrow-rowStart + (rowEnd-rowStart)*ii])*pUlDeltaVl*SpHarmonic;		     
					    }						
					}
				    }
				}
			    }			    
		      MatDenseRestoreArray(pSddft->YOrbNew,&arrGradPsiSeq);
		    }
		}
	      index_mvatm = index_mvatm+3;
	    }	    	 
	  for(l=0;l<=lmax;l++)
	    {
	      PetscFree(YDUlDeltaV[l]);	       	    
	    }
	  PetscFree(YDUlDeltaV); 
	    
	}
    }
 
  /*
   * sum the contributions across all the processors and compute the forces
   */     
  for(at=0;at<pSddft->Ntype;at++)
    {
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];
      if(lmax!=0)
	{	  
	  start = (int)floor(pSddft->startPos[at]/3);
	  end = (int)floor(pSddft->endPos[at]/3);
	  for(poscnt=start;poscnt<=end;poscnt++)
	    {	      
	      fJ_z=0.0;
	      for(i=0;i<pSddft->Nstates;i++)
		{  gi = smearing_FermiDirac(pSddft->Beta,pSddft->lambda[i],pSddft->lambda_f);
		  for(l=0;l<=lmax;l++)
		    {
		      if(l!=lloc)
			{
			  for(m=-l;m<=l;m++)
			    {			     
			      ierr =  MPI_Allreduce(MPI_IN_PLACE,&W[poscnt][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);			     
			      fJ_z += gi*W[poscnt][i][l][m+l]*T[poscnt][i][l][m+l];	     
			    }
			}
		    }		  
		}	      	      
	      fJ_z = -4.0*delta*delta*delta*fJ_z;	     
	      VecSetValue(pSddft->forces,3*poscnt+2,fJ_z,ADD_VALUES);	      
      	    }
	}
    }
  /*
   * end of z-component force calculation 
   */
  VecRestoreArray(pSddft->Atompos,&pAtompos);
  VecRestoreArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);          
  VecAssemblyBegin(pSddft->forces);
  VecAssemblyEnd(pSddft->forces);    
   

  /*
   * free memory
   */
  for(at=0;at<pSddft->Ntype;at++)
    {
      lmax = pSddft->psd[at].lmax;
     
      int start,end;
      start = (int)floor(pSddft->startPos[at]/3);
      end = (int)floor(pSddft->endPos[at]/3);
	  
      for(poscnt=start;poscnt<=end;poscnt++)
	{	      
	  if(lmax!=0)
	    {
	      for(i=0;i<pSddft->Nstates;i++)
		{		 
		  for(l=0;l<=lmax;l++)
		    {			     
		      PetscFree(W[poscnt][i][l]);
		      PetscFree(T[poscnt][i][l]);
			      
		    }			 
		  PetscFree(W[poscnt][i]);
		  PetscFree(T[poscnt][i]);			 
		}
	
	    }
	  PetscFree(W[poscnt]);
	  PetscFree(T[poscnt]);		
	}     
    }
  PetscFree(W);
  PetscFree(T);       
 
  return ierr;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                             Display_force: prints forces                                  // 
///////////////////////////////////////////////////////////////////////////////////////////////
void Display_force(SDDFT_OBJ* pSddft)
{    
  PetscScalar *pForces;
  PetscInt poscnt,Index=0;
  PetscScalar SumFx=0.0, SumFy=0.0, SumFz=0.0;
  PetscScalar Ct=27.211384523/0.529177249; // Hartree/Bohr to eV/Angstrom conversion

  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD,"                                 Forces                                    \n");
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  VecGetArray(pSddft->forces,&pForces); 
  PetscPrintf(PETSC_COMM_WORLD,"Forces (Hartree/Bohr) \n"); 
  for(poscnt=0;poscnt<pSddft->nAtoms;poscnt++)
    {
      PetscPrintf(PETSC_COMM_WORLD," %0.16lf \t %0.16lf \t %0.16lf \n",pForces[Index],pForces[Index+1],pForces[Index+2]);
      SumFx+=pForces[Index]; SumFy+=pForces[Index+1]; SumFz+=pForces[Index+2]; // sum of forces 
      Index = Index+3;
    }
  VecRestoreArray(pSddft->forces,&pForces);
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n\n");
      
  return;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//         Symmetrysize_force: Symmetricizes forces such that sum of forces is zero          //
///////////////////////////////////////////////////////////////////////////////////////////////
void  Symmetrysize_force(SDDFT_OBJ* pSddft)
{

  PetscScalar *pForces;
  PetscScalar *pmvAtmConstraint;
  PetscInt poscnt,Index=0;
  PetscScalar SumFx=0.0, SumFy=0.0, SumFz=0.0;

  /*
   * calculate the average force components
   */
  VecGetArray(pSddft->forces,&pForces); 
  VecGetArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);
   for(poscnt=0;poscnt<pSddft->nAtoms;poscnt++)
    {      
      SumFx+=pForces[Index]; 
      
      SumFy+=pForces[Index+1]; 
      
      SumFz+=pForces[Index+2];

      Index=Index+3;
    }
  Index=0;
  /*
   * subtract the average force components
   */
   for(poscnt=0;poscnt<pSddft->nAtoms;poscnt++)
    {
      if(pmvAtmConstraint[Index]==1)
	pForces[Index]=pForces[Index]-(1.0/pSddft->nAtoms)*SumFx;
      else
	pForces[Index]=0.0;


      if(pmvAtmConstraint[Index+1]==1)
	pForces[Index+1]=pForces[Index+1]-(1.0/pSddft->nAtoms)*SumFy;
      else
	pForces[Index+1]=0.0;

      if(pmvAtmConstraint[Index+2]==1)
	pForces[Index+2]=pForces[Index+2]-(1.0/pSddft->nAtoms)*SumFz;
      else
	pForces[Index+2]=0.0;
	       
      Index=Index+3;
	       
    }
  VecRestoreArray(pSddft->forces,&pForces);
  VecRestoreArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);

  return;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//              PeriodicCalculate_force: local component of force on atoms                   // 
///////////////////////////////////////////////////////////////////////////////////////////////
void PeriodicCalculate_force(SDDFT_OBJ* pSddft)
{
  PetscScalar ***PotPhiArrGlbIdx;
  PetscScalar *pAtompos;
  PetscScalar *pForces;
  PetscScalar *pmvAtmConstraint;
  PetscMPIInt comm_size,rank;
  PetscScalar *YD=NULL;

  PetscScalar ForceProcComponentz,ForceProcComponenty,ForceProcComponentx;
  Mat ForceProcMatz,ForceProcMaty,ForceProcMatx;

  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);  
  MatCreate(PETSC_COMM_WORLD,&ForceProcMatz);
  MatSetSizes(ForceProcMatz,PETSC_DECIDE,PETSC_DECIDE,comm_size,pSddft->nAtoms);
  MatSetType(ForceProcMatz,MATMPIDENSE);
  MatSetUp(ForceProcMatz);
  
  MatCreate(PETSC_COMM_WORLD,&ForceProcMaty);
  MatSetSizes(ForceProcMaty,PETSC_DECIDE,PETSC_DECIDE,comm_size,pSddft->nAtoms);
  MatSetType(ForceProcMaty,MATMPIDENSE);
  MatSetUp(ForceProcMaty);
  
  MatCreate(PETSC_COMM_WORLD,&ForceProcMatx);
  MatSetSizes(ForceProcMatx,PETSC_DECIDE,PETSC_DECIDE,comm_size,pSddft->nAtoms);
  MatSetType(ForceProcMatx,MATMPIDENSE);
  MatSetUp(ForceProcMatx);

  MatZeroEntries(ForceProcMatx);
  MatZeroEntries(ForceProcMaty);
  MatZeroEntries(ForceProcMatz);


  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0;
  PetscInt Xstart,Ystart,Zstart,Xend,Yend,Zend,XBJstart,YBJstart,ZBJstart,XBJend,YBJend,ZBJend;
  PetscScalar x0, y0, z0,X0,Y0,Z0,coeffs[MAX_ORDER+1],coeffs_grad[MAX_ORDER+1];
  PetscScalar GradBJx,GradBJy,GradBJz,PhimVpsJ,FORCE;
  PetscInt Nposx,Nposy,Nposz,PP,QQ,RR; 

  PetscScalar tableR[MAX_TABLE_SIZE],tableVps[MAX_TABLE_SIZE],Bval,noetot=pSddft->noetot;
  PetscScalar ***pVpsArray=NULL;
  PetscScalar ***pBArray=NULL;
  PetscScalar Dtemp;
      
  PetscInt  i=0,j,k,xi,yj,zk,offset,tablesize,l,I,J,K,p,nzVps,nyVps,nxVps,nzBJ,nyBJ,nxBJ,nzGradBJ,nyGradBJ,nxGradBJ,i0,j0,k0,a,poscnt,index=0,index_force=0,index_mvatm=0,count,at;
  PetscScalar delta=pSddft->delta,x,y,z,r,cutoffr,rmax;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt o = pSddft->order;
  Vec forcex;Vec forcey;Vec forcez; 
  
  VecCreate(PETSC_COMM_WORLD,&forcex);
  VecSetSizes(forcex,PETSC_DECIDE,comm_size);
  VecSetFromOptions(forcex);
   
  VecCreate(PETSC_COMM_WORLD,&forcey);
  VecSetSizes(forcey,PETSC_DECIDE,comm_size);
  VecSetFromOptions(forcey);
   
  VecCreate(PETSC_COMM_WORLD,&forcez);
  VecSetSizes(forcez,PETSC_DECIDE,comm_size);
  VecSetFromOptions(forcez);   

  VecZeroEntries(forcex);
  VecZeroEntries(forcey);
  VecZeroEntries(forcez);
  
  for(p=0;p<=o;p++)
    {
      coeffs[p] = pSddft->coeffs[p]/(2*M_PI);
      coeffs_grad[p] = pSddft->coeffs_grad[p];
    }  
 
  VecGetArray(pSddft->forces,&pForces); 
  VecGetArray(pSddft->mvAtmConstraint,&pmvAtmConstraint); 

  DMDAVecGetArray(pSddft->da,pSddft->potentialPhi,&PotPhiArrGlbIdx);      
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
   
  VecGetArray(pSddft->Atompos,&pAtompos); 
  /*
   * loop over different types of atoms 
   */
  for(at=0;at<pSddft->Ntype;at++)
    {   
      cutoffr=pSddft->CUTOFF[at]+o*delta;
      offset = (PetscInt)ceil(cutoffr/delta + 0.5);

      tableR[0]=0.0; tableVps[0]=pSddft->psd[at].Vloc[0]; 
      count=1;
      do{
	tableR[count] = pSddft->psd[at].RadialGrid[count-1];
	tableVps[count] = pSddft->psd[at].Vloc[count-1]; 
	count++;

      }while(tableR[count-1] <= 1.732*cutoffr); 
      rmax = tableR[count-1];
      int start,end;
      start = (int)floor(pSddft->startPos[at]/3);
      end = (int)floor(pSddft->endPos[at]/3);           

      /*
       * derivatives of the spline fit to the pseudopotential
       */
      PetscMalloc(sizeof(PetscScalar)*count,&YD);
      getYD_gen(tableR,tableVps,YD,count); 
  
    
      /*
       * loop over every atom of a given type
       */
      for(poscnt=start;poscnt<=end;poscnt++)
	{ 
	  X0 = pAtompos[index++];
	  Y0 = pAtompos[index++];
	  Z0 = pAtompos[index++];

	  Nposx = ceil(cutoffr/R_x);
	  Nposy = ceil(cutoffr/R_y);
	  Nposz = ceil(cutoffr/R_z);
	  
	  for(PP= -Nposx; PP<=Nposx; PP++)
	    for(QQ= -Nposy; QQ<=Nposy; QQ++)
	      for(RR= -Nposz; RR<=Nposz; RR++)
		{
		  x0 = X0 + PP*2.0*R_x;
		  y0 = Y0 + QQ*2.0*R_y;
		  z0 = Z0 + RR*2.0*R_z;

		  xstart=-1000; ystart=-1000; zstart=-1000; xend=-1000; yend=-1000; zend=-1000; overlap=0;
   
		  xi = (int)((x0 + R_x)/delta + 0.5); 
		  yj = (int)((y0 + R_y)/delta + 0.5);
		  zk = (int)((z0 + R_z)/delta + 0.5);
        
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
		      /*
		       * find the starting and ending indices of finite difference nodes in each 
		       * direction for calculation of pseudopotential and pseudocharge density 
		       */
      
		      Xstart = xstart+o; Xend = xend-o;  
		      Ystart = ystart+o; Yend = yend-o;
		      Zstart = zstart+o; Zend = zend-o;

		      XBJstart = xstart; XBJend = xend;
		      YBJstart = ystart; YBJend = yend;
		      ZBJstart = zstart; ZBJend = zend;
        
		      if(xstart == xcor)
			{
			  Xstart = xstart; XBJstart = xstart-o;
			}
		      if(xend == xcor+lxdim-1)
			{
			  Xend = xend; XBJend = xend+o;
			}
        
		      if(ystart == ycor)
			{
			  Ystart = ystart; YBJstart = ystart-o;
			}
		      if(yend == ycor+lydim-1)
			{
			  Yend = yend; YBJend = yend+o;
			}
 
		      if(zstart == zcor)
			{
			  Zstart = zstart; ZBJstart = zstart-o;
			}
		      if(zend == zcor+lzdim-1)
			{
			  Zend = zend; ZBJend = zend+o;
			}

		      nzBJ = ZBJend-ZBJstart+1;
		      nyBJ = YBJend-YBJstart+1;
		      nxBJ = XBJend-XBJstart+1;
        
		      nzVps = nzBJ+o*2;
		      nyVps = nyBJ+o*2;
		      nxVps = nxBJ+o*2;

		      nzGradBJ = Zend-Zstart+1;
		      nyGradBJ = Yend-Ystart+1;
		      nxGradBJ = Xend-Xstart+1;
              
       
		      PetscMalloc(sizeof(PetscScalar**)*nzVps,&pVpsArray);
       
		      if(pVpsArray == NULL)
			{
			  printf("Memory alocation fail");
			  exit(1);
			}
		      for(i = 0; i < nzVps; i++)
			{
			  PetscMalloc(sizeof(PetscScalar*)*nyVps,&pVpsArray[i]);
         
			  if(pVpsArray[i] == NULL)
			    {
			      printf("Memory alocation fail");
			      exit(1);
			    }
    
			  for(j=0;j<nyVps;j++)
			    {
			      PetscMalloc(sizeof(PetscScalar)*nxVps,&pVpsArray[i][j]);
            
			      if(pVpsArray[i][j] == NULL)
				{
				  printf("Memory alocation fail");
				  exit(1);
				} 
			    }      
			}
  
		      PetscMalloc(sizeof(PetscScalar**)*nzBJ,&pBArray);       
		      if(pBArray == NULL)
			{
			  printf("Memory alocation fail");
			  exit(1);
			}
		      for(i = 0; i < nzBJ; i++)
			{
			  PetscMalloc(sizeof(PetscScalar*)*nyBJ,&pBArray[i]);
         
			  if(pBArray[i] == NULL)
			    {
			      printf("Memory alocation fail");
			      exit(1);
			    }
    
			  for(j=0;j<nyBJ;j++)
			    {
			      PetscMalloc(sizeof(PetscScalar)*nxBJ,&pBArray[i][j]);
            
			      if(pBArray[i][j] == NULL)
				{
				  printf("Memory alocation fail");
				  exit(1);
				} 
			    }      
			}

		      i0 = XBJstart-o;
		      j0 = YBJstart-o;
		      k0 = ZBJstart-o;
		      /* 
		       * evaluate the pseudopotential at nodes in the overlap region + 
		       * finite-difference order in each direction
		       */
		      for(k=0;k<nzVps;k++) 
			for(j=0;j<nyVps;j++)  
			  for(i=0;i<nxVps;i++) 
			    {   
			      I = i+i0;
			      J = j+j0;
			      K = k+k0;
    
			      x = delta*I - R_x ;
			      y = delta*J - R_y ;
			      z = delta*K - R_z ;   
			      r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
              
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
		       * calculate pseudocharge density at nodes inside the overlap region from
		       * previously calculated pseudopotential values using finite difference stencil  
		       */
		      for(k=ZBJstart;k<=ZBJend;k++)
			for(j=YBJstart;j<=YBJend;j++)
			  for(i=XBJstart;i<=XBJend;i++)
			    {
			      Bval=0;        
			      I=i-i0;
			      J=j-j0;
			      K=k-k0;
                                   
			      Bval=pVpsArray[K][J][I]*coeffs[0];
			      for(a=1;a<=o;a++)
				{
				  Bval+=(pVpsArray[K][J][I-a] + pVpsArray[K][J][I+a] + pVpsArray[K][J-a][I] +        
					 pVpsArray[K][J+a][I] + pVpsArray[K-a][J][I] + pVpsArray[K+a][J][I])*coeffs[a];          
				}
			      pBArray[k-ZBJstart][j-YBJstart][i-XBJstart] = Bval;             
			    } 

		      ForceProcComponentz = 0.0;
		      ForceProcComponenty = 0.0;
		      ForceProcComponentx = 0.0;

		      /*
		       * calculate contribution of force	      
		       */
		      for(k=Zstart;k<=Zend;k++) 
			for(j=Ystart;j<=Yend;j++)  
			  for(i=Xstart;i<=Xend;i++) 
			    {  
			      GradBJx=0.0;GradBJy=0.0;GradBJz=0.0;          
			      I=i-i0;
			      J=j-j0;
			      K=k-k0;       
			      for(a=1;a<=o;a++)
				{
	   
				  GradBJz += (pBArray[K-o+a][J-o][I-o] - pBArray[K-o-a][J-o][I-o])*coeffs_grad[a];	 
				  GradBJy += (pBArray[K-o][J-o+a][I-o] - pBArray[K-o][J-o-a][I-o])*coeffs_grad[a];	   
				  GradBJx += (pBArray[K-o][J-o][I-o+a] - pBArray[K-o][J-o][I-o-a])*coeffs_grad[a];                   
				}
			      PhimVpsJ = PotPhiArrGlbIdx[k][j][i]-pVpsArray[K][J][I];

			      ForceProcComponentz+=GradBJz*PhimVpsJ;
			      ForceProcComponenty+=GradBJy*PhimVpsJ;
			      ForceProcComponentx+=GradBJx*PhimVpsJ; 

			    }    
		      MPI_Comm_rank(MPI_COMM_WORLD,&rank);	 
		      MatSetValue(ForceProcMatz,rank,poscnt,ForceProcComponentz,ADD_VALUES);	
		      MatSetValue(ForceProcMaty,rank,poscnt,ForceProcComponenty,ADD_VALUES);
		      MatSetValue(ForceProcMatx,rank,poscnt,ForceProcComponentx,ADD_VALUES);
		      for(i = 0; i < nzVps; i++)
			{
			  for(j=0;j<nyVps;j++)
			    {
			      PetscFree(pVpsArray[i][j]);                    
			    }
			  PetscFree(pVpsArray[i]);
			}  
		      PetscFree(pVpsArray);   
      
		      for(i = 0; i < nzBJ; i++)
			{
			  for(j=0;j<nyBJ;j++)
			    {
			      PetscFree(pBArray[i][j]);                    
			    }
			  PetscFree(pBArray[i]);
			}  
		      PetscFree(pBArray); 
       
		    }  
		}
	
	  index_mvatm = index_mvatm+3;

	}    
  
      PetscFree(YD); 
    }
  DMDAVecRestoreArray(pSddft->da,pSddft->potentialPhi,&PotPhiArrGlbIdx);
  VecRestoreArray(pSddft->Atompos,&pAtompos);

  MatAssemblyBegin(ForceProcMatx,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ForceProcMatx,MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ForceProcMaty,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ForceProcMaty,MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ForceProcMatz,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ForceProcMatz,MAT_FINAL_ASSEMBLY);	
	
  /*
   * sum contribution of forces from all processors
   */
  for(poscnt=0;poscnt<pSddft->nAtoms;poscnt++)
    {
      MatGetColumnVector(ForceProcMatx,forcex,poscnt);
      MatGetColumnVector(ForceProcMaty,forcey,poscnt);
      MatGetColumnVector(ForceProcMatz,forcez,poscnt);		
      VecSum(forcex,&FORCE);pForces[index_force++] = delta*delta*delta*FORCE;
      VecSum(forcey,&FORCE);pForces[index_force++] = delta*delta*delta*FORCE;
      VecSum(forcez,&FORCE);pForces[index_force++] = delta*delta*delta*FORCE;		
    }

  VecDestroy(&forcex);
  VecDestroy(&forcey);
  VecDestroy(&forcez);  

  MatDestroy(&ForceProcMatx);
  MatDestroy(&ForceProcMaty);
  MatDestroy(&ForceProcMatz);
   
  /*
   * adding force correction to forces 
   */
  for(poscnt=0;poscnt<3*pSddft->nAtoms;poscnt++)
    {
      pForces[poscnt] = pForces[poscnt]+pSddft->pForces_corr[poscnt];        
      VecSetValues(pSddft->forces,1,&poscnt,&pForces[poscnt],INSERT_VALUES);            
    }    
  VecRestoreArray(pSddft->forces,&pForces);
  VecRestoreArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);

  return;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//     PeriodicCalculate_forceCorrection: force correction on atoms for periodic systems     //
///////////////////////////////////////////////////////////////////////////////////////////////
void PeriodicCalculate_forceCorrection(SDDFT_OBJ* pSddft)
{
  
  PetscScalar ***PotPhicArrGlbIdx;
  PetscScalar ***BlcArrGlbIdx;
  PetscScalar ***BlcArrGlbIdx_TM;
  PetscScalar *pmvAtmConstraint;
  PetscMPIInt comm_size,rank;
  PetscScalar *YD=NULL;
  PetscScalar Dtemp;

  PetscScalar *pAtompos;  
  PetscScalar ForceProcComponentz,ForceProcComponenty,ForceProcComponentx;
  Mat ForceProcMatz,ForceProcMaty,ForceProcMatx;
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  
  MatCreate(PETSC_COMM_WORLD,&ForceProcMatz);
  MatSetSizes(ForceProcMatz,PETSC_DECIDE,PETSC_DECIDE,comm_size,pSddft->nAtoms);
  MatSetType(ForceProcMatz,MATMPIDENSE);
  MatSetUp(ForceProcMatz);
  
  MatCreate(PETSC_COMM_WORLD,&ForceProcMaty);
  MatSetSizes(ForceProcMaty,PETSC_DECIDE,PETSC_DECIDE,comm_size,pSddft->nAtoms);
  MatSetType(ForceProcMaty,MATMPIDENSE);
  MatSetUp(ForceProcMaty);
  
  MatCreate(PETSC_COMM_WORLD,&ForceProcMatx);
  MatSetSizes(ForceProcMatx,PETSC_DECIDE,PETSC_DECIDE,comm_size,pSddft->nAtoms);
  MatSetType(ForceProcMatx,MATMPIDENSE);
  MatSetUp(ForceProcMatx);

  MatZeroEntries(ForceProcMatx);
  MatZeroEntries(ForceProcMaty);
  MatZeroEntries(ForceProcMatz);

  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0;
  PetscInt Xstart,Ystart,Zstart,Xend,Yend,Zend,XBJstart,YBJstart,ZBJstart,XBJend,YBJend,ZBJend;
  PetscScalar x0, y0, z0,X0,Y0,Z0,coeffs[MAX_ORDER+1],coeffs_grad[MAX_ORDER+1];
  PetscScalar GradBJx,GradBJy,GradBJz,GradBJx_TM,GradBJy_TM,GradBJz_TM,PhicmVtmJ,PhicpVpsJ,BtmpBps,FORCE_corr;
  PetscScalar GradVJx,GradVJy,GradVJz,GradVJx_TM,GradVJy_TM,GradVJz_TM;
  
  PetscInt Nposx,Nposy,Nposz,PP,QQ,RR; 
  PetscScalar tableR[MAX_TABLE_SIZE],tableVps[MAX_TABLE_SIZE],Bval,Bval_TM,noetot=pSddft->noetot;
  PetscScalar ***pVpsArray=NULL;
  PetscScalar ***pBArray=NULL;
  PetscScalar ***pVpsArray_TM=NULL;
  PetscScalar ***pBArray_TM=NULL;
      
  PetscInt  i=0,j,k,xi,yj,zk,offset,tablesize,tablesize_TM,l,I,J,K,p,nzVps,nyVps,nxVps,nzBJ,nyBJ,nxBJ,  i0,j0,k0,a,poscnt,index=0,index_force=0,index_mvatm=0,at,count;
  PetscScalar delta=pSddft->delta,x,y,z,r,cutoffr,rmax,rmax_TM,rcut=pSddft->REFERENCE_CUTOFF;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt o = pSddft->order;
  Vec forcex_corr;Vec forcey_corr;Vec forcez_corr; 

  VecCreate(PETSC_COMM_WORLD,&forcex_corr);
  VecSetSizes(forcex_corr,PETSC_DECIDE,comm_size);
  VecSetFromOptions(forcex_corr);
   
  VecCreate(PETSC_COMM_WORLD,&forcey_corr);
  VecSetSizes(forcey_corr,PETSC_DECIDE,comm_size);
  VecSetFromOptions(forcey_corr);
   
  VecCreate(PETSC_COMM_WORLD,&forcez_corr);
  VecSetSizes(forcez_corr,PETSC_DECIDE,comm_size);
  VecSetFromOptions(forcez_corr);   

  VecZeroEntries(forcex_corr);
  VecZeroEntries(forcey_corr);
  VecZeroEntries(forcez_corr);
  
  for(p=0;p<=o;p++)
    {
      coeffs[p] = pSddft->coeffs[p]/(2*M_PI);
      coeffs_grad[p] = pSddft->coeffs_grad[p];
    }   
  PetscMalloc(sizeof(PetscScalar)*3*pSddft->nAtoms, &pSddft->pForces_corr);
  if(pSddft->pForces_corr == NULL)
    {
      printf("Memory alocation fail in pForces_corr");
      exit(1);
    }

  DMDAVecGetArray(pSddft->da,pSddft->Phi_c,&PotPhicArrGlbIdx); 
  DMDAVecGetArray(pSddft->da,pSddft->chrgDensB,&BlcArrGlbIdx);  
  DMDAVecGetArray(pSddft->da,pSddft->chrgDensB_TM,&BlcArrGlbIdx_TM); 
        
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);

  VecGetArray(pSddft->Atompos,&pAtompos);   
  VecGetArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);     

  /*
   * loop over different types of atoms 
   */
  for(at=0;at<pSddft->Ntype;at++)
    {   
      cutoffr=pSddft->CUTOFF[at]+o*delta;
      offset = (PetscInt)ceil(cutoffr/delta + 0.5);

      tableR[0]=0.0; tableVps[0]=pSddft->psd[at].Vloc[0]; 
      count=1;
      do{
	tableR[count] = pSddft->psd[at].RadialGrid[count-1];
	tableVps[count] = pSddft->psd[at].Vloc[count-1]; 
	count++;

      }while(tableR[count-1] <= 1.732*cutoffr); 
      rmax = tableR[count-1];
      int start,end;
      start = (int)floor(pSddft->startPos[at]/3);
      end = (int)floor(pSddft->endPos[at]/3);
     
      /*
       * derivatives of the spline fit to the pseudopotential
       */
      PetscMalloc(sizeof(PetscScalar)*count,&YD);
      getYD_gen(tableR,tableVps,YD,count); 
      
      /*
       * loop over every atom of a given type
       */
      for(poscnt=start;poscnt<=end;poscnt++)
	{ 

	  X0 = pAtompos[index++];
	  Y0 = pAtompos[index++];
	  Z0 = pAtompos[index++];
          
	  pSddft->pForces_corr[index_mvatm]=0.0;
	  pSddft->pForces_corr[index_mvatm+1]=0.0; 
	  pSddft->pForces_corr[index_mvatm+2]=0.0; 
      
	  Nposx = ceil(cutoffr/R_x);
	  Nposy = ceil(cutoffr/R_y);
	  Nposz = ceil(cutoffr/R_z);

	  for(PP= -Nposx; PP<=Nposx; PP++)
	    for(QQ= -Nposy; QQ<=Nposy; QQ++)
	      for(RR= -Nposz; RR<=Nposz; RR++)
		{
		  x0 = X0 + PP*2.0*R_x;
		  y0 = Y0 + QQ*2.0*R_y;
		  z0 = Z0 + RR*2.0*R_z;

		  xstart=-1000; ystart=-1000; zstart=-1000; xend=-1000; yend=-1000; zend=-1000; overlap=0;
        
     
		  xi = (int)((x0 + R_x)/delta + 0.5); 
		  yj = (int)((y0 + R_y)/delta + 0.5);
		  zk = (int)((z0 + R_z)/delta + 0.5);
     
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
       
		      /*
		       * find the starting and ending indices of finite difference nodes in each 
		       * direction for calculation of pseudopotential and pseudocharge density 
		       */       
		      Xstart = xstart+o; Xend = xend-o;  
		      Ystart = ystart+o; Yend = yend-o;
		      Zstart = zstart+o; Zend = zend-o;
      
		      XBJstart = xstart; XBJend = xend;
		      YBJstart = ystart; YBJend = yend;
		      ZBJstart = zstart; ZBJend = zend;        
		      if(xstart == xcor)
			{
			  Xstart = xstart; XBJstart = xstart-o;
			}
		      if(xend == xcor+lxdim-1)
			{
			  Xend = xend; XBJend = xend+o;
			}        
		      if(ystart == ycor)
			{
			  Ystart = ystart; YBJstart = ystart-o;
			}
		      if(yend == ycor+lydim-1)
			{
			  Yend = yend; YBJend = yend+o;
			} 
		      if(zstart == zcor)
			{
			  Zstart = zstart; ZBJstart = zstart-o;
			}
		      if(zend == zcor+lzdim-1)
			{
			  Zend = zend; ZBJend = zend+o;
			}

		      nzBJ = ZBJend-ZBJstart+1;
		      nyBJ = YBJend-YBJstart+1;
		      nxBJ = XBJend-XBJstart+1;
        
		      nzVps = nzBJ+o*2;
		      nyVps = nyBJ+o*2;
		      nxVps = nxBJ+o*2;
              
		      PetscMalloc(sizeof(PetscScalar**)*nzVps,&pVpsArray);       
		      if(pVpsArray == NULL)
			{
			  printf("Memory alocation fail");
			  exit(1);
			}
		      for(i = 0; i < nzVps; i++)
			{
			  PetscMalloc(sizeof(PetscScalar*)*nyVps,&pVpsArray[i]);
         
			  if(pVpsArray[i] == NULL)
			    {
			      printf("Memory alocation fail");
			      exit(1);
			    }
    
			  for(j=0;j<nyVps;j++)
			    {
			      PetscMalloc(sizeof(PetscScalar)*nxVps,&pVpsArray[i][j]);
            
			      if(pVpsArray[i][j] == NULL)
				{
				  printf("Memory alocation fail");
				  exit(1);
				} 
			    }      
			}
         
		      PetscMalloc(sizeof(PetscScalar**)*nzBJ,&pBArray);       
		      if(pBArray == NULL)
			{
			  printf("Memory alocation fail");
			  exit(1);
			}
		      for(i = 0; i < nzBJ; i++)
			{
			  PetscMalloc(sizeof(PetscScalar*)*nyBJ,&pBArray[i]);
         
			  if(pBArray[i] == NULL)
			    {
			      printf("Memory alocation fail");
			      exit(1);
			    }
    
			  for(j=0;j<nyBJ;j++)
			    {
			      PetscMalloc(sizeof(PetscScalar)*nxBJ,&pBArray[i][j]);
            
			      if(pBArray[i][j] == NULL)
				{
				  printf("Memory alocation fail");
				  exit(1);
				} 
			    }      
			}
       

		      PetscMalloc(sizeof(PetscScalar**)*nzVps,&pVpsArray_TM);       
		      if(pVpsArray_TM == NULL)
			{
			  printf("Memory alocation fail");
			  exit(1);
			}
		      for(i = 0; i < nzVps; i++)
			{
			  PetscMalloc(sizeof(PetscScalar*)*nyVps,&pVpsArray_TM[i]);
         
			  if(pVpsArray_TM[i] == NULL)
			    {
			      printf("Memory alocation fail");
			      exit(1);
			    }
    
			  for(j=0;j<nyVps;j++)
			    {
			      PetscMalloc(sizeof(PetscScalar)*nxVps,&pVpsArray_TM[i][j]);
            
			      if(pVpsArray_TM[i][j] == NULL)
				{
				  printf("Memory alocation fail");
				  exit(1);
				} 
			    }      
			}
  

		      PetscMalloc(sizeof(PetscScalar**)*nzBJ,&pBArray_TM);       
		      if(pBArray_TM == NULL)
			{
			  printf("Memory alocation fail");
			  exit(1);
			}
		      for(i = 0; i < nzBJ; i++)
			{
			  PetscMalloc(sizeof(PetscScalar*)*nyBJ,&pBArray_TM[i]);
         
			  if(pBArray_TM[i] == NULL)
			    {
			      printf("Memory alocation fail");
			      exit(1);
			    }
    
			  for(j=0;j<nyBJ;j++)
			    {
			      PetscMalloc(sizeof(PetscScalar)*nxBJ,&pBArray_TM[i][j]);
            
			      if(pBArray_TM[i][j] == NULL)
				{
				  printf("Memory alocation fail");
				  exit(1);
				} 
			    }      
			}
        
      
		      i0 = XBJstart-o;
		      j0 = YBJstart-o;
		      k0 = ZBJstart-o;
     
		      /* 
		       * evaluate the pseudopotential at nodes in the overlap region + 
		       * finite-difference order in each direction
		       */
		      for(k=0;k<nzVps;k++) 
			for(j=0;j<nyVps;j++)  
			  for(i=0;i<nxVps;i++) 
			    {   
			      I = i+i0;
			      J = j+j0;
			      K = k+k0;
    
			      x = delta*I - R_x ;
			      y = delta*J - R_y ;
			      z = delta*K - R_z ;   
			      r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
         
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
			      pVpsArray_TM[k][j][i] = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);          
			    }
 
      
		      /*
		       * calculate pseudocharge density and contribution to self energy at nodes 
		       * inside the overlap region from previously calculated pseudopotential values
		       * using finite difference stencil  
		       */
		      for(k=ZBJstart;k<=ZBJend;k++)
			for(j=YBJstart;j<=YBJend;j++)
			  for(i=XBJstart;i<=XBJend;i++)
			    {
			      Bval=0;Bval_TM=0;         
			      I=i-i0;
			      J=j-j0;
			      K=k-k0;
                                   
			      Bval=pVpsArray[K][J][I]*coeffs[0];
			      Bval_TM=pVpsArray_TM[K][J][I]*coeffs[0];
			      for(a=1;a<=o;a++)
				{
				  Bval+=(pVpsArray[K][J][I-a] + pVpsArray[K][J][I+a] + pVpsArray[K][J-a][I] +        
					 pVpsArray[K][J+a][I] + pVpsArray[K-a][J][I] + pVpsArray[K+a][J][I])*coeffs[a];

				  Bval_TM+=(pVpsArray_TM[K][J][I-a] + pVpsArray_TM[K][J][I+a] + pVpsArray_TM[K][J-a][I] +        
					    pVpsArray_TM[K][J+a][I] + pVpsArray_TM[K-a][J][I] + pVpsArray_TM[K+a][J][I])*coeffs[a];          
				}
                    
			      pBArray[k-ZBJstart][j-YBJstart][i-XBJstart] = Bval; 
			      pBArray_TM[k-ZBJstart][j-YBJstart][i-XBJstart] = Bval_TM; 
          
          
			    } 

		      ForceProcComponentz = 0.0;
		      ForceProcComponenty = 0.0;
		      ForceProcComponentx = 0.0;

		      /*
		       * calculate contribution of force	      
		       */
		      for(k=Zstart;k<=Zend;k++) 
			for(j=Ystart;j<=Yend;j++)  
			  for(i=Xstart;i<=Xend;i++) 
			    {  
			      GradBJx=0;GradBJy=0;GradBJz=0; 
			      GradBJx_TM=0;GradBJy_TM=0;GradBJz_TM=0; 
			      GradVJx=0,GradVJy=0,GradVJz=0;
			      GradVJx_TM=0,GradVJy_TM=0,GradVJz_TM=0;

			      I=i-i0;
			      J=j-j0;
			      K=k-k0;    
      
			      for(a=1;a<=o;a++)
				{
		
				  GradBJz += (pBArray[K-o+a][J-o][I-o] - pBArray[K-o-a][J-o][I-o])*coeffs_grad[a];
				  GradBJz_TM += (pBArray_TM[K-o+a][J-o][I-o] - pBArray_TM[K-o-a][J-o][I-o])*coeffs_grad[a];
				  GradVJz += (pVpsArray[K+a][J][I] - pVpsArray[K-a][J][I])*coeffs_grad[a];
				  GradVJz_TM += (pVpsArray_TM[K+a][J][I] - pVpsArray_TM[K-a][J][I])*coeffs_grad[a];
	    
				  GradBJy += (pBArray[K-o][J-o+a][I-o] - pBArray[K-o][J-o-a][I-o])*coeffs_grad[a];
				  GradBJy_TM += (pBArray_TM[K-o][J-o+a][I-o] - pBArray_TM[K-o][J-o-a][I-o])*coeffs_grad[a];
				  GradVJy += (pVpsArray[K][J+a][I] - pVpsArray[K][J-a][I])*coeffs_grad[a];
      
				  GradVJy_TM += (pVpsArray_TM[K][J+a][I] - pVpsArray_TM[K][J-a][I])*coeffs_grad[a];
             
				  GradBJx += (pBArray[K-o][J-o][I-o+a] - pBArray[K-o][J-o][I-o-a])*coeffs_grad[a];                 
				  GradBJx_TM += (pBArray_TM[K-o][J-o][I-o+a] - pBArray_TM[K-o][J-o][I-o-a])*coeffs_grad[a];     
				  GradVJx += (pVpsArray[K][J][I+a] - pVpsArray[K][J][I-a])*coeffs_grad[a];
                     
				  GradVJx_TM += (pVpsArray_TM[K][J][I+a] - pVpsArray_TM[K][J][I-a])*coeffs_grad[a];
	                
				}

			      PhicmVtmJ= PotPhicArrGlbIdx[k][j][i]- pVpsArray_TM[K][J][I];
			      PhicpVpsJ= PotPhicArrGlbIdx[k][j][i]+ pVpsArray[K][J][I];
			      BtmpBps  = BlcArrGlbIdx_TM[k][j][i] + BlcArrGlbIdx[k][j][i];

			      ForceProcComponentz+= -GradBJz_TM*PhicmVtmJ-GradBJz*PhicpVpsJ-(GradVJz_TM-GradVJz)*BtmpBps
				-GradVJz*pVpsArray[K][J][I]+GradVJz_TM*pVpsArray_TM[K][J][I];

			      ForceProcComponenty+= -GradBJy_TM*PhicmVtmJ-GradBJy*PhicpVpsJ-(GradVJy_TM-GradVJy)*BtmpBps
				-GradVJy*pVpsArray[K][J][I]+GradVJy_TM*pVpsArray_TM[K][J][I];

			      ForceProcComponentx+= -GradBJx_TM*PhicmVtmJ-GradBJx*PhicpVpsJ-(GradVJx_TM-GradVJx)*BtmpBps
				-GradVJx*pVpsArray[K][J][I]+GradVJx_TM*pVpsArray_TM[K][J][I];  
          
			    }
      
		      MPI_Comm_rank(MPI_COMM_WORLD,&rank);	 
		      MatSetValue(ForceProcMatz,rank,poscnt,ForceProcComponentz,ADD_VALUES);	
		      MatSetValue(ForceProcMaty,rank,poscnt,ForceProcComponenty,ADD_VALUES);
		      MatSetValue(ForceProcMatx,rank,poscnt,ForceProcComponentx,ADD_VALUES);

		      for(i = 0; i < nzVps; i++)
			{
			  for(j=0;j<nyVps;j++)
			    {
			      PetscFree(pVpsArray[i][j]);                    
			    }
			  PetscFree(pVpsArray[i]);
			}  
		      PetscFree(pVpsArray);   
     
		      for(i = 0; i < nzBJ; i++)
			{
			  for(j=0;j<nyBJ;j++)
			    {
			      PetscFree(pBArray[i][j]);                    
			    }
			  PetscFree(pBArray[i]);
			}  
		      PetscFree(pBArray); 
       
		    } 
   
		}
 
	  index_mvatm = index_mvatm+3;
	}  
      PetscFree(YD); 
    }

  DMDAVecRestoreArray(pSddft->da,pSddft->Phi_c,&PotPhicArrGlbIdx);   
  DMDAVecRestoreArray(pSddft->da,pSddft->chrgDensB,&BlcArrGlbIdx);   
  DMDAVecRestoreArray(pSddft->da,pSddft->chrgDensB_TM,&BlcArrGlbIdx_TM); 
  VecRestoreArray(pSddft->Atompos,&pAtompos);  
  VecRestoreArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);

  MatAssemblyBegin(ForceProcMatx,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ForceProcMatx,MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ForceProcMaty,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ForceProcMaty,MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ForceProcMatz,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ForceProcMatz,MAT_FINAL_ASSEMBLY);	

  for(poscnt=0;poscnt<pSddft->nAtoms;poscnt++)
    {
      MatGetColumnVector(ForceProcMatx,forcex_corr,poscnt);
      MatGetColumnVector(ForceProcMaty,forcey_corr,poscnt);
      MatGetColumnVector(ForceProcMatz,forcez_corr,poscnt);
		
      VecSum(forcex_corr,&FORCE_corr);pSddft->pForces_corr[index_force++] = -0.5*delta*delta*delta*FORCE_corr;
      VecSum(forcey_corr,&FORCE_corr);pSddft->pForces_corr[index_force++] = -0.5*delta*delta*delta*FORCE_corr;
      VecSum(forcez_corr,&FORCE_corr);pSddft->pForces_corr[index_force++] = -0.5*delta*delta*delta*FORCE_corr;
    }

  VecDestroy(&forcex_corr);
  VecDestroy(&forcey_corr);
  VecDestroy(&forcez_corr);  

  MatDestroy(&ForceProcMatx);
  MatDestroy(&ForceProcMaty);
  MatDestroy(&ForceProcMatz);
      
  return;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//  PeriodicForce_Nonlocal: nonlocal component of force on atoms for periodic systems        // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode PeriodicForce_Nonlocal(SDDFT_OBJ* pSddft,Mat* Psi)
{
  PetscScalar *pAtompos; 
  PetscScalar *pmvAtmConstraint;

  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0,colidx;
  PetscScalar x0, y0, z0, X0,Y0,Z0,cutoffr,max1,max2,max=0;
  int start,end,lmax,lloc,p,a;  
  PetscScalar tableR[MAX_TABLE_SIZE],tableUlDeltaV[4][MAX_TABLE_SIZE],coeffs_grad[MAX_ORDER+1],tableU[4][MAX_TABLE_SIZE];
  
  PetscScalar pUlDeltaVl;
  PetscInt rowStart,rowEnd;
  PetscScalar gi;
  PetscScalar *arrPsiSeq,*arrGradPsiSeq;

  PetscScalar ****W=NULL;
  PetscScalar ****T=NULL;

  PetscScalar *Dexact=NULL,Dtemp;      
  PetscScalar **YDUlDeltaV=NULL;
  PetscScalar **YDU=NULL;
  PetscErrorCode ierr;
  PetscScalar SpHarmonic,UlDelVl,Ul,fJ_x,fJ_y,fJ_z;

  PetscScalar Rcut;
  PetscScalar dr=1e-3;

  int OrbitalSize,rank;
  PetscInt Nposx,Nposy,Nposz,PP,QQ,RR; 

  PetscInt  i=0,j,k,xi,yj,zk,offset,I,J,K,II,JJ,KK,nzVps,nyVps,nxVps,nzVpsloc,nyVpsloc,nxVpsloc,i0,j0,k0,poscnt,index=0,at,ii,jj,kk,l,m,index_mvatm=0;
  PetscInt o = pSddft->order;
  PetscScalar delta=pSddft->delta,x,y,z,r,rmax,rtemp;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;

  int count,coord;
  PetscMPIInt comm_size;
  PetscInt s;PetscScalar temp;
      
  PetscInt xm,ym,zm,starts[3],dims[3];
  ISLocalToGlobalMapping ltog;
    
  AO aodmda1;
  ierr = DMDAGetAO(pSddft->da,&aodmda1);
  PetscInt LIrow;
 
  PetscMalloc(sizeof(PetscScalar***)*pSddft->nAtoms,&W);
  PetscMalloc(sizeof(PetscScalar***)*pSddft->nAtoms,&T);
  if(T == NULL || W == NULL)
    {
      printf("memory allocation failed in T or W for forces");
      exit(1);
    }
  for(i=0;i<pSddft->nAtoms;i++)
    {     
      PetscMalloc(sizeof(PetscScalar**)*pSddft->Nstates,&W[i]);
      PetscMalloc(sizeof(PetscScalar**)*pSddft->Nstates,&T[i]);
      if(T[i] == NULL || W[i] == NULL)
	{
	  printf("memory allocation fail");
	  exit(1);
	}
    }
      
  MatGetOwnershipRange(*Psi,&rowStart,&rowEnd);       
  DMDAGetCorners(pSddft->da,0,0,0,&xm,&ym,&zm);
  DMGetLocalToGlobalMapping(pSddft->da,&ltog);
      
  DMDAGetGhostCorners(pSddft->da,&starts[0],&starts[1],&starts[2],&dims[0],&dims[1],&dims[2]);

  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
   
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
 
  PetscScalar OrbScalingFactor;
  PetscReal norms[pSddft->Nstates];
  MatGetColumnNorms(*Psi,NORM_2,norms);
  /*
   * scaling factor for normalization of orbitals
   */
  OrbScalingFactor = 1.0/(sqrt(delta*delta*delta)*norms[3]);  
  MatDestroy(&pSddft->YOrbNew);
  ierr = MatDuplicate(*Psi,MAT_SHARE_NONZERO_PATTERN,&pSddft->YOrbNew);
  CHKERRQ(ierr);

  VecGetArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);          
  VecGetArray(pSddft->Atompos,&pAtompos);        

  /*
   * Calculate x-component of forces
   */
  index_mvatm=0;index=0;
  /*
   * Calculate gradient of orbitals in x-direction
   */
  MatMatMultSymbolic(pSddft->gradient_x,*Psi,PETSC_DEFAULT,&pSddft->YOrbNew);
  MatMatMultNumeric(pSddft->gradient_x,*Psi,pSddft->YOrbNew);	
    
    	    
  for(at=0;at<pSddft->Ntype;at++)
    {                          
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];	            
	   
      start = floor(pSddft->startPos[at]/3);
      end = floor(pSddft->endPos[at]/3);

      /*
       * if the atom is hydrogen, then we omit the calculation of non-local forces
       */
      if(lmax==0) 
	{
	  index = index + 3*(end-start+1); 
	  index_mvatm = index_mvatm + 3*(end-start+1);
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
	  }while(tableR[count-1] <= 1.732*pSddft->CUTOFF[at]+10.0); 
	  rmax = tableR[count-1];
                                            
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDUlDeltaV);
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDU);           
	  for(l=0;l<=lmax;l++)
	    {
	      PetscMalloc(sizeof(PetscScalar)*count,&YDUlDeltaV[l]);
	      PetscMalloc(sizeof(PetscScalar)*count,&YDU[l]);   
	      /*
	       * derivatives of the spline fit to the projectors an pseudowavefunctions
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
		      ispline_gen(tableR,tableUlDeltaV[l],count,&rtemp,&UlDelVl,&Dtemp,1,YDUlDeltaV[l]);
		      ispline_gen(tableR,tableU[l],count,&rtemp,&Ul,&Dtemp,1,YDU[l]);
											
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

	      for(i=0;i<pSddft->Nstates;i++)
		{				    
		  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&W[poscnt][i]);
		  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&T[poscnt][i]);
		  if( W[poscnt][i] == NULL || T[poscnt][i] == NULL )
		    {
		      printf("W[][] || T"); exit(1);
		    }
			     			    
		  for(l=0;l<=lmax;l++)
		    {			
		      PetscMalloc(sizeof(PetscScalar)*(2*l+1),&W[poscnt][i][l]);
		      PetscMalloc(sizeof(PetscScalar)*(2*l+1),&T[poscnt][i][l]);
		      if( W[poscnt][i][l] == NULL || T[poscnt][i][l] == NULL )
			{
			  printf("W[][][] T[][][]"); exit(1);
			}				 			 
		      
		      for(m=-l;m<=l;m++)
			{	     
			  W[poscnt][i][l][m+l]=0.0;
			  T[poscnt][i][l][m+l]=0.0;				      
			}			
		    }
		}
	      
	      Nposx = ceil(cutoffr/R_x);
	      Nposy = ceil(cutoffr/R_y);
	      Nposz = ceil(cutoffr/R_z);
	  
	      for(PP= -Nposx; PP<=Nposx; PP++)
		for(QQ= -Nposy; QQ<=Nposy; QQ++)
		  for(RR= -Nposz; RR<=Nposz; RR++)
		    {
		      x0 = X0 + PP*2.0*R_x;
		      y0 = Y0 + QQ*2.0*R_y;
		      z0 = Z0 + RR*2.0*R_z;

		      xstart=-1000; ystart=-1000; zstart=-1000; xend=-1000; yend=-1000; zend=-1000; overlap=0;
		      
		      xi = roundf((x0 + R_x)/delta); 
		      yj = roundf((y0 + R_y)/delta);
		      zk = roundf((z0 + R_z)/delta);
		      
		      xs = xi-offset; xl = xi+offset;
		      ys = yj-offset; yl = yj+offset;
		      zs = zk-offset; zl = zk+offset;
     
		      /*
		       * find if domain of influence of pseudopotential overlaps with the domain 
		       * stored by processor  
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

			  i0 = xs-o;
			  j0 = ys-o;
			  k0 = zs-o;
			    							
			  /*
			   * Get local pointer (column wise) to the wavefunctions and gradient of 
			   * wavefunctions  
			   */		    			     
			  MatDenseGetArray(*Psi,&arrPsiSeq);		    
			  MatDenseGetArray(pSddft->YOrbNew,&arrGradPsiSeq); 

			  /* 
			   * evaluate the the quantities that contribute to nonlocal force at 
			   * nodes in the overlap region
			   */
			  for(k=zstart;k<=zend;k++)
			    for(j=ystart;j<=yend;j++)
			      for(i=xstart;i<=xend;i++)
				{				  
				  I=i-i0;
				  J=j-j0;
				  K=k-k0; 
				 
				  x = delta*i-R_x;
				  y = delta*j-R_y;
				  z = delta*k-R_z;
				  r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
				 
				  LIrow = k*n_x*n_y + j*n_x + i; 
				  AOApplicationToPetsc(aodmda1,1,&LIrow);
				  for(l=0;l<=lmax;l++)
				    {
				      if(l!=lloc)
					{ 
					  if(r == tableR[0])
					    {
					      pUlDeltaVl = tableUlDeltaV[l][0];	
					    }else{
					    ispline_gen(tableR,tableUlDeltaV[l],count,&r,&pUlDeltaVl,&Dtemp,1,YDUlDeltaV[l]);
					  }						 
					  if(l==0)
					    Rcut= pSddft->psd[at].rc_s;
					  if(l==1)
					    Rcut= pSddft->psd[at].rc_p;
					  if(l==2)
					    Rcut= pSddft->psd[at].rc_d;
					  if(l==3)
					    Rcut= pSddft->psd[at].rc_f;		      

					  for(m=-l;m<=l;m++)
					    {	
					      SpHarmonic=SphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut+1.0);
					      for(ii=0;ii<pSddft->Nstates;ii++)
						{
						  OrbScalingFactor = 1.0/(sqrt(delta*delta*delta)*norms[ii]);
						  W[poscnt][ii][l][m+l] += OrbScalingFactor*(arrGradPsiSeq[LIrow-rowStart + (rowEnd-rowStart)*ii])*pUlDeltaVl*SpHarmonic;	    
						  T[poscnt][ii][l][m+l] += OrbScalingFactor*(arrPsiSeq[LIrow-rowStart + (rowEnd-rowStart)*ii])*pUlDeltaVl*SpHarmonic/Dexact[l];
					     						     
						}						
					    }
					}
				    }
				}			    
			  MatDenseRestoreArray(*Psi,&arrPsiSeq); 
			  MatDenseRestoreArray(pSddft->YOrbNew,&arrGradPsiSeq);			     
			     
			}
		    }	     
	      index_mvatm = index_mvatm+3;
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
   * sum the contributions across all the processors and compute the forces
   */      
  for(at=0;at<pSddft->Ntype;at++)
    {
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];
      if(lmax!=0)
	{	  
	  start = (int)floor(pSddft->startPos[at]/3);
	  end = (int)floor(pSddft->endPos[at]/3);
	  for(poscnt=start;poscnt<=end;poscnt++)
	    {	      
	      fJ_x=0.0;
	      for(i=0;i<pSddft->Nstates;i++)
		{  gi = smearing_FermiDirac(pSddft->Beta,pSddft->lambda[i],pSddft->lambda_f);	
		  for(l=0;l<=lmax;l++)
		    {
		      if(l!=lloc)
			{
			  for(m=-l;m<=l;m++)
			    {			      
			      ierr =  MPI_Allreduce(MPI_IN_PLACE,&T[poscnt][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
			      ierr =  MPI_Allreduce(MPI_IN_PLACE,&W[poscnt][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);			     
			      fJ_x += gi*W[poscnt][i][l][m+l]*T[poscnt][i][l][m+l];			     
			    }
			}
		    }		  
		}
	      	      
	      fJ_x = -4.0*delta*delta*delta*fJ_x;	     
	      VecSetValue(pSddft->forces,3*poscnt,fJ_x,ADD_VALUES);	      
      	    }
	}
    }
  /*
   * end of x-component force calculation 
   */

  /*
   * Calculate y-component of forces
   */

  /* NOTE: we do no need to calculate T since it has already been calculated while 
   * calculating the x-component of forces. Hence we do not need to calculate the exact 
   * denominator. Also, memory for W has already been created, we dont need to allocate again,
   * just need to set it to zero.
   */

  index_mvatm=0;index=0;
  MatMatMultNumeric(pSddft->gradient_y,*Psi,pSddft->YOrbNew);	
  /*
   * loop over different types of atoms 
   */   	    
  for(at=0;at<pSddft->Ntype;at++)
    {                          
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];	

      start = floor(pSddft->startPos[at]/3);
      end = floor(pSddft->endPos[at]/3);

      /*
       * if the atom is hydrogen, then we omit the calculation of non-local forces
       */
      if(lmax==0)
	{
	  index = index + 3*(end-start+1); 
	  index_mvatm = index_mvatm + 3*(end-start+1);
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
	      if(l==0)
		{
		  tableUlDeltaV[0][0]=pSddft->psd[at].Us[0]*(pSddft->psd[at].Vs[0]-pSddft->psd[at].Vloc[0]);		
		}
	      if(l==1)
		{
		  tableUlDeltaV[1][0]=0.0; 			     
		}
	      if(l==2)
		{
		  tableUlDeltaV[2][0]=0.0; 			      
		}
	      if(l==3)
		{
		  tableUlDeltaV[3][0]=0.0; 		   	  
		}
	    }
	  count=1;
	  do{
	    tableR[count] = pSddft->psd[at].RadialGrid[count-1];
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
		    tableUlDeltaV[0][count]=pSddft->psd[at].Us[count-1]*(pSddft->psd[at].Vs[count-1]-pSddft->psd[at].Vloc[count-1]);			             
		  }
		if(l==1)
		  {
		    tableUlDeltaV[1][count]=pSddft->psd[at].Up[count-1]*(pSddft->psd[at].Vp[count-1]-pSddft->psd[at].Vloc[count-1]);	         	       
		  }
		if(l==2)
		  {
		    tableUlDeltaV[2][count]=pSddft->psd[at].Ud[count-1]*(pSddft->psd[at].Vd[count-1]-pSddft->psd[at].Vloc[count-1]);			   	       
		  }   
		if(l==3)
		  {
		    tableUlDeltaV[3][count]=pSddft->psd[at].Uf[count-1]*(pSddft->psd[at].Vf[count-1]-pSddft->psd[at].Vloc[count-1]);			            
		  }  
	      }
	    count++;
	  }while(tableR[count-1] <= 1.732*pSddft->CUTOFF[at]+10.0); 
	  rmax = tableR[count-1];
                                         
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDUlDeltaV);	  
	  for(l=0;l<=lmax;l++)
	    {
	      /*
	       * derivatives of the spline fit to the projectors
	       */
	      PetscMalloc(sizeof(PetscScalar)*count,&YDUlDeltaV[l]);		         
	      getYD_gen(tableR,tableUlDeltaV[l],YDUlDeltaV[l],count);             
	    } 	  
	
	  /*
	   * loop over every atom of a given type
	   */
	  for(poscnt=start;poscnt<=end;poscnt++)
	    { 
		  
	      X0 = pAtompos[index++];
	      Y0 = pAtompos[index++];
	      Z0 = pAtompos[index++];

	      for(i=0;i<pSddft->Nstates;i++)
		{    			    
		  for(l=0;l<=lmax;l++)
		    { 
		      for(m=-l;m<=l;m++)
			{	     
			  W[poscnt][i][l][m+l]=0.0;		      
			}			
		    }
		}       

	      Nposx = ceil(cutoffr/R_x);
	      Nposy = ceil(cutoffr/R_y);
	      Nposz = ceil(cutoffr/R_z);
	  
	      for(PP= -Nposx; PP<=Nposx; PP++)
		for(QQ= -Nposy; QQ<=Nposy; QQ++)
		  for(RR= -Nposz; RR<=Nposz; RR++)
		    {
		      x0 = X0 + PP*2.0*R_x;
		      y0 = Y0 + QQ*2.0*R_y;
		      z0 = Z0 + RR*2.0*R_z;			

		      xstart=-1000; ystart=-1000; zstart=-1000; xend=-1000; yend=-1000; zend=-1000; overlap=0;
		      
		      xi = roundf((x0 + R_x)/delta); 
		      yj = roundf((y0 + R_y)/delta);
		      zk = roundf((z0 + R_z)/delta);
		      		      
		      xs = xi-offset; xl = xi+offset;
		      ys = yj-offset; yl = yj+offset;
		      zs = zk-offset; zl = zk+offset;
     
		      /*
		       * find if domain of influence of pseudopotential overlaps with the domain
		       * stored by processor  
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
			  i0 = xs-o;
			  j0 = ys-o;
			  k0 = zs-o;
			  			  
			  /*
			   * Get local pointer (column wise) to the gradient of wavefunctions  
			   */
			  MatDenseGetArray(pSddft->YOrbNew,&arrGradPsiSeq); 
			    
			  for(k=zstart;k<=zend;k++)
			    for(j=ystart;j<=yend;j++)
			      for(i=xstart;i<=xend;i++)
				{				 
				  I=i-i0;
				  J=j-j0;
				  K=k-k0; 
				  
				  x = delta*i-R_x;
				  y = delta*j-R_y;
				  z = delta*k-R_z;
				  r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
				  
				  LIrow = k*n_x*n_y + j*n_x + i; 
				  AOApplicationToPetsc(aodmda1,1,&LIrow);
				  for(l=0;l<=lmax;l++)
				    {
				      if(l!=lloc)
					{ 					  
					  if(r == tableR[0])
					    {
					      pUlDeltaVl = tableUlDeltaV[l][0];	
					    }else{
					    ispline_gen(tableR,tableUlDeltaV[l],count,&r,&pUlDeltaVl,&Dtemp,1,YDUlDeltaV[l]);
					  }						 
					  if(l==0)
					    Rcut= pSddft->psd[at].rc_s;
					  if(l==1)
					    Rcut= pSddft->psd[at].rc_p;
					  if(l==2)
					    Rcut= pSddft->psd[at].rc_d;
					  if(l==3)
					    Rcut= pSddft->psd[at].rc_f;
					  for(m=-l;m<=l;m++)
					    {	
					      SpHarmonic=SphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut+1.0);
					      for(ii=0;ii<pSddft->Nstates;ii++)
						{
						  OrbScalingFactor = 1.0/(sqrt(delta*delta*delta)*norms[ii]);
						  W[poscnt][ii][l][m+l] += OrbScalingFactor*(arrGradPsiSeq[LIrow-rowStart + (rowEnd-rowStart)*ii])*pUlDeltaVl*SpHarmonic;		     
						}						
					    }
					}
				    }
				}
			    
			  MatDenseRestoreArray(pSddft->YOrbNew,&arrGradPsiSeq);

			}
		    }	     
	      index_mvatm = index_mvatm+3;
	    }	    
	 
	  for(l=0;l<=lmax;l++)
	    {
	      PetscFree(YDUlDeltaV[l]);	       	    
	    }
	  PetscFree(YDUlDeltaV); 
	    
	}
    }     
  
  /*
   * sum the contributions across all the processors and compute the forces
   */
  for(at=0;at<pSddft->Ntype;at++)
    {
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];
      if(lmax!=0)
	{
	  start = (int)floor(pSddft->startPos[at]/3);
	  end = (int)floor(pSddft->endPos[at]/3);
	  for(poscnt=start;poscnt<=end;poscnt++)
	    {	      
	      fJ_y=0.0;
	      for(i=0;i<pSddft->Nstates;i++)
		{  gi = smearing_FermiDirac(pSddft->Beta,pSddft->lambda[i],pSddft->lambda_f);
		  for(l=0;l<=lmax;l++)
		    {
		      if(l!=lloc)
			{
			  for(m=-l;m<=l;m++)
			    {			     
			      ierr =  MPI_Allreduce(MPI_IN_PLACE,&W[poscnt][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);			     
			      fJ_y += gi*W[poscnt][i][l][m+l]*T[poscnt][i][l][m+l];			     
			    }
			}
		    }		  
		}	     	      
	      fJ_y = -4.0*delta*delta*delta*fJ_y;	     
	      VecSetValue(pSddft->forces,3*poscnt+1,fJ_y,ADD_VALUES);	      
      	    }
	}
    }
  /*
   * end of y-component force calculation 
   */


  /*
   * Calculate z-component of forces
   */

  /* we do no need to calculate T since it has already been calculated while calculating the 
   * x-component of forces. Hence we do not need to calculate the exact denominator. Also,
   * memory for W has already been created, we dont need to allocate again, just need to set it
   * to zero
   */

  index_mvatm=0;index=0;  
  /*
   * Calculate gradient of orbitals in z-direction
   */
  MatMatMultNumeric(pSddft->gradient_z,*Psi,pSddft->YOrbNew);	
  /*
   * loop over different types of atoms 
   */
  for(at=0;at<pSddft->Ntype;at++)
    {                          
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];	 
	   
      start = floor(pSddft->startPos[at]/3);
      end = floor(pSddft->endPos[at]/3);

      if(lmax==0)
	{
	  index = index + 3*(end-start+1);
	  index_mvatm = index_mvatm + 3*(end-start+1);
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
		}
	      if(l==1)
		{
		  tableUlDeltaV[1][0]=0.0; 			     
		}
	      if(l==2)
		{
		  tableUlDeltaV[2][0]=0.0; 				      
		}
	      if(l==3)
		{
		  tableUlDeltaV[3][0]=0.0; 	   	  
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
		  }
		if(l==1)
		  {
		    tableUlDeltaV[1][count]=pSddft->psd[at].Up[count-1]*(pSddft->psd[at].Vp[count-1]-pSddft->psd[at].Vloc[count-1]); 		         	       
		  }
		if(l==2)
		  {
		    tableUlDeltaV[2][count]=pSddft->psd[at].Ud[count-1]*(pSddft->psd[at].Vd[count-1]-pSddft->psd[at].Vloc[count-1]);			   	       
		  }   
		if(l==3)
		  {
		    tableUlDeltaV[3][count]=pSddft->psd[at].Uf[count-1]*(pSddft->psd[at].Vf[count-1]-pSddft->psd[at].Vloc[count-1]);		            
		  }  
	      }
	    count++;
	  }while(tableR[count-1] <= 1.732*pSddft->CUTOFF[at]+10.0);
	  rmax = tableR[count-1];
                                      
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDUlDeltaV);	  
	  for(l=0;l<=lmax;l++)
	    {
	      /*
	       * derivatives of the spline fit to the projectors
	       */
	      PetscMalloc(sizeof(PetscScalar)*count,&YDUlDeltaV[l]);		         
	      getYD_gen(tableR,tableUlDeltaV[l],YDUlDeltaV[l],count);             
	    } 	  
	
	  /*
	   * loop over every atom of a given type
	   */
	  for(poscnt=start;poscnt<=end;poscnt++)
	    { 
	      X0 = pAtompos[index++];
	      Y0 = pAtompos[index++];
	      Z0 = pAtompos[index++];

	      for(i=0;i<pSddft->Nstates;i++)
		{    			    
		  for(l=0;l<=lmax;l++)
		    { 
		      for(m=-l;m<=l;m++)
			{	     
			  W[poscnt][i][l][m+l]=0.0;		      
			}			
		    }
		}
	       
	      Nposx = ceil(cutoffr/R_x);
	      Nposy = ceil(cutoffr/R_y);
	      Nposz = ceil(cutoffr/R_z);
	  
	      for(PP= -Nposx; PP<=Nposx; PP++)
		for(QQ= -Nposy; QQ<=Nposy; QQ++)
		  for(RR= -Nposz; RR<=Nposz; RR++)
		    {
		      x0 = X0 + PP*2.0*R_x;
		      y0 = Y0 + QQ*2.0*R_y;
		      z0 = Z0 + RR*2.0*R_z;		       

		      /*
		       * find if domain of influence of pseudopotential overlaps with the domain
		       * stored by processor  
		       */
		      xstart=-1000; ystart=-1000; zstart=-1000; xend=-1000; yend=-1000; zend=-1000; overlap=0;
		     
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

			  i0 = xs-o;
			  j0 = ys-o;
			  k0 = zs-o;
			  /*
			   * Get local pointer (column wise) to the gradient of wavefunctions  
			   */				   		    
			  MatDenseGetArray(pSddft->YOrbNew,&arrGradPsiSeq); 
			    
			  /* 
			   * evaluate the the quantities that contribute to nonlocal force at 
			   * nodes in the overlap region
			   */
			  for(k=zstart;k<=zend;k++)
			    for(j=ystart;j<=yend;j++)
			      for(i=xstart;i<=xend;i++)
				{
				  I=i-i0;
				  J=j-j0;
				  K=k-k0; 
				 
				  x = delta*i-R_x;
				  y = delta*j-R_y;
				  z = delta*k-R_z;
				  r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));

				  LIrow = k*n_x*n_y + j*n_x + i; 
				  AOApplicationToPetsc(aodmda1,1,&LIrow);
				  for(l=0;l<=lmax;l++)
				    {
				      if(l!=lloc)
					{ 
					  if(r == tableR[0])
					    {
					      pUlDeltaVl = tableUlDeltaV[l][0];	
					    }else{
					    ispline_gen(tableR,tableUlDeltaV[l],count,&r,&pUlDeltaVl,&Dtemp,1,YDUlDeltaV[l]);
					  }						 
					  if(l==0)
					    Rcut= pSddft->psd[at].rc_s;
					  if(l==1)
					    Rcut= pSddft->psd[at].rc_p;
					  if(l==2)
					    Rcut= pSddft->psd[at].rc_d;
					  if(l==3)
					    Rcut= pSddft->psd[at].rc_f;
					  for(m=-l;m<=l;m++)
					    {	
					      SpHarmonic=SphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut+1.0);
					      for(ii=0;ii<pSddft->Nstates;ii++)
						{
						  OrbScalingFactor = 1.0/(sqrt(delta*delta*delta)*norms[ii]); 
						  W[poscnt][ii][l][m+l] += OrbScalingFactor*(arrGradPsiSeq[LIrow-rowStart + (rowEnd-rowStart)*ii])*pUlDeltaVl*SpHarmonic;		     
						}						
					    }
					}
				    }
				}		    
			  MatDenseRestoreArray(pSddft->YOrbNew,&arrGradPsiSeq);	
			}
		    }	     
	      index_mvatm = index_mvatm+3;
	    }	  
	  for(l=0;l<=lmax;l++)
	    {
	      PetscFree(YDUlDeltaV[l]);	       	    
	    }
	  PetscFree(YDUlDeltaV); 	    
	}
    }     
 
  /*
   * sum the contributions across all the processors and compute the forces
   */
  for(at=0;at<pSddft->Ntype;at++)
    {
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];
      if(lmax!=0)
	{
	  start = (int)floor(pSddft->startPos[at]/3);
	  end = (int)floor(pSddft->endPos[at]/3);
	  for(poscnt=start;poscnt<=end;poscnt++)
	    {	      
	      fJ_z=0.0;
	      for(i=0;i<pSddft->Nstates;i++)
		{  gi = smearing_FermiDirac(pSddft->Beta,pSddft->lambda[i],pSddft->lambda_f);
		  for(l=0;l<=lmax;l++)
		    {
		      if(l!=lloc)
			{
			  for(m=-l;m<=l;m++)
			    {			     
			      ierr =  MPI_Allreduce(MPI_IN_PLACE,&W[poscnt][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);			     
			      fJ_z += gi*W[poscnt][i][l][m+l]*T[poscnt][i][l][m+l];			     
			    }
			}
		    }		  
		}
	      	      
	      fJ_z = -4.0*delta*delta*delta*fJ_z;	     
	      VecSetValue(pSddft->forces,3*poscnt+2,fJ_z,ADD_VALUES);	      
      	    }
	}
    }
  /*
   * end of z-component force calculation 
   */
  VecRestoreArray(pSddft->Atompos,&pAtompos);
  VecRestoreArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);          
  VecAssemblyBegin(pSddft->forces);
  VecAssemblyEnd(pSddft->forces);    
   
  /*
   * free memory
   */
  for(at=0;at<pSddft->Ntype;at++)
    {
      lmax = pSddft->psd[at].lmax;
      
      int start,end;
      start = (int)floor(pSddft->startPos[at]/3);
      end = (int)floor(pSddft->endPos[at]/3);
	  
      for(poscnt=start;poscnt<=end;poscnt++)
	{	      
	  if(lmax!=0)
	    {
	      for(i=0;i<pSddft->Nstates;i++)
		{		 
		  for(l=0;l<=lmax;l++)
		    {			     
		      PetscFree(W[poscnt][i][l]);
		      PetscFree(T[poscnt][i][l]);			      
		    }			 
		  PetscFree(W[poscnt][i]);
		  PetscFree(T[poscnt][i]);
			 
		}
	
	    }
	  PetscFree(W[poscnt]);
	  PetscFree(T[poscnt]);		
	}
	 
      
    }
  PetscFree(W);
  PetscFree(T);       
  
  return ierr;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//   kPointPeriodicForce_Nonlocal: nonlocal component of force on atoms for k-point code     // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode kPointPeriodicForce_Nonlocal(SDDFT_OBJ* pSddft)
{
  PetscScalar *pAtompos; 
  PetscScalar *pmvAtmConstraint;

  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0,colidx;
  PetscScalar x0, y0, z0, X0,Y0,Z0,cutoffr,max1,max2,max=0,X,Y,Z;
  int start,end,lmax,lloc,p,a;  
  PetscScalar tableR[MAX_TABLE_SIZE],tableUlDeltaV[4][MAX_TABLE_SIZE],coeffs_grad[MAX_ORDER+1],tableU[4][MAX_TABLE_SIZE];
   
  PetscScalar pUlDeltaVl;
  PetscInt rowStart,rowEnd;
  PetscScalar gi;
  PetscScalar *arrPsi1Seq,*arrGradPsi1Seq,*arrPsi2Seq,*arrGradPsi2Seq;
 
  PetscScalar *****W1=NULL;
  PetscScalar *****T1=NULL;
  PetscScalar *****W2=NULL;
  PetscScalar *****T2=NULL;
      
  PetscScalar *Dexact=NULL,Dtemp;      
  PetscScalar **YDUlDeltaV=NULL;
  PetscScalar **YDU=NULL;
  PetscErrorCode ierr;
  PetscScalar SH_real,SH_imag,UlDelVl,Ul,fJ_x,fJ_y,fJ_z;

  PetscScalar Rcut;
  PetscScalar dr=1e-3;
  int kpt;
  PetscScalar k1,k2,k3;

  int OrbitalSize,rank;
  PetscInt Nposx,Nposy,Nposz,PP,QQ,RR; // number of positions we need to move the atom to get the right forces.

  PetscInt  i=0,j,k,xi,yj,zk,offset,I,J,K,II,JJ,KK,nzVps,nyVps,nxVps,nzVpsloc,nyVpsloc,nxVpsloc,i0,j0,k0,poscnt,index=0,at,ii,jj,kk,l,m,index_mvatm=0;
  PetscInt o = pSddft->order;
  PetscScalar delta=pSddft->delta,x,y,z,r,rmax,rtemp;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscScalar a1,a2,b1,b2;
  int nk1,nk2,nk3;

  int count,coord;
  PetscMPIInt comm_size;
  PetscInt s;PetscScalar temp;
      
  PetscInt xm,ym,zm,starts[3],dims[3];
  ISLocalToGlobalMapping ltog;
  
  AO aodmda1;
  ierr = DMDAGetAO(pSddft->da,&aodmda1);
  PetscInt LIrow;
   
 
  PetscMalloc(sizeof(PetscScalar****)*pSddft->nAtoms,&W1);
  PetscMalloc(sizeof(PetscScalar****)*pSddft->nAtoms,&T1);
  PetscMalloc(sizeof(PetscScalar****)*pSddft->nAtoms,&W2);
  PetscMalloc(sizeof(PetscScalar****)*pSddft->nAtoms,&T2);

  if(T1 == NULL || W1 == NULL || T2 == NULL || W2 == NULL)
    {
      printf("memory allocation failed in T or W for forces");
      exit(1);
    }
  for(i=0;i<pSddft->nAtoms;i++)
    {      
      PetscMalloc(sizeof(PetscScalar***)*pSddft->Nkpts_sym,&W1[i]);
      PetscMalloc(sizeof(PetscScalar***)*pSddft->Nkpts_sym,&T1[i]);
      PetscMalloc(sizeof(PetscScalar***)*pSddft->Nkpts_sym,&W2[i]);
      PetscMalloc(sizeof(PetscScalar***)*pSddft->Nkpts_sym,&T2[i]);

      if(T1[i] == NULL || W1[i] == NULL || T2[i] == NULL || W2[i] == NULL)
	{
	  printf("memory allocation fail");
	  exit(1);
	}
    }
      
  MatGetOwnershipRange(pSddft->XOrb1[0],&rowStart,&rowEnd); 
      
  DMDAGetCorners(pSddft->da,0,0,0,&xm,&ym,&zm);
  DMGetLocalToGlobalMapping(pSddft->da,&ltog);
      
  DMDAGetGhostCorners(pSddft->da,&starts[0],&starts[1],&starts[2],&dims[0],&dims[1],&dims[2]);

  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
   
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  
  PetscScalar OrbScalingFactor;
  PetscReal norms[pSddft->Nstates];
  PetscReal norms1[pSddft->Nstates];
 
  MatGetColumnNorms(pSddft->XOrb1[0],NORM_2,norms);
  MatGetColumnNorms(pSddft->XOrb2[0],NORM_2,norms1);
  
  VecGetArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);          
  VecGetArray(pSddft->Atompos,&pAtompos);        

  /*
   * Calculate x-component of forces
   */
  index_mvatm=0;index=0;
  /*
   * Calculate gradient of orbitals in x-direction
   */
  for(k=0;k<pSddft->Nkpts_sym;k++)
    {
      MatZeroEntries(pSddft->YOrb1[k]);
      MatZeroEntries(pSddft->YOrb2[k]);
      MatMatMultSymbolic(pSddft->gradient_x,pSddft->XOrb1[k],PETSC_DEFAULT,&pSddft->YOrb1[k]);
      MatMatMultNumeric(pSddft->gradient_x,pSddft->XOrb1[k],pSddft->YOrb1[k]);	
      MatMatMultSymbolic(pSddft->gradient_x,pSddft->XOrb2[k],PETSC_DEFAULT,&pSddft->YOrb2[k]);
      MatMatMultNumeric(pSddft->gradient_x,pSddft->XOrb2[k],pSddft->YOrb2[k]);	
    }

  /*
   * loop over different types of atoms 
   */
  for(at=0;at<pSddft->Ntype;at++)
    {                          
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];	            
	   
      start = floor(pSddft->startPos[at]/3);
      end = floor(pSddft->endPos[at]/3);

      if(lmax==0) 
	{
	  index = index + 3*(end-start+1);
	  index_mvatm = index_mvatm + 3*(end-start+1);
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
	  }while(tableR[count-1] <= 1.732*pSddft->CUTOFF[at]+10.0); 
	  rmax = tableR[count-1];            
	
	                                 
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDUlDeltaV);
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDU);           
	  for(l=0;l<=lmax;l++)
	    {
	      PetscMalloc(sizeof(PetscScalar)*count,&YDUlDeltaV[l]);
	      PetscMalloc(sizeof(PetscScalar)*count,&YDU[l]);             
	      getYD_gen(tableR,tableUlDeltaV[l],YDUlDeltaV[l],count); 
	      getYD_gen(tableR,tableU[l],YDU[l],count);             
	    }        	 				    	
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
		      ispline_gen(tableR,tableUlDeltaV[l],count,&rtemp,&UlDelVl,&Dtemp,1,YDUlDeltaV[l]);
		      ispline_gen(tableR,tableU[l],count,&rtemp,&Ul,&Dtemp,1,YDU[l]);
											
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
	      
	      for(k=0;k<pSddft->Nkpts_sym;k++)
		{  			     
		  PetscMalloc(sizeof(PetscScalar**)*(pSddft->Nstates),&W1[poscnt][k]);
		  PetscMalloc(sizeof(PetscScalar**)*(pSddft->Nstates),&T1[poscnt][k]);
		  PetscMalloc(sizeof(PetscScalar**)*(pSddft->Nstates),&W2[poscnt][k]);
		  PetscMalloc(sizeof(PetscScalar**)*(pSddft->Nstates),&T2[poscnt][k]);
		  if( W1[poscnt][k] == NULL || T1[poscnt][k] == NULL || W2[poscnt][k] == NULL || T2[poscnt][k] == NULL)
		    {
		      printf("W[][] || T"); exit(1);
		    }		  
		  for(i=0;i<pSddft->Nstates;i++)
		    {

		      PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&W1[poscnt][k][i]);
		      PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&T1[poscnt][k][i]);
		      PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&W2[poscnt][k][i]);
		      PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&T2[poscnt][k][i]);
		      if( W1[poscnt][k][i] == NULL || T1[poscnt][k][i] == NULL || W2[poscnt][k][i] == NULL || T2[poscnt][k][i] == NULL)
			{
			  printf("W[][][] || T"); exit(1);
			}
			     			    
		      for(l=0;l<=lmax;l++)
			{			
			  PetscMalloc(sizeof(PetscScalar)*(2*l+1),&W1[poscnt][k][i][l]);
			  PetscMalloc(sizeof(PetscScalar)*(2*l+1),&T1[poscnt][k][i][l]);
			  PetscMalloc(sizeof(PetscScalar)*(2*l+1),&W2[poscnt][k][i][l]);
			  PetscMalloc(sizeof(PetscScalar)*(2*l+1),&T2[poscnt][k][i][l]);
			  if( W1[poscnt][k][i][l] == NULL || T1[poscnt][k][i][l] == NULL || W2[poscnt][k][i][l] == NULL || T2[poscnt][k][i][l] == NULL )
			    {
			      printf("W[][][][] T[][][][]"); exit(1);
			    }				 
			  for(m=-l;m<=l;m++)
			    {	     
			      W1[poscnt][k][i][l][m+l]=0.0;
			      T1[poscnt][k][i][l][m+l]=0.0;
			      W2[poscnt][k][i][l][m+l]=0.0;
			      T2[poscnt][k][i][l][m+l]=0.0;
			    }			
			}
		    }
		}
		 			 
	      Nposx = ceil(cutoffr/R_x);
	      Nposy = ceil(cutoffr/R_y);
	      Nposz = ceil(cutoffr/R_z);
	  
	      for(PP= -Nposx; PP<=Nposx; PP++)
		for(QQ= -Nposy; QQ<=Nposy; QQ++)
		  for(RR= -Nposz; RR<=Nposz; RR++)
		    {
		      x0 = X0 + PP*2.0*R_x;
		      y0 = Y0 + QQ*2.0*R_y;
		      z0 = Z0 + RR*2.0*R_z;

		      xstart=-1000; ystart=-1000; zstart=-1000; xend=-1000; yend=-1000; zend=-1000; overlap=0;
		      
		      xi = roundf((x0 + R_x)/delta); 
		      yj = roundf((y0 + R_y)/delta);
         	      zk = roundf((z0 + R_z)/delta);
	     

		      xs = xi-offset; xl = xi+offset;
		      ys = yj-offset; yl = yj+offset;
		      zs = zk-offset; zl = zk+offset;
		      
		      /*
		       * find if domain of influence of pseudopotential overlaps with the domain 
		       * stored by processor  
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

			  i0 = xs-o;
			  j0 = ys-o;
			  k0 = zs-o;			    							
			  
			  for(k=zstart;k<=zend;k++)
			    for(j=ystart;j<=yend;j++)
			      for(i=xstart;i<=xend;i++)
				{				  
				  I=i-i0;
				  J=j-j0;
				  K=k-k0; 
				 
				  x = delta*i-R_x;
				  y = delta*j-R_y;
				  z = delta*k-R_z;
				  r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));

				  /*
				   * coordinates of the "original finite difference node (unshifted, not periodic image)"
				   */				  
				  X = x - PP*2.0*R_x;
				  Y = y - QQ*2.0*R_y;
				  Z = z - RR*2.0*R_z;
				 
				  LIrow = k*n_x*n_y + j*n_x + i; 
				  AOApplicationToPetsc(aodmda1,1,&LIrow);
				  for(l=0;l<=lmax;l++)
				    {
				      if(l!=lloc)
					{ 					 
					  if(r == tableR[0])
					    {
					      pUlDeltaVl = tableUlDeltaV[l][0];	
					    }else{
					    ispline_gen(tableR,tableUlDeltaV[l],count,&r,&pUlDeltaVl,&Dtemp,1,YDUlDeltaV[l]);
					  }						 
					  if(l==0)
					    Rcut= pSddft->psd[at].rc_s;
					  if(l==1)
					    Rcut= pSddft->psd[at].rc_p;
					  if(l==2)
					    Rcut= pSddft->psd[at].rc_d;
					  if(l==3)
					    Rcut= pSddft->psd[at].rc_f;		      

					  for(m=-l;m<=l;m++)
					    {					      
					      ComplexSphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut+1.0,&SH_real,&SH_imag);
					      
					      kpt=0;
					      for(nk1=1;nk1<=pSddft->Kx;nk1++)
						for(nk2=1;nk2<=pSddft->Ky;nk2++)
						  for(nk3=1;nk3<=ceil(pSddft->Kz/2.0);nk3++)
						    {						      
						      k1=((2.0*nk1-pSddft->Kx-1.0)/(2.0*pSddft->Kx))*(M_PI/R_x);
						      k2=((2.0*nk2-pSddft->Ky-1.0)/(2.0*pSddft->Ky))*(M_PI/R_y);
						      k3=((2.0*nk3-pSddft->Kz-1.0)/(2.0*pSddft->Kz))*(M_PI/R_z);
						     		    			     
						      MatDenseGetArray(pSddft->XOrb1[kpt],&arrPsi1Seq); 
						      MatDenseGetArray(pSddft->XOrb2[kpt],&arrPsi2Seq); 
						      MatDenseGetArray(pSddft->YOrb1[kpt],&arrGradPsi1Seq); 
						      MatDenseGetArray(pSddft->YOrb2[kpt],&arrGradPsi2Seq);
						 						      
						      for(ii=0;ii<pSddft->Nstates;ii++)
							{
							  OrbScalingFactor = 1.0/(sqrt(delta*delta*delta)*sqrt(norms[ii]*norms[ii]+norms1[ii]*norms1[ii]));  
							  a1=OrbScalingFactor*(cos(k1*X+k2*Y+k3*Z)*arrGradPsi1Seq[LIrow-rowStart+(rowEnd-rowStart)*ii]-sin(k1*X+k2*Y+k3*Z)*arrGradPsi2Seq[LIrow-rowStart +(rowEnd-rowStart)*ii]);
							  a2=OrbScalingFactor*(sin(k1*X+k2*Y+k3*Z)*arrGradPsi1Seq[LIrow-rowStart+(rowEnd-rowStart)*ii]+cos(k1*X+k2*Y+k3*Z)*arrGradPsi2Seq[LIrow-rowStart +(rowEnd-rowStart)*ii]);
		 
							  b1=OrbScalingFactor*(cos(k1*X+k2*Y+k3*Z)*arrPsi1Seq[LIrow-rowStart+(rowEnd-rowStart)*ii]-sin(k1*X+k2*Y+k3*Z)*arrPsi2Seq[LIrow-rowStart +(rowEnd-rowStart)*ii]);
							  b2=OrbScalingFactor*(sin(k1*X+k2*Y+k3*Z)*arrPsi1Seq[LIrow-rowStart+(rowEnd-rowStart)*ii]+cos(k1*X+k2*Y+k3*Z)*arrPsi2Seq[LIrow-rowStart +(rowEnd-rowStart)*ii]);

							 
							  W1[poscnt][kpt][ii][l][m+l] += (a1*pUlDeltaVl*SH_real + a2*pUlDeltaVl*SH_imag);
							  W2[poscnt][kpt][ii][l][m+l] += (a2*pUlDeltaVl*SH_real - a1*pUlDeltaVl*SH_imag);
							 
							  T1[poscnt][kpt][ii][l][m+l] += (b1*pUlDeltaVl*SH_real + b2*pUlDeltaVl*SH_imag)/Dexact[l];
							  T2[poscnt][kpt][ii][l][m+l] += (b1*pUlDeltaVl*SH_imag - b2*pUlDeltaVl*SH_real)/Dexact[l];
							}
						      MatDenseRestoreArray(pSddft->XOrb1[kpt],&arrPsi1Seq); 
						      MatDenseRestoreArray(pSddft->XOrb2[kpt],&arrPsi2Seq);
						      MatDenseRestoreArray(pSddft->YOrb1[kpt],&arrGradPsi1Seq); 
						      MatDenseRestoreArray(pSddft->YOrb2[kpt],&arrGradPsi2Seq); 
						     
						      kpt++;
						    }
					    }
					}
				    }
				}			     
			}
		    }	     
	      index_mvatm = index_mvatm+3;
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
  
  for(at=0;at<pSddft->Ntype;at++)
    {
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];
      if(lmax!=0)
	{	  
	  start = (int)floor(pSddft->startPos[at]/3);
	  end = (int)floor(pSddft->endPos[at]/3);
	  for(poscnt=start;poscnt<=end;poscnt++)
	    {	      
	      fJ_x=0.0;
	      for(k=0;k<pSddft->Nkpts_sym;k++)
		{
		  for(i=0;i<pSddft->Nstates;i++)
		    {  gi = smearing_FermiDirac(pSddft->Beta,pSddft->lambdakpt[k][i],pSddft->lambda_f);	// occupation
		      for(l=0;l<=lmax;l++)
			{
			  if(l!=lloc)
			    {
			      for(m=-l;m<=l;m++)
				{			      
				  ierr =  MPI_Allreduce(MPI_IN_PLACE,&T1[poscnt][k][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
				  ierr =  MPI_Allreduce(MPI_IN_PLACE,&W1[poscnt][k][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);	
				  ierr =  MPI_Allreduce(MPI_IN_PLACE,&T2[poscnt][k][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
				  ierr =  MPI_Allreduce(MPI_IN_PLACE,&W2[poscnt][k][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);	
				  fJ_x += gi*pSddft->kptWts[k]*(W1[poscnt][k][i][l][m+l]*T1[poscnt][k][i][l][m+l] - W2[poscnt][k][i][l][m+l]*T2[poscnt][k][i][l][m+l]);	      			     
				}
			    }
			}		  
		    }
		} 	      	      
	      fJ_x = -4.0*delta*delta*delta*fJ_x/pSddft->Nkpts;	     
	      VecSetValue(pSddft->forces,3*poscnt,fJ_x,ADD_VALUES);	      
      	    }
	}
    }
  
  /*
   * end of x-component force calculation 
   */


  /*
   * Calculate y-component of forces
   */

  /* NOTE: we do no need to calculate T since it has already been calculated while 
   * calculating the x-component of forces. Hence we do not need to calculate the exact 
   * denominator. Also, memory for W has already been created, we dont need to allocate again,
   * just need to set it to zero.
   */

  index_mvatm=0;index=0;
 	
  for(k=0;k<pSddft->Nkpts_sym;k++)
    {      
      MatMatMultNumeric(pSddft->gradient_y,pSddft->XOrb1[k],pSddft->YOrb1[k]);      
      MatMatMultNumeric(pSddft->gradient_y,pSddft->XOrb2[k],pSddft->YOrb2[k]);	
    }
	    
  for(at=0;at<pSddft->Ntype;at++)
    {                          
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];	

      start = floor(pSddft->startPos[at]/3);
      end = floor(pSddft->endPos[at]/3);

      if(lmax==0) 
	{
	  index = index + 3*(end-start+1); 
	  index_mvatm = index_mvatm + 3*(end-start+1);
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
	      if(l==0)
		{
		  tableUlDeltaV[0][0]=pSddft->psd[at].Us[0]*(pSddft->psd[at].Vs[0]-pSddft->psd[at].Vloc[0]);		
		}
	      if(l==1)
		{
		  tableUlDeltaV[1][0]=0.0; 			     
		}
	      if(l==2)
		{
		  tableUlDeltaV[2][0]=0.0; 				      
		}
	      if(l==3)
		{
		  tableUlDeltaV[3][0]=0.0; 		   	  
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
		  }
		if(l==1)
		  {
		    tableUlDeltaV[1][count]=pSddft->psd[at].Up[count-1]*(pSddft->psd[at].Vp[count-1]-pSddft->psd[at].Vloc[count-1]); 		         	       
		  }
		if(l==2)
		  {
		    tableUlDeltaV[2][count]=pSddft->psd[at].Ud[count-1]*(pSddft->psd[at].Vd[count-1]-pSddft->psd[at].Vloc[count-1]); 			   	       
		  }   
		if(l==3)
		  {
		    tableUlDeltaV[3][count]=pSddft->psd[at].Uf[count-1]*(pSddft->psd[at].Vf[count-1]-pSddft->psd[at].Vloc[count-1]);			            
		  }  
	      }
	    count++;
	  }while(tableR[count-1] <= 1.732*pSddft->CUTOFF[at]+10.0); 
	  rmax = tableR[count-1];
                             
	                                  
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDUlDeltaV);	  
	  for(l=0;l<=lmax;l++)
	    {
	      /*
	       * derivatives of the spline fit to the projectors
	       */
	      PetscMalloc(sizeof(PetscScalar)*count,&YDUlDeltaV[l]);		         
	      getYD_gen(tableR,tableUlDeltaV[l],YDUlDeltaV[l],count);           
	    } 	  
	
	  /*
	   * loop over every atom of a given type
	   */
	  for(poscnt=start;poscnt<=end;poscnt++)
	    { 		  
	      X0 = pAtompos[index++];
	      Y0 = pAtompos[index++];
	      Z0 = pAtompos[index++];

	      for(k=0;k<pSddft->Nkpts_sym;k++)
		{
		  for(i=0;i<pSddft->Nstates;i++)
		    {    			    
		      for(l=0;l<=lmax;l++)
			{ 
			  for(m=-l;m<=l;m++)
			    {	     
			      W1[poscnt][k][i][l][m+l]=0.0;
			      W2[poscnt][k][i][l][m+l]=0.0;
			    }			
			}
		    }
		}
	      
	      Nposx = ceil(cutoffr/R_x);
	      Nposy = ceil(cutoffr/R_y);
	      Nposz = ceil(cutoffr/R_z);
	  
	      for(PP= -Nposx; PP<=Nposx; PP++)
		for(QQ= -Nposy; QQ<=Nposy; QQ++)
		  for(RR= -Nposz; RR<=Nposz; RR++)
		    {
		      x0 = X0 + PP*2.0*R_x;
		      y0 = Y0 + QQ*2.0*R_y;
		      z0 = Z0 + RR*2.0*R_z;
			
		      xstart=-1000; ystart=-1000; zstart=-1000; xend=-1000; yend=-1000; zend=-1000; overlap=0;
		      
		      xi = roundf((x0 + R_x)/delta); 
		      yj = roundf((y0 + R_y)/delta);
		      zk = roundf((z0 + R_z)/delta);
		       
		      
		      xs = xi-offset; xl = xi+offset;
		      ys = yj-offset; yl = yj+offset;
		      zs = zk-offset; zl = zk+offset;
     
		      /*
		       * find if domain of influence of pseudopotential overlaps with the domain
		       * stored by processor  
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

			  i0 = xs-o;
			  j0 = ys-o;
			  k0 = zs-o;
			    							
			  /*
			   * Get local pointer (column wise) to the gradient of wavefunctions  
			   */			    
			  for(k=zstart;k<=zend;k++)
			    for(j=ystart;j<=yend;j++)
			      for(i=xstart;i<=xend;i++)
				{
				  
				  I=i-i0;
				  J=j-j0;
				  K=k-k0; 
				 
				  x = delta*i-R_x;
				  y = delta*j-R_y;
				  z = delta*k-R_z;
				  r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
				 
				  X = x - PP*2.0*R_x;
				  Y = y - QQ*2.0*R_y;
				  Z = z - RR*2.0*R_z;
				  
				  LIrow = k*n_x*n_y + j*n_x + i; 
				  AOApplicationToPetsc(aodmda1,1,&LIrow);
				  for(l=0;l<=lmax;l++)
				    {
				      if(l!=lloc)
					{ 					  
					  if(r == tableR[0])
					    {
					      pUlDeltaVl = tableUlDeltaV[l][0];	
					    }else{
					    ispline_gen(tableR,tableUlDeltaV[l],count,&r,&pUlDeltaVl,&Dtemp,1,YDUlDeltaV[l]);
					  }						 
					  if(l==0)
					    Rcut= pSddft->psd[at].rc_s;
					  if(l==1)
					    Rcut= pSddft->psd[at].rc_p;
					  if(l==2)
					    Rcut= pSddft->psd[at].rc_d;
					  if(l==3)
					    Rcut= pSddft->psd[at].rc_f;
					  for(m=-l;m<=l;m++)
					    {					      
					      ComplexSphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut+1.0,&SH_real,&SH_imag);
					      kpt=0;
					      for(nk1=1;nk1<=pSddft->Kx;nk1++)
						for(nk2=1;nk2<=pSddft->Ky;nk2++)
						  for(nk3=1;nk3<=ceil(pSddft->Kz/2.0);nk3++)
						    {						      
						      k1=((2.0*nk1-pSddft->Kx-1.0)/(2.0*pSddft->Kx))*(M_PI/R_x);
						      k2=((2.0*nk2-pSddft->Ky-1.0)/(2.0*pSddft->Ky))*(M_PI/R_y);
						      k3=((2.0*nk3-pSddft->Kz-1.0)/(2.0*pSddft->Kz))*(M_PI/R_z);			 						      
						      MatDenseGetArray(pSddft->YOrb1[kpt],&arrGradPsi1Seq); 
						      MatDenseGetArray(pSddft->YOrb2[kpt],&arrGradPsi2Seq); 
						      
						      for(ii=0;ii<pSddft->Nstates;ii++)
							{
							  
							  OrbScalingFactor = 1.0/(sqrt(delta*delta*delta)*sqrt(norms[ii]*norms[ii]+norms1[ii]*norms1[ii]));
							  a1=OrbScalingFactor*(cos(k1*X+k2*Y+k3*Z)*arrGradPsi1Seq[LIrow-rowStart+(rowEnd-rowStart)*ii]-sin(k1*X+k2*Y+k3*Z)*arrGradPsi2Seq[LIrow-rowStart +(rowEnd-rowStart)*ii]);
							  a2=OrbScalingFactor*(sin(k1*X+k2*Y+k3*Z)*arrGradPsi1Seq[LIrow-rowStart+(rowEnd-rowStart)*ii]+cos(k1*X+k2*Y+k3*Z)*arrGradPsi2Seq[LIrow-rowStart +(rowEnd-rowStart)*ii]);
							  
							  W1[poscnt][kpt][ii][l][m+l] += (a1*pUlDeltaVl*SH_real + a2*pUlDeltaVl*SH_imag);
							  W2[poscnt][kpt][ii][l][m+l] += (a2*pUlDeltaVl*SH_real - a1*pUlDeltaVl*SH_imag);
							}
						      MatDenseRestoreArray(pSddft->YOrb1[kpt],&arrGradPsi1Seq); 
						      MatDenseRestoreArray(pSddft->YOrb2[kpt],&arrGradPsi2Seq); 
						      kpt++;
						    }
					    }
					}
				    }
				} 		
			}
		    }
	      
	      index_mvatm = index_mvatm+3;
	    }	  
	  for(l=0;l<=lmax;l++)
	    {
	      PetscFree(YDUlDeltaV[l]);	       	    
	    }
	  PetscFree(YDUlDeltaV); 
	    
	}
    }
  
  /*
   * sum the contributions across all the processors and compute the forces
   */   
  for(at=0;at<pSddft->Ntype;at++)
    {
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];
      if(lmax!=0)
	{
	  start = (int)floor(pSddft->startPos[at]/3);
	  end = (int)floor(pSddft->endPos[at]/3);
	  for(poscnt=start;poscnt<=end;poscnt++)
	    {	      
	      fJ_y=0.0;
	      for(k=0;k<pSddft->Nkpts_sym;k++)
		{
		  for(i=0;i<pSddft->Nstates;i++)
		    {  gi = smearing_FermiDirac(pSddft->Beta,pSddft->lambdakpt[k][i],pSddft->lambda_f);	
		      for(l=0;l<=lmax;l++)
			{
			  if(l!=lloc)
			    {
			      for(m=-l;m<=l;m++)
				{			     
				  ierr =  MPI_Allreduce(MPI_IN_PLACE,&W1[poscnt][k][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
				  ierr =  MPI_Allreduce(MPI_IN_PLACE,&W2[poscnt][k][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);
				  fJ_y += gi*pSddft->kptWts[k]*(W1[poscnt][k][i][l][m+l]*T1[poscnt][k][i][l][m+l]-W2[poscnt][k][i][l][m+l]*T2[poscnt][k][i][l][m+l]);			     
				}
			    }
			}		  
		    }
		}	      	      
	      fJ_y = -4.0*delta*delta*delta*fJ_y/pSddft->Nkpts;	     
	      VecSetValue(pSddft->forces,3*poscnt+1,fJ_y,ADD_VALUES);	      
      	    }
	}
    }
  /*
   * end of y-component force calculation 
   */


  /*
   * Calculate z-component of forces
   */

  /* we do no need to calculate T since it has already been calculated while calculating the 
   * x-component of forces. Hence we do not need to calculate the exact denominator. Also,
   * memory for W has already been created, we dont need to allocate again, just need to set it
   * to zero
   */

  index_mvatm=0;index=0;  

  /*
   * Calculate gradient of orbitals in z-direction
   */
  for(k=0;k<pSddft->Nkpts_sym;k++)
    {      
      MatMatMultNumeric(pSddft->gradient_z,pSddft->XOrb1[k],pSddft->YOrb1[k]);	     
      MatMatMultNumeric(pSddft->gradient_z,pSddft->XOrb2[k],pSddft->YOrb2[k]);	
    }
  /*
   * loop over different types of atoms 
   */    	    
  for(at=0;at<pSddft->Ntype;at++)
    {                          
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];	 
	   
      start = floor(pSddft->startPos[at]/3);
      end = floor(pSddft->endPos[at]/3);

      /*
       * if the atom is hydrogen, then we omit the calculation of non-local forces
       */
      if(lmax==0) 
	{
	  index = index + 3*(end-start+1); 
	  index_mvatm = index_mvatm + 3*(end-start+1);
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
	      if(l==0)
		{
		  tableUlDeltaV[0][0]=pSddft->psd[at].Us[0]*(pSddft->psd[at].Vs[0]-pSddft->psd[at].Vloc[0]);		
		}
	      if(l==1)
		{
		  tableUlDeltaV[1][0]=0.0; 			     
		}
	      if(l==2)
		{
		  tableUlDeltaV[2][0]=0.0; 				      
		}
	      if(l==3)
		{
		  tableUlDeltaV[3][0]=0.0; 		   	  
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
		  }
		if(l==1)
		  {
		    tableUlDeltaV[1][count]=pSddft->psd[at].Up[count-1]*(pSddft->psd[at].Vp[count-1]-pSddft->psd[at].Vloc[count-1]);		         	       
		  }
		if(l==2)
		  {
		    tableUlDeltaV[2][count]=pSddft->psd[at].Ud[count-1]*(pSddft->psd[at].Vd[count-1]-pSddft->psd[at].Vloc[count-1]); 			   	       
		  }   
		if(l==3)
		  {
		    tableUlDeltaV[3][count]=pSddft->psd[at].Uf[count-1]*(pSddft->psd[at].Vf[count-1]-pSddft->psd[at].Vloc[count-1]); 		            
		  }  
	      }
	    count++;
	  }while(tableR[count-1] <= 1.732*pSddft->CUTOFF[at]+10.0); 
	  rmax = tableR[count-1];
                            
	                                
	  PetscMalloc(sizeof(PetscScalar*)*(lmax+1),&YDUlDeltaV);	  
	  for(l=0;l<=lmax;l++)
	    {
	      /*
	       * derivatives of the spline fit to the projectors
	       */
	      PetscMalloc(sizeof(PetscScalar)*count,&YDUlDeltaV[l]);		         
	      getYD_gen(tableR,tableUlDeltaV[l],YDUlDeltaV[l],count);           
	    } 	  
	
	  /*
	   * loop over every atom of a given type
	   */
	  for(poscnt=start;poscnt<=end;poscnt++)
	    { 
	     
	      X0 = pAtompos[index++];
	      Y0 = pAtompos[index++];
	      Z0 = pAtompos[index++];
	      
	      for(k=0;k<pSddft->Nkpts_sym;k++)
		{
		  for(i=0;i<pSddft->Nstates;i++)
		    {    			    
		      for(l=0;l<=lmax;l++)
			{ 
			  for(m=-l;m<=l;m++)
			    {	     
			      W1[poscnt][k][i][l][m+l]=0.0;
			      W2[poscnt][k][i][l][m+l]=0.0;
			    }			
			}
		    }
		}

	      
	      Nposx = ceil(cutoffr/R_x);
	      Nposy = ceil(cutoffr/R_y);
	      Nposz = ceil(cutoffr/R_z);
	  
	      for(PP= -Nposx; PP<=Nposx; PP++)
		for(QQ= -Nposy; QQ<=Nposy; QQ++)
		  for(RR= -Nposz; RR<=Nposz; RR++)
		    {
		      x0 = X0 + PP*2.0*R_x;
		      y0 = Y0 + QQ*2.0*R_y;
		      z0 = Z0 + RR*2.0*R_z;		       

		      xstart=-1000; ystart=-1000; zstart=-1000; xend=-1000; yend=-1000; zend=-1000; overlap=0;
		      
		      xi = roundf((x0 + R_x)/delta); 
		      yj = roundf((y0 + R_y)/delta);
		      zk = roundf((z0 + R_z)/delta);

		      xs = xi-offset; xl = xi+offset;
		      ys = yj-offset; yl = yj+offset;
		      zs = zk-offset; zl = zk+offset;
     
		      /*
		       * find if domain of influence of pseudopotential overlaps with the domain
		       * stored by processor  
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
			  i0 = xs-o;
			  j0 = ys-o;
			  k0 = zs-o;
			  
			  /* 
			   * evaluate the the quantities that contribute to nonlocal force at 
			   * nodes in the overlap region
			   */
			  for(k=zstart;k<=zend;k++)
			    for(j=ystart;j<=yend;j++)
			      for(i=xstart;i<=xend;i++)
				{
				  
				  I=i-i0;
				  J=j-j0;
				  K=k-k0; 
				 
				  x = delta*i-R_x;
				  y = delta*j-R_y;
				  z = delta*k-R_z;
				  r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
			  
				  X = x - PP*2.0*R_x;
				  Y = y - QQ*2.0*R_y;
				  Z = z - RR*2.0*R_z;
				 
				  LIrow = k*n_x*n_y + j*n_x + i; 
				  AOApplicationToPetsc(aodmda1,1,&LIrow);
				  for(l=0;l<=lmax;l++)
				    {
				      if(l!=lloc)
					{ 					  
					  if(r == tableR[0])
					    {
					      pUlDeltaVl = tableUlDeltaV[l][0];	
					    }else{
					    ispline_gen(tableR,tableUlDeltaV[l],count,&r,&pUlDeltaVl,&Dtemp,1,YDUlDeltaV[l]);
					  }						 
					  if(l==0)
					    Rcut= pSddft->psd[at].rc_s;
					  if(l==1)
					    Rcut= pSddft->psd[at].rc_p;
					  if(l==2)
					    Rcut= pSddft->psd[at].rc_d;
					  if(l==3)
					    Rcut= pSddft->psd[at].rc_f;
					  for(m=-l;m<=l;m++)
					    {					     
					      ComplexSphericalHarmonic(x-x0,y-y0,z-z0,l,m,Rcut+1.0,&SH_real,&SH_imag);
					      kpt=0;
					      for(nk1=1;nk1<=pSddft->Kx;nk1++)
						for(nk2=1;nk2<=pSddft->Ky;nk2++)
						  for(nk3=1;nk3<=ceil(pSddft->Kz/2.0);nk3++)
						    {				 
						      
						      k1=((2.0*nk1-pSddft->Kx-1.0)/(2.0*pSddft->Kx))*(M_PI/R_x);
						      k2=((2.0*nk2-pSddft->Ky-1.0)/(2.0*pSddft->Ky))*(M_PI/R_y);
						      k3=((2.0*nk3-pSddft->Kz-1.0)/(2.0*pSddft->Kz))*(M_PI/R_z);

						      MatDenseGetArray(pSddft->YOrb1[kpt],&arrGradPsi1Seq); 
						      MatDenseGetArray(pSddft->YOrb2[kpt],&arrGradPsi2Seq); 
						     
						      for(ii=0;ii<pSddft->Nstates;ii++)
							{
							                                                    
							  OrbScalingFactor = 1.0/(sqrt(delta*delta*delta)*sqrt(norms[ii]*norms[ii]+norms1[ii]*norms1[ii]));
							  
							  a1=OrbScalingFactor*(cos(k1*X+k2*Y+k3*Z)*arrGradPsi1Seq[LIrow-rowStart+(rowEnd-rowStart)*ii]-sin(k1*X+k2*Y+k3*Z)*arrGradPsi2Seq[LIrow-rowStart +(rowEnd-rowStart)*ii]);
							  a2=OrbScalingFactor*(sin(k1*X+k2*Y+k3*Z)*arrGradPsi1Seq[LIrow-rowStart+(rowEnd-rowStart)*ii]+cos(k1*X+k2*Y+k3*Z)*arrGradPsi2Seq[LIrow-rowStart +(rowEnd-rowStart)*ii]);
							  
							  W1[poscnt][kpt][ii][l][m+l] += (a1*pUlDeltaVl*SH_real + a2*pUlDeltaVl*SH_imag);
							  W2[poscnt][kpt][ii][l][m+l] += (a2*pUlDeltaVl*SH_real - a1*pUlDeltaVl*SH_imag);
							}
						      MatDenseRestoreArray(pSddft->YOrb1[kpt],&arrGradPsi1Seq); 
						      MatDenseRestoreArray(pSddft->YOrb2[kpt],&arrGradPsi2Seq); 
						      kpt++;

						    }
					    }
					}
				    }
				}   			    
			  
			}
		    }	      
	      index_mvatm = index_mvatm+3;
	    }	    
	 
	  for(l=0;l<=lmax;l++)
	    {
	      PetscFree(YDUlDeltaV[l]);	       	    
	    }
	  PetscFree(YDUlDeltaV); 
	    
	}
    }
  
  /*
   * sum the contributions across all the processors and compute the forces
   */
  for(at=0;at<pSddft->Ntype;at++)
    {
      lmax = pSddft->psd[at].lmax;
      lloc = pSddft->localPsd[at];
      if(lmax!=0)
	{	  
	  start = (int)floor(pSddft->startPos[at]/3);
	  end = (int)floor(pSddft->endPos[at]/3);
	  for(poscnt=start;poscnt<=end;poscnt++)
	    {	      
	      fJ_z=0.0;
	      for(k=0;k<pSddft->Nkpts_sym;k++)
		{
		  for(i=0;i<pSddft->Nstates;i++)
		    {  gi = smearing_FermiDirac(pSddft->Beta,pSddft->lambdakpt[k][i],pSddft->lambda_f);	
		      for(l=0;l<=lmax;l++)
			{
			  if(l!=lloc)
			    {
			      for(m=-l;m<=l;m++)
				{			     
				  ierr =  MPI_Allreduce(MPI_IN_PLACE,&W1[poscnt][k][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);		
				  ierr =  MPI_Allreduce(MPI_IN_PLACE,&W2[poscnt][k][i][l][m+l],1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  CHKERRQ(ierr);		
				  fJ_z += gi*pSddft->kptWts[k]*(W1[poscnt][k][i][l][m+l]*T1[poscnt][k][i][l][m+l]-W2[poscnt][k][i][l][m+l]*T2[poscnt][k][i][l][m+l]);			     
				}
			    }
			}		  
		    }
		}
	      	      
	      fJ_z = -4.0*delta*delta*delta*fJ_z/pSddft->Nkpts;	     
	      VecSetValue(pSddft->forces,3*poscnt+2,fJ_z,ADD_VALUES);	      
      	    }
	}
    }
  
  /*
   * end of z-component force calculation 
   */
  VecRestoreArray(pSddft->Atompos,&pAtompos);
  VecRestoreArray(pSddft->mvAtmConstraint,&pmvAtmConstraint);          
  VecAssemblyBegin(pSddft->forces);
  VecAssemblyEnd(pSddft->forces);    
 
  /*
   * free memory
   */
  for(at=0;at<pSddft->Ntype;at++)
    {
      lmax = pSddft->psd[at].lmax;      
      int start,end;
      start = (int)floor(pSddft->startPos[at]/3);
      end = (int)floor(pSddft->endPos[at]/3);	  
      for(poscnt=start;poscnt<=end;poscnt++)
	{	      
	  if(lmax!=0)
	    {
	      for(k=0;k<pSddft->Nkpts_sym;k++)
		{
		  for(i=0;i<pSddft->Nstates;i++)
		    {		 
		      for(l=0;l<=lmax;l++)
			{			     
			  PetscFree(W1[poscnt][k][i][l]);
			  PetscFree(T1[poscnt][k][i][l]);
			  PetscFree(W2[poscnt][k][i][l]);
			  PetscFree(T2[poscnt][k][i][l]);			      
			}			 
		      PetscFree(W1[poscnt][k][i]);
		      PetscFree(T1[poscnt][k][i]);
		      PetscFree(W2[poscnt][k][i]);
		      PetscFree(T2[poscnt][k][i]);			 
		    }
		  PetscFree(W1[poscnt][k]);
		  PetscFree(T1[poscnt][k]);
		  PetscFree(W2[poscnt][k]);
		  PetscFree(T2[poscnt][k]);
		}
	    }
	  PetscFree(W1[poscnt]);
	  PetscFree(T1[poscnt]);
	  PetscFree(W2[poscnt]);
	  PetscFree(T2[poscnt]);
	}      
    }
  PetscFree(W1);
  PetscFree(T1);
  PetscFree(W2);
  PetscFree(T2);
  
  return ierr;
}
