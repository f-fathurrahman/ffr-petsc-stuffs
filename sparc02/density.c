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
  | file name: density.cc          
  |
  | Description: This file contains the functions required for calculation of pseudocharge
  | density, self energy, initial guess electron density, reference pseudocharge density,
  | reference self energy and reference pseudopotential. 
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana, Deepa Phanish
  |
  | Last Modified: 2/15/2016   
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "petscsys.h"

#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


///////////////////////////////////////////////////////////////////////////////////////////////
//                          ChargDensB_VecInit: Pseudocharge density calculation.            //
//                                  Additionally calculates the self energy                  //
///////////////////////////////////////////////////////////////////////////////////////////////
void ChargDensB_VecInit(SDDFT_OBJ* pSddft)
{    
  PetscScalar ***BlcArrGlbIdx;  
  PetscScalar ***BlcVpsArrGlbIdx;
  PetscScalar *pAtompos;
  PetscScalar *YD=NULL;

  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0;
  PetscScalar x0, y0, z0,coeffs[MAX_ORDER+1],cutoffr;
    
  PetscScalar tableR[MAX_TABLE_SIZE];
  PetscScalar tableVps[MAX_TABLE_SIZE], Bval, elecno, noetot = pSddft->noetot, rhosum;
  PetscScalar ***pVpsArray = NULL;
  PetscScalar Dtemp;
      
  PetscInt i=0,j,k,xi,yj,zk,offset, Ii, J, K, p, nzVps,nyVps,nxVps;
  PetscInt i0,j0,k0,a,poscnt,index=0,at;
  PetscScalar delta=pSddft->delta, x, y, z, rmax;
  PetscScalar r;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt o = pSddft->order;
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  int count;
  for(p=0;p<=o;p++)
    {
      coeffs[p] = pSddft->coeffs[p]/(2*M_PI);
    }  
  
  DMDAVecGetArray(pSddft->da,pSddft->chrgDensB,&BlcArrGlbIdx);  
  DMDAVecGetArray(pSddft->da,pSddft->bjVj,&BlcVpsArrGlbIdx);  
  
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
  VecGetArray(pSddft->Atompos,&pAtompos);  

  /*
   * loop over different types of atoms 
   */
  for(at=0;at<pSddft->Ntype;at++)
    {   
      cutoffr=pSddft->CUTOFF[at];
      offset = (PetscInt)ceil(cutoffr/delta + 0.5);

      /*
       * since the pseudopotential table read from the file does not have a value at r=0,
       * we assume the pseudopotential value at r=0 to be same as the first entry read
       * from the file. 
       */
      tableR[0]=0.0; tableVps[0]=pSddft->psd[at].Vloc[0]; // at r=0
      count=1;
      do{
      	tableR[count] = pSddft->psd[at].RadialGrid[count-1];
      	tableVps[count] = pSddft->psd[at].Vloc[count-1]; //Vloc
      	count++;
      } while( PetscRealPart(tableR[count-1]) <= (pSddft->CUTOFF[at] + 4.0) );
      
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
#ifdef _DEBUG
        printf("Bcube xs:%d, ys:%d, zs:%d\n",xs,ys,zs);
        printf("Overlap xstart:%d, ystart:%d, zstart:%d\n", xstart, ystart,zstart);
        printf("overlap xend:%d, yend:%d, zend:%d\n", xend,yend,zend);
#endif
      
        nzVps = zend-zstart+1+o*2;
        nyVps = yend-ystart+1+o*2;
        nxVps = xend-xstart+1+o*2;
    
        PetscMalloc(sizeof(PetscScalar**)*nzVps,&pVpsArray);
         
        if(pVpsArray == NULL)
    {
      //cout<<"Memory allocation fail";
      exit(1);
    }
        for(i = 0; i < nzVps; i++)
    {
      PetscMalloc(sizeof(PetscScalar*)*nyVps,&pVpsArray[i]);
     
      if(pVpsArray[i] == NULL)
        {
          //cout<<"Memory allocation fail";
          exit(1);
        }
      
      for(j=0;j<nyVps;j++)
        {
          PetscMalloc(sizeof(PetscScalar)*nxVps,&pVpsArray[i][j]);
        
          if(pVpsArray[i][j] == NULL)
      {
        //cout<<"Memory allocation fail";
        exit(1);
      } 
        }      
    }
    
        i0 = xstart-o;
        j0 = ystart-o;
        k0 = zstart-o;
  
        /* 
         * evaluate the pseudopotential at nodes in the overlap region + 
         * finite-difference order in each direction
         */
        for(k=0;k<nzVps;k++) 
    for(j=0;j<nyVps;j++)  
      for(i=0;i<nxVps;i++) 
        {   
          Ii = i + i0;
          J = j + j0;
          K = k + k0;
      
          x = delta*I - R_x ;
          y = delta*J - R_y ;
          z = delta*K - R_z ;   
          r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
            
          if(r == tableR[0])
      {
        pVpsArray[k][j][i] = tableVps[0];
      }
          else if(PetscRealPart(r) > PetscRealPart(rmax))
      {
        pVpsArray[k][j][i] = -pSddft->noe[at]/r;
      }
          else
      {
        ispline_gen(tableR,tableVps,count,&r,&pVpsArray[k][j][i],&Dtemp,1,YD);
      }
      
        }
    
        /*
         * calculate pseudocharge density and contribution to self energy at nodes 
         * inside the overlap region from previously calculated pseudopotential values 
         * using finite difference stencil  
         */
        for(k=zstart;k<=zend;k++)
    for(j=ystart;j<=yend;j++)
      for(i=xstart;i<=xend;i++)
        {
          Bval=0;  
          Ii = i-i0; // ffr: what's wrong with I?
          J = j-j0;
          K = k-k0;
           
          Bval=pVpsArray[K][J][Ii]*coeffs[0];
          for(a=1;a<=o;a++)
      {
        Bval+=(pVpsArray[K][J][Ii-a] + pVpsArray[K][J][Ii+a] + pVpsArray[K][J-a][Ii] +        
         pVpsArray[K][J+a][Ii] + pVpsArray[K-a][J][Ii] + pVpsArray[K+a][J][Ii])*coeffs[a];      
      }      
          BlcArrGlbIdx[k][j][i] += Bval;
          BlcVpsArrGlbIdx[k][j][i] += (pVpsArray[K][J][Ii])*Bval;       
        }        
     
        for(i = 0; i < nzVps; i++)
    {
      for(j=0;j<nyVps;j++)
        {
          PetscFree(pVpsArray[i][j]);                    
        }
      PetscFree(pVpsArray[i]);
    }  
        PetscFree(pVpsArray);        
      }  
  }
      
      PetscFree(YD);   
    }

  DMDAVecRestoreArray(pSddft->da,pSddft->chrgDensB,&BlcArrGlbIdx);  
  DMDAVecRestoreArray(pSddft->da,pSddft->bjVj,&BlcVpsArrGlbIdx);
  VecRestoreArray(pSddft->Atompos,&pAtompos);  
  /*
   * calculate self energy term
   */
  VecSum(pSddft->bjVj,&pSddft->Eself); 
  pSddft->Eself = -0.5*pSddft->Eself*delta*delta*delta;  
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD,"           Pseudocharge, self energy and electrostatic correction          \n");
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD,"Self energy of nuclei:%9.9f (Hartree)\n",pSddft->Eself);
     
  /* 
   * check if pseudocharge density has the correct sign
   */
  VecSum(pSddft->chrgDensB,&elecno);
  PetscPrintf(PETSC_COMM_WORLD,"sum b: %lf\n",elecno);
  assert( PetscRealPart(elecno) < 0);  

  /*
   * scale electron density appropriately with respect to pseudocharge density
   */
  VecSum(pSddft->elecDensRho,&rhosum);
  rhosum=-elecno/rhosum;
  VecScale(pSddft->elecDensRho,rhosum); 


  /*
   * calculate the total enclosed valence charge inside the simulation domain
   */
  elecno *= delta*delta*delta;
  pSddft->elecN = -elecno; 
  PetscPrintf(PETSC_COMM_WORLD,"Total valence electronic charge:%.9f\n",elecno);
  
#if 1 
  assert(
    ( PetscRealPart((noetot + elecno)/noetot) < Z_ACC ) &&
    ( PetscRealPart((-elecno - noetot)/elecno) < Z_ACC )
  );
#endif
    
  return;
}


///////////////////////////////////////////////////////////////////////////////////////////////
//                  PeriodicChargDensB_VecInit: Pseudocharge density calculation.            //
//                                  Additionally calculates the self energy                  //
///////////////////////////////////////////////////////////////////////////////////////////////
void PeriodicChargDensB_VecInit(SDDFT_OBJ* pSddft)
{
    
  PetscScalar ***BlcArrGlbIdx;  
  PetscScalar ***BlcVpsArrGlbIdx;
  PetscScalar *pAtompos;
  PetscScalar *YD=NULL;

  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0;
  PetscScalar x0, y0, z0,X0,Y0,Z0,coeffs[MAX_ORDER+1],cutoffr;
    
  PetscScalar tableR[MAX_TABLE_SIZE],tableVps[MAX_TABLE_SIZE],Bval,elecno,noetot=pSddft->noetot,rhosum;
  PetscScalar ***pVpsArray=NULL;
  PetscScalar Dtemp;
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    
  PetscInt  i=0,j,k,xi,yj,zk,offset,Ii, J, K, p, nzVps,nyVps,nxVps,
    i0,j0,k0,a,poscnt,index=0,at;
  PetscScalar delta=pSddft->delta,x,y,z,r,rmax;
  
  // ffr: not used
  //PetscInt n_x = pSddft->numPoints_x;
  //PetscInt n_y = pSddft->numPoints_y;
  //PetscInt n_z = pSddft->numPoints_z;

  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt o = pSddft->order;
  
  PetscInt PP,QQ,RR;
  PetscInt Imax_x,Imin_x,Imax_y,Imin_y,Imax_z,Imin_z;

  int count;
  for(p=0;p<=o;p++)
    {
      coeffs[p] = pSddft->coeffs[p]/(2*M_PI);
    }  
 
  DMDAVecGetArray(pSddft->da,pSddft->chrgDensB,&BlcArrGlbIdx);  
  DMDAVecGetArray(pSddft->da,pSddft->bjVj,&BlcVpsArrGlbIdx);  
  
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
#ifdef _DEBUG
  printf("Gxdim:%d,Gydim:%d,Gzdim:%d\n",gxdim,gydim,gzdim); 
  printf("xcor:%d,ycor:%d,zcor:%d\n",xcor,ycor,zcor); 
  printf("lxdim:%d,lydim:%d,lzdim:%d\n",lxdim,lydim,lzdim); 
#endif      
    
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
      }while( PetscRealPart(tableR[count-1]) <= pSddft->CUTOFF[at]+4.0);
      
      rmax = tableR[count-1];
      int start,end;
      start = (int)floor(pSddft->startPos[at]/3);
      end = (int)floor(pSddft->endPos[at]/3);
          
      PetscMalloc(sizeof(PetscScalar)*count,&YD);
      getYD_gen(tableR, tableVps, YD, count);
      
      /*
       * loop over every atom of a given type
       */
      for(poscnt=start;poscnt<=end;poscnt++)
  {       
    X0 = pAtompos[index++];
    Y0 = pAtompos[index++];
    Z0 = pAtompos[index++];
       
    xi = (int)((X0 + R_x)/delta + 0.5); 
    yj = (int)((Y0 + R_y)/delta + 0.5);
    zk = (int)((Z0 + R_z)/delta + 0.5);
    
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

      xstart=-1; ystart=-1; zstart=-1; xend=-1; yend=-1; zend=-1; overlap=0;
      /*
       * periodic map of atomic position
       */
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
       * find if domain of influence of pseudocharge overlaps with the domain stored by
       * processor  
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
          nzVps = zend-zstart+1+o*2;
          nyVps = yend-ystart+1+o*2;
          nxVps = xend-xstart+1+o*2;       
          PetscMalloc(sizeof(PetscScalar**)*nzVps,&pVpsArray);
         
          if(pVpsArray == NULL)
      {
        //cout<<"Memory allocation fail";
        exit(1);
      }
          for(i = 0; i < nzVps; i++)
      {
        PetscMalloc(sizeof(PetscScalar*)*nyVps,&pVpsArray[i]);
     
        if(pVpsArray[i] == NULL)
          {
            //cout<<"Memory allocation fail";
            exit(1);
          }
      
        for(j=0;j<nyVps;j++)
          {
            PetscMalloc(sizeof(PetscScalar)*nxVps,&pVpsArray[i][j]);
        
            if(pVpsArray[i][j] == NULL)
        {
          //cout<<"Memory allocation fail";
          exit(1);
        } 
          }      
      }
    
          i0 = xstart-o;
          j0 = ystart-o;
          k0 = zstart-o;
          /* 
           * evaluate the pseudopotential at nodes in the overlap region + 
           * finite-difference order in each direction
           */
          for(k=0;k<nzVps;k++) 
      for(j=0;j<nyVps;j++)  
        for(i=0;i<nxVps;i++) 
          {   
            Ii = i+i0;
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
            else if( PetscRealPart(r) > PetscRealPart(rmax) )
        {
          pVpsArray[k][j][i] = -pSddft->noe[at]/r;
        }
            else
        {
          ispline_gen(tableR, tableVps, count, &r, &pVpsArray[k][j][i], &Dtemp, 1, YD);
        }
      
          }
    
          /*
           * calculate pseudocharge density and contribution to self energy at nodes 
           * inside the overlap region from previously calculated pseudopotential values 
           * using finite difference stencil  
           */
          for(k=zstart;k<=zend;k++)
      for(j=ystart;j<=yend;j++)
        for(i=xstart;i<=xend;i++)
          {
            Bval=0;     
            Ii = i - i0;
            J = j - j0;
            K = k - k0;
           
            Bval=pVpsArray[K][J][Ii]*coeffs[0];
            for(a=1;a<=o;a++)
        {
          Bval+=(pVpsArray[K][J][Ii-a] + pVpsArray[K][J][Ii+a] + pVpsArray[K][J-a][Ii] +        
           pVpsArray[K][J+a][Ii] + pVpsArray[K-a][J][Ii] + pVpsArray[K+a][J][Ii])*coeffs[a];
      
        }
      
            BlcArrGlbIdx[k][j][i] += Bval;
            BlcVpsArrGlbIdx[k][j][i] += (pVpsArray[K][J][Ii])*Bval;  
      
          }        
     
          for(i = 0; i < nzVps; i++)
      {
        for(j=0;j<nyVps;j++)
          {
            PetscFree(pVpsArray[i][j]);                    
          }
        PetscFree(pVpsArray[i]);
      }  
          PetscFree(pVpsArray);        
        } 
    }
  }      
      PetscFree(YD);   
    }

  DMDAVecRestoreArray(pSddft->da,pSddft->chrgDensB,&BlcArrGlbIdx);  
  DMDAVecRestoreArray(pSddft->da,pSddft->bjVj,&BlcVpsArrGlbIdx);
  VecRestoreArray(pSddft->Atompos,&pAtompos);  
  
  /*
   * calculate self energy term
   */
  VecSum(pSddft->bjVj,&pSddft->Eself); 
  pSddft->Eself = -0.5*pSddft->Eself*delta*delta*delta;
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD,"           Pseudocharge, self energy and electrostatic correction          \n");
  PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD,"Self energy of nuclei:%9.9f (Hartree)\n",pSddft->Eself);
       
  /* 
   * check if pseudocharge density has the correct sign
   */
  VecSum(pSddft->chrgDensB,&elecno);
  assert( PetscRealPart(elecno) < 0 );  
 
  /*
   * scale electron density appropriately with respect to pseudocharge density
   */
  VecSum(pSddft->elecDensRho,&rhosum);
  rhosum=-elecno/rhosum;
  VecScale(pSddft->elecDensRho,rhosum); 
  /*
   * calculate the total enclosed valence charge inside the simulation domain
   */
  elecno *= delta*delta*delta;
  pSddft->elecN = -elecno; 
  PetscPrintf(PETSC_COMM_WORLD,"Total valence electronic charge:%.9f\n",elecno);
  
#if 1
  assert(
    ( PetscRealPart((noetot + elecno)/noetot) < Z_ACC) &&
    ( PetscRealPart((-elecno - noetot)/elecno) < Z_ACC)
  );    
#endif
   
  return;
}



///////////////////////////////////////////////////////////////////////////////////////////////
//     SuperpositionAtomicCharge_VecInit: calculation of initial guess electron density      //
//                using superposition of single atom electron densities                      //
///////////////////////////////////////////////////////////////////////////////////////////////
void SuperpositionAtomicCharge_VecInit(SDDFT_OBJ* pSddft)
{ 
  PetscScalar ***RholcArrGlbIdx;
  PetscScalar ***pRhoArray=NULL;
  PetscScalar *pAtompos;
  PetscScalar *YD=NULL;
   
  int count;
  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0;
  PetscScalar x0, y0, z0;  
  PetscScalar tableRho[MAX_TABLE_SIZE],tableR[MAX_TABLE_SIZE],rhosum;
  PetscScalar Dtemp;

  PetscInt i=0, j, k, xi, yj, zk, offset, Ii, J, K, nzVps, nyVps, nxVps, i0, j0, k0, poscnt, index=0, at;
  PetscScalar delta=pSddft->delta,x,y,z,r,cutoffr,rmax;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt o = pSddft->order;
  
  DMDAVecGetArray(pSddft->da,pSddft->SuperposAtRho,&RholcArrGlbIdx);

  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
  VecGetArray(pSddft->Atompos,&pAtompos);
  
  /*
   * loop over different types of atoms 
   */
  for(at=0;at<pSddft->Ntype;at++)
    {   
      cutoffr=pSddft->CUTOFF[at];
      offset = (PetscInt)ceil(cutoffr/delta + 0.5);
      /*
       * since the single atom electron density table read from the file does not have a value
       * at r=0, we assume the value at r=0 to be 0 
       */
      tableR[0]=0.0; tableRho[0]=0.0; 
      count=1;
      do{
  tableR[count] = pSddft->psd[at].RadialGrid[count-1];
  tableRho[count] = pSddft->psd[at].uu[count-1]; 
  count++;
      }while( PetscRealPart(tableR[count-1]) <= pSddft->CUTOFF[at]+4.0);
      

      rmax = tableR[count-1];
      int start,end;
      start = (int)floor(pSddft->startPos[at]/3);
      end = (int)floor(pSddft->endPos[at]/3);

      /*
       * derivatives of the spline fit to the single atom electron density
       */      
      PetscMalloc(sizeof(PetscScalar)*count,&YD);
      getYD_gen(tableR,tableRho,YD,count); 
      
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
     * find if domain of influence of pseudocharge overlaps with the domain stored by
     * processor  
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
    
        nzVps = zend-zstart+1+o*2;
        nyVps = yend-ystart+1+o*2;
        nxVps = xend-xstart+1+o*2;
  
        PetscMalloc(sizeof(PetscScalar**)*nzVps,&pRhoArray);       
        if(pRhoArray == NULL)
    {
      //cout<<"Memory allocation fail";
      exit(1);
    }
        for(i = 0; i < nzVps; i++)
    {
      PetscMalloc(sizeof(PetscScalar*)*nyVps,&pRhoArray[i]);          
      if(pRhoArray[i] == NULL)
        {
          //cout<<"Memory allocation fail";
          exit(1);
        }
    
      for(j=0;j<nyVps;j++)
        {
          PetscMalloc(sizeof(PetscScalar)*nxVps,&pRhoArray[i][j]);            
          if(pRhoArray[i][j] == NULL)
      {
        //cout<<"Memory allocation fail";
        exit(1);
      } 
        }      
    }

        i0 = xstart-o;
        j0 = ystart-o;
        k0 = zstart-o;
        
        /* 
         * calculate the single atom electron density at nodes in the overlap region
         */         
        for(k=o;k<nzVps-o;k++) 
    for(j=o;j<nyVps-o;j++)  
      for(i=o;i<nxVps-o;i++) 
        {    
          Ii = i+i0;
          J = j+j0;
          K = k+k0;
    
          x = delta*I - R_x ;
          y = delta*J - R_y ;
          z = delta*K - R_z ;   
          r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
                 
          if(r == tableR[0])
      {
        pRhoArray[k][j][i] = tableRho[0];
      }
          else if( PetscRealPart(r) <= PetscRealPart(rmax) )
      {
        ispline_gen(tableR,tableRho,count,&r,&pRhoArray[k][j][i],&Dtemp,1,YD);
      }  
          else  
      {
        pRhoArray[k][j][i] = 0.0;
      }
          
        }
        /*
         * superpose single atom electron densities for each node in the overlap region
         */        
        for(k=zstart;k<=zend;k++)
    for(j=ystart;j<=yend;j++)
      for(i=xstart;i<=xend;i++)
        {         
          Ii = i - i0;
          J=j-j0;
          K=k-k0;    
          RholcArrGlbIdx[k][j][i] += pRhoArray[K][J][Ii];
        }   

        for(i = 0; i < nzVps; i++)
    {
      for(j=0;j<nyVps;j++)
        {
          PetscFree(pRhoArray[i][j]);          
        }
      PetscFree(pRhoArray[i]);         
    }  
        PetscFree(pRhoArray);        
      }  
  }      
      PetscFree(YD);   
    }
    
  DMDAVecRestoreArray(pSddft->da,pSddft->SuperposAtRho,&RholcArrGlbIdx);
  VecRestoreArray(pSddft->Atompos,&pAtompos);
   
  VecSum(pSddft->SuperposAtRho,&rhosum); 
 
  /*
   * scale the initial guess electron density with respect to pseudocharge density 
   */
  rhosum = pSddft->noetot/(delta*delta*delta*rhosum); 
  VecScale(pSddft->SuperposAtRho,rhosum);    
  VecSum(pSddft->SuperposAtRho,&rhosum);
  PetscPrintf(PETSC_COMM_WORLD,"Total charge enclosed by guess electron density:%lf\n",rhosum*delta*delta*delta);

  return;
}




void PeriodicSuperpositionAtomicCharge_VecInit(SDDFT_OBJ* pSddft)
{ 
  PetscScalar ***RholcArrGlbIdx;
  PetscScalar ***pRhoArray=NULL;
  PetscScalar *pAtompos;
  PetscScalar *YD=NULL;
   
  int count;
  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0;
  PetscScalar x0, y0, z0, X0,Y0,Z0; 
  PetscScalar tableRho[MAX_TABLE_SIZE],tableR[MAX_TABLE_SIZE],rhosum;
  PetscScalar Dtemp;

  PetscInt  i=0,j,k,xi,yj,zk,offset,Ii,J,K,nzVps,nyVps,nxVps,i0,j0,k0,poscnt,index=0,at;
  PetscScalar delta=pSddft->delta,x,y,z,r,cutoffr,rmax;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt o = pSddft->order;
  PetscInt PP,QQ,RR;
  PetscInt Imax_x,Imin_x,Imax_y,Imin_y,Imax_z,Imin_z;
  
  DMDAVecGetArray(pSddft->da,pSddft->SuperposAtRho,&RholcArrGlbIdx);
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
  
  VecGetArray(pSddft->Atompos,&pAtompos);
     
  /*
   * loop over different types of atoms 
   */ 
  for(at=0;at<pSddft->Ntype;at++)
    {   
      cutoffr=pSddft->CUTOFF[at];
      offset = (PetscInt)ceil(cutoffr/delta + 0.5);
      /*
       * since the single atom electron density table read from the file does not have a value
       * at r=0, we assume the value at r=0 to be 0 
       */
      tableR[0]=0.0; tableRho[0]=0.0; 
      count = 1;
      do{
        tableR[count] = pSddft->psd[at].RadialGrid[count-1];
        tableRho[count] = pSddft->psd[at].uu[count-1]; 
        count++;
      } while( PetscRealPart(tableR[count-1]) <= pSddft->CUTOFF[at]+4.0);
      
      rmax = tableR[count-1];
      int start,end;
      start = (int)floor(pSddft->startPos[at]/3);
      end = (int)floor(pSddft->endPos[at]/3);
            
      /*
       * derivatives of the spline fit to the single atom electron density
       */  
      PetscMalloc(sizeof(PetscScalar)*count,&YD);
      getYD_gen(tableR,tableRho,YD,count); 

      /*
       * loop over every atom of a given type
       */
      for(poscnt=start;poscnt<=end;poscnt++)
  { 
    X0 = pAtompos[index++];
    Y0 = pAtompos[index++];
    Z0 = pAtompos[index++];
      
    xi = (int)((X0 + R_x)/delta + 0.5); 
    yj = (int)((Y0 + R_y)/delta + 0.5);
    zk = (int)((Z0 + R_z)/delta + 0.5);

    Imax_x=0; Imin_x=0; Imax_y=0; Imin_y=0; Imax_z=0; Imin_z=0;

    /*
     * find if domain of influence of pseudocharge overlaps with the domain stored by
     * processor  
     */
    if(xi-offset <= 0)
      Imax_x= ceil(cutoffr/R_x);

    if(xi+offset > n_x)
      Imin_x= -ceil(cutoffr/R_x);

    if(yj-offset <= 0)
      Imax_y= ceil(cutoffr/R_y);

    if(yj+offset > n_y)
      Imin_y= -ceil(cutoffr/R_y);

    if(zk-offset <= 0)
      Imax_z= ceil(cutoffr/R_z);

    if(zk+offset > n_z)  
      Imin_z= -ceil(cutoffr/R_z);

    for(PP=Imin_x;PP<=Imax_x;PP++)
      for(QQ=Imin_y;QQ<=Imax_y;QQ++)
        for(RR=Imin_z;RR<=Imax_z;RR++)
    {
  
      xstart=-1; ystart=-1; zstart=-1; xend=-1; yend=-1; zend=-1; overlap=0;
       
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
       * find if domain of influence of pseudocharge overlaps with the domain stored by
       * processor  
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
#ifdef _DEBUG
          printf("Bcube xs:%d, ys:%d, zs:%d\n",xs,ys,zs);
          printf("Overlap xstart:%d, ystart:%d, zstart:%d\n", xstart, ystart,zstart);
          printf("overlap xend:%d, yend:%d, zend:%d\n", xend,yend,zend);
#endif
    
          nzVps = zend-zstart+1+o*2;
          nyVps = yend-ystart+1+o*2;
          nxVps = xend-xstart+1+o*2;
  
          PetscMalloc(sizeof(PetscScalar**)*nzVps,&pRhoArray);       
          if(pRhoArray == NULL)
      {
        //cout<<"Memory allocation fail";
        exit(1);
      }
          for(i = 0; i < nzVps; i++)
      {
        PetscMalloc(sizeof(PetscScalar*)*nyVps,&pRhoArray[i]);          
        if(pRhoArray[i] == NULL)
          {
            //cout<<"Memory allocation fail";
            exit(1);
          }
    
        for(j=0;j<nyVps;j++)
          {
            PetscMalloc(sizeof(PetscScalar)*nxVps,&pRhoArray[i][j]);            
            if(pRhoArray[i][j] == NULL)
        {
          //cout<<"Memory allocation fail";
          exit(1);
        } 
          }      
      }

          i0 = xstart-o;
          j0 = ystart-o;
          k0 = zstart-o;
        
          /* 
           * calculate the single atom electron density at nodes in the overlap region
           */
          for(k=o;k<nzVps-o;k++) 
      for(j=o;j<nyVps-o;j++)  
        for(i=o;i<nxVps-o;i++) 
          {    
            Ii = i+i0;
            J = j+j0;
            K = k+k0;
    
            x = delta*Ii - R_x ;
            y = delta*J - R_y ;
            z = delta*K - R_z ;   
            r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
                 
            if(r == tableR[0])
        {
          pRhoArray[k][j][i] = tableRho[0];
        }
            else if( PetscRealPart(r) <= PetscRealPart(rmax) )
        {
          ispline_gen(tableR,tableRho,count,&r,&pRhoArray[k][j][i],&Dtemp,1,YD);
        }  
            else  
        {
          pRhoArray[k][j][i] = 0.0; 
        }
          
          }
      
          /*
           * superpose single atom electron densities for each node in the overlap region
           */
          for(k=zstart;k<=zend;k++)
      for(j=ystart;j<=yend;j++)
        for(i=xstart;i<=xend;i++)
          {        
            Ii = i - i0;
            J = j - j0;
            K = k - k0;    
            RholcArrGlbIdx[k][j][i] += pRhoArray[K][J][Ii];
          }   

          for(i = 0; i < nzVps; i++)
      {
        for(j=0;j<nyVps;j++)
          {
            PetscFree(pRhoArray[i][j]);          
          }
        PetscFree(pRhoArray[i]);         
      }  
          PetscFree(pRhoArray);        
        } 
    }
  }
      PetscFree(YD); 
  
    }
        
  
  DMDAVecRestoreArray(pSddft->da,pSddft->SuperposAtRho,&RholcArrGlbIdx);
  VecRestoreArray(pSddft->Atompos,&pAtompos);  
  VecSum(pSddft->SuperposAtRho,&rhosum); 

  /*
   * scale the initial guess electron density with respect to pseudocharge density 
   */ 
  rhosum = pSddft->noetot/(delta*delta*delta*rhosum); 
  VecScale(pSddft->SuperposAtRho,rhosum); 
  VecSum(pSddft->SuperposAtRho,&rhosum);
  PetscPrintf(PETSC_COMM_WORLD,"Total charge enclosed by guess electron density:%lf\n",rhosum*delta*delta*delta);

  return;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                ChargDensB_TM_VecInit: Reference pseudocharge density calculation.         //
//                         Additionally calculates the reference self energy                 //
///////////////////////////////////////////////////////////////////////////////////////////////
void ChargDensB_TM_VecInit(SDDFT_OBJ* pSddft)
{
  
  //FILE *fRvsVFile;
  PetscScalar ***BlcArrGlbIdx_TM;
  PetscScalar ***BlcVpsArrGlbIdx_TM;
  PetscScalar *pAtompos;

  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0;
  PetscScalar x0, y0, z0,coeffs[MAX_ORDER+1];
  
  //char RvsVFile[100]= "./pseudopotential/";
  PetscScalar Bval_TM, intb_TM; 
  PetscScalar pVpsArray_TM_KJI, pVpsArray_TM_KJIma, pVpsArray_TM_KJIpa;
  PetscScalar pVpsArray_TM_KJmaI, pVpsArray_TM_KJpaI, pVpsArray_TM_KmaJI, pVpsArray_TM_KpaJI;
  int at;
  PetscInt i=0;
  PetscInt j, k, xi, yj, zk, offset, p;
  //PetscInt nzVps,nyVps,nxVps;
  //PetscInt i0,j0,k0;
  PetscInt a,poscnt;
  PetscInt index = 0;
  PetscScalar delta=pSddft->delta,x,y,z,r,cutoffr,rcut=pSddft->REFERENCE_CUTOFF;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt o = pSddft->order; 

  DMDAVecGetArray(pSddft->da,pSddft->chrgDensB_TM,&BlcArrGlbIdx_TM);
  DMDAVecGetArray(pSddft->da,pSddft->bjVj_TM,&BlcVpsArrGlbIdx_TM);  
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
 
  for(p=0;p<=o;p++)
    {
      coeffs[p] = pSddft->coeffs[p]/(2*M_PI);
    }  
     
  VecGetArray(pSddft->Atompos,&pAtompos);

  /*
   * loop over different types of atoms 
   */ 
  for(at=0;at<pSddft->Ntype;at++)
    {
      cutoffr=pSddft->CUTOFF[at];
      offset = (PetscInt)ceil(cutoffr/delta + 0.5);
      int start,end;
      start = (int)floor(pSddft->startPos[at]/3);
      end = (int)floor(pSddft->endPos[at]/3);

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
     * find if domain of influence of pseudocharge overlaps with the domain stored by 
     * processor  
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
    
        //nzVps = zend-zstart+1+o*2;
        //nyVps = yend-ystart+1+o*2;
        //nxVps = xend-xstart+1+o*2;
        
        /*
         * use finite difference stencil and reference pseudopotential values to 
         * calculate reference pseudocharge density and contribution to reference self
         * energy at nodes inside the overlap region   
         */
        for(k=zstart;k<=zend;k++)
    for(j=ystart;j<=yend;j++)
      for(i=xstart;i<=xend;i++)
        {

          Bval_TM=0;          
          x = delta*i- R_x; y = delta*j - R_y; z = delta*k - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
          pVpsArray_TM_KJI = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
          Bval_TM=pVpsArray_TM_KJI*coeffs[0];
          for(a=1;a<=o;a++)
      {       
        x = delta*(i-a)- R_x; y = delta*j - R_y; z = delta*k - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
        pVpsArray_TM_KJIma = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
        
        x = delta*(i+a)- R_x; y = delta*j - R_y; z = delta*k - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
        pVpsArray_TM_KJIpa = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
       
        x = delta*i- R_x; y = delta*(j-a) - R_y; z = delta*k - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
        pVpsArray_TM_KJmaI = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
        
        x = delta*i- R_x; y = delta*(j+a) - R_y; z = delta*k - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
        pVpsArray_TM_KJpaI = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
        
        x = delta*i- R_x; y = delta*j - R_y; z = delta*(k-a) - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
        pVpsArray_TM_KmaJI = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
        
        x = delta*i- R_x; y = delta*j - R_y; z = delta*(k+a) - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
        pVpsArray_TM_KpaJI = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
          
        Bval_TM+=(pVpsArray_TM_KJIma + pVpsArray_TM_KJIpa + pVpsArray_TM_KJmaI +        
            pVpsArray_TM_KJpaI + pVpsArray_TM_KmaJI + pVpsArray_TM_KpaJI)*coeffs[a];
      }
              
          BlcArrGlbIdx_TM[k][j][i] += Bval_TM;
          BlcVpsArrGlbIdx_TM[k][j][i] += (pVpsArray_TM_KJI)*Bval_TM;
        }           
      }
  }    
    }

  DMDAVecRestoreArray(pSddft->da,pSddft->chrgDensB_TM,&BlcArrGlbIdx_TM);  
  DMDAVecRestoreArray(pSddft->da,pSddft->bjVj_TM,&BlcVpsArrGlbIdx_TM);
    
  VecSum(pSddft->bjVj_TM,&pSddft->Eself_TM); 
  pSddft->Eself_TM = -0.5*pSddft->Eself_TM*delta*delta*delta;
  PetscPrintf(PETSC_COMM_WORLD,"Reference Self energy of nuclei:%9.9f (Hartree)\n",pSddft->Eself_TM);
  
  VecRestoreArray(pSddft->Atompos,&pAtompos); 
 
  /*
   * check sign of reference pseudocharge density
   */
  VecSum(pSddft->chrgDensB_TM,&intb_TM);
  assert( PetscRealPart(intb_TM) < 0);

  intb_TM *= delta*delta*delta;
  PetscPrintf(PETSC_COMM_WORLD,"Total reference valence electronic charge: %.9f\n",intb_TM);
  
  return;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//        PeriodicChargDensB_TM_VecInit: Reference pseudocharge density calculation.         //
//                         Additionally calculates the reference self energy                 //
///////////////////////////////////////////////////////////////////////////////////////////////
void PeriodicChargDensB_TM_VecInit(SDDFT_OBJ* pSddft)
{  
  //FILE *fRvsVFile;
  PetscScalar ***BlcArrGlbIdx_TM;
  PetscScalar ***BlcVpsArrGlbIdx_TM;
  PetscScalar *pAtompos;

  PetscInt xcor,ycor,zcor,gxdim,gydim,gzdim,lxdim,lydim,lzdim;
  PetscInt xs,ys,zs,xl,yl,zl,xstart=-1,ystart=-1,zstart=-1,xend=-1,yend=-1,zend=-1,overlap=0;
  PetscScalar x0, y0, z0,X0,Y0,Z0,coeffs[MAX_ORDER+1];
  
  //char RvsVFile[100]= "./pseudopotential/";
  PetscScalar Bval_TM;  
  PetscScalar intb_TM;
  PetscScalar pVpsArray_TM_KJI, pVpsArray_TM_KJIma, pVpsArray_TM_KJIpa;
  PetscScalar pVpsArray_TM_KJmaI, pVpsArray_TM_KJpaI, pVpsArray_TM_KmaJI, pVpsArray_TM_KpaJI;
  int at;
  PetscInt i=0;
  PetscInt j, k, xi, yj, zk, offset, p;
  //PetscInt nzVps, nyVps, nxVps;
  //PetscInt i0,j0,k0;
  PetscInt a,poscnt,index=0;
  PetscInt PP,QQ,RR;
  PetscInt Imax_x,Imin_x,Imax_y,Imin_y,Imax_z,Imin_z;
  PetscScalar delta=pSddft->delta,x,y,z,r,cutoffr,rcut=pSddft->REFERENCE_CUTOFF;
  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt o = pSddft->order;

  DMDAVecGetArray(pSddft->da,pSddft->chrgDensB_TM,&BlcArrGlbIdx_TM);
  DMDAVecGetArray(pSddft->da,pSddft->bjVj_TM,&BlcVpsArrGlbIdx_TM);
  
  DMDAGetInfo(pSddft->da,0,&gxdim,&gydim,&gzdim,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);

  for(p=0;p<=o;p++)
    {
      coeffs[p] = pSddft->coeffs[p]/(2*M_PI);
    }           
     
  VecGetArray(pSddft->Atompos,&pAtompos);
   
  /*
   * loop over different types of atoms 
   */ 
  for(at=0;at<pSddft->Ntype;at++)
    {
      cutoffr=pSddft->CUTOFF[at];
      offset = (PetscInt)ceil(cutoffr/delta + 0.5);
      int start,end;
      start = (int)floor(pSddft->startPos[at]/3);
      end = (int)floor(pSddft->endPos[at]/3);

      /*
       * loop over every atom of a given type
       */
      for(poscnt=start;poscnt<=end;poscnt++)
  { 
     
    X0 = pAtompos[index++];
    Y0 = pAtompos[index++];
    Z0 = pAtompos[index++];
  
    xi = (int)((X0 + R_x)/delta + 0.5); 
    yj = (int)((Y0 + R_y)/delta + 0.5);
    zk = (int)((Z0 + R_z)/delta + 0.5);

    Imax_x=0; Imin_x=0; Imax_y=0; Imin_y=0; Imax_z=0; Imin_z=0;

    if(xi-offset <= 0)
      Imax_x= ceil(cutoffr/R_x);
    if(xi+offset > n_x) 
      Imin_x= -ceil(cutoffr/R_x);
    if(yj-offset <= 0)
      Imax_y= ceil(cutoffr/R_y);
    if(yj+offset > n_y)
      Imin_y= -ceil(cutoffr/R_y);
    if(zk-offset <= 0)
      Imax_z= ceil(cutoffr/R_z);
    if(zk+offset > n_z)  
      Imin_z= -ceil(cutoffr/R_z);

    /*
     * loop over the periodic images of atoms
     */
    for(PP=Imin_x;PP<=Imax_x;PP++)
      for(QQ=Imin_y;QQ<=Imax_y;QQ++)
        for(RR=Imin_z;RR<=Imax_z;RR++)
    {  
      xstart=-1; ystart=-1; zstart=-1; xend=-1; yend=-1; zend=-1; overlap=0;
       
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
       * find if domain of influence of pseudocharge overlaps with the domain stored by 
       * processor  
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
           
          //nzVps = zend-zstart+1+o*2;
          //nyVps = yend-ystart+1+o*2;
          //nxVps = xend-xstart+1+o*2;
      
          /*
           * use finite difference stencil and reference pseudopotential values to 
           * calculate reference pseudocharge density and contribution to reference self
           * energy at nodes inside the overlap region   
           */
          for(k=zstart;k<=zend;k++)
      for(j=ystart;j<=yend;j++)
        for(i=xstart;i<=xend;i++)
          {

            Bval_TM=0;         
            x = delta*i- R_x; y = delta*j - R_y; z = delta*k - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
            pVpsArray_TM_KJI = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
            Bval_TM=pVpsArray_TM_KJI*coeffs[0];
            for(a=1;a<=o;a++)
        {
     
          x = delta*(i-a)- R_x; y = delta*j - R_y; z = delta*k - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
          pVpsArray_TM_KJIma = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
     
          x = delta*(i+a)- R_x; y = delta*j - R_y; z = delta*k - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
          pVpsArray_TM_KJIpa = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
     
          x = delta*i- R_x; y = delta*(j-a) - R_y; z = delta*k - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
          pVpsArray_TM_KJmaI = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
    
          x = delta*i- R_x; y = delta*(j+a) - R_y; z = delta*k - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
          pVpsArray_TM_KJpaI = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
    
          x = delta*i- R_x; y = delta*j - R_y; z = delta*(k-a) - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
          pVpsArray_TM_KmaJI = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
     
          x = delta*i- R_x; y = delta*j - R_y; z = delta*(k+a) - R_z; r = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0));
          pVpsArray_TM_KpaJI = PseudopotReference(r,rcut,-1.0*pSddft->noe[at]);
          
          Bval_TM+=(pVpsArray_TM_KJIma + pVpsArray_TM_KJIpa + pVpsArray_TM_KJmaI +        
              pVpsArray_TM_KJpaI + pVpsArray_TM_KmaJI + pVpsArray_TM_KpaJI)*coeffs[a];
        }
              
            BlcArrGlbIdx_TM[k][j][i] += Bval_TM;
            BlcVpsArrGlbIdx_TM[k][j][i] += (pVpsArray_TM_KJI)*Bval_TM;                   
          }        
  
        }  
    }
  }
    
    }

  DMDAVecRestoreArray(pSddft->da,pSddft->chrgDensB_TM,&BlcArrGlbIdx_TM);  
  DMDAVecRestoreArray(pSddft->da,pSddft->bjVj_TM,&BlcVpsArrGlbIdx_TM);
    
  VecSum(pSddft->bjVj_TM,&pSddft->Eself_TM); 
  pSddft->Eself_TM = -0.5*pSddft->Eself_TM*delta*delta*delta;
  PetscPrintf(PETSC_COMM_WORLD,"Eself_TM:%9.9f\n",pSddft->Eself_TM);
  
  VecRestoreArray(pSddft->Atompos,&pAtompos); 
   
  /*
   * check sign of reference pseudocharge density
   */
  VecSum(pSddft->chrgDensB_TM,&intb_TM);
  assert( PetscRealPart(intb_TM) < 0 );
  intb_TM *= delta*delta*delta;
  PetscPrintf(PETSC_COMM_WORLD,"Total reference valence electronic charge: %.9f\n",intb_TM);
      
  return;
} 


///////////////////////////////////////////////////////////////////////////////////////////////
//                          PseudopotReference: reference pseudopotential                    //
//       Reference: Linear scaling solution of the all-electron Coulomb problem in solids;   //
//                              J.E.Pask, N. Sukumar and S.E. Mousavi                        //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar PseudopotReference(PetscScalar r,PetscScalar rcut,PetscScalar Znucl)
{
  PetscScalar Vr;

  if( PetscAbsScalar(r) <= PetscAbsScalar(rcut) )
    {
      Vr=(9.0*pow(r,7) - 30.0*pow(r,6)*rcut + 28.0*pow(r,5)*rcut*rcut - 14.0*pow(r,2)*pow(rcut,5) + 12.0*pow(rcut,7))/(5.0*pow(rcut,8));
    }
  else if( PetscAbsScalar(r) > PetscAbsScalar(rcut) )
    {
      Vr = 1.0/r;
    }
  else
    {
      PetscPrintf(PETSC_COMM_WORLD,"Incorrect r in correction, r = %.16lf \n",r);
      exit(1);
    }  
  Vr = Znucl*Vr;
  return Vr;

}
