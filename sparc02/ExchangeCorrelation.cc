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
  | file name: ExchangeCorrelation.cc          
  |
  | Description: This file contains the functions required for calculation of exchange 
  | correlation potential and energy 
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana, Deepa Phanish
  |
  | Last Modified: 2/26/2016   
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "petscsys.h"
#include "math.h"
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
///////////////////////////////////////////////////////////////////////////////////////////////
//              Vxc_Calc_CA: LDA Ceperley-Alder Exchange-correlation potential               // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode Vxc_Calc_CA(SDDFT_OBJ* pSddft)
{
  if(strcmp(pSddft->XC,"LDA")==0 || strcmp(pSddft->XC,"LDA_PW")==0) // Perdew-Wang
    { 
      Vxc_Calc_CA_PW(pSddft);
    }else if(strcmp(pSddft->XC,"LDA_PZ")==0) // Perdew-Zunger
    {
      Vxc_Calc_CA_PZ(pSddft);
    }else{
    printf("currently only LDA, LDA_PW and LDA_PZ options supported for exchange correlation\n");
    exit(1);
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//              Exc_Calc_CA: LDA Ceperley-Alder Exchange-correlation energy                  //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode Exc_Calc_CA(SDDFT_OBJ* pSddft)
{

  if(strcmp(pSddft->XC,"LDA")==0 || strcmp(pSddft->XC,"LDA_PW")==0)  // Perdew-Wang 
    {
      Exc_Calc_CA_PW(pSddft);
    }else if(strcmp(pSddft->XC,"LDA_PZ")==0) //Perdew-Zunger
    {
      Exc_Calc_CA_PZ(pSddft);
    }else{
    printf("currently only LDA, LDA_PW and LDA_PZ options supported for exchange correlation\n");
    exit(1);
  }
  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//        Vxc_Calc_CA_PW: LDA Ceperley-Alder Perdew-Wang Exchange-correlation potential      // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode Vxc_Calc_CA_PW(SDDFT_OBJ* pSddft)
{
  PetscInt  i,j,k;
  PetscScalar C3,rhoi;
  PetscScalar A,alpha1,beta1,beta2,beta3,beta4,Vxci;
  PetscScalar ***rholc,***Vxclc;
  PetscInt xcor,ycor,zcor,lxdim,lydim,lzdim;
  PetscScalar p;
  
  p = 1.0 ;
  A = 0.031091 ;
  alpha1 = 0.21370 ;
  beta1 = 7.5957 ;
  beta2 = 3.5876 ;
  beta3 = 1.6382 ;
  beta4 = 0.49294 ;

  C3 = 0.9847450218427; 
  
  DMDAVecGetArray(pSddft->da,pSddft->elecDensRho,&rholc);
  DMDAVecGetArray(pSddft->da,pSddft->Vxc,&Vxclc);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
 
  for(k=zcor; k<zcor+lzdim; k++)
    for(j=ycor; j<ycor+lydim; j++)
      for(i=xcor; i<xcor+lxdim; i++)
	{
	  rhoi = rholc[k][j][i];
    
   
	  if (rhoi==0)
	    {
	      Vxci = 0.0 ;	      
	    }
	  else
	    {	
	      Vxci = pow((0.75/(M_PI*rhoi)),(1.0/3.0)) ;
	      Vxci = (-2.0*A*(1.0+alpha1*Vxci))*log(1.0+1.0/(2.0*A*(beta1*pow(Vxci,0.5) + beta2*Vxci + beta3*pow(Vxci,1.5) + beta4*pow(Vxci,(p+1.0))))) 
		- (Vxci/3.0)*(-2.0*A*alpha1*log(1.0+1.0/(2.0*A*( beta1*pow(Vxci,0.5) + beta2*Vxci + beta3*pow(Vxci,1.5) + beta4*pow(Vxci,(p+1.0))))) 
			      - ((-2.0*A*(1.0+alpha1*Vxci))*(A*( beta1*pow(Vxci,-0.5)+ 2.0*beta2 + 3.0*beta3*pow(Vxci,0.5) + 2.0*(p+1.0)*beta4*pow(Vxci,p) ))) 
			      /((2.0*A*( beta1*pow(Vxci,0.5) + beta2*Vxci + beta3*pow(Vxci,1.5) + beta4*pow(Vxci,(p+1.0)) ) )*(2.0*A*( beta1*pow(Vxci,0.5) + beta2*Vxci + beta3*pow(Vxci,1.5) + beta4*pow(Vxci,(p+1.0)) ) )+(2.0*A*( beta1*pow(Vxci,0.5) + beta2*Vxci + beta3*pow(Vxci,1.5) + beta4*pow(Vxci,(p+1.0)) ) )) ) ;
	      
	    } 	
	  Vxci = Vxci - C3*pow(rhoi,1.0/3.0) ;     
	  Vxclc[k][j][i] = Vxci;    
    
	}
  
  DMDAVecRestoreArray(pSddft->da,pSddft->elecDensRho,&rholc);
  DMDAVecRestoreArray(pSddft->da,pSddft->Vxc,&Vxclc); 
   
  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//        Exc_Calc_CA_PW: LDA Ceperley-Alder Perdew-Wang Exchange-correlation energy         // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode Exc_Calc_CA_PW(SDDFT_OBJ* pSddft)
{
  PetscInt xcor,ycor,zcor,lxdim,lydim,lzdim,i,j,k;
  PetscScalar delVol,A,alpha1,beta1,beta2,beta3,beta4,C2,Exc,rhoi,Ec,Ex,rs,p;
  Vec vecExc;
  PetscScalar ***rholc,***Exclc;

  p = 1.0;
  A = 0.031091 ;
  alpha1 = 0.21370 ;
  beta1 = 7.5957 ;
  beta2 = 3.5876 ;
  beta3 = 1.6382 ;
  beta4 = 0.49294 ;
  C2 = 0.73855876638202 ;

  delVol = pow(pSddft->delta,3.0);
  DMDAVecGetArray(pSddft->da,pSddft->elecDensRho,&rholc);
  VecDuplicate(pSddft->elecDensRho,&vecExc);

  DMDAVecGetArray(pSddft->da,pSddft->elecDensRho,&rholc);
  DMDAVecGetArray(pSddft->da,vecExc,&Exclc);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
    
  for(k=zcor; k<zcor+lzdim; k++)
    for(j=ycor; j<ycor+lydim; j++)
      for(i=xcor; i<xcor+lxdim; i++)
	{
	  rhoi = rholc[k][j][i];
	  Ex = -C2*pow(rhoi,1.0/3.0);
	
	  if (rhoi==0)
	    {
	      Ec = 0.0 ;
	    }
	  else
	    {
	      Ec = pow(0.75/(M_PI*rhoi),(1.0/3.0));
	      Ec = -2.0*A*(1.0+alpha1*Ec)*log(1.0+1.0/(2.0*A*( beta1*pow(Ec,0.5) + beta2*Ec + beta3*pow(Ec,1.5) + beta4*pow(Ec,(p+1.0))))) ; 
	    }    
	  Exclc[k][j][i] = Ex+Ec; 
	}
       
  DMDAVecRestoreArray(pSddft->da,pSddft->elecDensRho,&rholc); 
  DMDAVecRestoreArray(pSddft->da,vecExc,&Exclc);
  VecDot(vecExc,pSddft->elecDensRho,&Exc);
  Exc*=delVol;
  pSddft->Exc = Exc; 
  VecDestroy(&vecExc);  

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//        Vxc_Calc_CA_PZ: LDA Ceperley-Alder Perdew-Zunger Exchange-correlation potential    // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode Vxc_Calc_CA_PZ(SDDFT_OBJ* pSddft)
{
  PetscInt  i,j,k;
  PetscScalar C3,rhoi,rs;
  PetscScalar A,B,C,D,gamma1,beta1,beta2,Vxci;
  PetscScalar ***rholc,***Vxclc;
  PetscInt xcor,ycor,zcor,lxdim,lydim,lzdim;
  
  A = 0.0311;
  B = -0.048 ;
  C = 0.002 ;
  D = -0.0116 ;
  gamma1 = -0.1423 ;
  beta1 = 1.0529 ;
  beta2 = 0.3334 ; 
 
  C3 = 0.9847450218427;
   
  DMDAVecGetArray(pSddft->da,pSddft->elecDensRho,&rholc);
  DMDAVecGetArray(pSddft->da,pSddft->Vxc,&Vxclc);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
 
  for(k=zcor; k<zcor+lzdim; k++)
    for(j=ycor; j<ycor+lydim; j++)
      for(i=xcor; i<xcor+lxdim; i++)
	{
	  rhoi = rholc[k][j][i];
    
   
	  if (rhoi==0)
	    {
	      Vxci = 0.0 ;	     
	    }
	  else
	    {	     
	      rs = pow(0.75/(M_PI*rhoi),(1.0/3.0));
	      if (rs<1.0)
		{
		  Vxci = log(rs)*(A+(2.0/3.0)*C*rs) + (B-(1.0/3.0)*A) + (1.0/3.0)*(2.0*D-C)*rs; 
		}
	      else
		{
		  Vxci = (gamma1 + (7.0/6.0)*gamma1*beta1*pow(rs,0.5) + (4.0/3.0)*gamma1*beta2*rs)/pow(1+beta1*pow(rs,0.5)+beta2*rs,2.0) ;		  
		}
	    } 	
	  Vxci = Vxci - C3*pow(rhoi,1.0/3.0) ;     
	  Vxclc[k][j][i] = Vxci;        
	}
  
  DMDAVecRestoreArray(pSddft->da,pSddft->elecDensRho,&rholc);
  DMDAVecRestoreArray(pSddft->da,pSddft->Vxc,&Vxclc); 
   
  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//        Exc_Calc_CA_PZ: LDA Ceperley-Alder Perdew-Zunger Exchange-correlation energy       // 
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode Exc_Calc_CA_PZ(SDDFT_OBJ* pSddft)
{  
  PetscInt xcor,ycor,zcor,lxdim,lydim,lzdim,i,j,k;
  PetscScalar delVol,A,B,C,D,gamma1,beta1,beta2,C2,Exc,rhoi,Ec,Ex,rs;
  Vec vecExc;
  PetscScalar ***rholc,***Exclc;

  A = 0.0311;
  B = -0.048 ;
  C = 0.002 ;
  D = -0.0116 ;
  gamma1 = -0.1423 ;
  beta1 = 1.0529 ;
  beta2 = 0.3334 ; 
  C2 = 0.73855876638202;

  delVol = pow(pSddft->delta,3.0);
  VecDuplicate(pSddft->elecDensRho,&vecExc);

  DMDAVecGetArray(pSddft->da,pSddft->elecDensRho,&rholc);
  DMDAVecGetArray(pSddft->da,vecExc,&Exclc);
  DMDAGetCorners(pSddft->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
    
  for(k=zcor; k<zcor+lzdim; k++)
    for(j=ycor; j<ycor+lydim; j++)
      for(i=xcor; i<xcor+lxdim; i++)
	{
	  rhoi = rholc[k][j][i];
	  Ex = -C2*pow(rhoi,1.0/3.0);
	
	  if (rhoi==0)
	    {
	      Ec = 0.0 ;
	    }
	  else
	    {
	      rs = pow(0.75/(M_PI*rhoi),(1.0/3.0));
	      if (rs<1.0)
		{
		  Ec = A*log(rs) + B + C*rs*log(rs) + D*rs ;
		}
	      else
		{
		  Ec = gamma1/(1.0+beta1*pow(rs,0.5)+beta2*rs) ;
		}
	    }    
	  Exclc[k][j][i] = Ex+Ec; 
	}
       
  DMDAVecRestoreArray(pSddft->da,pSddft->elecDensRho,&rholc); 
  DMDAVecRestoreArray(pSddft->da,vecExc,&Exclc);
  VecDot(vecExc,pSddft->elecDensRho,&Exc);
  Exc*=delVol;
  pSddft->Exc = Exc;    
  VecDestroy(&vecExc); 

  return 0;
}


