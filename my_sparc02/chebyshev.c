#include "sddft.h"
#include "petscsys.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SIGN(a,b) ((b) >= 0.0 ? PetscAbsScalar(a) : -PetscAbsScalar(a))

///////////////////////////////////////////////////////////////////////////////////////////////
//                         ChebyshevFiltering: Chebychev filter function                     //
///////////////////////////////////////////////////////////////////////////////////////////////
void ChebyshevFiltering(SDDFT_OBJ* pSddft,int m,PetscScalar a,PetscScalar b,PetscScalar a0)
{
  /*
   * a0 is minimum eigenvalue
   * a is cutoff eigenvalue
   * b is maximum eigenvalue
   * m is degree of Chebyshev polynomial
   */
  PetscScalar e,c,sigma,sigma1,sigma2,gamma;
  int i;
  e=(b-a)/2.0;  c=(b+a)/2.0;
  sigma = e/(a0-c); sigma1 = sigma; gamma = 2.0/sigma1;
  /*
   * shift the Hamiltonian by constant 	
   */	   
  MatShift(pSddft->HamiltonianOpr,-c); 	 
  //MatMatMultNumeric(pSddft->HamiltonianOpr,pSddft->XOrb, pSddft->YOrb);
  MatProductCreate(pSddft->HamiltonianOpr, pSddft->XOrb, NULL, &pSddft->YOrb);
  MatProductSetType(pSddft->YOrb, MATPRODUCT_AB);
  MatProductSetFromOptions(pSddft->YOrb);
  MatProductSymbolic(pSddft->YOrb);
  MatProductNumeric(pSddft->YOrb);


  MatScale(pSddft->YOrb,sigma1/e);
	          
  for(i=2;i<=m;i++)
    {
      sigma2 = 1.0/(gamma-sigma);	     
      //MatMatMultNumeric(pSddft->HamiltonianOpr, pSddft->YOrb, pSddft->YOrbNew);
      MatProductCreate(pSddft->HamiltonianOpr, pSddft->YOrb, NULL, &pSddft->YOrbNew);
      MatProductSetType(pSddft->YOrbNew, MATPRODUCT_AB);
      MatProductSetFromOptions(pSddft->YOrbNew);
      MatProductSymbolic(pSddft->YOrbNew);
      MatProductNumeric(pSddft->YOrbNew);

      MatScale(pSddft->YOrbNew,2*sigma2/e);
      MatAXPY(pSddft->YOrbNew,-sigma*sigma2,pSddft->XOrb,SAME_NONZERO_PATTERN);
			 
      MatCopy(pSddft->YOrb,pSddft->XOrb,SAME_NONZERO_PATTERN);
      MatCopy(pSddft->YOrbNew,pSddft->YOrb,SAME_NONZERO_PATTERN);			                           			  
      sigma = sigma2;
    }	  
  /*
   * shift back the Hamiltonian by subtracting the constant previously added
   */
  MatShift(pSddft->HamiltonianOpr,c);	 
  pSddft->ChebyshevCallCounter++;
	   	     
  return;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//               kPointChebyshevFiltering: Chebychev filter function for k-point             //
///////////////////////////////////////////////////////////////////////////////////////////////
void kPointChebyshevFiltering(SDDFT_OBJ* pSddft,int m,PetscScalar a,PetscScalar b,PetscScalar a0,int k)
{

  /*
   * a0 is minimum eigenvalue
   * a is cutoff eigenvalue
   * b is maximum eigenvalue
   * m is degree of Chebyshev polynomial
   */
  PetscScalar e,c,sigma,sigma1,sigma2,gamma;
  int i;
  e=(b-a)/2.0;  c=(b+a)/2.0;
  sigma = e/(a0-c); sigma1 = sigma; gamma = 2.0/sigma1;
	   
  /*
   * shift the Hamiltonian by constant 	
   */
	   
  /*
   * Real part
   */

  MatShift(pSddft->HamiltonianOpr1,-c); 
  //MatMatMultNumeric(pSddft->HamiltonianOpr1,pSddft->XOrb1[k],pSddft->YOrb1[k]);
  MatProductCreate(pSddft->HamiltonianOpr1, pSddft->XOrb1[k], NULL, &pSddft->YOrb1[k]);
  MatProductSetType(pSddft->YOrb1[k], MATPRODUCT_AB);
  MatProductSetFromOptions(pSddft->YOrb1[k]);
  MatProductSymbolic(pSddft->YOrb1[k]);
  MatProductNumeric(pSddft->YOrb1[k]);

  //MatMatMultNumeric(pSddft->HamiltonianOpr2, pSddft->XOrb2[k], pSddft->ZOrb1);
  MatProductCreate(pSddft->HamiltonianOpr2, pSddft->XOrb2[k], NULL, &pSddft->ZOrb1);
  MatProductSetType(pSddft->ZOrb1, MATPRODUCT_AB);
  MatProductSetFromOptions(pSddft->ZOrb1);
  MatProductSymbolic(pSddft->ZOrb1);
  MatProductNumeric(pSddft->ZOrb1);
  
  MatAXPY(pSddft->YOrb1[k],-1.0,pSddft->ZOrb1,DIFFERENT_NONZERO_PATTERN);	   
  MatScale(pSddft->YOrb1[k],sigma1/e);   
 

  /*
   * Imaginary part
   */
  //MatMatMultNumeric(pSddft->HamiltonianOpr1,pSddft->XOrb2[k],pSddft->YOrb2[k]);
  MatProductCreate(pSddft->HamiltonianOpr1, pSddft->XOrb2[k], NULL, &pSddft->YOrb2[k]);
  MatProductSetType(pSddft->YOrb2[k], MATPRODUCT_AB);
  MatProductSetFromOptions(pSddft->YOrb2[k]);
  MatProductSymbolic(pSddft->YOrb2[k]);
  MatProductNumeric(pSddft->YOrb2[k]);
  
  //MatMatMultNumeric(pSddft->HamiltonianOpr2,pSddft->XOrb1[k],pSddft->ZOrb2);
  MatProductCreate(pSddft->HamiltonianOpr2, pSddft->XOrb1[k], NULL, &pSddft->ZOrb2);
  MatProductSetType(pSddft->ZOrb2, MATPRODUCT_AB);
  MatProductSetFromOptions(pSddft->ZOrb2);
  MatProductSymbolic(pSddft->ZOrb2);
  MatProductNumeric(pSddft->ZOrb2);

  MatAXPY(pSddft->YOrb2[k],1.0,pSddft->ZOrb2,DIFFERENT_NONZERO_PATTERN);	   
  MatScale(pSddft->YOrb2[k],sigma1/e);   
   
  for(i=2;i<=m;i++)
    {
      sigma2 = 1.0/(gamma-sigma);			  
	     
      /*
       * Real
       */
      //MatMatMultNumeric(pSddft->HamiltonianOpr1,pSddft->YOrb1[k],pSddft->YOrbNew1);
      MatProductCreate(pSddft->HamiltonianOpr1, pSddft->YOrb1[k], NULL, &pSddft->YOrbNew1);
      MatProductSetType(pSddft->YOrbNew1, MATPRODUCT_AB);
      MatProductSetFromOptions(pSddft->YOrbNew1);
      MatProductSymbolic(pSddft->YOrbNew1);
      MatProductNumeric(pSddft->YOrbNew1);

      //MatMatMultNumeric(pSddft->HamiltonianOpr2,pSddft->YOrb2[k],pSddft->ZOrbNew1);
      MatProductCreate(pSddft->HamiltonianOpr2, pSddft->XOrb2[k], NULL, &pSddft->ZOrbNew1);
      MatProductSetType(pSddft->ZOrbNew1, MATPRODUCT_AB);
      MatProductSetFromOptions(pSddft->ZOrbNew1);
      MatProductSymbolic(pSddft->ZOrbNew1);
      MatProductNumeric(pSddft->ZOrbNew1);

      
      MatAXPY(pSddft->YOrbNew1,-1.0,pSddft->ZOrbNew1,DIFFERENT_NONZERO_PATTERN); 
      MatScale(pSddft->YOrbNew1,2*sigma2/e);
      MatAXPY(pSddft->YOrbNew1,-sigma*sigma2,pSddft->XOrb1[k],SAME_NONZERO_PATTERN);
	       
      /*
       * Imaginary
       */
      //MatMatMultNumeric(pSddft->HamiltonianOpr1,pSddft->YOrb2[k],pSddft->YOrbNew2);
      MatProductCreate(pSddft->HamiltonianOpr1, pSddft->YOrb2[k], NULL, &pSddft->YOrbNew2);
      MatProductSetType(pSddft->YOrbNew2, MATPRODUCT_AB);
      MatProductSetFromOptions(pSddft->YOrbNew2);
      MatProductSymbolic(pSddft->YOrbNew2);
      MatProductNumeric(pSddft->YOrbNew2);
      
      //MatMatMultNumeric(pSddft->HamiltonianOpr2,pSddft->YOrb1[k],pSddft->ZOrbNew2);
      MatProductCreate(pSddft->HamiltonianOpr2, pSddft->YOrb1[k], NULL, &pSddft->ZOrbNew2);
      MatProductSetType(pSddft->ZOrbNew2, MATPRODUCT_AB);
      MatProductSetFromOptions(pSddft->ZOrbNew2);
      MatProductSymbolic(pSddft->ZOrbNew2);
      MatProductNumeric(pSddft->ZOrbNew2);
      
      MatAXPY(pSddft->YOrbNew2,1.0,pSddft->ZOrbNew2,DIFFERENT_NONZERO_PATTERN); 
      MatScale(pSddft->YOrbNew2,2*sigma2/e);
      MatAXPY(pSddft->YOrbNew2,-sigma*sigma2,pSddft->XOrb2[k],SAME_NONZERO_PATTERN);
		      
      /*
       * Swap real
       */
      MatCopy(pSddft->YOrb1[k],pSddft->XOrb1[k],SAME_NONZERO_PATTERN);
      MatCopy(pSddft->YOrbNew1,pSddft->YOrb1[k],SAME_NONZERO_PATTERN);
	     
      /*
       * Swap imaginary
       */
      MatCopy(pSddft->YOrb2[k],pSddft->XOrb2[k],SAME_NONZERO_PATTERN);
      MatCopy(pSddft->YOrbNew2,pSddft->YOrb2[k],SAME_NONZERO_PATTERN);
      sigma = sigma2;
    }   	   
  MatShift(pSddft->HamiltonianOpr1,c);
  pSddft->ChebyshevCallCounter++; 

	   	     
  return;
}
//////////////////////////////////////////////////////////////////////////////////////////////
//         Lanczos:  Lanczos algorithm for calculating the minimum and maximum              //
//                                  eigenvalues of Hamiltonian                              //
//        see W.H. Press, Numerical recepies 3rd edition: The art of scientific computing,  //
//                                 Cambridge university press, 2007                         //
/////////////////////////////////////////////////////////////////////////////////////////////  
void Lanczos(SDDFT_OBJ* pSddft, PetscScalar* EigenMin, PetscScalar* EigenMax)
{
  
  PetscScalar tolLanczos=pSddft->LANCZOSTOL;
  PetscScalar Vnorm;
  PetscInt Nx,Ny,Nz;
  PetscScalar Lk,Mk,Lkp1,Mkp1,deltaL,deltaM;	
  int k;
	
  Nx = pSddft->numPoints_x;
  Ny = pSddft->numPoints_y;
  Nz = pSddft->numPoints_z;
	

  PetscScalar *a,*b;
  PetscMalloc(sizeof(PetscScalar)*(Nx*Ny*Nz),&a);
  PetscMalloc(sizeof(PetscScalar)*(Nx*Ny*Nz),&b);

  Vec Vk;
  Vec Vkm1;
  Vec Vkp1;

  VecDuplicate(pSddft->elecDensRho,&Vk);
  VecDuplicate(pSddft->elecDensRho,&Vkm1);
  VecDuplicate(pSddft->elecDensRho,&Vkp1);
	
  VecSet(Vkm1,1.0);
  VecNorm(Vkm1,NORM_2,&Vnorm);
  VecScale(Vkm1,1.0/Vnorm);
  MatMult(pSddft->HamiltonianOpr, Vkm1, Vk);
	
  VecDot(Vkm1, Vk, &a[0]);
  VecAXPY(Vk, -a[0], Vkm1);
  VecNorm(Vk,NORM_2, &b[0]);
  VecScale(Vk, 1.0/b[0]);
	
  k=0;
  Lk=0.0;
  Mk=0.0;
  deltaL=1.0;
  deltaM=1.0;
	
  while( (PetscAbsScalar(deltaL) > PetscAbsScalar(tolLanczos)) ||
         (PetscAbsScalar(deltaM) > PetscAbsScalar(tolLanczos)) )
    {
      MatMult(pSddft->HamiltonianOpr,Vk,Vkp1);
      VecDot(Vk,Vkp1,&a[k+1]);
      VecAXPY(Vkp1,-a[k+1],Vk); // Vkp1 = Vkp1 - ak+1 Vk
      VecAXPY(Vkp1,-b[k],Vkm1); // Vkp1 = Vkp1 -bk Vkm1
      VecCopy(Vk,Vkm1); // Vkm1=Vk
      VecNorm(Vkp1,NORM_2,&b[k+1]);
		
      VecCopy(Vkp1,Vk);
      VecScale(Vk,1.0/b[k+1]); // Vk=Vkp1/b[k+1]
		
      /*
       * Call function to find eigenvalue of Tridiagonal matrix here minimum eigenvalue is 
       * Lkp1, maximum eigenvalue is Mkp1
       */	
      TridiagEigenSolve(a,b,k+2,&Lkp1,&Mkp1);		
      deltaL = PetscAbsScalar(Lkp1-Lk);
      deltaM = PetscAbsScalar(Mkp1-Mk);
      Lk=Lkp1;
      Mk=Mkp1;
      k++;		
    }
  *EigenMin = Lkp1;
  *EigenMax = Mkp1;
	
  PetscFree(a);
  PetscFree(b);
  VecDestroy(&Vk); 
  VecDestroy(&Vkm1);
  VecDestroy(&Vkp1);

  return;
}
//////////////////////////////////////////////////////////////////////////////////////////////
//   kPointLanczos:  Lanczos algorithm for calculating the minimum and maximum              //
//                                eigenvalues of k-point Hamiltonian                        //
//        see W.H. Press, Numerical recepies 3rd edition: The art of scientific computing,  //
//                                 Cambridge university press, 2007                         //
/////////////////////////////////////////////////////////////////////////////////////////////  
void kPointLanczos(SDDFT_OBJ* pSddft,PetscScalar* EigenMin,PetscScalar* EigenMax)
{

  PetscScalar tolLanczos=pSddft->LANCZOSTOL;
  PetscScalar rVnorm,iVnorm;
  PetscInt Nx,Ny,Nz;
  PetscScalar Lk,Mk,Lkp1,Mkp1,deltaL,deltaM;
  PetscScalar temp;
  int k;
	
  Nx = pSddft->numPoints_x;
  Ny = pSddft->numPoints_y;
  Nz = pSddft->numPoints_z;
	
  PetscScalar *a,*b;
  PetscMalloc(sizeof(PetscScalar)*(Nx*Ny*Nz),&a);
  PetscMalloc(sizeof(PetscScalar)*(Nx*Ny*Nz),&b);

  /*
   * real
   */
  Vec rVk;
  Vec rVkm1;
  Vec rVkp1;

  /*
   * imaginary
   */
  Vec iVk;
  Vec iVkm1;
  Vec iVkp1;
  Vec tempVec;

  VecDuplicate(pSddft->elecDensRho,&rVk);
  VecDuplicate(pSddft->elecDensRho,&rVkm1);
  VecDuplicate(pSddft->elecDensRho,&rVkp1);

  VecDuplicate(pSddft->elecDensRho,&iVk);
  VecDuplicate(pSddft->elecDensRho,&iVkm1);
  VecDuplicate(pSddft->elecDensRho,&iVkp1);

  VecDuplicate(pSddft->elecDensRho,&tempVec);
	
  VecSet(rVkm1,1.0);
  VecSet(iVkm1,1.0);

  VecNorm(rVkm1,NORM_2,&rVnorm);
  VecNorm(iVkm1,NORM_2,&iVnorm);
  VecScale(rVkm1,1.0/(sqrt(rVnorm*rVnorm + iVnorm*iVnorm)));
  VecScale(iVkm1,1.0/(sqrt(rVnorm*rVnorm + iVnorm*iVnorm)));
	
  MatMult(pSddft->HamiltonianOpr1,rVkm1,rVk);
  MatMult(pSddft->HamiltonianOpr2,iVkm1,tempVec);
  VecAXPY(rVk,-1.0,tempVec);

  MatMult(pSddft->HamiltonianOpr2,rVkm1,iVk);
  MatMult(pSddft->HamiltonianOpr1,iVkm1,tempVec);
  VecAXPY(iVk,1.0,tempVec); 
	
  VecDot(rVkm1,rVk,&a[0]);
  VecDot(iVkm1,iVk,&temp);
  a[0]=a[0]+temp;

  VecAXPY(rVk,-a[0],rVkm1);
  VecAXPY(iVk,-a[0],iVkm1);

  VecNorm(rVk,NORM_2,&b[0]);
  VecNorm(iVk,NORM_2,&temp);
  b[0] = sqrt(b[0]*b[0]+ temp*temp);

  VecScale(rVk,1.0/b[0]);
  VecScale(iVk,1.0/b[0]);
	
  k=0;
  Lk=0.0;
  Mk=0.0;
  deltaL=1.0;
  deltaM=1.0;
		
  while( (PetscAbsScalar(deltaL) > PetscAbsScalar(tolLanczos)) ||
         (PetscAbsScalar(deltaM) > PetscAbsScalar(tolLanczos)) )
    {
      MatMult(pSddft->HamiltonianOpr1,rVk,rVkp1);
      MatMult(pSddft->HamiltonianOpr2,iVk,tempVec);
      VecAXPY(rVkp1,-1.0,tempVec); 
		
      MatMult(pSddft->HamiltonianOpr2,rVk,iVkp1);
      MatMult(pSddft->HamiltonianOpr1,iVk,tempVec);
      VecAXPY(iVkp1,1.0,tempVec); 

      VecDot(rVk,rVkp1,&a[k+1]);
      VecDot(iVk,iVkp1,&temp);
      a[k+1]=a[k+1]+temp;

      VecAXPY(rVkp1,-a[k+1],rVk); // Vkp1 = Vkp1 - ak+1 Vk
      VecAXPY(iVkp1,-a[k+1],iVk); // Vkp1 = Vkp1 - ak+1 Vk

      VecAXPY(rVkp1,-b[k],rVkm1); // Vkp1 = Vkp1 -bk Vkm1
      VecAXPY(iVkp1,-b[k],iVkm1); // Vkp1 = Vkp1 -bk Vkm1

      VecCopy(rVk,rVkm1); // Vkm1=Vk
      VecCopy(iVk,iVkm1); // Vkm1=Vk

      VecNorm(rVkp1,NORM_2,&b[k+1]);
      VecNorm(iVkp1,NORM_2,&temp);
      b[k+1] = sqrt(b[k+1]*b[k+1]+ temp*temp);		
		
      VecCopy(rVkp1,rVk);
      VecCopy(iVkp1,iVk);

      VecScale(rVk,1.0/b[k+1]); // Vk=Vkp1/b[k+1]
      VecScale(iVk,1.0/b[k+1]); // Vk=Vkp1/b[k+1]
		
      /*
       * Call function to find eigenvalue of Tridiagonal matrix here minimum eigenvalue is 
       * Lkp1, maximum eigenvalue is Mkp1
       */
      TridiagEigenSolve(a,b,k+2,&Lkp1,&Mkp1);
		
      deltaL = PetscAbsScalar(Lkp1-Lk);
      deltaM = PetscAbsScalar(Mkp1-Mk);
      Lk=Lkp1;
      Mk=Mkp1;
      k++;	
    }
  *EigenMin = Lkp1;
  *EigenMax = Mkp1;

  PetscFree(a);
  PetscFree(b);
  VecDestroy(&rVk); 
  VecDestroy(&rVkm1);
  VecDestroy(&rVkp1);

  VecDestroy(&iVk); 
  VecDestroy(&iVkm1);
  VecDestroy(&iVkp1);
  VecDestroy(&tempVec);	

  return;
}


///////////////////////////////////////////////////////////////////////////////////////////////
//       TridiagEigenSolve: Tridiagonal eigen solver for calculating the eigenvalues of      //
//                                         tridiagonal matrix.                               //
//       see W.H. Press, Numerical recepies 3rd edition: The art of scientific computing,    //
//                                  Cambridge university press, 2007                         //
///////////////////////////////////////////////////////////////////////////////////////////////
void TridiagEigenSolve(
  PetscScalar diag[],
  PetscScalar subdiag[],
  int n, PetscScalar* EigenMin, PetscScalar* EigenMax)
{
          
       
  int m,l,iter,i;
  PetscScalar s,r,p,g,f,dd,c,b;
              
 
  PetscScalar *d, *e; // d has diagonal and e has subdiagonal
  PetscMalloc(sizeof(PetscScalar)*n, &d);
  PetscMalloc(sizeof(PetscScalar)*n, &e);
  /*
   * create copy of diag and subdiag in d and e
   */
  for(i=0;i<n;i++)
    {
      d[i] = diag[i];
      e[i] = subdiag[i];
    }
            
  /*
   * e has the subdiagonal elements, ignore last element(n-1) of e by making it zero
   */
  e[n-1] = 0.0;         
  for(l=0;l<=n-1;l++)
    {
      iter=0;
      do{
	for(m=l;m<=n-2;m++)
	  {              
	    dd = PetscAbsScalar(d[m]) + PetscAbsScalar(d[m+1]);
	    if((PetscScalar)(PetscAbsScalar(e[m])+dd) == dd) break;
	  }
	if(m != l) {
	  if(iter++ == 50) {  PetscPrintf(PETSC_COMM_SELF,"Too many iterations in Tridiagonal solver\n");exit(1);}
	  g = (d[l+1]-d[l])/(2.0*e[l]);
	  r = sqrt(g*g+1.0); 
	  g = d[m] - d[l] + e[l]/( g + SIGN(r, PetscAbsScalar(g)) );
	  s = 1.0;
    c = 1.0;
	  p = 0.0;
                 
	  for(i=m-1;i>=l;i--) {
	    f=s*e[i];
	    b=c*e[i];
	    e[i+1] = (r=sqrt(g*g+f*f));
	    if (r == 0.0) {
	      d[i+1] -=p;
	      e[m]=0.0;
	      break;
	    }
	    s=f/r;
	    c=g/r;
	    g=d[i+1]-p;
	    r=(d[i]-g)*s +2.0*c*b;
	    d[i+1]=g+(p=s*r);
	    g=c*r-b;
                                                
	  }   
	  if(r==0.0 && i>=l) continue;
	  d[l] -= p;
	  e[l]=g;
	  e[m]=0.0;
	}
      }while(m!=l);
    }        
        
  /*
   * go over the array d to find the smallest and largest eigenvalue
   */
  *EigenMin = d[0];
  *EigenMax = d[0];
        
  for(i=1;i<n;i++)
    {
      if( PetscRealPart(d[i]) > PetscRealPart(*EigenMax) )
	{
	  *EigenMax = d[i];
	}
      else if( PetscRealPart(d[i]) < PetscRealPart(*EigenMin) )
	{
	  *EigenMin = d[i];
	}          
    }                  
    
  PetscFree(d);
  PetscFree(e);
                
  return;
}
