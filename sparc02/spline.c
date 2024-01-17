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
  | file name: spline.cc          
  |
  | Description: This file contains the functions required for spline interpolation
  | Reference: Cubic Spline Interpolation: A Review, George Wolberg, Technical Report,
  | Department of Computer Science, Columbia University
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana
  |
  | Last Modified: 1/26/2016   
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "petscsys.h"
////////////////////////////////////////////////////////////////////////////////////////////////
//                            ispline_gen: cubic spline interpolation                         //
////////////////////////////////////////////////////////////////////////////////////////////////
void ispline_gen(PetscScalar *X1, PetscScalar *Y1, int len1, PetscScalar *X2,
  PetscScalar *Y2,PetscScalar *DY2,int len2,PetscScalar *YD)
{
  int i,j;
  PetscScalar A0, A1, A2, A3, x, dx, dy, p1;
  PetscScalar p2, p3;
            
  if( PetscRealPart(X2[0]) < PetscRealPart(X1[0]) ||
      PetscRealPart(X2[len2-1]) > PetscRealPart(X1[len1-1]) ) // check range
    {
      printf("%lf, %lf, %lf, %lf\n",X1[0],X2[0],X1[len1-1],X2[len2-1]);
      printf("out of range in spline interpolation\n");
      exit(1);
    }
        
  /*
   * p1 is left endpoint of the interval
   * p2 is resampling position
   * p3 is right endpoint of interval
   * j is input index of current interval
   */
    
  p3=X2[0]-1;    
  for(i=j=0;i<len2;i++)
    {
      /*
       * check if in new interval
       */
      p2=X2[i];
      if( PetscRealPart(p2) > PetscRealPart(p3) ){
  /*
   * find interval which contains p2 
   */
  for(; j < len1 && PetscRealPart(p2) > PetscRealPart(X1[j]); j++);
  if( PetscRealPart(p2) < PetscRealPart(X1[j]) ) j--;
  p1=X1[j];
  p3=X1[j+1]; 

  /*
   * compute spline coefficients
   */
  dx = 1.0/(X1[j+1]-X1[j]);
  dy = (Y1[j+1]-Y1[j])*dx;
  A0 = Y1[j];
  A1 = YD[j];
  A2 = dx*(3.0*dy - 2.0*YD[j]-YD[j+1]);
  A3 = dx*dx*(-2.0*dy+YD[j] + YD[j+1]);  
      }
      /*
       * use Horner's rule to calculate cubic polynomial
       */
      x =  p2-p1;
      Y2[i] = ((A3*x +A2)*x + A1)*x + A0;        
      DY2[i] =(3.0*A3*x + 2.0*A2)*x + A1; // analytical derivative
    }
  
  return;  
}

////////////////////////////////////////////////////////////////////////////////////////////////
//                  getYD_gen: derivatives required for spline interpolation                  //
////////////////////////////////////////////////////////////////////////////////////////////////
void getYD_gen(PetscScalar *X, PetscScalar *Y, PetscScalar *YD,int len)
{
  int i;
  PetscScalar h0,h1,r0,r1,*A,*B,*C;
          
  PetscMalloc(sizeof(PetscScalar)*len,&A);
  PetscMalloc(sizeof(PetscScalar)*len,&B);
  PetscMalloc(sizeof(PetscScalar)*len,&C);
          
          
  h0 =  X[1]-X[0]; h1 = X[2]-X[1];
  r0 = (Y[1]-Y[0])/h0; r1=(Y[2]-Y[1])/h1;
  B[0] = h1*(h0+h1);
  C[0] = (h0+h1)*(h0+h1);
  YD[0] = r0*(3*h0*h1 + 2*h1*h1) + r1*h0*h0;
               
  for(i=1;i<len-1;i++) {
    h0 = X[i]-X[i-1]; h1=X[i+1]-X[i];
    r0 = (Y[i]-Y[i-1])/h0;  r1=(Y[i+1]-Y[i])/h1;
    A[i] = h1;
    B[i] = 2*(h0+h1);
    C[i] = h0;
    YD[i] = 3*(r0*h1 + r1*h0);
  }
           
  A[i] = (h0+h1)*(h0+h1);
  B[i] = h0*(h0+h1);
  YD[i] = r0*h1*h1 + r1*(3*h0*h1 + 2*h0*h0);
          
  tridiag_gen(A,B,C,YD,len);
 
  PetscFree(A);
  PetscFree(B);
  PetscFree(C);                                     

  return;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//                  tridiag_gen: Solves a tridiagonal system using Gauss Elimination          //
////////////////////////////////////////////////////////////////////////////////////////////////
void tridiag_gen(PetscScalar *A,PetscScalar *B,PetscScalar *C,PetscScalar *D,int len)
{
  int i;
  PetscScalar b, *F;
  PetscMalloc(sizeof(PetscScalar)*len,&F);

  /*
   * Gauss elimination; forward substitution
   */
  b = B[0];
  D[0] = D[0]/b;
  for(i=1;i<len;i++){
    F[i] = C[i-1]/b;
    b= B[i] - A[i]*F[i];
    if(b==0) {
      printf("Divide by zero in tridiag_gen\n"); 
      exit(1);
    }
    D[i] =(D[i] - D[i-1]*A[i])/b;
  }

  /*
   * backsubstitution 
   */
  for(i=len-2;i >= 0;i--)
    D[i] -= (D[i+1]*F[i+1]);
           
  PetscFree(F);
  return;
}



