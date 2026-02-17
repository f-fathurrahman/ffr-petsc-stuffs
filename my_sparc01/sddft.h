/*=============================================================================================
  | Simulation Package for Ab-initio Real-space Calculations (SPARC) 
  | Copyright (C) 2016 Material Physics & Mechanics Group at Georgia Tech.
  |
  | S. Ghosh, P. Suryanarayana, SPARC: Accurate and efficient finite-difference formulation and
  | parallel implementation of Density Functional Theory. Part I: Isolated clusters, Computer
  | Physics Communications
  |
  | file name: sddft.h          
  |
  | Description: This file contains the function declarations for SPARC
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana
  |
  | Last Modified: 2/9/2016   
  |-------------------------------------------------------------------------------------------*/

#ifndef _SDDFT_
#define _SDDFT_

#include "petsc.h"
#include "petscksp.h"
#include "isddft.h"
#include "assert.h"

#undef _DEBUG 
#define KSP_TYPE KSPGMRES
#define MAX_TABLE_SIZE 11000
#define Z_ACC 0.001
#define ITMAXBRENTS 100
#define EPSILON 1e-16

void SddftObjInitialize(SDDFT_OBJ* pSddft);
void Read_parameters(SDDFT_OBJ* pSddft);
PetscScalar fract(PetscInt n,PetscInt k);
void Read_ion(SDDFT_OBJ* pSddft);
void Read_relax(SDDFT_OBJ* pSddft);
void Read_pseudopotential(SDDFT_OBJ* pSddft);
PetscErrorCode Objects_Create(SDDFT_OBJ* pSddft);
PetscErrorCode Laplace_matInit(SDDFT_OBJ* pSddft);
PetscErrorCode Gradient_matInit(SDDFT_OBJ* pSddft);

void ispline_gen(PetscScalar *X1,PetscScalar *Y1,int len1,PetscScalar *X2,PetscScalar *Y2,PetscScalar *DY2,int len2,PetscScalar *YD);
void getYD_gen(PetscScalar *X, PetscScalar *Y, PetscScalar *YD,int len);
void tridiag_gen(PetscScalar *A,PetscScalar *B,PetscScalar *C,PetscScalar *D,int len);
void SuperpositionAtomicCharge_VecInit(SDDFT_OBJ* pSddft);

void ChargDensB_cutoff(SDDFT_OBJ* pSddft); 
void ChargDensB_VecInit(SDDFT_OBJ* pSddft);
PetscErrorCode LaplacianNonlocalPseudopotential_MatInit(SDDFT_OBJ* pSddft);
PetscScalar SphericalHarmonic(PetscScalar x,PetscScalar y,PetscScalar z,int l,int m,PetscScalar rc);
PetscScalar SphericalHarmonic_Derivatives(PetscScalar x,PetscScalar y,PetscScalar z,int l,int m,PetscScalar rc, PetscScalar *Dx,PetscScalar *Dy,PetscScalar *Dz);

void EstimateNonZerosNonlocalPseudopot(SDDFT_OBJ* pSddft);
void Objects_Destroy(SDDFT_OBJ* pOfdft);
PetscErrorCode Wavefunctions_MatInit(SDDFT_OBJ* pSddft);

void ChebyshevFiltering(SDDFT_OBJ* pSddft,int m,PetscScalar a,PetscScalar b,PetscScalar a0);
void Lanczos(SDDFT_OBJ* pSddft,PetscScalar* EigenMin,PetscScalar* EigenMax);
void TridiagEigenSolve(PetscScalar diag[],PetscScalar subdiag[],int n,PetscScalar* EigenMin,PetscScalar* EigenMax);

PetscScalar constraint(SDDFT_OBJ* pSddft,PetscScalar lambdaf);
PetscScalar smearing_FermiDirac(PetscScalar bet, PetscScalar lambda, PetscScalar lambdaf);

PetscErrorCode CalculateDensity(SDDFT_OBJ* pSddft,Mat* Psi);
PetscErrorCode CalculateDensity1(SDDFT_OBJ* pSddft,Mat* Psi);
 
PetscScalar findRootBrent(SDDFT_OBJ* pSddft,PetscScalar x1,PetscScalar x2,PetscScalar tol);
PetscErrorCode RotatePsi(SDDFT_OBJ* pSddft, Mat* Psi, Mat* Q, Mat* PsiQ);


PetscErrorCode Vxc_Calc_CA(SDDFT_OBJ* pSddft); 
PetscErrorCode Exc_Calc_CA(SDDFT_OBJ* pSddft); 
PetscErrorCode Vxc_Calc_CA_PZ(SDDFT_OBJ* pSddft);
PetscErrorCode Exc_Calc_CA_PZ(SDDFT_OBJ* pSddft);
PetscErrorCode Vxc_Calc_CA_PW(SDDFT_OBJ* pSddft);
PetscErrorCode Exc_Calc_CA_PW(SDDFT_OBJ* pSddft);

PetscErrorCode Vxc_Calc(SDDFT_OBJ* pSddft);
PetscErrorCode Exc_Calc(SDDFT_OBJ* pSddft);

PetscErrorCode SolvePoisson(SDDFT_OBJ* pSddft);

PetscScalar mix(SDDFT_OBJ* pSddft, PetscInt its, Vec fk); 
void SystemEnergy_Calc(SDDFT_OBJ* pSddft);
void Display_Energy(SDDFT_OBJ* pSddft);

void ChargDensB_TM_VecInit(SDDFT_OBJ* pSddft);
PetscScalar PseudopotReference(PetscScalar r,PetscScalar rcut,PetscScalar Znucl);
void CorrectionEnergy_Calc(SDDFT_OBJ* pSddft);

PetscErrorCode ProjectMatrices(SDDFT_OBJ* pSddft, Mat* Psi,Mat *Hsub,Mat* Msub); 

PetscScalar SelfConsistentField(SDDFT_OBJ* pSddft);
void Calculate_force(SDDFT_OBJ* pSddft);
PetscErrorCode Force_Nonlocal(SDDFT_OBJ* pSddft,Mat* Psi);
void Calculate_forceCorrection(SDDFT_OBJ* pSddft);
void Display_force(SDDFT_OBJ* pSddft);
void  Symmetrysize_force(SDDFT_OBJ* pSddft);

PetscErrorCode FormFunction_relaxAtoms(SDDFT_OBJ* pSddft);
void NLCG_relaxAtoms(SDDFT_OBJ* pSddft);
void Display_Atompos(SDDFT_OBJ* pSddft);
void Display_Relax(SDDFT_OBJ* pSddft);

PetscErrorCode SDDFT_Nonperiodic(SDDFT_OBJ* pSddft);
void Set_VecZero(SDDFT_OBJ* pSddft);
PetscErrorCode Wavefunctions_MatMatMultSymbolic(SDDFT_OBJ* pSddft); 

PetscErrorCode MultipoleExpansion_Phi(SDDFT_OBJ* pSddft,Vec* RhopBvec);
PetscErrorCode CalculateDipoleMoment(SDDFT_OBJ* pSddft);

#endif
