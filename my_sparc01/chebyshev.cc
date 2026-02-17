/*=============================================================================================
  | Description: This file contains the functions required for Chebyshev filtering and lanczos
  | algorithm for finding the maximum and minimum eigenvalues
  |-------------------------------------------------------------------------------------------*/
#include "sddft.h"
#include "petscsys.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SIGN(a, b) ((b) >= 0.0 ? PetscAbsScalar(a) : -PetscAbsScalar(a))

///////////////////////////////////////////////////////////////////////////////////////////////
//                         ChebyshevFiltering: Chebychev filter function                     //
///////////////////////////////////////////////////////////////////////////////////////////////
void ChebyshevFiltering(SDDFT_OBJ *pSddft, int m, PetscScalar a, PetscScalar b, PetscScalar a0) {

  /*
   * a0 is minimum eigenvalue
   * a is cutoff eigenvalue
   * b is maximum eigenvalue
   * m is degree of Chebyshev polynomial
   */

  PetscScalar e, c, sigma, sigma1, sigma2, gamma;
  int i;

  e = (b - a) / 2.0;
  c = (b + a) / 2.0;
  sigma = e / (a0 - c);
  sigma1 = sigma;
  gamma = 2.0 / sigma1;

  /*
   * shift the Hamiltonian by constant
   */

  MatShift(pSddft->HamiltonianOpr, -c);

  //MatMatMultNumeric(pSddft->HamiltonianOpr, pSddft->XOrb, pSddft->YOrb);
  MatProductCreate(pSddft->HamiltonianOpr, pSddft->XOrb, NULL, &pSddft->YOrb);
  MatProductSetType(pSddft->YOrb, MATPRODUCT_AB);
  MatProductSetFromOptions(pSddft->YOrb);
  MatProductSymbolic(pSddft->YOrb);
  MatProductNumeric(pSddft->YOrb);

  MatScale(pSddft->YOrb, sigma1 / e);

  for (i = 2; i <= m; i++) {
    sigma2 = 1.0 / (gamma - sigma);

    //MatMatMultNumeric(pSddft->HamiltonianOpr, pSddft->YOrb, pSddft->YOrbNew);
    //MatProductCreateWithMat(pSddft->HamiltonianOpr, pSddft->YOrb, NULL, pSddft->YOrbNew);

    MatProductCreate(pSddft->HamiltonianOpr, pSddft->YOrb, NULL, &pSddft->YOrbNew);
    MatProductSetType(pSddft->YOrbNew, MATPRODUCT_AB);
    MatProductSetFromOptions(pSddft->YOrbNew);
    MatProductSymbolic(pSddft->YOrbNew);
    MatProductNumeric(pSddft->YOrbNew);

    MatScale(pSddft->YOrbNew, 2 * sigma2 / e);
    MatAXPY(pSddft->YOrbNew, -sigma * sigma2, pSddft->XOrb, SAME_NONZERO_PATTERN);

    MatCopy(pSddft->YOrb, pSddft->XOrb, SAME_NONZERO_PATTERN);
    MatCopy(pSddft->YOrbNew, pSddft->YOrb, SAME_NONZERO_PATTERN);
    sigma = sigma2;
  }

  /*
   * shift back the Hamiltonian by subtracting the constant previously added
   */

  MatShift(pSddft->HamiltonianOpr, c);

  pSddft->ChebyshevCallCounter++;
  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//         Lanczos:  Lanczos algorithm for calculating the minimum and maximum              //
//                                  eigenvalues of Hamiltonian                              //
//        see W.H. Press, Numerical recepies 3rd edition: The art of scientific computing,  //
//                                 Cambridge university press, 2007                         //
/////////////////////////////////////////////////////////////////////////////////////////////
void Lanczos(SDDFT_OBJ *pSddft, PetscScalar *EigenMin, PetscScalar *EigenMax) {

  PetscScalar tolLanczos = pSddft->LANCZOSTOL;
  PetscScalar Vnorm;
  PetscInt Nx, Ny, Nz;
  PetscScalar Lk, Mk, Lkp1, Mkp1, deltaL, deltaM;

  int k;

  Nx = pSddft->numPoints_x;
  Ny = pSddft->numPoints_y;
  Nz = pSddft->numPoints_z;

  PetscScalar *a, *b;
  PetscMalloc(sizeof(PetscScalar) * (Nx * Ny * Nz), &a);
  PetscMalloc(sizeof(PetscScalar) * (Nx * Ny * Nz), &b);

  Vec Vk;
  Vec Vkm1;
  Vec Vkp1;

  VecDuplicate(pSddft->elecDensRho, &Vk);
  VecDuplicate(pSddft->elecDensRho, &Vkm1);
  VecDuplicate(pSddft->elecDensRho, &Vkp1);

  VecSet(Vkm1, 1.0);
  VecNorm(Vkm1, NORM_2, &Vnorm);
  VecScale(Vkm1, 1.0 / Vnorm);
  MatMult(pSddft->HamiltonianOpr, Vkm1, Vk);

  VecDot(Vkm1, Vk, &a[0]);
  VecAXPY(Vk, -a[0], Vkm1);
  VecNorm(Vk, NORM_2, &b[0]);
  VecScale(Vk, 1.0 / b[0]);

  k = 0;
  Lk = 0.0;
  Mk = 0.0;
  deltaL = 1.0;
  deltaM = 1.0;

  while (deltaL > tolLanczos || deltaM > tolLanczos) {
    MatMult(pSddft->HamiltonianOpr, Vk, Vkp1);
    VecDot(Vk, Vkp1, &a[k + 1]);
    VecAXPY(Vkp1, -a[k + 1], Vk); // Vkp1 = Vkp1 - ak+1 Vk
    VecAXPY(Vkp1, -b[k], Vkm1);   // Vkp1 = Vkp1 -bk Vkm1
    VecCopy(Vk, Vkm1);            // Vkm1=Vk
    VecNorm(Vkp1, NORM_2, &b[k + 1]);

    VecCopy(Vkp1, Vk);
    VecScale(Vk, 1.0 / b[k + 1]); // Vk=Vkp1/b[k+1]

    /*
     * Call function to find eigenvalue of Tridiagonal matrix here minimum eigenvalue is
     * Lkp1, maximum eigenvalue is Mkp1
     */
    TridiagEigenSolve(a, b, k + 2, &Lkp1, &Mkp1);
    deltaL = PetscAbsScalar(Lkp1 - Lk);
    deltaM = PetscAbsScalar(Mkp1 - Mk);
    Lk = Lkp1;
    Mk = Mkp1;
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

///////////////////////////////////////////////////////////////////////////////////////////////
//       TridiagEigenSolve: Tridiagonal eigen solver for calculating the eigenvalues of      //
//                                         tridiagonal matrix.                               //
//       see W.H. Press, Numerical recepies 3rd edition: The art of scientific computing,    //
//                                  Cambridge university press, 2007                         //
///////////////////////////////////////////////////////////////////////////////////////////////
void TridiagEigenSolve(PetscScalar diag[], PetscScalar subdiag[], int n, PetscScalar *EigenMin,
                       PetscScalar *EigenMax) {

  int m, l, iter, i;
  PetscScalar s, r, p, g, f, dd, c, b;

  PetscScalar *d, *e; // d has diagonal and e has subdiagonal
  PetscMalloc(sizeof(PetscScalar) * n, &d);
  PetscMalloc(sizeof(PetscScalar) * n, &e);
  /*
   * create copy of diag and subdiag in d and e
   */
  for (i = 0; i < n; i++) {
    d[i] = diag[i];
    e[i] = subdiag[i];
  }

  /*
   * e has the subdiagonal elements, ignore last element(n-1) of e by making it zero
   */
  e[n - 1] = 0.0;
  for (l = 0; l <= n - 1; l++) {
    iter = 0;
    do {
      for (m = l; m <= n - 2; m++) {
        dd = PetscAbsScalar(d[m]) + PetscAbsScalar(d[m + 1]);
        if ((PetscScalar)(PetscAbsScalar(e[m]) + dd) == dd)
          break;
      }
      if (m != l) {
        if (iter++ == 50) {
          PetscPrintf(PETSC_COMM_SELF, "Too many iterations in Tridiagonal solver\n");
          exit(1);
        }
        g = (d[l + 1] - d[l]) / (2.0 * e[l]);
        r = sqrt(g * g + 1.0);
        g = d[m] - d[l] + e[l] / (g + SIGN(r, g));
        s = c = 1.0;
        p = 0.0;

        for (i = m - 1; i >= l; i--) {
          f = s * e[i];
          b = c * e[i];
          e[i + 1] = (r = sqrt(g * g + f * f));
          if (r == 0.0) {
            d[i + 1] -= p;
            e[m] = 0.0;
            break;
          }
          s = f / r;
          c = g / r;
          g = d[i + 1] - p;
          r = (d[i] - g) * s + 2.0 * c * b;
          d[i + 1] = g + (p = s * r);
          g = c * r - b;
        }
        if (r == 0.0 && i >= l)
          continue;
        d[l] -= p;
        e[l] = g;
        e[m] = 0.0;
      }
    } while (m != l);
  }

  /*
   * go over the array d to find the smallest and largest eigenvalue
   */
  *EigenMin = d[0];
  *EigenMax = d[0];

  for (i = 1; i < n; i++) {
    if (d[i] > *EigenMax) {
      *EigenMax = d[i];
    } else if (d[i] < *EigenMin) {
      *EigenMin = d[i];
    }
  }

  PetscFree(d);
  PetscFree(e);

  return;
}
