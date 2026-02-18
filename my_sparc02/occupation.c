
#include "math.h"
#include "petscsys.h"
#include "sddft.h"

#include "cblas.h"
//#include "mkl_lapacke.h"
//#include "mkl.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SIGN(a, b) ((b) >= 0.0 ? PetscAbsScalar(a) : -PetscAbsScalar(a))
///////////////////////////////////////////////////////////////////////////////////////////////
//                          constraint: constraint on occupation //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar constraint(SDDFT_OBJ *pSddft, PetscScalar lambdaf) {
  PetscInt k, N;
  PetscScalar g = 0;
  PetscScalar ans, Ne;

  N = pSddft->Nstates;
  Ne = 1.0 * pSddft->elecN;
  for (k = 0; k < N; k++) {
    g += 2.0 * smearing_FermiDirac(pSddft->Beta, pSddft->lambda[k], lambdaf);
  }
  ans = g - Ne;
  return ans;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                    kPointconstraint: constraint on occupation //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar kPointconstraint(SDDFT_OBJ *pSddft, PetscScalar lambdaf) {
  PetscInt k, n, Ns, Nk;
  PetscScalar g = 0;
  PetscScalar ans, Ne;

  Ns = pSddft->Nstates;
  Nk = pSddft->Nkpts_sym;
  Ne = 1.0 * pSddft->elecN;
  for (k = 0; k < Nk; k++)
    for (n = 0; n < Ns; n++) {
      g += 2.0 * pSddft->kptWts[k] *
           smearing_FermiDirac(pSddft->Beta, pSddft->lambdakpt[k][n], lambdaf);
    }
  ans = (1.0 / pSddft->Nkpts) * g - Ne;
  return ans;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                          smearing_FermiDirac: Fermi Dirac function //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar smearing_FermiDirac(PetscScalar bet, PetscScalar lambda,
                                PetscScalar lambdaf) {
  PetscScalar g;
  g = 1.0 / (1.0 + exp(bet * (lambda - lambdaf)));
  return g;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                CalculateDensity: Calculates electron density from
//                wavefunctions           //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode CalculateDensity(SDDFT_OBJ *pSddft, Mat *Psi) {

  /*
   * the orbitals (Psi) is a parallel (row major) dense matrix storing the
   * subspace rotated orbitals after chebyshev filtering
   */

  PetscInt Nstates = pSddft->Nstates;
  PetscScalar elecno, rhosum;
  PetscInt rowStart, rowEnd;
  int i, j, rank, ierr;
  IS irow, icol;
  PetscScalar gi;
  PetscScalar *arrPsiSeq;
  PetscScalar denVal;

  MatGetOwnershipRange(*Psi, &rowStart, &rowEnd);
  ISCreateStride(MPI_COMM_SELF, rowEnd - rowStart, rowStart, 1, &irow);
  ISCreateStride(MPI_COMM_SELF, Nstates, 0, 1, &icol);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  MatDenseGetArray(*Psi, &arrPsiSeq);

  /*
   * loop over local rows in each processor
   */

  for (i = rowStart; i < rowEnd; i++) {
    denVal = 0.0;
    /*
     * loop over all the states
     */
    for (j = 0; j < Nstates; j++) {
      gi = 2.0 * smearing_FermiDirac(pSddft->Beta, pSddft->lambda[j],
                                     pSddft->lambda_f);
      denVal =
          denVal + gi * (arrPsiSeq[i - rowStart + (rowEnd - rowStart) * j]) *
                       (arrPsiSeq[i - rowStart + (rowEnd - rowStart) * j]);
    }
    VecSetValue(pSddft->elecDensRho, i, denVal, INSERT_VALUES);
  }

  VecAssemblyBegin(pSddft->elecDensRho);
  VecAssemblyEnd(pSddft->elecDensRho);
  MatDenseRestoreArray(*Psi, &arrPsiSeq);
  /*
   * scaling of electron density
   */
  VecSum(pSddft->chrgDensB, &elecno);
  VecSum(pSddft->elecDensRho, &rhosum);
  rhosum = -elecno / rhosum;
  VecScale(pSddft->elecDensRho, rhosum);

  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
// kPointCalculateDensity: Calculates electron density from wavefunctions (k
// point sampling) //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode kPointCalculateDensity(SDDFT_OBJ *pSddft, Mat *U1, Mat *U2,
                                      int k) {
  /*
   * the (U1,U2) are parallel (row major) dense matrix storing the subspace
   * rotated periodic part of orbitals after chebyshev filtering. U1 stores real
   * and U2 imaginary
   */
  PetscInt Nstates = pSddft->Nstates;

  PetscScalar elecno, rhosum;
  PetscInt rowStart, rowEnd;
  int i, j, rank, ierr;
  IS irow, icol;
  PetscScalar gi;
  PetscScalar *arrUSeq1, *arrUSeq2;
  PetscScalar denVal;

  MatGetOwnershipRange(*U1, &rowStart, &rowEnd);
  ISCreateStride(MPI_COMM_SELF, rowEnd - rowStart, rowStart, 1, &irow);
  ISCreateStride(MPI_COMM_SELF, Nstates, 0, 1, &icol);

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  MatDenseGetArray(*U1, &arrUSeq1);
  MatDenseGetArray(*U2, &arrUSeq2);
  /*
   * loop over local rows in each processor
   */
  for (i = rowStart; i < rowEnd; i++) {
    denVal = 0.0;
    /*
     * loop over all the states
     */
    for (j = 0; j < Nstates; j++) {
      gi = (2.0 / pSddft->Nkpts) * smearing_FermiDirac(pSddft->Beta,
                                                       pSddft->lambdakpt[k][j],
                                                       pSddft->lambda_f);
      denVal = denVal +
               pSddft->kptWts[k] *
                   (gi * (arrUSeq1[i - rowStart + (rowEnd - rowStart) * j]) *
                        (arrUSeq1[i - rowStart + (rowEnd - rowStart) * j]) +
                    gi * (arrUSeq2[i - rowStart + (rowEnd - rowStart) * j]) *
                        (arrUSeq2[i - rowStart + (rowEnd - rowStart) * j]));
    }
    VecSetValue(pSddft->elecDensRho, i, denVal, ADD_VALUES);
  }

  VecAssemblyBegin(pSddft->elecDensRho);
  VecAssemblyEnd(pSddft->elecDensRho);
  MatDenseRestoreArray(*U1, &arrUSeq1);
  MatDenseRestoreArray(*U2, &arrUSeq2);
  /*
   * electron density is scaled appropriately here such that:
   * 1)integral of electron density = total valence electronic charge
   * 2)integral of (electron density + pseudocharge density) = 0
   */
  if (k + 1 == pSddft->Nkpts_sym) {
    VecSum(pSddft->chrgDensB, &elecno);
    VecSum(pSddft->elecDensRho, &rhosum);
    rhosum = -elecno / rhosum;
    VecScale(pSddft->elecDensRho, rhosum);
  }
  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                          findRootBrent: Brents method for finding root //
//    See: W.H. Press, Numerical recepies 3rd edition: The art of scientific
//    computing,      //
//                                Cambridge university press, 2007 //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar findRootBrent(SDDFT_OBJ *pSddft, PetscScalar x1, PetscScalar x2,
                          PetscScalar tol) {
  /*
   * this function finds the root of the function "constraint"
   */
  int iter;
  PetscScalar tol1q, eq;
  PetscScalar a = x1, b = x2, c = x2, d, e, min1, min2;

  PetscScalar fa = constraint(pSddft, a), fb = constraint(pSddft, b), fc, p, q,
              r, s, tol1, xm;

  if ( (PetscRealPart(fa) > 0.0 && PetscRealPart(fb) > 0.0) ||
       (PetscRealPart(fa) < 0.0 && PetscRealPart(fb) < 0.0)) {
    printf("Root must be bracketed in Brent's method\n");
    exit(1);
  }
  fc = fb;
  for (iter = 1; iter <= ITMAXBRENTS; iter++) {

    if ((PetscRealPart(fb) > 0.0 && PetscRealPart(fc) > 0.0) ||
        (PetscRealPart(fb) < 0.0 && PetscRealPart(fc) < 0.0)) {
      c = a;
      fc = fa;
      e = d = b - a;
    }

    if (PetscAbsScalar((PetscScalar)fc) < PetscAbsScalar((PetscScalar)fb)) {
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }
    tol1 = 2.0 * EPSILON * PetscAbsScalar((PetscScalar)b) + 0.5 * tol;
    xm = 0.5 * (c - b);
    if (PetscAbsScalar((PetscScalar)xm) <= PetscRealPart(tol1) || PetscRealPart(fb) == 0.0)
      return b;

    if (PetscAbsScalar((PetscScalar)e) >= PetscRealPart(tol1) &&
        PetscAbsScalar((PetscScalar)fa) > PetscAbsScalar((PetscScalar)fb)) {
      /*
       * attempt inverse quadratic interpolation
       */
      s = fb / fa;
      if (a == c) {
        p = 2.0 * xm * s;
        q = 1.0 - s;
      } else {
        q = fa / fc;
        r = fb / fc;
        p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
        q = (q - 1.0) * (r - 1.0) * (s - 1.0);
      }
      if (PetscRealPart(p) > 0.0) {
        /*
         * check whether in bounds
         */
        q = -q;
      }
      p = PetscAbsScalar((PetscScalar)p);

      tol1q = tol1 * q;

      min1 = 3.0 * xm * q - PetscAbsScalar((PetscScalar)tol1q);
      eq = e * q;
      min2 = PetscAbsScalar((PetscScalar)eq);
      if (2.0 * PetscRealPart(p) < (
          PetscRealPart(min1) < PetscRealPart(min2) ?
          PetscRealPart(min1) : PetscRealPart(min2)
          )
      ) { // accept interpolation
        e = d;
        d = p / q;
      } else { /*
                * Bounds decreasing too slowly, use bisection
                */
        d = xm;
        e = d;
      }
    } else {
      d = xm;
      e = d;
    }
    /*
     * move last best guess to a
     */
    a = b;
    fa = fb;
    if (PetscAbsScalar((PetscScalar)d) > PetscRealPart(tol1)) {
      /*
       * evaluate new trial root
       */
      b += d;
    } else {
      b += SIGN(tol1, PetscRealPart(xm));
    }
    fb = constraint(pSddft, b);
  }
  printf(
      "Maximum iterations exceeded in brents root finding method...exiting\n");
  exit(1);
  return 0.0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//    kPointfindRootBrent: Brents method for finding root for k point sampling
//    // See: W.H. Press, Numerical recepies 3rd edition: The art of scientific
//    computing,      //
//                                Cambridge university press, 2007 //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscScalar kPointfindRootBrent(SDDFT_OBJ *pSddft, PetscScalar x1,
                                PetscScalar x2, PetscScalar tol) {
  /*
   * this function finds the root of the function "kPointconstraint"
   */
  int iter;
  PetscScalar tol1q, eq;
  PetscScalar a = x1, b = x2, c = x2, d, e, min1, min2;

  PetscScalar fa = kPointconstraint(pSddft, a),
              fb = kPointconstraint(pSddft, b), fc, p, q, r, s, tol1, xm;

  if ( ( PetscRealPart(fa) > 0.0 && PetscRealPart(fb) > 0.0) ||
       ( PetscRealPart(fa) < 0.0 && PetscRealPart(fb) < 0.0)
  ) {
    printf("Root must be bracketed in Brent's method\n");
    exit(1);
  }
  fc = fb;
  for (iter = 1; iter <= ITMAXBRENTS; iter++) {

    if ( (PetscRealPart(fb) > 0.0 && PetscRealPart(fc) > 0.0) ||
         (PetscRealPart(fb) < 0.0 && PetscRealPart(fc) < 0.0)) {
      c = a;
      fc = fa;
      e = d = b - a;
    }

    if (PetscAbsScalar((PetscScalar)fc) < PetscAbsScalar((PetscScalar)fb)) {
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }
    tol1 = 2.0 * EPSILON * PetscAbsScalar((PetscScalar)b) + 0.5 * tol;
    xm = 0.5 * (c - b);
    if (PetscAbsScalar((PetscScalar)xm) <= PetscRealPart(tol1) || PetscRealPart(fb) == 0.0)
      return b;

    if (PetscAbsScalar((PetscScalar)e) >= PetscRealPart(tol1) &&
        PetscAbsScalar((PetscScalar)fa) > PetscAbsScalar((PetscScalar)fb)) {
      /*
       * attempt inverse quadratic interpolation
       */
      s = fb / fa;
      if (a == c) {
        p = 2.0 * xm * s;
        q = 1.0 - s;
      } else {
        q = fa / fc;
        r = fb / fc;
        p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
        q = (q - 1.0) * (r - 1.0) * (s - 1.0);
      }
      if (PetscRealPart(p) > 0.0) {
        /*
         * check whether in bounds
         */
        q = -q;
      }
      p = PetscAbsScalar((PetscScalar)p);

      tol1q = tol1 * q;

      min1 = 3.0 * xm * q - PetscAbsScalar((PetscScalar)tol1q);
      eq = e * q;
      min2 = PetscAbsScalar((PetscScalar)eq);
      if (2.0 * PetscRealPart(p) < ( PetscRealPart(min1) < PetscRealPart(min2) ? PetscRealPart(min1) : PetscRealPart(min2) )) {
        /*
         * accept interpolation
         */
        e = d;
        d = p / q;
      } else {
        /*
         * Bounds decreasing too slowly, use bisection
         */
        d = xm;
        e = d;
      }
    } else {
      d = xm;
      e = d;
    }
    /*
     * move last best guess to a
     */
    a = b;
    fa = fb;
    if (PetscAbsScalar((PetscScalar)d) > PetscRealPart(tol1)) {
      /*
       * evaluate new trial root
       */
      b += d;
    } else {
      b += SIGN(tol1, PetscRealPart(xm));
    }
    fb = kPointconstraint(pSddft, b);
  }
  printf(
      "Maximum iterations exceeded in brents root finding method...exiting\n");
  exit(1);
  return 0.0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                          RotatePsi: subspace rotation of filtered
//                          wavefunctions           //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode RotatePsi(SDDFT_OBJ *pSddft, Mat *Psi, Mat *Q, Mat *PsiQ) {
  /*
   * Psi is the eigenvectors to be rotated. Parallel distributed dense matrix
   * (Npts x Nstates) PsiQ is the rotated eigenvectors. Parallel distributed
   * dense (Npts x Nstates) Q is the rotation matrix. Serial dense (Nstates x
   * Nstates)
   */

  PetscInt rowStart, rowEnd;
  IS irow, icol;
  double alpha, beta;
  PetscScalar *arrPsiSeq, *arrPsiQSeq, *arrQ;
  int i, rank;
  PetscInt *idxm, *idxn;
  PetscInt Nstates = pSddft->Nstates;
  int M, N, K;
  alpha = 1.0;
  beta = 0.0;

  MatGetOwnershipRange(*Psi, &rowStart, &rowEnd);
  ISCreateStride(MPI_COMM_SELF, rowEnd - rowStart, rowStart, 1, &irow);
  ISCreateStride(MPI_COMM_SELF, Nstates, 0, 1, &icol);

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  /*
   * get local array
   */
  MatDenseGetArray(*Psi, &arrPsiSeq);
  MatDenseGetArray(*Q, &arrQ);
  MatDenseGetArray(*PsiQ, &arrPsiQSeq);

  M = rowEnd - rowStart;
  N = Nstates;
  K = Nstates;
  /*
   * call BLAS function for matrix matrix multiplication
   */
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
              arrPsiSeq, M, arrQ, K, beta, arrPsiQSeq, M);
  /*
   * restore local arrays
   */
  MatDenseRestoreArray(*Psi, &arrPsiSeq);
  MatDenseRestoreArray(*Q, &arrQ);
  MatDenseRestoreArray(*PsiQ, &arrPsiQSeq);
  MatAssemblyBegin(*PsiQ, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*PsiQ, MAT_FINAL_ASSEMBLY);

  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////
//                    kPointRotatePsi: subspace rotation of filtered
//                    wavefunctions           //
///////////////////////////////////////////////////////////////////////////////////////////////
PetscErrorCode kPointRotatePsi(SDDFT_OBJ *pSddft, Mat *U1, Mat *U2, Mat *Q1,
                               Mat *Q2, Mat *UQ1, Mat *UQ2) {

  /*
   * U1, U2 are the periodic orbitals to be rotated. Parallel distributed dense
   * matrix (Npts x Nstates) UQ1, UQ2 are the rotated orbitals (real and
   * imaginary respectively). UQ1, UQ2 is parallel distributed dense (Npts x
   * Nstates) Q1, Q2 is the rotation matrix. Serial dense (Nstates x Nstates)
   */

  PetscInt rowStart, rowEnd;
  IS irow, icol;
  double alpha, beta;

  PetscScalar *arrUSeq1, *arrUQSeq1, *arrQ1;
  PetscScalar *arrUSeq2, *arrQ2;
  PetscScalar *arrTemp;
  Mat tempMat;
  int i, rank;
  PetscInt *idxm, *idxn;
  PetscInt Nstates = pSddft->Nstates;
  int M, N, K;
  alpha = 1.0;
  beta = 0.0;
  MatGetOwnershipRange(*U1, &rowStart, &rowEnd);

  ISCreateStride(MPI_COMM_SELF, rowEnd - rowStart, rowStart, 1, &irow);
  ISCreateStride(MPI_COMM_SELF, Nstates, 0, 1, &icol);

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  M = rowEnd - rowStart;
  N = Nstates;
  K = Nstates;

  /*
   * get local array
   */
  MatDenseGetArray(*U1, &arrUSeq1);
  MatDenseGetArray(*Q1, &arrQ1);
  MatDenseGetArray(*U2, &arrUSeq2);
  MatDenseGetArray(*Q2, &arrQ2);

  /*
   * calculation of real part
   */
  MatDenseGetArray(*UQ1, &arrUQSeq1);
  MatDenseGetArray(pSddft->YOrbNew1, &arrTemp);

  /*
   * call BLAS function for matrix matrix multiplication
   */
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
              arrUSeq1, M, arrQ1, K, beta, arrUQSeq1, M); // U1Q1
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
              arrUSeq2, M, arrQ2, K, beta, arrTemp, M); // U2Q2

  MatDenseRestoreArray(*UQ1, &arrUQSeq1);
  MatDenseRestoreArray(pSddft->YOrbNew1, &arrTemp);

  MatAssemblyBegin(*UQ1, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*UQ1, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(pSddft->YOrbNew1, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pSddft->YOrbNew1, MAT_FINAL_ASSEMBLY);

  MatAXPY(*UQ1, -1.0, pSddft->YOrbNew1,
          SAME_NONZERO_PATTERN); // stores the real part (UQ1 = U1Q1-U2Q2)

  /*
   * calculation of real part
   */
  MatDenseGetArray(*UQ2, &arrUQSeq1); // reuse arrUQSeq1
  MatDenseGetArray(pSddft->YOrbNew1, &arrTemp);

  /*
   * call BLAS function for matrix matrix multiplication
   */
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
              arrUSeq2, M, arrQ1, K, beta, arrUQSeq1, M); // U2Q1
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
              arrUSeq1, M, arrQ2, K, beta, arrTemp, M); // U1Q2

  MatDenseRestoreArray(*UQ2, &arrUQSeq1);
  MatDenseRestoreArray(pSddft->YOrbNew1, &arrTemp);

  MatAssemblyBegin(*UQ2, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*UQ2, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(pSddft->YOrbNew1, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(pSddft->YOrbNew1, MAT_FINAL_ASSEMBLY);

  MatAXPY(*UQ2, 1.0, pSddft->YOrbNew1,
          SAME_NONZERO_PATTERN); // stores the imaginary part (UQ2 = U2Q1+U1Q2)

  /*
   * restore local arrays
   */
  MatDenseRestoreArray(*U1, &arrUSeq1);
  MatDenseRestoreArray(*Q1, &arrQ1);
  MatDenseRestoreArray(*U2, &arrUSeq2);
  MatDenseRestoreArray(*Q2, &arrQ2);

  return 0;
}
