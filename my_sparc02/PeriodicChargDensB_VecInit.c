
#include "petscsys.h"
#include "sddft.h"

#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// PeriodicChargDensB_VecInit: Pseudocharge density calculation.
//                 Additionally calculates the self energy      
void PeriodicChargDensB_VecInit(SDDFT_OBJ *pSddft) {

  PetscScalar ***BlcArrGlbIdx;
  PetscScalar ***BlcVpsArrGlbIdx;
  PetscScalar *pAtompos;
  PetscScalar *YD = NULL;

  PetscInt xcor, ycor, zcor, gxdim, gydim, gzdim, lxdim, lydim, lzdim;
  PetscInt xs, ys, zs, xl, yl, zl, xstart = -1, ystart = -1, zstart = -1, xend = -1, yend = -1, zend = -1, overlap = 0;
  PetscScalar x0, y0, z0, X0, Y0, Z0, coeffs[MAX_ORDER + 1], cutoffr;

  PetscScalar tableR[MAX_TABLE_SIZE], tableVps[MAX_TABLE_SIZE], Bval, elecno, noetot = pSddft->noetot, rhosum;
  PetscScalar ***pVpsArray = NULL;
  PetscScalar Dtemp;
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscInt i = 0, j, k, xi, yj, zk, offset, Ii, J, K, p, nzVps, nyVps, nxVps, i0, j0, k0, a, poscnt, index = 0, at;
  PetscScalar delta = pSddft->delta, x, y, z, r, rmax;

  // ffr: not used
  // PetscInt n_x = pSddft->numPoints_x;
  // PetscInt n_y = pSddft->numPoints_y;
  // PetscInt n_z = pSddft->numPoints_z;

  PetscScalar R_x = pSddft->range_x;
  PetscScalar R_y = pSddft->range_y;
  PetscScalar R_z = pSddft->range_z;
  PetscInt o = pSddft->order;

  PetscInt PP, QQ, RR;
  PetscInt Imax_x, Imin_x, Imax_y, Imin_y, Imax_z, Imin_z;

  int count;
  for (p = 0; p <= o; p++) {
    coeffs[p] = pSddft->coeffs[p] / (2 * M_PI);
  }

  DMDAVecGetArray(pSddft->da, pSddft->chrgDensB, &BlcArrGlbIdx);
  DMDAVecGetArray(pSddft->da, pSddft->bjVj, &BlcVpsArrGlbIdx);

  DMDAGetInfo(pSddft->da, 0, &gxdim, &gydim, &gzdim, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  DMDAGetCorners(pSddft->da, &xcor, &ycor, &zcor, &lxdim, &lydim, &lzdim);
#ifdef _DEBUG
  printf("Gxdim:%d,Gydim:%d,Gzdim:%d\n", gxdim, gydim, gzdim);
  printf("xcor:%d,ycor:%d,zcor:%d\n", xcor, ycor, zcor);
  printf("lxdim:%d,lydim:%d,lzdim:%d\n", lxdim, lydim, lzdim);
#endif

  VecGetArray(pSddft->Atompos, &pAtompos);
  for (at = 0; at < pSddft->Ntype; at++) {
    cutoffr = pSddft->CUTOFF[at];
    offset = (PetscInt)ceil(cutoffr / delta + 0.5);

    tableR[0] = 0.0;
    tableVps[0] = pSddft->psd[at].Vloc[0];
    count = 1;

    do {
      tableR[count] = pSddft->psd[at].RadialGrid[count - 1];
      tableVps[count] = pSddft->psd[at].Vloc[count - 1];
      count++;
    } while (PetscRealPart(tableR[count - 1]) <= pSddft->CUTOFF[at] + 4.0);

    rmax = tableR[count - 1];
    int start, end;
    start = (int)floor(pSddft->startPos[at] / 3);
    end = (int)floor(pSddft->endPos[at] / 3);

    PetscMalloc(sizeof(PetscScalar) * count, &YD);
    getYD_gen(tableR, tableVps, YD, count);

    /*
     * loop over every atom of a given type
     */
    for (poscnt = start; poscnt <= end; poscnt++) {
      X0 = pAtompos[index++];
      Y0 = pAtompos[index++];
      Z0 = pAtompos[index++];

      xi = (int)((X0 + R_x) / delta + 0.5);
      yj = (int)((Y0 + R_y) / delta + 0.5);
      zk = (int)((Z0 + R_z) / delta + 0.5);

      Imax_x = 0;
      Imin_x = 0;
      Imax_y = 0;
      Imin_y = 0;
      Imax_z = 0;
      Imin_z = 0;

      Imax_x = ceil(cutoffr / R_x);
      Imin_x = -ceil(cutoffr / R_x);
      Imax_y = ceil(cutoffr / R_y);
      Imin_y = -ceil(cutoffr / R_y);
      Imax_z = ceil(cutoffr / R_z);
      Imin_z = -ceil(cutoffr / R_z);

      for (PP = Imin_x; PP <= Imax_x; PP++)
        for (QQ = Imin_y; QQ <= Imax_y; QQ++)
          for (RR = Imin_z; RR <= Imax_z; RR++) {

            xstart = -1;
            ystart = -1;
            zstart = -1;
            xend = -1;
            yend = -1;
            zend = -1;
            overlap = 0;
            /*
             * periodic map of atomic position
             */
            x0 = X0 + PP * 2.0 * R_x;
            y0 = Y0 + QQ * 2.0 * R_y;
            z0 = Z0 + RR * 2.0 * R_z;

            xi = roundf((x0 + R_x) / delta);
            yj = roundf((y0 + R_y) / delta);
            zk = roundf((z0 + R_z) / delta);

            xs = xi - offset;
            xl = xi + offset;
            ys = yj - offset;
            yl = yj + offset;
            zs = zk - offset;
            zl = zk + offset;

            /*
             * find if domain of influence of pseudocharge overlaps with the domain stored by
             * processor
             */
            if (xs >= xcor && xs <= xcor + lxdim - 1)
              xstart = xs;
            else if (xcor >= xs && xcor <= xl)
              xstart = xcor;

            if (xl >= xcor && xl <= xcor + lxdim - 1)
              xend = xl;
            else if (xcor + lxdim - 1 >= xs && xcor + lxdim - 1 <= xl)
              xend = xcor + lxdim - 1;

            if (ys >= ycor && ys <= ycor + lydim - 1)
              ystart = ys;
            else if (ycor >= ys && ycor <= yl)
              ystart = ycor;

            if (yl >= ycor && yl <= ycor + lydim - 1)
              yend = yl;
            else if (ycor + lydim - 1 >= ys && ycor + lydim - 1 <= yl)
              yend = ycor + lydim - 1;

            if (zs >= zcor && zs <= zcor + lzdim - 1)
              zstart = zs;
            else if (zcor >= zs && zcor <= zl)
              zstart = zcor;

            if (zl >= zcor && zl <= zcor + lzdim - 1)
              zend = zl;
            else if (zcor + lzdim - 1 >= zs && zcor + lzdim - 1 <= zl)
              zend = zcor + lzdim - 1;

            if ((xstart != -1) && (xend != -1) && (ystart != -1) && (yend != -1) && (zstart != -1) && (zend != -1))
              overlap = 1;

            if (overlap) {
              nzVps = zend - zstart + 1 + o * 2;
              nyVps = yend - ystart + 1 + o * 2;
              nxVps = xend - xstart + 1 + o * 2;
              PetscMalloc(sizeof(PetscScalar **) * nzVps, &pVpsArray);

              if (pVpsArray == NULL) {
                // cout<<"Memory allocation fail";
                exit(1);
              }
              for (i = 0; i < nzVps; i++) {
                PetscMalloc(sizeof(PetscScalar *) * nyVps, &pVpsArray[i]);

                if (pVpsArray[i] == NULL) {
                  // cout<<"Memory allocation fail";
                  exit(1);
                }

                for (j = 0; j < nyVps; j++) {
                  PetscMalloc(sizeof(PetscScalar) * nxVps, &pVpsArray[i][j]);

                  if (pVpsArray[i][j] == NULL) {
                    // cout<<"Memory allocation fail";
                    exit(1);
                  }
                }
              }

              i0 = xstart - o;
              j0 = ystart - o;
              k0 = zstart - o;
              /*
               * evaluate the pseudopotential at nodes in the overlap region +
               * finite-difference order in each direction
               */
              for (k = 0; k < nzVps; k++)
                for (j = 0; j < nyVps; j++)
                  for (i = 0; i < nxVps; i++) {
                    Ii = i + i0;
                    J = j + j0;
                    K = k + k0;

                    x = delta * I - R_x;
                    y = delta * J - R_y;
                    z = delta * K - R_z;
                    r = sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));

                    if (r == tableR[0]) {
                      pVpsArray[k][j][i] = tableVps[0];
                    } else if (PetscRealPart(r) > PetscRealPart(rmax)) {
                      pVpsArray[k][j][i] = -pSddft->noe[at] / r;
                    } else {
                      ispline_gen(tableR, tableVps, count, &r, &pVpsArray[k][j][i], &Dtemp, 1, YD);
                    }
                  }

              /*
               * calculate pseudocharge density and contribution to self energy at nodes
               * inside the overlap region from previously calculated pseudopotential values
               * using finite difference stencil
               */
              for (k = zstart; k <= zend; k++)
                for (j = ystart; j <= yend; j++)
                  for (i = xstart; i <= xend; i++) {
                    Bval = 0;
                    Ii = i - i0;
                    J = j - j0;
                    K = k - k0;

                    Bval = pVpsArray[K][J][Ii] * coeffs[0];
                    for (a = 1; a <= o; a++) {
                      Bval += (pVpsArray[K][J][Ii - a] + pVpsArray[K][J][Ii + a] + pVpsArray[K][J - a][Ii] +
                               pVpsArray[K][J + a][Ii] + pVpsArray[K - a][J][Ii] + pVpsArray[K + a][J][Ii]) *
                              coeffs[a];
                    }

                    BlcArrGlbIdx[k][j][i] += Bval;
                    BlcVpsArrGlbIdx[k][j][i] += (pVpsArray[K][J][Ii]) * Bval;
                  }

              for (i = 0; i < nzVps; i++) {
                for (j = 0; j < nyVps; j++) {
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

  DMDAVecRestoreArray(pSddft->da, pSddft->chrgDensB, &BlcArrGlbIdx);
  DMDAVecRestoreArray(pSddft->da, pSddft->bjVj, &BlcVpsArrGlbIdx);
  VecRestoreArray(pSddft->Atompos, &pAtompos);

  /*
   * calculate self energy term
   */
  VecSum(pSddft->bjVj, &pSddft->Eself);
  PetscPrintf(PETSC_COMM_WORLD, "After VecSum Eself=%9.9f (Hartree)\n", PetscRealPart(pSddft->Eself));
  pSddft->Eself = -0.5 * pSddft->Eself * delta * delta * delta;
  PetscPrintf(PETSC_COMM_WORLD, "***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD, "           Pseudocharge, self energy and electrostatic correction          \n");
  PetscPrintf(PETSC_COMM_WORLD, "***************************************************************************\n");
  PetscPrintf(PETSC_COMM_WORLD, "Self energy of nuclei:%9.9f (Hartree)\n", PetscRealPart(pSddft->Eself));

  /*
   * check if pseudocharge density has the correct sign
   */
  VecSum(pSddft->chrgDensB, &elecno);
  PetscPrintf(PETSC_COMM_WORLD, "elecno = [%18.10f,%18.10f]\n", PetscRealPart(elecno), PetscImaginaryPart(elecno));
  assert(PetscRealPart(elecno) < 0);
  
  /*
   * scale electron density appropriately with respect to pseudocharge density
   */
  VecSum(pSddft->elecDensRho, &rhosum);
  rhosum = -elecno / rhosum;
  VecScale(pSddft->elecDensRho, rhosum);

  // calculate the total enclosed valence charge inside the simulation domain
  PetscPrintf(PETSC_COMM_WORLD, "delta = [%18.10f,%18.10f]\n",
    PetscRealPart(delta), PetscImaginaryPart(delta));
  PetscPrintf(PETSC_COMM_WORLD, "delta^3 = [%18.10f,%18.10f]\n",
    PetscRealPart(delta*delta*delta), PetscImaginaryPart(delta*delta*delta));
  elecno *= (delta * delta * delta);
  PetscPrintf(PETSC_COMM_WORLD, "after mult with delta^3 elecno = [%18.10f,%18.10f]\n",
    PetscRealPart(elecno), PetscImaginaryPart(elecno));

  pSddft->elecN = -elecno;
  PetscPrintf(PETSC_COMM_WORLD, "Total valence electronic charge: %.9e\n", PetscRealPart(elecno));

  PetscPrintf(PETSC_COMM_WORLD, "noetot = %18.10f\n", PetscRealPart(noetot));
  PetscPrintf(PETSC_COMM_WORLD, "elecno = %18.10f\n", PetscRealPart(elecno));
  PetscPrintf(PETSC_COMM_WORLD, "Z_ACC  = %18.10f\n", PetscRealPart(Z_ACC));

  PetscPrintf(PETSC_COMM_WORLD, "assert1 = %f\n", PetscRealPart((noetot + elecno)/noetot));
  PetscPrintf(PETSC_COMM_WORLD, "assert2 = %f\n", PetscRealPart((-elecno - noetot)/elecno));

#if 0
  assert((PetscRealPart((noetot + elecno) / noetot) < Z_ACC) && (PetscRealPart((-elecno - noetot) / elecno) < Z_ACC));
#endif

  return;
}

