#include <petsc.h>

int main(int argc,char **args)
{
  Mat A,B,C;
  PetscInt i;
  PetscMPIInt rank;

  PetscInitialize(&argc,&args,NULL,NULL);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  /* Create A */
  MatCreate(PETSC_COMM_WORLD,&A);
  MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,3,3);
  MatSetFromOptions(A);
  MatSetUp(A);

  for (i=0;i<3;i++) {
    PetscScalar val = 2.0;
    MatSetValue(A,i,i,val,INSERT_VALUES);
  }

  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

  /* Create B */
  MatCreate(PETSC_COMM_WORLD,&B);
  MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,3,3);
  MatSetFromOptions(B);
  MatSetUp(B);

  for (i=0;i<3;i++) {
    PetscScalar val = 3.0;
    MatSetValue(B,i,i,val,INSERT_VALUES);
  }

  MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);

  /* Create result matrix C */
  MatCreate(PETSC_COMM_WORLD,&C);
  MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,3,3);
  MatSetFromOptions(C);

  /* ---- MAT PRODUCT API ---- */

  /* Tell PETSc we want C = A * B */
  MatProductCreate(A, B, NULL, &C);
  MatProductSetType(C, MATPRODUCT_AB);

  MatProductSetFromOptions(C);

  /* Step 1 — symbolic phase */
  MatProductSymbolic(C);

  /* Step 2 — numeric phase */
  MatProductNumeric(C);

  /* View result */
  if (!rank) {
    PetscPrintf(PETSC_COMM_SELF,"\nResult matrix C:\n");
  }
  MatView(C,PETSC_VIEWER_STDOUT_WORLD);

  /* Cleanup */
  MatDestroy(&A);
  MatDestroy(&B);
  MatDestroy(&C);

  PetscFinalize();
  return 0;
}
