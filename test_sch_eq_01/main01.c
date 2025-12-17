#include <petsc.h>
#include <slepceps.h>

int main(int argc,char **argv)
{

  SlepcInitialize(&argc, &argv, NULL, NULL);

  Mat H;
  EPS eps;
  PetscInt N = 500, i;
  PetscScalar h, v;
  PetscReal xmin = -10.0, xmax = 10.0;

  h = (xmax - xmin)/(N-1);

  /* Create Hamiltonian matrix */
  MatCreate(PETSC_COMM_WORLD,&H);
  MatSetSizes(H,PETSC_DECIDE,PETSC_DECIDE,N,N);
  MatSetFromOptions(H);
  MatSetUp(H);

  for (i=0; i<N; i++) {
    PetscReal x = xmin + i*h;
    /* Potential: harmonic oscillator V = x^2 / 2 */
    v = 0.5 * x * x;
    MatSetValue(H, i, i, 1.0/(h*h) + v,INSERT_VALUES);
    if( i > 0 ) {
      MatSetValue(H, i, i-1, -0.5/(h*h),INSERT_VALUES);
    }
    if( i < N-1 ) {
      MatSetValue(H, i, i+1, -0.5/(h*h),INSERT_VALUES);
    }
  }

  MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);

  /* Setup eigenproblem */
  EPSCreate(PETSC_COMM_WORLD,&eps);
  EPSSetOperators(eps,H,NULL);
  EPSSetProblemType(eps,EPS_HEP);   /* Hermitian */
  EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);
  EPSSetFromOptions(eps);

  EPSSolve(eps);

  PetscInt nconv;
  EPSGetConverged(eps,&nconv);
  PetscPrintf(PETSC_COMM_WORLD,"Converged eigenpairs: %d\n",nconv);

  for(i=0; i < nconv; i++) {
    PetscScalar eig;
    EPSGetEigenvalue(eps,i,&eig,NULL);
    PetscPrintf(PETSC_COMM_WORLD,"E[%d] = %g\n",i,(double)PetscRealPart(eig));
  }

  EPSDestroy(&eps);
  MatDestroy(&H);
  SlepcFinalize();
  return 0;
}
