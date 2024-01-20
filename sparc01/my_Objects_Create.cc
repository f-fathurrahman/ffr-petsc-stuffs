// Objects_Create: Creates the PETSc objects                         //
PetscErrorCode Objects_Create(SDDFT_OBJ *pSddft) {

  PetscPrintf(PETSC_COMM_WORLD, "-----------------------\n");
  PetscPrintf(PETSC_COMM_WORLD, "ENTER my_Objects_Create\n");
  PetscPrintf(PETSC_COMM_WORLD, "-----------------------\n");

  PetscInt n_x = pSddft->numPoints_x;
  PetscInt n_y = pSddft->numPoints_y;
  PetscInt n_z = pSddft->numPoints_z;
  PetscInt o = pSddft->order;
  int MAX_ITS_ANDERSON = pSddft->MixingHistory;

  PetscInt xcor, ycor, zcor, lxdim, lydim, lzdim;
  int i;
  Mat A;
  PetscMPIInt comm_size;
  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD, "num_points = %d, %d, %d\n", n_x, n_y, n_z);
  PetscPrintf(PETSC_COMM_WORLD, "order = %d\n", o);

  // DM for electron density
  ierr = DMDACreate3d(PETSC_COMM_WORLD,
    DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
    DMDA_STENCIL_STAR, n_x, n_y, n_z, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
    1, o, 0, 0, 0, &pSddft->da);
  CHKERRQ(ierr);
  ierr = DMSetUp(pSddft->da); CHKERRQ(ierr);

  // DM for gradient
  ierr = DMDACreate3d(PETSC_COMM_WORLD,
    DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
    DMDA_STENCIL_STAR, n_x, n_y, n_z, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, o, 0,
    0, 0, &pSddft->da_grad);
  CHKERRQ(ierr);
  ierr = DMSetUp(pSddft->da_grad); CHKERRQ(ierr);

  // Create electron density
  ierr = DMCreateGlobalVector(pSddft->da, &pSddft->elecDensRho); // error
  CHKERRQ(ierr);

  VecDuplicate(pSddft->elecDensRho, &pSddft->SuperposAtRho);
  VecDuplicate(pSddft->elecDensRho, &pSddft->chrgDensB);
  VecDuplicate(pSddft->elecDensRho, &pSddft->chrgDensB_TM);
  VecDuplicate(pSddft->elecDensRho, &pSddft->potentialPhi);
  VecDuplicate(pSddft->elecDensRho, &pSddft->Phi_c);
  VecDuplicate(pSddft->elecDensRho, &pSddft->twopiRhoPB);
  VecDuplicate(pSddft->elecDensRho, &pSddft->twopiBTMmBPS);
  VecDuplicate(pSddft->elecDensRho, &pSddft->Veff);
  VecDuplicate(pSddft->elecDensRho, &pSddft->bjVj);
  VecDuplicate(pSddft->elecDensRho, &pSddft->bjVj_TM);
  VecDuplicate(pSddft->elecDensRho, &pSddft->Vxc);

  VecDuplicate(pSddft->elecDensRho, &pSddft->tempVec);
  VecDuplicate(pSddft->elecDensRho, &pSddft->PoissonRHSAdd);

  VecDuplicate(pSddft->elecDensRho, &pSddft->xkprev);
  VecDuplicate(pSddft->elecDensRho, &pSddft->xk);
  VecDuplicate(pSddft->elecDensRho, &pSddft->fkprev);


  PetscMalloc(sizeof(Vec) * (MAX_ITS_ANDERSON), &pSddft->Xk);
  PetscMalloc(sizeof(Vec) * (MAX_ITS_ANDERSON), &pSddft->Fk);
  PetscMalloc(sizeof(Vec) * (MAX_ITS_ANDERSON), &pSddft->XpbF);

  for (i = 0; i < MAX_ITS_ANDERSON; i++) {
    VecDuplicate(pSddft->elecDensRho, &pSddft->Xk[i]);
    VecDuplicate(pSddft->elecDensRho, &pSddft->Fk[i]);
    VecDuplicate(pSddft->elecDensRho, &pSddft->XpbF[i]);
  }

  MPI_Comm_size(PETSC_COMM_WORLD, &comm_size);
  if (comm_size == 1) {
    DMCreateMatrix(pSddft->da, &pSddft->laplaceOpr);
    DMSetMatType(pSddft->da, MATSEQSBAIJ);
  } else {
    DMCreateMatrix(pSddft->da, &pSddft->laplaceOpr);
    DMSetMatType(pSddft->da, MATMPISBAIJ);
  }
  A = pSddft->laplaceOpr;
  KSPCreate(PETSC_COMM_WORLD, &pSddft->ksp);
  KSPSetType(pSddft->ksp, KSP_TYPE);
  KSPSetOperators(pSddft->ksp, A, A);
  KSPSetTolerances(pSddft->ksp, pSddft->KSPTOL, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
  KSPSetFromOptions(pSddft->ksp);

  // creating gradient operators
  
  if (comm_size == 1) {
    DMCreateMatrix(pSddft->da_grad, &pSddft->gradient_x);
    DMCreateMatrix(pSddft->da_grad, &pSddft->gradient_y);
    DMCreateMatrix(pSddft->da_grad, &pSddft->gradient_z);
    DMSetMatType(pSddft->da_grad, MATSEQBAIJ);
  } else {
    DMCreateMatrix(pSddft->da_grad, &pSddft->gradient_x);
    DMCreateMatrix(pSddft->da_grad, &pSddft->gradient_y);
    DMCreateMatrix(pSddft->da_grad, &pSddft->gradient_z);
    DMSetMatType(pSddft->da_grad, MATMPIBAIJ);
  }

  DMDAGetCorners(pSddft->da, &xcor, &ycor, &zcor, &lxdim, &lydim, &lzdim);
  PetscMalloc(sizeof(PetscInt) * (lzdim * lydim * lxdim), &pSddft->nnzDArray);
  PetscMalloc(sizeof(PetscInt) * (lzdim * lydim * lxdim), &pSddft->nnzODArray);


  PetscPrintf(PETSC_COMM_WORLD, "----------------------\n");
  PetscPrintf(PETSC_COMM_WORLD, "EXIT my_Objects_Create\n");
  PetscPrintf(PETSC_COMM_WORLD, "----------------------\n");

  return 0;
}