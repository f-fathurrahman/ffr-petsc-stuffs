/*=============================================================================================
  | Simulation Package for Ab-initio Real-space Calculations (SPARC) 
  | Copyright (C) 2016 Material Physics & Mechanics Group at Georgia Tech.
  |
  | S. Ghosh, P. Suryanarayana, SPARC: Accurate and efficient finite-difference formulation and
  | parallel implementation of Density Functional Theory. Part I: Isolated clusters, Computer
  | Physics Communications
  |
  | file name: isddft.h          
  |
  | Description: This file contains the variables required by SPARC
  |
  | Authors: Swarnava Ghosh, Phanish Suryanarayana
  |
  | Last Modified: 2/9/2016   
  |-------------------------------------------------------------------------------------------*/

#ifndef _IODFD_
#define _IODFD_
#include "petsc.h"
#include "petscdmda.h"
#define MAX_ORDER 10 
#define EPSNEV 1
/*
 * structure storing the pseudopotential information 
 */
typedef struct 
{
  
  PetscScalar *Vloc; ///< stores local part of pseudopotential 
  PetscScalar *Vs;  ///< stores s component of pseudopotential
  PetscScalar *Vp;  ///< stores p component of pseudopotential
  PetscScalar *Vd; ///< stores d component of pseudopotential
  PetscScalar *Vf; ///< stores f component of pseudopotential
  PetscScalar *Us; ///< stores s component of pseudowavefunction
  PetscScalar *Up; ///< stores p component of pseudowavefunction
  PetscScalar *Ud; ///< stores d component of pseudowavefunction
  PetscScalar *Uf; ///< stores f component of pseudowavefunction
  PetscScalar *uu;  ///< stores single atom electron density
  PetscScalar *RadialGrid;  ///< stores the radial grid
  PetscScalar rc_s; ///< s component pseudopotential cutoff
  PetscScalar rc_p; ///< p component pseudopotential cutoff
  PetscScalar rc_d; ///< d component pseudopotential cutoff
  PetscScalar rc_f; ///< f component pseudopotential cutoff
  PetscInt lmax; ///< maximum pseudopotential component
  PetscInt size; ///< size of the arrays storing the pseudopotentials

}PSD_OBJ;

/*
 * structure storing the variables required by the functions of SPARC 
 */
typedef struct
{
  PetscInt  numPoints_x; ///< stores the number of nodes in x-direction
  PetscInt  numPoints_y; ///< stores the number of nodes in y-direction
  PetscInt  numPoints_z; ///< stores the number of nodes in z-direction
  PetscInt  order; ///< stores half of finite difference order
  PetscInt  nAtoms; ///< stores the total number of atoms in the simulation
  PetscInt *noa,*noe; ///< For each type of atom, noa stores the number of atoms of that type, noe stores the number of valence electrons of that type
  PetscInt noetot; ///< stores the total number of valence electrons in the simulation   
  PetscInt ChebDegree; ///< stores the degree of Chebychev polynomial used for the Chebychev filter 
  PetscInt MAXITSCF; ///< Maximum number of SCF iterations	
  PetscInt MAXIT_NLCG; ///< Maximum number of NLCG iterations	

  PetscScalar range_x; ///< stores half of domain length in x direction
  PetscScalar range_y; ///< stores half of domain length in y direction
  PetscScalar range_z; ///< stores half of domain length in z direction
  PetscScalar delta; ///< mesh spacing 
  PetscScalar elecN; ///< stores integral of total pseudocharge density 
  PetscScalar Eatom; ///< stores total energy per atom  
  PetscScalar Eself; ///< stores self energy of the nuclei
  PetscScalar Eself_TM; ///< stores self energy of the nuclei calculated using reference pseudopotential
  PetscScalar Exc; ///< stores exchange correlation energy
  PetscScalar Ecorrection; ///< stores electrostatic correction energy
  PetscScalar Entropy; ///< stores the electronic entropy
  PetscScalar Eband; ///< stores the band structure energy
  PetscScalar TOLSCF; ///< stores the scf tolerence
  PetscScalar KSPTOL; ///< stores the tolerence for solving the Poisson equation
  PetscScalar NLCGTOL; ///< stores the tolerance for the non-linear conjugate gradient method for atomic relaxation
  PetscScalar LANCZOSTOL; ///< stores the tolerence for the lanczos iteration
  PetscScalar PseudochargeRadiusTOL; ///< stores the tolerance for calculation of pseudocharge density radius
  PetscScalar REFERENCE_CUTOFF; ///< stores the cutoff for the reference pseudopotential

  PetscScalar coeffs[MAX_ORDER+1]; ///< stores the weights of the finite-difference laplacian
  PetscScalar coeffs_grad[MAX_ORDER+1]; ///< stores the weights of the finite difference gradient
  PetscScalar Beta; ///< stores the electronic smearing (1/(k_B * T)) (units: 1/Ha)
  PetscInt Nstates;  ///< stores total number of electronic states used in the simulation
  PetscInt Ntype; ///< stores total number of types of atoms used for the simulation
  PetscInt SCFNewRhoCalcCtr; ///< stores the number of times Chebychev filtering is done in the first SCF iteration before updation of electron density
  
  int RelaxCount; ///< stores the number of relaxation steps done. This is incremented for every relaxation step.
  PetscScalar MixingParameter; ///< stores mixing parameter for Anderson Mixing
  int MixingHistory; ///< stores mixing history for Anderson Mixing 

  PetscScalar lambda_f; ///< stores the Fermi energy
  PetscScalar *lambda; ///< stores the eigenvalues in the subspace eigen problem. Size of the array is Nstates
  PetscScalar *CUTOFF; ///< stores the pseudocharge cutoff \f$ r_J^b \f$. Size of array is nAtoms. 
  PetscInt *startPos; ///< stores the starting position of a particular type of atom in the list of atom positions (stored in Vec Atompos)
  PetscInt *endPos; ///< stores the ending position of a particular type of atom in the list of atom positions (stored in Vec Atompos)
  char **atomType; ///< stores the string containing the atom type name for every type of atom. example H for Hydrogen, He for Helium, Li for Lithium, etc. This is read from the .ion file  
  char **psdName; ///< stores the string containing the pseudopotential name for every type of atom. This is read from the .inpt file  
  int *localPsd; ///< stores the respective local component of pseudopotential for each type of atom. 0 for s, 1 for p, 2 for d, 3 for f.
  char file[30]; ///< stores the input filename 
  char XC[30]; ///< stores the exchange correlation name
  int RelaxFlag; ///< flag for relaxation of atoms. 1=relaxation, any other number=NO relaxation

  int ChebyshevCallCounter; ///< variable for storing the number of times chebyshev filter has been called 
  
  PetscInt *nnzDArray;  ///< stores the number of nonzeros in the diagonal block of the nonlocal matrix
  PetscInt *nnzODArray; ///< stores the number of nonzeros in the off diagonal block of the nonlocal matrix

  DM da; ///< DMDA for finite difference laplacian. see PETSc DMDA for more details
  DM da_grad; ///< DMDA for finite difference laplacian. see PETSc DMDA for more details  
  Vec elecDensRho; ///< PETSc vector for storing the electron density
  Vec SuperposAtRho; ///< PETSc vector for storing the guess electron density as obtained from superposition of atoms 
  Vec chrgDensB; ///< PETSc vector for storing the pseudocharge density. This is calculated from the pseudopotential files given as user input
  Vec chrgDensB_TM; ///< PETSc vector for storing the reference pseudocharge density. This is used for calculation of energy and force corrections
  Vec potentialPhi; ///< PETSc vector for storing the electrostatic potential
  Vec twopiRhoPB; ///< PETSc vector for storing \f$ 2*\pi*(\rho+b) \f$ (the right hand side of the poisson equation solves for the elctrostatic potential)
  Vec Veff; ///< PETSc vector for storing effective potential \f$ V_{eff} = \phi + V_{xc} \f$
  Vec Vxc; ///< PETSc vector for storing exchange-correlation potential \f$ V_{xc} \f$
  Vec bjVj; ///< PETSc vector for storing the nodal contribution of the repulsive energy
  Vec bjVj_TM; ///< PETSc vector for storing the nodal contribution of the reference repulsive energy
  Vec Phi_c; ///< PETSc vector for storing the correction electrostatic potential
  Vec twopiBTMmBPS; ///< PETSc vector for storing the difference between the reference pseudocharge density and the pseudocharge density
  Vec PoissonRHSAdd; ///< PETSc vector for storing the correction term in multipole expansion

  PetscScalar *pForces_corr; ///< stores the correction in forces on atoms
  Vec forces; ///< PETSc vector storing the forces on the atoms. Size of this vector is 3*nAtoms. The first, second and third entries are the x,y and z coordinates of the atomic forces on first atom, the fourth, fifth and sixth entries are the x,y,z coordinates on the atomic positions of second atom and so on.
  
  Vec Atompos; ///< PETSc vector storing the atomic positions. Size of this vector is 3*nAtoms. The first, second and third entries are the x,y and z coordinates of the atomic positions of first atom, the fourth, fifth and sixth entries are the x,y,z coordinates of the atomic positions of second atom and so on.
   
  Vec mvAtmConstraint;  ///< PETSc vector storing the constraints for movement of atoms. Size of this vector is 3*nAtoms. An entry of 1 means the atom is "movable", hence "net" force on it is non-zero, an entry of 0 means the atom is "fixed" and the "net" force on it is zero.
   
  Mat laplaceOpr; ///< PETSc distributed sparse matrix (row major format) for storing -1/2 laplacian. This is a PETSc DMDA type matrix. See PETSc manuals for details on DMDA matrices
  Mat HamiltonianOpr;	///< PETSc distributed sparse matrix (row major format) for storing the Hamiltonian operator
  Mat gradient_x; ///< PETSc distributed sparse matrix (row major format) for storing x component of the gradient operator. This is a PETSc DMDA type matrix. See PETSc manuals for details on DMDA matrices
  Mat gradient_y;  ///< PETSc distributed sparse matrix (row major format) for storing y component of the gradient operator. This is a PETSc DMDA type matrix. See PETSc manuals for details on DMDA matrices
  Mat gradient_z;  ///< PETSc distributed sparse matrix (row major format) for storing z component of the gradient operator. This is a PETSc DMDA type matrix. See PETSc manuals for details on DMDA matrices
  
  /* 
   * matrices for storing orbitals
   */
  Mat XOrb; ///< PETSc distributed dense matrix (row major format) for storing the orbitals. Stores input to the chebychev filter
  Mat YOrb; ///< PETSc distributed dense matrix (row major format) for storing the orbitals. Stores the filtered orbitals
  Mat YOrbNew; ///< PETSc distributed dense matrix (row major format) for storing the orbitals. Stores a temporary copy of the orbitals in chebychev filtering
    
  /*
   * Vectors for Anderson mixing
   */
  Vec xkprev; 
  Vec xk;
  Vec fkprev;
  Vec *Xk;
  Vec *Fk;
  Vec *XpbF;   

  Vec tempVec; ///< temporary vector used for storing copy of vector during computation
 
  KSP ksp; ///< PETSc (Krylov subspace) ksp object for solving the Poisson equation. See PETSc manual for more details on ksp

  PSD_OBJ *psd; ///< datatype for the pseudopotentials. This is defined for each type of atom.
     
}SDDFT_OBJ;
#endif
