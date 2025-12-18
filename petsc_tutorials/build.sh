set -x

REPO_DIR="/home/efefer/WORKS/my_github_repos/ffr-petsc-stuffs"

#PETSC_HOME="${REPO_DIR}/install/petsc-3.24.2_opt"
PETSC_HOME="${REPO_DIR}/install/petsc-3.24.2"
#PETSC_HOME="${REPO_DIR}/install/petsc-3.24.2_complex"

#PETSC_HOME="${REPO_DIR}/install/petsc-3.12.5_opt"
#PETSC_HOME="${REPO_DIR}/install/petsc-3.12.5"
#PETSC_HOME="${REPO_DIR}/install/petsc-3.12.5_complex"

basnam=`basename $1 .c`

# Can be obtained by `make`-ing some tutorials:
mpicc -I${PETSC_HOME}/include $1 \
-Wl,-rpath,${PETSC_HOME}/lib -L${PETSC_HOME}/lib \
-lpetsc -llapack -lblas -lm -o ${basnam}.x
