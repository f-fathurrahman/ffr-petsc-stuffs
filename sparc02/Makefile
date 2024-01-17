all:sparc

PETSC_DIR1 = /home/efefer/parallel_c/petsc-3.5.4
PETSC_DIR2 = /home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug
#PETSC_DIR2 = /home/efefer/WORKS/PETSC/petsc-3.5.3/arch-linux2-c-opt

#MKLROOT = /home/efefer/mysoftwares/LIBMKL/

FFLAGS =
#CPPFLAGS = -I${PETSC_DIR}/include -I ./inc -I ${MKLROOT}/include -L ${MKLROOT}/lib/ -llapack-addons -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread 

CPPFLAGS = -std=c++11 -I ${PETSC_DIR1}/include -I ${PETSC_DIR2}/include -I ./inc -I ${MKLROOT}/include

SOURCEH = ./inc/sddft.h ./inc/isddft.h

OBJSC = ./src/main.o ./src/initObjs.o ./src/spline.o ./src/nonlocal.o ./src/density.o ./src/readfiles.o ./src/energy.o ./src/ExchangeCorrelation.o ./src/occupation.o ./src/poisson.o ./src/chebyshev.o ./src/scf.o ./src/mixing.o ./src/forces.o ./src/relaxatoms.o ./src/multipole.o ./src/kPointHamiltonian.o

PETSC_LIB = -Wl,-rpath,${PETSC_DIR2}/lib: ${PETSC_DIR2}/lib/libpetsc.so -L${MKLROOT} -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread

CLEANFILES = ./lib/sparc ${OBJSC}


sparc: ${OBJSC}
	${CXX} -Wall -o lib/sparc ${OBJSC} ${PETSC_LIB}
	${RM} ${OBJSC}
