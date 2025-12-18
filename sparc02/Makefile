REPO_DIR = /home/efefer/WORKS/my_github_repos/ffr-petsc-stuffs
PETSC_HOME = $(REPO_DIR)/install/petsc-3.12.5_complex

CC = mpicc

INC_FLAGS = -I$(PETSC_HOME)/include 
CFLAGS = -Wall -O3

LDLIBS  = -Wl,-rpath,$(PETSC_HOME)/lib \
-L$(PETSC_HOME)/lib \
-lpetsc \
-llapacke -llapack -lblas -lm

MY_OBJS = 

OBJLIBS = \
ExchangeCorrelation.o density.o forces.o kPointHamiltonian.o  mixing.o nonlocal.o poisson.o \
relaxatoms.o  spline.o chebyshev.o energy.o initObjs.o  multipole.o  occupation.o  readfiles.o  scf.o \
$(MY_OBJS)

all: lib

# Note the implicit rule to compile '.cc' files into '.o' files is
%.o : %.c
	$(CC) -c $(CFLAGS) $(INC_FLAGS) $< -o $@

lib: $(OBJLIBS)
	ar rcs libsparc.a $(OBJLIBS)

main_debug: lib
	$(CC) $(CFLAGS) $(INC_FLAGS) $(LDFLAGS) main_debug.c libsparc.a $(LDLIBS) -o main_debug.x

main: lib
	$(CC) $(CFLAGS) $(INC_FLAGS) $(LDFLAGS) main.c libsparc.a $(LDLIBS) -o main.x

.PHONY: clean
clean:
	rm -fv *.x *.o libsparc.a
