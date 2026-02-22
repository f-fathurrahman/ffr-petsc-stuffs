REPO_DIR = /home/efefer/WORKS/my_github_repos/ffr-petsc-stuffs
PETSC_HOME = $(REPO_DIR)/install/petsc-3.24.2_complex

CC = mpicc

INC_FLAGS = -I$(PETSC_HOME)/include 
#CFLAGS = -Wall -O3 -D_DEBUG -g # -O0 -fsanitize=address
CFLAGS = -Wall -O0 -g -O0 #-fsanitize=address,undefined

LDLIBS  = -Wl,-rpath,$(PETSC_HOME)/lib \
-L$(PETSC_HOME)/lib \
-lpetsc \
-llapacke -llapack -lblas -lm

MY_OBJS = 

# XXX Need to use .o extension
OBJLIBS = \
ExchangeCorrelation.o density.o forces.o kPointHamiltonian.o  mixing.o nonlocal.o poisson.o \
relaxatoms.o  spline.o chebyshev.o energy.o initObjs.o  multipole.o  occupation.o  readfiles.o  scf.o \
per_Laplacian_nlpsp_mat_init.o \
PeriodicChargDensB_VecInit.o \
$(MY_OBJS)

all: lib

#nonlocal.o: nonlocal.c
#	$(CC) -c $(CFLAGS) -O0 -fsanitize=address $(INC_FLAGS) nonlocal.c -o nonlocal.o

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
