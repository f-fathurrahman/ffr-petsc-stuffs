REPO_DIR = /home/efefer/WORKS/my_github_repos/ffr-petsc-stuffs
PETSC_HOME = $(REPO_DIR)/install/petsc-3.24.2

CPPFLAGS = -I$(PETSC_HOME)/include 

LDLIBS  = -Wl,-rpath,$(PETSC_HOME)/lib \
-L$(PETSC_HOME)/lib \
-Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib \
-L/usr/lib/x86_64-linux-gnu/openmpi/lib -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 \
-L/usr/lib/gcc/x86_64-linux-gnu/9 -Wl,-rpath,/usr/lib/x86_64-linux-gnu \
-L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/lib/x86_64-linux-gnu \
-L/lib/x86_64-linux-gnu -lpetsc \
-llapacke -llapack -lblas -lm -lstdc++ \
-ldl -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lgfortran \
-lm -lgfortran -lm -lgcc_s -lquadmath -lpthread -lquadmath -lstdc++ -ldl

#CFLAGS = -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing \
-Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3  -fPIC -Wall -Wwrite-strings \
-Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3

CFLAGS = -Wall -O3

MY_OBJS = \
my_SddftObjInitialize.o \
my_Read_parameters.o my_Read_ion.o my_Read_relax.o my_Read_pseudopotential.o \
my_Objects_Create.o


OBJLIBS = \
ExchangeCorrelation.o  density.o  forces.o    \
mixing.o     nonlocal.o    poisson.o    relaxatoms.o  spline.o \
chebyshev.o   energy.o   initObjs.o  multipole.o  occupation.o  readfiles.o  scf.o \
$(MY_OBJS)

OBJSC = main.o $(OBJLIBS)

EXECUTABLE = main.x
TESTBASE = ../.ci

override CC=mpicc

all: lib

# Note the implicit rule to compile '.cc' files into '.o' files is
%.o : %.cc
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

lib: $(OBJLIBS)
	ar rcs libsparc.a $(OBJLIBS)

main_debug: lib
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) main_debug.cc libsparc.a $(LDLIBS) -o main_debug.x

main: $(OBJSC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) main.cc libsparc.a $(LDLIBS) -o main.x

.PHONY: clean
clean:
	rm -f  $(OBJSC) $(EXECUTABLE) *.x

test: ../tests/SPARC_testing_script.py
	cd ../tests; python SPARC_testing_script.py
