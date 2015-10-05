#
# Makefile for mysvm
#

# if you get memory errors using mySVM (segmentation fault, bus error,...)
# compile mySVM with less or without optimization (setting CFLAGS = -Wall).
CFLAGS = -Wall -O4
CC = g++
OBJ = smo.o svm_nu.o svm_c.o globals.o example_set.o parameters.o kernel.o
BINDIR = bin/`uname -s`

# -static for suns
# -pg for gprof

all: dirs mysvm predict

predict: globals.o predict.o svm_c.o parameters.o kernel.o example_set.o
	$(CC) $(CFLAGS) -o $(BINDIR)/predict $(OBJ) predict.o

mysvm: smo.o svm_nu.o globals.o learn.o svm_c.o parameters.o kernel.o example_set.o 
	$(CC) $(CFLAGS) -o $(BINDIR)/mysvm $(OBJ) learn.o

svm_nu.o: svm_nu.h svm_nu.cpp
	$(CC) $(CFLAGS) -c svm_nu.cpp	

globals.o: globals.h globals.cpp
	$(CC) $(CFLAGS) -c globals.cpp	

predict.o: predict.cpp globals.h example_set.h svm_c.h parameters.h kernel.h
	$(CC) $(CFLAGS) -c predict.cpp

learn.o: learn.cpp globals.h example_set.h svm_c.h parameters.h kernel.h
	$(CC) $(CFLAGS) -c learn.cpp

smo.o: smo.h smo.cpp
	$(CC) $(CFLAGS) -c smo.cpp

svm_c.o: globals.h svm_c.h svm_c.cpp example_set.h parameters.h kernel.h
	$(CC) $(CFLAGS) -c svm_c.cpp

parameters.o: globals.h parameters.h parameters.cpp
	$(CC) $(CFLAGS) -c parameters.cpp

kernel.o: globals.h kernel.h kernel.cpp example_set.h parameters.h
	$(CC) $(CFLAGS) -c kernel.cpp


example_set.o: globals.h example_set.h example_set.cpp
	$(CC) $(CFLAGS) -c example_set.cpp

dirs:
	test ! -d bin/ && mkdir bin/ ; true
	test ! -d $(BINDIR) && mkdir $(BINDIR) ; true

clean:
	rm -f $(OBJ) learn.o predict.o bin/$(HOSTTYPE)/mysvm ; true


