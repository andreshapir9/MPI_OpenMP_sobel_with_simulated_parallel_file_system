INCLUDE_DIRS = -I/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/include/
LIB_DIRS = -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib/debug -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib
LIB_DIRS = -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib/release -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib
CFLAGSOPEN=  -fopenmp 
CC = mpicc
CXX = mpicxx
CPP = g++

CDEFS=
CFLAGS= -g -O3 -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= sobelMPI2 sobel_MPI_scalable
HFILES=  
CFILES=  sobelMPI2.cpp sobel_MPI_scalable.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

# sobel:	sobel.cpp
# 	$(CPP)  -o $@ sobel.cpp

# sobelMPI:	sobelMPI.cpp
# 	$(CXX) $(CFLAGS) -o $@ sobelMPI.cpp $(LIB_DIRS)



#This is the working version
sobel_MPI_scalable: sobel_MPI_scalable.cpp
	$(CXX) $(CFLAGS) $(CFLAGSOPEN) -o $@ sobel_MPI_scalable.cpp $(LIB_DIRS)
	
sobelMPI2:	sobelMPI2.cpp
	$(CXX) $(CFLAGS) -o $@ sobelMPI2.cpp $(LIB_DIRS)
