CUDAC = nvcc
CUDAOPT = -std=c++17 -lineinfo -g
CUDAFLAGS = $(CUDAOPT)

CXX = mpic++
CFLAGS = -g 
INCLUDE = -I/opt/spack/opt/spack/linux-debian11-cascadelake/gcc-10.2.1/cuda-11.7.0-42vtagqikeyleemufys4fi54tnjh75by/include -I/home/cyj/szq/mpi_spmm/include -I/opt/spack/opt/spack/linux-debian11-cascadelake/gcc-10.2.1/openmpi-4.1.3-byrqipvj7nkafujsgj7q6mtjz2jqydnn/include
CUDAPATH = $(CUDA_HOME)


LDFLAGS = -lm -lstdc++ -ldl
LDLIBS = $(LDFLAGS) -L$(CUDAPATH)/lib64 -lcudart

targets = benchmark-cusparse benchmark-optimized 
commonobj = utils.o main.o
objects = $(commonobj) spgemm-cusparse.o spgemm-optimized.o

.PHONY : default
default : all

.PHONY : all
all : $(targets)

utils.o : utils.cu 
	$(CUDAC) -c $(CUDAFLAGS) $(INCLUDE) $< -o $@

spgemm-cusparse.o : spgemm-cusparse.cu 
	$(CUDAC) -c $(CUDAFLAGS) $(INCLUDE) $< -o $@

spgemm-optimized.o : spgemm-optimized.cu 
	$(CUDAC) -O0 -c $(CUDAFLAGS) $(INCLUDE) $< -o $@

main.o: main.cpp
	$(CXX) -c  -O0 $(CFLAGS) $(INCLUDE) $< -o $@

benchmark-cusparse : main.o spgemm-cusparse.o utils.o
	$(CXX) $^ $(LDLIBS) -lcusparse -o $@

benchmark-optimized: main.o spgemm-optimized.o utils.o
	$(CXX) $^ $(LDLIBS) -lcusparse -o $@

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)
