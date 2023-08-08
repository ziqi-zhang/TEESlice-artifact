#ifndef SGXDNN_UTILS_HPP_
#define SGXDNN_UTILS_HPP_

#include <unsupported/Eigen/CXX11/Tensor>

#ifdef USE_SGX
#include "Enclave_t.h"
#else
#include <chrono>
#endif

bool TIMING = true;

Eigen::array<Eigen::IndexPair<int>, 1> MATRIX_PRODUCT = { Eigen::IndexPair<int>(1, 0) };
Eigen::array<Eigen::IndexPair<int>, 1> INNER_PRODUCT = { Eigen::IndexPair<int>(0, 0) };
Eigen::array<int, 2> TRANSPOSE2D = {{1, 0}};

typedef Eigen::array<long, 1> array1d;
typedef Eigen::array<long, 2> array2d;
typedef Eigen::array<long, 3> array3d;
typedef Eigen::array<long, 4> array4d;

typedef long long int64;

template <typename T, int NDIMS = 1>
using Tensor = typename Eigen::Tensor<T, NDIMS, Eigen::RowMajor, Eigen::DenseIndex>;

template <typename T>
using Matrix = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using MatrixMap = typename Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
using VectorMap = typename Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>>;

#ifndef USE_SGX
typedef std::chrono::time_point<std::chrono::high_resolution_clock> sgx_time_t;
std::chrono::time_point<std::chrono::high_resolution_clock> get_time() {
	if (TIMING) {
        return std::chrono::high_resolution_clock::now();
	}
	return std::chrono::time_point<std::chrono::high_resolution_clock>();
}

std::chrono::time_point<std::chrono::high_resolution_clock> get_time_force() {
	return std::chrono::high_resolution_clock::now();
}

double get_elapsed_time(std::chrono::time_point<std::chrono::high_resolution_clock> start,
						std::chrono::time_point<std::chrono::high_resolution_clock> end) {

	std::chrono::duration<double> elapsed = end - start;
	return elapsed.count();
}
#else
typedef double sgx_time_t;
double get_time() {
	if (TIMING) {
		double res;
		ocall_get_time(&res);
		return res;
	}
	return 0.0;
}

double get_time_force() {
	double res;
	ocall_get_time(&res);
	return res;
}

double get_elapsed_time(double start, double end) {
	return (end - start) / (1000.0 * 1000.0);
}
#endif

#endif /* SGXDNN_UTILS_HPP_ */
