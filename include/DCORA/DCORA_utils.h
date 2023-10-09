/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DCORAUTILS_H
#define DCORAUTILS_H

#include <DCORA/DCORA_types.h>
#include <DCORA/RelativeSEMeasurement.h>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <chrono>

// ROPTLIB includes
#include "Manifolds/Stiefel/Stiefel.h"

namespace DCORA {

class SimpleTimer {
 public:
  /**
   * @brief Start timer
   */
  void tic();
  /**
   * @brief Return elapsed time since last tic in ms
   * @return
   */
  double toc();
  /**
   * @brief Start timer and return starting time
   * @return
   */
  static std::chrono::time_point<std::chrono::high_resolution_clock> Tic();
  /**
   * @brief Return time elapsed since input start_time
   * @param start_time
   * @return elapsed time in ms
   */
  static double Toc(const std::chrono::time_point<std::chrono::high_resolution_clock> &start_time);
 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;
};

/**
 * @brief Write a dense Eigen matrix to file
 * @param M
 * @param filename
 */
void writeMatrixToFile(const Matrix &M, const std::string &filename);

/**
 * @brief Write a sparse matrix to file
 * @param M
 * @param filename
 */
void writeSparseMatrixToFile(const SparseMatrix &M, const std::string &filename);

/**
Helper function to read a dataset in .g2o format
*/
std::vector<RelativeSEMeasurement> read_g2o_file(const std::string &filename,
                                                 size_t &num_poses);

/**
 * @brief
 * @param measurements
 * @param dimension
 * @param num_poses
 */
void get_dimension_and_num_poses(const std::vector<RelativeSEMeasurement> &measurements,
                                 size_t &dimension,
                                 size_t &num_poses);

/**
Helper function to construct connection laplacian matrix in SE(d)
*/
void constructOrientedConnectionIncidenceMatrixSE(
    const std::vector<RelativeSEMeasurement> &measurements, SparseMatrix &AT,
    DiagonalMatrix &OmegaT);

/**
Helper function to construct connection laplacian matrix in SE(d)
*/
SparseMatrix constructConnectionLaplacianSE(
    const std::vector<RelativeSEMeasurement> &measurements);

/**
Given a vector of relative pose measurements, this function computes and returns
the B matrices defined in equation (69) of the tech report
*/
void constructBMatrices(const std::vector<RelativeSEMeasurement> &measurements,
                        SparseMatrix &B1, SparseMatrix &B2, SparseMatrix &B3);

/**
Given the measurement matrices B1 and B2 and a matrix R of rotational state
estimates, this function computes and returns the corresponding optimal
translation estimates
*/
Matrix recoverTranslations(const SparseMatrix &B1, const SparseMatrix &B2,
                           const Matrix &R);

/**
Project a given matrix to the rotation group
*/
Matrix projectToRotationGroup(const Matrix &M);

/**
 * @brief project an input matrix M to the Stiefel manifold
 * @param M
 * @return orthogonal projection of M to Stiefel manifold
 */
Matrix projectToStiefelManifold(const Matrix &M);

/**
  Generate a fixed element of the Stiefel element
  The returned value is guaranteed to be the same for each d and r
*/
Matrix fixedStiefelVariable(unsigned d, unsigned r);

/**
 * @brief Generate a random element of the Stiefel manifold
 * @param d
 * @param r
 * @return
 */
Matrix randomStiefelVariable(unsigned d, unsigned r);

/**
 * @brief Compute the error term (weighted squared residual)
 * @param m measurement
 * @param R1 rotation of first pose
 * @param t1 translation of first pose
 * @param R2 rotation of second pose
 * @param t2 translation of second pose
 * @return
 */
double computeMeasurementError(const RelativeSEMeasurement &m,
                               const Matrix &R1, const Matrix &t1,
                               const Matrix &R2, const Matrix &t2);

/**
 * @brief Quantile of chi-squared distribution with given degrees of freedom at probability alpha.
 * Equivalent to chi2inv in Matlab.
 * @param quantile
 * @param dof
 * @return
 */
double chi2inv(double quantile, size_t dof);

/**
 * @brief For SO(3), convert angular distance in radian to chordal distance
 * @param rad input angular distance in radian
 * @return
 */
double angular2ChordalSO3(double rad);

/**
 * @brief Verify that the input matrix is a valid rotation
 * @param R
 */
void checkRotationMatrix(const Matrix &R);

/**
 * @brief Check that the input matrix of dimension r-by-d is a valid element of the Stiefel manifold
 * @param Y
 */
void checkStiefelMatrix(const Matrix &Y);

}  // namespace DCORA

#endif