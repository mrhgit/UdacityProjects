#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
  rmse.setZero();

  int n = estimations.size();
  int m = ground_truth.size();

  if (n==0 || m==0) return rmse;
  if (n != m) return rmse;
  if (estimations[0].size() != ground_truth[0].size()) return rmse;
  
  VectorXd sum_sqr_res(4);
  sum_sqr_res.setZero();
  for (int i=0; i < n; ++i){
    VectorXd sqr_res(4);
    sqr_res = (estimations[i] - ground_truth[i]).array().pow(2);
    sum_sqr_res += sqr_res;
  }
  VectorXd mse = 1./n * sum_sqr_res.array();

  rmse = mse.array().sqrt();

  return rmse;
}
