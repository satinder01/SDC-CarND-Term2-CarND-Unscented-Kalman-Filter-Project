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
  rmse << 0,0,0,0;

  if(estimations.size() == 0) {
    cout << "estimation size is zero estimations.size()"<< endl;
    return rmse;
  }
        
  if(estimations.size() != ground_truth.size()){
    cout << "estimation and groud_truth size mismatch" << endl;
    return rmse;
  }


  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){
    VectorXd residuals = estimations[i] - ground_truth[i];
    residuals = residuals.array()*residuals.array();
    rmse += residuals;
  }

  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();
  return rmse;

}
