#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  Tools tools;

  noise_ax = 9; // process noise - acceleration "velocity noise" in x
  noise_ay = 9; // process noise - acceleration "velocity noise" in y

  VectorXd x(4); // our state - px,py,vx,vy position and velocity in x,y directions
  x.setZero();

  MatrixXd P(4, 4); // uncertainty covariance - initialize to big values
  P << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1000, 0,
       0, 0, 0, 1000;

  MatrixXd F(4,4);
  F.setZero(); // need delta time to populate

  H_laser_ << 1, 0, 0, 0, // our laser measurement is px and py from our state vector px,py,vx,vy
              0, 1, 0, 0;

  Hj_.setZero(); // need the actual state to calculate this, since it is NOT a linear state->meas mapping

  MatrixXd Q(4,4);
  Q.setZero(); // need delta time to populate

  ekf_.Init(x,P,F,H_laser_,R_laser_,Q);


}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::UpdateJacobian(){
  Hj_ = tools.CalculateJacobian(ekf_.x_);
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
// These lines allow experimenting with "turning off" a certain type of sensor.
//if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) return; // turn off LIDAR
//if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) return; // turn off RADAR


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1; // px,py,vx,vy
    previous_timestamp_ = measurement_pack.timestamp_;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float px,py,vx,vy;
      float range, bearing, drange_dt;
      range = measurement_pack.raw_measurements_[0];
      bearing = measurement_pack.raw_measurements_[1];
      drange_dt = measurement_pack.raw_measurements_[2];
      float cos_b = cos(bearing*3.14159265/180);
      float sin_b = sin(bearing*3.14159265/180);
      px = range * cos_b;
      py = range * sin_b;
      vx = 0;//drange_dt * cos_b;
      vy = 0;//drange_dt * sin_b;
      ekf_.x_ << px, py, vx, vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      float px,py,vx,vy;
      px = measurement_pack.raw_measurements_[0]; // px measured directly
      py = measurement_pack.raw_measurements_[1]; // py measured directly
      vx = 0; // LIDAR doesn't give us velocity
      vy = 0; // LIDAR doesn't give us velocity
      ekf_.x_ << px, py, vx, vy;
   }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1.0e6;
  previous_timestamp_ = measurement_pack.timestamp_;


  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  // update F (state transition matrix) to reflect delta time - position accumlates over time as a function of velocity

  ekf_.F_ << 1,0,dt, 0,
             0,1, 0,dt,
             0,0, 1, 0,
             0,0, 0, 1;

  float dt2 = pow(dt,2.);
  float dt3 = dt2*dt/2; // = pow(dt,3.)/2;
  float dt4 = dt3*dt/2; // = pow(dt,4.)/4;
  float sx = noise_ax;
  float sy = noise_ay;

  // update Q to reflect delta time - noise accumulates over time as a function of process noise (acceleration, in this case)
  ekf_.Q_ << dt4*sx, 0, dt3*sx, 0,
        0, dt4*sy, 0, dt3*sy,
        dt3*sx, 0, dt2*sx, 0,
        0, dt3*sy, 0, dt2*sy;

  ekf_.Predict(); // our process model is linear, so we can use same approach for classic and extended KF

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */


  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    VectorXd z = VectorXd(3);
    z << measurement_pack.raw_measurements_[0] , measurement_pack.raw_measurements_[1] , measurement_pack.raw_measurements_[2];

    UpdateJacobian(); // Update Hj_
    ekf_.H_ = Hj_; // set the model H to our Hj (radar Jacobian)
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(z);
  } else {
    // Laser updates
    VectorXd z = VectorXd(2);
    z << measurement_pack.raw_measurements_[0] , measurement_pack.raw_measurements_[1];
    ekf_.H_ = H_laser_; // set the model H to our H_laser
    ekf_.R_ = R_laser_;
    ekf_.Update(z);
 }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
