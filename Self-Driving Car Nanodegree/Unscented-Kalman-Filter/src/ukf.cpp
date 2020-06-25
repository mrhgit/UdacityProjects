#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

double fix_angle(double ang_in){
    double pi = 3.1415926535;
    while (ang_in > pi) ang_in -= 2.*pi;
    while (ang_in < -pi) ang_in += 2.*pi;
    return ang_in;
}


/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5; // ADJUSTED

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.25; // ADJUSTED
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  n_x_ = 5;
  n_aug_ = 7;

  previous_timestamp_ = 0;
}

UKF::~UKF() {}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

  //define spreading parameter
  double lambda = 3 - n_aug_;

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1); // one at mean and 2 per state dimension

  int n_q = n_aug_ - n_x_;
  MatrixXd Q = MatrixXd(n_q, n_q);
  Q << pow(std_a_,2) , 0, 0, pow(std_yawdd_,2); // assume zero-covariance between noise

  P_aug.setZero();
  P_aug.block(0,0,n_x_,n_x_) = P_;  // upper-left block is the original P uncertainty
  P_aug.block(n_x_,n_x_,n_q,n_q) = Q; // lower-right block is the noise Q

  //create augmented mean state
  x_aug << x_.array(), 0, 0; // assume noises have zero means

  //create augmented covariance matrix

  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  A *= sqrt(lambda + n_aug_);

  //create augmented sigma points
  Xsig_aug = x_aug.rowwise().replicate(2*n_aug_ + 1);
  Xsig_aug.block(0,1,n_aug_,n_aug_) += A;
  Xsig_aug.block(0,1+n_aug_,n_aug_,n_aug_) -= A;

  //write result
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd* Xsig_aug, MatrixXd* Xsig_out, double delta_t) {
  // CTRV prediction

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //predict sigma points
  Xsig_pred.block(0, 0, n_x_,2 * n_aug_ + 1) = Xsig_aug->block(0, 0, n_x_,2 * n_aug_ + 1);
  
  for (int i=0; i < Xsig_aug->cols(); ++i){ // for each augmented sigma point
      double px = (*Xsig_aug)(0,i);
      double py = (*Xsig_aug)(1,i);
      double v  = (*Xsig_aug)(2,i);
      double psi = (*Xsig_aug)(3,i);
      double psi_dot = (*Xsig_aug)(4,i);
      double nu_a = (*Xsig_aug)(5,i);
      double nu_psidd = (*Xsig_aug)(6,i);
      
      double cos_psi = cos(psi);
      double sin_psi = sin(psi);
      double delta_t2 = pow(delta_t,2);
      
      // Add deterministic state progression
      if (fabs(psi_dot) < 1e-6){
          Xsig_pred(0,i) += v * cos_psi * delta_t;
          Xsig_pred(1,i) += v * sin_psi * delta_t;
          //Xsig_pred(3,i) += psi_dot * delta_t; // This will be zero
      }
      else {
          Xsig_pred(0,i) += v / psi_dot * (sin(psi + psi_dot*delta_t) - sin_psi);
          Xsig_pred(1,i) += v / psi_dot * (-cos(psi + psi_dot*delta_t) + cos_psi);
          Xsig_pred(3,i) += psi_dot * delta_t;
      }
      //Xsig_pred(2,i) += 0; // CTRV model
      //Xsig_pred(4,i) += 0; // CTRV model
      
      // Add process noise
      double half_delta_t2 = 0.5 * pow(delta_t,2);
      Xsig_pred(0,i) += half_delta_t2 * cos_psi * nu_a;
      Xsig_pred(1,i) += half_delta_t2 * sin_psi * nu_a;
      Xsig_pred(2,i) += delta_t * nu_a;
      Xsig_pred(3,i) += half_delta_t2 * nu_psidd;
      Xsig_pred(4,i) += delta_t * nu_psidd;
  }
  
  //write result
  *Xsig_out = Xsig_pred;

}


void UKF::PredictMeanAndCovariance(MatrixXd* Xsig_pred, VectorXd* x_out, MatrixXd* P_out) {

  //define spreading parameter
  double lambda = 3 - n_aug_;

  //create vector for predicted state
  VectorXd x_pred = VectorXd(n_x_);

  //create covariance matrix for prediction
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);

  //set weights
  double w0 = lambda / (lambda + n_aug_);
  double wn = 0.5 / (lambda + n_aug_);
  
  //predict state mean
  x_pred = w0 * Xsig_pred->col(0);
  for (int i=1; i < Xsig_pred->cols(); ++i)
      x_pred += wn * Xsig_pred->col(i);

  //predict state covariance matrix
  MatrixXd pred_error = *Xsig_pred - x_pred.rowwise().replicate(Xsig_pred->cols());
  
  pred_error(3,0) = fix_angle(pred_error(3,0));
  P_pred = w0 * (pred_error.col(0) * pred_error.col(0).transpose());
  for (int i=1; i < Xsig_pred->cols(); ++i){
      pred_error(3,i) = fix_angle(pred_error(3,i));
      P_pred += wn * (pred_error.col(i) * pred_error.col(i).transpose());
  }

  //write result
  *x_out = x_pred;
  *P_out = P_pred;
}

void UKF::PredictRadarMeasurement(MatrixXd* Xsig_pred, VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out) {

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //define spreading parameter
  double lambda = 3 - n_aug_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig_pred = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S_pred = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  for (int i=0; i < Xsig_pred->cols(); ++i){
      double px = (*Xsig_pred)(0,i);
      double py = (*Xsig_pred)(1,i);
      double v = (*Xsig_pred)(2,i);
      double psi = (*Xsig_pred)(3,i);
      double psi_dot = (*Xsig_pred)(4,i);
      
      double rho = sqrt(pow(px,2) + pow(py,2));
      double phi = atan2(py,px);
      double rho_dot = v*(px*cos(psi) + py*sin(psi))/ rho;
      Zsig_pred(0,i) = rho;
      Zsig_pred(1,i) = phi;
      Zsig_pred(2,i) = rho_dot;
  }
  //calculate mean predicted measurement
  double w0 = lambda / (lambda + n_aug_);
  double wn = 0.5 / (lambda + n_aug_);
  
  z_pred = w0 * Zsig_pred.col(0);
  for (int i=1; i < Zsig_pred.cols(); ++i)
      z_pred += wn * Zsig_pred.col(i);
  
  //calculate innovation covariance matrix S
  MatrixXd Z_diff = Zsig_pred - z_pred.rowwise().replicate(Zsig_pred.cols());
  for (int i=0; i < Z_diff.cols(); ++i)
    Z_diff(1,i) = fix_angle(Z_diff(1,i));
  
  S_pred = w0 * (Z_diff.col(0) * Z_diff.col(0).transpose());
  for (int i=1; i < Zsig_pred.cols(); ++i)
    S_pred += wn * (Z_diff.col(i) * Z_diff.col(i).transpose());
    
  // Add Radar Noise - R (additive noise)
  MatrixXd R(n_z,n_z);
  R << pow(std_radr_,2) , 0, 0,
       0, pow(std_radphi_,2), 0,
       0, 0, pow(std_radrd_,2);
  S_pred += R;
  
  //write result
  *z_out = z_pred;
  *S_out = S_pred;
  *Zsig_out = Zsig_pred;
}

void UKF::PredictLidarMeasurement(MatrixXd* Xsig_pred, VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out) {

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 2;

  //define spreading parameter
  double lambda = 3 - n_aug_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig_pred = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S_pred = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  Zsig_pred.block(0,0,2,Xsig_pred->cols()) = Xsig_pred->block(0,0,2,Xsig_pred->cols()); // px -> px, py -> py

  //calculate mean predicted measurement
  double w0 = lambda / (lambda + n_aug_);
  double wn = 0.5 / (lambda + n_aug_);
  
  z_pred = Zsig_pred.col(0) * w0;
  for (int i=1; i < Zsig_pred.cols(); ++i)
      z_pred += Zsig_pred.col(i) * wn;
  
  //calculate innovation covariance matrix S
  MatrixXd meas_error = Zsig_pred - z_pred.rowwise().replicate(Zsig_pred.cols());
  
  S_pred = w0 * (meas_error.col(0) * meas_error.col(0).transpose());
  for (int i=1; i < Zsig_pred.cols(); ++i)
    S_pred += wn * (meas_error.col(i) * meas_error.col(i).transpose());
    
  // Add Radar Noise - R (additive noise)
  MatrixXd R(n_z,n_z);
  R << pow(std_laspx_,2), 0,
       0, pow(std_laspy_,2);
  S_pred += R;
  
  //write result
  *z_out = z_pred;
  *S_out = S_pred;
  *Zsig_out = Zsig_pred;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MatrixXd *Zsig_pred, VectorXd *z_pred, MatrixXd *S_pred, VectorXd *z_true, VectorXd* x_pred, MatrixXd* P_pred) {
  /**
  Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //define spreading parameter
  double lambda = 3 - n_aug_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  double w0 = lambda / (lambda + n_aug_);
  double wn = 0.5 / (lambda + n_aug_);
  VectorXd z_diff = Zsig_pred->col(0) - *z_pred;
  VectorXd x_diff = Xsig_pred_.col(0) - *x_pred; //x_;
  z_diff(1) = fix_angle(z_diff(1));
  x_diff(3) = fix_angle(x_diff(3));
  x_diff(4) = fix_angle(x_diff(4));
  Tc = w0 * (x_diff * z_diff.transpose());
  for (int i=1; i < Xsig_pred_.cols(); ++i){
      z_diff = Zsig_pred->col(i) - *z_pred;
      z_diff(1) = fix_angle(z_diff(1));

      x_diff = Xsig_pred_.col(i) - *x_pred;//x_;
      x_diff(3) = fix_angle(x_diff(3));
      x_diff(4) = fix_angle(x_diff(4));

      Tc += wn * (x_diff * z_diff.transpose());
  }
  
  //calculate Kalman gain K;
  MatrixXd K(n_x_, n_z);
  MatrixXd S_pred_inv;
  S_pred_inv = S_pred->inverse();
  K = Tc * S_pred_inv;
  
  //update state mean and covariance matrix
  VectorXd z_err;
  z_err = *z_true - *z_pred;
  z_err(1) = fix_angle(z_err(1));

  x_ = *x_pred + K * z_err;
  x_(3) = fix_angle(x_(3));
  x_(4) = fix_angle(x_(4));
  P_ = *P_pred - K * (*S_pred) * K.transpose();

  // Check NIS (Normalized Innovation Squared)
  double epsilon = z_err.transpose() * S_pred_inv * z_err;
  if (epsilon > 7.815)
    num_above_NIS95_++;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MatrixXd *Zsig_pred, VectorXd *z_pred, MatrixXd *S_pred, VectorXd *z_true, VectorXd* x_pred, MatrixXd* P_pred) {
  /**
  Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  //set measurement dimension, lidar measure px, py
  int n_z = 2;

  //define spreading parameter
  double lambda = 3 - n_aug_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  double w0 = lambda / (lambda + n_aug_);
  double wn = 0.5 / (lambda + n_aug_);
  VectorXd z_diff = (*Zsig_pred).col(0) - *z_pred;
  Tc = w0 * ((Xsig_pred_.col(0) - *x_pred) * z_diff.transpose());
  for (int i=1; i < Xsig_pred_.cols(); ++i){
      z_diff = (*Zsig_pred).col(i) - *z_pred;
      Tc += wn * ((Xsig_pred_.col(i) - *x_pred) * z_diff.transpose());
  }
  
  //calculate Kalman gain K;
  MatrixXd K(n_x_, n_z);
  MatrixXd S_pred_inv;
  S_pred_inv = S_pred->inverse();
  K = Tc * S_pred_inv;
  
  //update state mean and covariance matrix
  VectorXd z_err;
  z_err = *z_true - *z_pred;

  x_ = *x_pred + K * z_err;
  x_(3) = fix_angle(x_(3));
  x_(4) = fix_angle(x_(4));
  P_ = *P_pred - K * (*S_pred) * K.transpose();

  // Check NIS (Normalized Innovation Squared)
  double epsilon = z_err.transpose() * S_pred_inv * z_err;
  if (epsilon > 5.991)
    num_above_NIS95_++;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1.0e6;
  if (fabs(dt) > 5) {
    cout << "Timestamp jump detected - resetting (dt="<<dt<<" sec.)" << endl;
    x_.setZero(); // we have no idea about our state
    P_.setIdentity();
    P_ *= 0.5;
    P_(4,4) = 0.1;
    P_(3,3) = 0.1;
    num_meas_ = 0;
    num_above_NIS95_ = 0;
    is_initialized_ = false;
  }

  if ((measurement_pack.sensor_type_ == MeasurementPackage::RADAR) && (!use_radar_)){
      cout << "Skipping measurement because RADAR processing is disabled" << endl;
      return;
  }
  if ((measurement_pack.sensor_type_ == MeasurementPackage::LASER) && (!use_laser_)){
      cout << "Skipping measurement because LIDAR processing is disabled" << endl;
      return;
  }

  num_meas_++;

  previous_timestamp_ = measurement_pack.timestamp_;

  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    x_ = VectorXd(n_x_);
    x_.setZero(); // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    float px,py,vabs,yaw_angle,yaw_rate; // our state
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float range, bearing, drange_dt; // our measurements
      range = measurement_pack.raw_measurements_[0];
      bearing = measurement_pack.raw_measurements_[1];
      drange_dt = measurement_pack.raw_measurements_[2];

      float cos_b = cos(bearing);//*3.14159265/180);
      float sin_b = sin(bearing);//*3.14159265/180);

      px = range * cos_b;
      py = range * sin_b;
      vabs = 0; // RADAR doesn't give us the velocity we're storing here
      yaw_angle = 0; // RADAR doesn't give us object yaw
      yaw_rate = 0; // RADAR doesn't give us object yaw rate
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      px = measurement_pack.raw_measurements_[0]; // px measured directly
      py = measurement_pack.raw_measurements_[1]; // py measured directly
      vabs = 0; // LIDAR doesn't give us velocity
      yaw_angle = 0; // LIDAR doesn't give us object yaw
      yaw_rate = 0; // LIDAR doesn't give us object yaw rate
    }
    x_ << px, py, vabs, yaw_angle, yaw_rate;


    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  MatrixXd Xsigma_points;
  VectorXd x_pred;
  MatrixXd P_pred;

  // Because prediction is no longer a linear process, we approximate our new mean and covariance using
  // "sigma points," points chosen along each dimension of the state and spread out by some multiple (or fraction)
  // of the standard deviation along that dimension.
  AugmentedSigmaPoints(&Xsigma_points);
  SigmaPointPrediction(&Xsigma_points, &Xsig_pred_, dt); // predict those augmented points in state (k+1)
  PredictMeanAndCovariance(&Xsig_pred_, &x_pred, &P_pred); // Calculate Mean = x (pred. state), Covar = P (pred. uncertainty matrix)

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if ((measurement_pack.sensor_type_ == MeasurementPackage::RADAR) && (use_radar_)) {
    // Predict what the measurement is - a non-linear prediction
    VectorXd z_pred;
    MatrixXd S_pred;
    MatrixXd Zsig_pred;
    PredictRadarMeasurement(&Xsig_pred_, &z_pred, &S_pred, &Zsig_pred);

    // Get the true measurement
    VectorXd z_true = VectorXd(3);
    z_true << measurement_pack.raw_measurements_[0] , // range
              measurement_pack.raw_measurements_[1] , // bearing
              measurement_pack.raw_measurements_[2];  // d(range)/dt
    //z_true(1) *= 3.14159265/180; // DO NOT Convert bearing to radians from degrees

    // Use the difference to calc Kalman gain, use Kalman gain to update state
    UpdateRadar(&Zsig_pred, &z_pred, &S_pred, &z_true, &x_pred, &P_pred);
  }

  if ((measurement_pack.sensor_type_ == MeasurementPackage::LASER) && (use_laser_)) {
    // LIDAR updates

    // Predict what the measurement is - a non-linear prediction
    VectorXd z_pred;
    MatrixXd S_pred;
    MatrixXd Zsig_pred;
    PredictLidarMeasurement(&Xsig_pred_, &z_pred, &S_pred, &Zsig_pred);

    // Get the true measurement
    VectorXd z_true = VectorXd(2);
    z_true << measurement_pack.raw_measurements_[0] , // px
              measurement_pack.raw_measurements_[1];  // py

    // Use the difference to calc Kalman gain, use Kalman gain to update state
    UpdateLidar(&Zsig_pred, &z_pred, &S_pred, &z_true, &x_pred, &P_pred);
 }

  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
  cout << "num_meas = " << num_meas_ << " num_above_NIS95 = "
       << num_above_NIS95_ << " containment = " << 100.0-100.0*num_above_NIS95_/num_meas_ << "%" << endl;

}

