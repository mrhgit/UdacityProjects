#include "PID.h"
#include "math.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    p_error = 0;
    i_error = 0; // reset integral error to 0
    this->Kp = Kp;
    this->Ki = Ki;
    this->Kd = Kd;
    first_run = true;
}

void PID::UpdateError(double cte) {
    if (first_run) {
    //if (fabs(i_error) < 0.0001) { // is this the first run?
        d_error = 0;
        first_run = false;
    }
    else {
        d_error = cte - p_error;
    }
    i_error += cte;
    p_error = cte; // store this cte for next time
}

double PID::TotalError() {
    return Kp * p_error + Ki * i_error + Kd * d_error; // this is the actual error - will be negated by the time it reaches steering
}

