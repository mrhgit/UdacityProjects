#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
size_t N = 12;
double dt = 0.05; // seconds.

const size_t n_state = 6;
const size_t n_actuator = 2;

const size_t x_start = 0;
const size_t y_start = x_start + N;
const size_t psi_start = y_start + N;
const size_t v_start = psi_start + N;
const size_t cte_start = v_start + N;
const size_t epsi_start = cte_start + N;
const size_t delta_start = epsi_start + N;
const size_t a_start = delta_start + N - 1;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;
const double Lfinv = 1.0/Lf;

const double ref_v = 30.0;// / 3600.0 * 1609.344;
class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // This operator() (BLANK) function is where the COST and CONSTRAINTS calculations go!
    //   cost = the error to minimize.  It is stored in fg[0], so ADD to it.
    //   constraints = the Model.  That is, we will be calculating updated positions/vel/orientation in this section
    //   and constraining the one side of the equation minus the other side of the equation to zero.
  
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.

    // There are N sets of reference states, each representing a step in time.
    //  We throw together all the error terms in fg[0].
    //  Similar to TensorFlow or other custom frameworks, we have to play with their special
    //    functions when we play in their sandbox... ex:  pow(x,y) -> CppAD::pow(x,y)
    fg[0] = 0;
    for (size_t t=0; t < N; ++t){
        fg[0] += 0.5*CppAD::pow(vars[cte_start + t],2); // cross-track error
        fg[0] += CppAD::pow(vars[epsi_start + t], 2); // heading error
        fg[0] += 0.5*CppAD::pow(vars[v_start + t] - ref_v, 2); // here ref_v is the target speed. this error ensures a non-zero-speed solution
    }
    
    // There are (N-1) sets of actuator values, each representing a step in time.
    //  We throw together all the error terms in fg[0]
    for (size_t t=0; t < N-1; ++t){
        fg[0] += 800*CppAD::pow(vars[delta_start + t], 2); // minimize the steering we'll do
        fg[0] += CppAD::pow(vars[a_start + t], 2); // minimize the acceleration necessary
    }
    
    // There are (N-2) gaps between sets of actuator values, each representing a step in time.
    //  We throw together all the error terms in fg[0]
    /*for (size_t t=0; t < N-2; ++t){
        fg[0] += CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2); // smooth out steering
        fg[0] += CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2); // minimize sudden acceleration changes (aka "jerk")
    }*/

    // Transfer initial state over directly    
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // Calculate the states for time 1 -> N (0 is initial state)
    for (size_t t=1; t < N; ++t){
        AD<double> x1 = vars[x_start + t];
        AD<double> y1 = vars[y_start + t];
        AD<double> psi1 = vars[psi_start + t];
        AD<double> v1 = vars[v_start + t];
        AD<double> cte1 = vars[cte_start + t];
        AD<double> epsi1 = vars[epsi_start + t];

        AD<double> x0 = vars[x_start + t - 1]; // previous x
        AD<double> y0 = vars[y_start + t - 1]; // previous y
        AD<double> psi0 = vars[psi_start + t - 1]; // previous psi
        AD<double> v0 = vars[v_start + t - 1]; // previous v
        AD<double> cte0 = vars[cte_start + t - 1]; // previous cte
        AD<double> epsi0 = vars[epsi_start + t - 1]; // previous epsi

        AD<double> delta0 = vars[delta_start + t - 1]; // previous delta
        AD<double> a0 = vars[a_start + t - 1]; // previous a

        
        AD<double> f0 = coeffs[0] + x0 * (coeffs[1]  + x0 * (coeffs[2] + x0*coeffs[3])); // the y position of the line based on the waypoints poly
        // 3rd order poly, ax^3 + bx^2 + cx + d = y
        // 1st derivative is 3*a*x^2 + 2*b*x + c = y'
        AD<double> psi_des0 = CppAD::atan(coeffs[1] + x0 * (2*coeffs[2] + x0*3*coeffs[3])); // the tangent to the polynomial at x - this is our desired psi
        
        fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt); // 0 = x1 - (...)  --> x1 = (...)
        fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
        fg[1 + psi_start + t] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
        fg[1 + v_start + t] = v1 - (v0 + a0 * dt);
        fg[1 + cte_start + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
        fg[1 + epsi_start + t] = epsi1 - ((psi_des0 - psi0) + v0 * delta0 / Lf * dt); // TODO check if (psi_des0 - psi0)
        
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  // state === initial state (after actuation latency)
  // coeffs === fitted polynomial coefficients of the 3rd order curve representing the desired waypoints
  
  bool ok = true;
  //size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;
  
  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];
  // TODO: Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9
  // The four state variables are:  x, y, orientation, speed
  // The two actuator variables are:  steering, acceleration
  size_t n_vars = n_state * N + n_actuator * (N-1);
  // TODO: Set the number of constraints
  size_t n_constraints = N * n_state;
  

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (size_t i = 0; i < n_vars; i++) {
    vars[i] = 0.0;
  }
  
  // Each state variable is repeated N times individually.  Therefore, initial state variable initialization must
  // step by N in order to initialize in the correct locations.
  /*for (size_t i = 0; i < n_state; i++) {
    vars[N*i] = state[i];
  }*/
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // TODO: Set lower and upper limits for variables.
  // First, set all to unbounded
  for (size_t i = 0; i < delta_start; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }
  // Now, set all steering limitations to -25 degrees to +25 degrees
  //double delta_bound = 0.436332; //25 * pi / 180;
  for (size_t i = delta_start; i < a_start; ++i) {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] =  0.436332;
  }
  // Now, set all acceleration limitations to +/-1
  for (size_t i = a_start; i < n_vars; ++i) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] =  1.0;
  }
  

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (size_t i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;
  
  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;
  

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.
  vector<double> solution_vec(n_state + n_actuator);
  solution_vec[0] = solution.x[x_start + 1];
  solution_vec[1] = solution.x[y_start + 1];
  solution_vec[2] = solution.x[psi_start + 1];
  solution_vec[3] = solution.x[v_start + 1];
  solution_vec[4] = solution.x[cte_start + 1];
  solution_vec[5] = solution.x[epsi_start + 1];
  solution_vec[6] = solution.x[delta_start + 1];
  solution_vec[7] = solution.x[a_start + 1];
  
  return solution_vec;
    
}
