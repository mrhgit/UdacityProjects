#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

const size_t n_state = 6;
const size_t n_actuator = 2;
const double Lf = 2.67;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"]; // global x pos of the waypoints
          vector<double> ptsy = j[1]["ptsy"]; // global y pos of the waypoints
          double px = j[1]["x"]; // global x pos of vehicle
          double py = j[1]["y"]; // global y pos of vehicle
          double psi = j[1]["psi"]; // orientation of vehicle in radians
          //double psi_unity = j[1]["psi_unity"]; // orientation of vehicle in radians
          double v = j[1]["speed"]; // velocity in mph
          double a = j[1]["throttle"]; // velocity in mph
          double str = j[1]["steering_angle"];
          //v = v / 3600.0 * 1609.344; // * 1hr/3600 sec * 1609.344 meters/mile

          //Display the waypoints/reference line (Yellow line in the simulator)
          Eigen::VectorXd next_x_vals;
          Eigen::VectorXd next_y_vals;
          
          next_x_vals.resize(ptsx.size());
          next_y_vals.resize(ptsx.size());
          for (size_t i=0; i < ptsx.size(); ++i){
            double dx = ptsx[i] - px;
            double dy = ptsy[i] - py;
            double r = sqrt(dx*dx + dy*dy); // range from vehicle to points
            double theta_mc = atan2(dy,dx); // angle from vehicle to points (map coordinates)
            double theta_vc = theta_mc - psi; // angle from vehicle to points (vehicle coordinates)
            
            next_x_vals[i] = r*cos(theta_vc); // x-value of points (vehicle coordinates)
            next_y_vals[i] = r*sin(theta_vc); // y-value of points (vehicle coordinates)
          }
          
          // ALL CALCULATIONS FROM HERE ON WILL BE IN THE VEHICLE-COORDINATE SYSTEM (straight = +x-axis, left turn = +y-axis)
          
          // Generalize the waypoints as a 3rd-Order polynomial
          Eigen::VectorXd coeffs = polyfit(next_x_vals, next_y_vals, 3);
          //cout << "Poly Coeffs = " << coeffs << endl;
          

          // Advance the position of the car by the actuator latency (100 mS)
          //   this will be the earliest we can affect the course of events
          double latency = 0.100; // 100 milliSeconds
          double f0 = coeffs[0];
          double psi_des0 = atan(coeffs[1]);
          double epsi0 = psi_des0;
          double delta0 = str;
          
          Eigen::VectorXd latent_state(n_state);
          latent_state[0] = v * latency * cos(0); // x advances at angle & speed
          latent_state[1] = v * latency * sin(0); // y advances at angle & speed
          latent_state[2] = 0 + latency*v/Lf*delta0; // steering advances
          latent_state[3] = v + a*latency; // velocity
          latent_state[4] = (f0 - 0) + v * sin(epsi0) * latency; // cte
          latent_state[5] = (psi_des0 - 0) + v*delta0 / Lf * latency;
          
          /*
          * Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */
          MPC mpc;
          vector<double> solution = mpc.Solve(latent_state, coeffs);
          double steer_value = solution[6];//-0.05;
          double throttle_value = solution[7];//0.25;
          //cout << "Steering = " << steer_value << " Throttle = " << throttle_value << endl;

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = -steer_value / 0.436332; // 0.436332 = deg2rad(25)
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory (Green line in the simulator)
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          mpc_x_vals.resize(8);
          mpc_y_vals.resize(8);

          double tstep = 0.2;
          for (size_t i=0; i < 8; ++i){
            //double y = polyeval(coeffs, i*5.0);
            double t = i*tstep;
            double v1 = v + throttle_value * t;
            double orientation = t*v1/Lf*steer_value;
            mpc_x_vals[i] = v1 * t * cos(orientation);
            mpc_y_vals[i] = v1 * t * sin(orientation);
          }
          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line
          vector<double> next_x_vd(next_x_vals.size());
          vector<double> next_y_vd(next_y_vals.size());
          for (int i=0; i < next_x_vals.size(); ++i){
            next_x_vd[i] = next_x_vals[i];
            next_y_vd[i] = next_y_vals[i];
          }
          
          msgJson["next_x"] = next_x_vd;
          msgJson["next_y"] = next_y_vd;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
