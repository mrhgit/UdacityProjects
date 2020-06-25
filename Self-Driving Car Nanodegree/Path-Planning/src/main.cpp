#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Dense"
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include <cmath>
#include "spline.h"
// #include "Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

#define MAX_S  6945.554
#define MPH2MPS 0.44704
#define CLEARLANE_TGTVEL_MPH 49.7
#define CLEARLANE_TGTVEL_MPS CLEARLANE_TGTVEL_MPH*MPH2MPS
#define CAR_MVMNT_TIME 0.020
#define LANE_CHANGE_TIME 2.5
#define N (LANE_CHANGE_TIME/CAR_MVMNT_TIME)

#define MAX_JERK 9.0
#define CRUISE_JERK 6.0
#define MAX_ACCEL 9.0
#define CRUISE_ACCEL 0.75

#define LANE_WIDTH 4.0
#define CAR_LENGTH 2.5

#define S_MARGIN 15
#define D_MARGIN LANE_WIDTH*0.25
#define TAILGATE_MARGIN 20
#define LANECHANGE_LOOKAHEAD_DISTANCE 3*TAILGATE_MARGIN
#define LEFT_LANE 0
#define CENTER_LANE 1
#define RIGHT_LANE 2

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
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double S1MinusS2(double s1, double s2){
    double dist;
    if (s1 > MAX_S * .75) // if s1 is in the final part of the loop
        if (s2 < MAX_S * .25){ // is s2 is in the first part of the loop
            dist = s1 - (s2 + MAX_S);
            return dist;
        }

    if (s2 > MAX_S * .75) // if s2 is in the final part of the loop
        if (s1 < MAX_S * .25){ // is s1 is in the first part of the loop
            dist = (s1 + MAX_S) - s2;
            return dist;
        }

    dist = s1 - s2;
    return dist;
}

bool S1GreaterThanS2(double s1, double s2){
    double dist = S1MinusS2(s1,s2);
    bool isgreater = dist > 0;
    return isgreater;
}

double magnitude(double dx, double dy){
    return sqrt(dx*dx + dy*dy);
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2((map_y-y),(map_x-x));

	double angle = fabs(theta-heading);
  angle = min(2*pi() - angle, angle);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  if (closestWaypoint == maps_x.size())
  {
    closestWaypoint = 0;
  }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y,bool use_spline=true)
{
    static tk::spline s_theta, s_r;
    static bool first_run = true;
    static int end = maps_s.size()-1;
    //static double max_s = maps_s[end] + distance(maps_x[end],maps_y[end],maps_x[0],maps_y[0]);

    if (use_spline){
        if (first_run) {
            vector<double> theta_vec, r_vec, s_vec;
            
            for (int i=0; i < maps_x.size(); i++){
                double r = magnitude(maps_x[i],maps_y[i]); // distance from center
                double theta = atan2(maps_y[i],maps_x[i]); // angle from 0
                
                s_vec.push_back(maps_s[i]);
                r_vec.push_back(r);
                theta_vec.push_back(theta);
            }
            // repeat the first two points (two because it's a cubic spline) so we can close the curve
            
            s_vec.push_back(s_vec[0] + MAX_S);
            s_vec.push_back(s_vec[1] + MAX_S);
            r_vec.push_back(r_vec[0]);
            r_vec.push_back(r_vec[1]);
            theta_vec.push_back(theta_vec[0]);
            theta_vec.push_back(theta_vec[1]);
            
            //for (int i=0; i < s_vec.size(); i++){
            //    cout << "s = " << s_vec[i] << " r = " << r_vec[i] << " theta = " << theta_vec[i] << endl;
            //}
            
            s_theta.set_points(s_vec,theta_vec); // checked - do not need to "unwrap" theta before sending it here
            s_r.set_points(s_vec,r_vec);
            
            first_run = false;
        }
        
        while (s > MAX_S) s -= MAX_S;
        
        // First we get the r and theta using s and the splines we made
        double r = s_r(s);
        double theta = s_theta(s);
        
        double x = r*cos(theta);
        double y = r*sin(theta);
        
        // find tangent line and shift 'd' distance along perpendicular
        double next_r = s_r(s + 0.01);
        double next_theta = s_theta(s + 0.01);
        double next_x = next_r*cos(next_theta);
        double next_y = next_r*sin(next_theta);
        
        double heading = atan2((next_y-y),(next_x-x));
        double perp_heading = heading-pi()/2;

        x += d*cos(perp_heading);
        y += d*sin(perp_heading);
       
        return {x,y};
    }
    else {
        // First find two waypoints around the point of interest
	    int prev_wp = -1;

	    while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) )) {
		    prev_wp++;
	    }

	    int next_wp = (prev_wp+1)%maps_x.size();
	
        double heading = atan2((maps_y[next_wp]-maps_y[prev_wp]),(maps_x[next_wp]-maps_x[prev_wp]));
        // the x,y,s along the segment
        double seg_s = (s-maps_s[prev_wp]);

        double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
        double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

        double perp_heading = heading-pi()/2;

        double x = seg_x + d*cos(perp_heading);
        double y = seg_y + d*sin(perp_heading);
        return {x,y};
    }
}

bool isLaneOpen(int lane, double s, double speed, vector<vector<double>> sensor_fusion) {
    // determines if there's room for us to change lanes
    int lane_d = LANE_WIDTH/2.0 + lane * LANE_WIDTH;
    for (int v=0; v < sensor_fusion.size(); v++){
        double veh_vx = sensor_fusion[v][3];
        double veh_vy = sensor_fusion[v][4];
        double veh_speed = magnitude(veh_vx,veh_vy);
        double veh_s = sensor_fusion[v][5];
        double veh_d = sensor_fusion[v][6];

        bool closeToS = false;
        if (S1GreaterThanS2(veh_s,s)){ // if it's ahead of us
            //if (veh_speed < speed)
            //    veh_s -= (speed - veh_speed)*1; // if they are slower, they are "closer"
            if (S1MinusS2(veh_s,s) < S_MARGIN)
                closeToS = true;

        }
        else { // if it's behind us
            //if (veh_speed > speed)
            //    veh_s += (veh_speed - speed)*1; // if they are faster, they are "closer"
            if (S1MinusS2(s,veh_s) < (S_MARGIN/2))
                closeToS = true;
        }

        // if vehicle is too close to where we are - no, not open!
        bool inTheLane = fabs(lane_d - veh_d) < D_MARGIN;
        if (inTheLane && closeToS)
            return false;
    }
    return true; // default is yes, it's open!
}

int getClosestCar(int lane, double s, vector<vector<double>> sensor_fusion) {
    // finds closest car ahead of us in lane
    int lane_d = LANE_WIDTH/2.0 + lane * LANE_WIDTH;

    double closest_dist = 9999999;
    int closest_idx = -1;
    for (int v=0; v < sensor_fusion.size(); v++){
        double veh_s = sensor_fusion[v][5];
        double veh_d = sensor_fusion[v][6];
        
        // if vehicle is too close to where we are - no, not open!
        bool inTheLane = fabs(lane_d - veh_d) < D_MARGIN;
        bool inFrontOfS = S1GreaterThanS2(veh_s,s);
        bool nextToOrInFrontOfS = inFrontOfS;

        if (inTheLane && nextToOrInFrontOfS){ // if it's in our lane and next to or in front of the s_position
            double dist = S1MinusS2(veh_s,s);
            if (dist < closest_dist) {
                closest_dist = dist;
                closest_idx = v;
            }
        }
    }
    return closest_idx; // couldn't find a car
}

double getSensorSpeed(int v, vector<vector<double>> sensor_fusion) {
    double veh_vx = sensor_fusion[v][3];
    double veh_vy = sensor_fusion[v][4];
    double veh_speed = magnitude(veh_vx,veh_vy);
    return veh_speed;
}

double getSensorDistance(int v, vector<vector<double>> sensor_fusion, double s) {
    double veh_s = sensor_fusion[v][5];
    return S1MinusS2(veh_s,s);
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  vector<double> last_svec, last_dvec;//, last_accel = 0.0, last_velocity = 0.0;
  vector<double> last_accels, last_vels;
  int target_lane;
  double target_velocity;
  bool first_run = true;
  int doing_lane_change = 0;

  h.onMessage([&doing_lane_change,&first_run,&target_lane,&target_velocity,&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy,&last_svec,&last_dvec,&last_accels,&last_vels](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
            cout << "-------------------------------------------------------------------" << endl;
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];
          	cout << "car_s = " << car_s << " car_d = " << car_d << endl;
          	
          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];

          	json msgJson;

          	vector<double> next_x_vals;
          	vector<double> next_y_vals;
          	
          	int M = previous_path_x.size();
            
      	    // clear out points that have already been passed
      	    if (M > 0){
          	    // N-M new points, so get rid of past data
          	    last_accels.erase(last_accels.begin(), last_accels.begin() + N-M);
          	    last_vels.erase(last_vels.begin(), last_vels.begin() + N-M);
          	    last_svec.erase(last_svec.begin(), last_svec.begin() + N-M);
          	    last_dvec.erase(last_dvec.begin(), last_dvec.begin() + N-M);
      	    }
            
          	
          	// A couple useful calculations
            int current_lane = car_d / LANE_WIDTH;
            double car_speed_mps = car_speed * MPH2MPS;
            
            
            // If this is the first run, we'll initialize our lane and velocity choices to get going
            if (first_run) {
                target_lane = current_lane;
                target_velocity = CLEARLANE_TGTVEL_MPS; //car_speed * MPH2MPS;
                first_run = false;
                //car_s = 3000;
            }
            else {
                // advance the "doing_lane_change" counter - necessary to avoid constantly doing lane changes
                doing_lane_change -= (N-M);
                if (doing_lane_change < 0) doing_lane_change = 0;
            }
            
            
            // DETERMINE TARGET_LANE AND TARGET_VELOCITY
            
            double best_speed = 0, best_dist = 0;
            int best_lane = target_lane;
            double lane_speeds[3];
            bool center_lane_open = isLaneOpen(CENTER_LANE,car_s,car_speed_mps,sensor_fusion);
            
            for (int i=0; i < 3; i++){ // for each lane
                // make it a priority to be in the middle lane
                int lane = CENTER_LANE;
                if (i==1) lane = LEFT_LANE;
                if (i==2) lane = RIGHT_LANE;
            
                if (!center_lane_open){
                    if (abs(lane-target_lane) > 1) continue; // TODO consider lane next to available lane
                    if (abs(lane-current_lane) > 1) continue; // TODO consider lane next to available lane
                }
                bool lane_open = isLaneOpen(lane,car_s,car_speed_mps,sensor_fusion); // determine if lane is open
                if (lane==target_lane or lane==current_lane or lane_open) { // candidate lane
                    int closest_car_idx = getClosestCar(lane,car_s,sensor_fusion);
                    double closest_car_speed = CLEARLANE_TGTVEL_MPS;
                    double closest_car_dist = 999999;
                    if (closest_car_idx >= 0){
                        closest_car_speed = getSensorSpeed(closest_car_idx,sensor_fusion) - 0.25;
                        closest_car_dist = getSensorDistance(closest_car_idx, sensor_fusion, car_s);
                    }
                    cout << "mylane = " << current_lane << " lane = " << lane << " speed = " << closest_car_speed << " dist = " << closest_car_dist << endl;
                    
                        
                    if (lane==current_lane && closest_car_dist > TAILGATE_MARGIN){
                        closest_car_speed = CLEARLANE_TGTVEL_MPS;
                        cout << "*";
                    }
                    else if (lane!=current_lane && closest_car_dist > TAILGATE_MARGIN*5){ // too far away to matter
                        closest_car_speed = CLEARLANE_TGTVEL_MPS;
                        cout << "**";
                    }
                    
                    if (abs(lane-current_lane) > 1) // slight nudge to encourage car to avoid lane search priority
                        closest_car_speed -= 0.1;
                    
                    cout << "lane = " << lane << " idx = " << closest_car_idx << " speed = " << closest_car_speed << " dist = " << closest_car_dist << endl;
                    lane_speeds[lane] = closest_car_speed;

                    if (closest_car_speed > best_speed || (best_dist < TAILGATE_MARGIN*2 && closest_car_dist > best_dist)) {
                        best_speed = closest_car_speed;
                        best_lane = lane;
                        best_dist = closest_car_dist;
                    }
                }
            }

            // dont' change lanes in the middle of a lane change
            if (doing_lane_change > 0){
                best_lane = target_lane;
                best_speed = target_velocity;
            }
            // avoid actually changing two lanes over
            if (abs(best_lane - target_lane) > 1){
                best_lane = CENTER_LANE;
            }
            
            best_speed = (best_speed > CLEARLANE_TGTVEL_MPS) ? CLEARLANE_TGTVEL_MPS : best_speed; // don't exceed speed limit

            int B = M; // to start, use all of the previous path
            if (best_lane != target_lane || fabs(best_speed - target_velocity) > 0.1) {
                // we have a change to perform - restart the path from scratch!
                if (best_lane != target_lane)
                    doing_lane_change = N;
                target_lane = best_lane;
                best_speed = lane_speeds[target_lane];
                target_velocity = best_speed;
                B = (M > 8) ? 8 : M; // the minimum of the next n points or M
            }
            
            //if ((3000 < car_s) && (car_s < 5700)) target_velocity = 100;
            
            cout << "target_lane = " << target_lane << endl << "target_speed = " << target_velocity << endl;
            
            // We've chosen to recalculate the path starting at B, so get rid of previous calculated points after B
      	    if ((M > 0) && (B != M)){
          	    last_accels.erase(last_accels.begin()+B, last_accels.end());
          	    last_vels.erase(last_vels.begin()+B, last_vels.end());
          	    last_svec.erase(last_svec.begin()+B, last_svec.end());
          	    last_dvec.erase(last_dvec.begin()+B, last_dvec.end());
      	    }
            
            // For purposes of appending a new path, retrieve the previous velocity and acceleration values
            double current_accel = 0.0;
            if (last_accels.size() > 0)
                current_accel = last_accels[B-1];//0.0;
            double current_velocity = 0; //car_speed * MPH2MPS;
            if (last_vels.size() > 0)
                current_velocity = last_vels[B-1];//0.0;
            cout << "current_vel = " << current_velocity << endl << "current_accel = " << current_accel << endl;

            // For purposes of appending a new path, retrieve the previous s and d values
            double ref_s = car_s, ref_d = car_d;
            if (last_svec.size() > 0)
                ref_s = last_svec[B-1];//0.0;
            if (last_dvec.size() > 0)
                ref_d = last_dvec[B-1];//0.0;

          	// define a path made up of (x,y) points that the car will visit sequentially every .02 seconds

          	double new_s = ref_s, new_d = ref_d;
          	double target_d = target_lane * LANE_WIDTH + LANE_WIDTH/2.0;
          	target_d -= 0.125*target_lane; // WORKAROUND FOR BAD INTERPOLATION IN SIMULATOR
          	
      	    // tack on the previous path to the "new path"
      	    for (int i=0; i < B; i++){
      	        next_x_vals.push_back(previous_path_x[i]);
      	        next_y_vals.push_back(previous_path_y[i]);
      	    }
      	    
      	    // Generate the lane change motion
      	    tk::spline d_transition;
      	    if (1) { // useful to create a scope
      	        vector<double> i_pts,d_pts;
      	        i_pts.push_back(-.2);
      	        i_pts.push_back(-.1);
      	        i_pts.push_back(0);
      	        if ((N-B) <= 1){
          	        i_pts.push_back(1);
          	        i_pts.push_back(1+0.1);
          	        i_pts.push_back(1+0.2);
      	        }
      	        else {
          	        i_pts.push_back(N-B-1);
          	        i_pts.push_back((N-B-1)+0.1);
          	        i_pts.push_back((N-B-1)+0.2);
      	        }
      	        
      	        d_pts.push_back(ref_d);
      	        d_pts.push_back(ref_d);
      	        d_pts.push_back(ref_d);
      	        d_pts.push_back(target_d);
      	        d_pts.push_back(target_d);
      	        d_pts.push_back(target_d);
      	        
      	        d_transition.set_points(i_pts,d_pts);
      	    }
      	    
      	    // Create the points, adjusting velocity and acceleration as we go
            double target_accel = 0.0;
          	for (int i=0; i < N - B; i++){
          	
          	    // Adjust target acceleration if our velocity isn't right
          	    if (current_velocity < target_velocity)
          	        if (current_velocity > target_velocity - 10.0)
              	        target_accel = CRUISE_ACCEL; // ease on into position
              	    else
              	        target_accel = MAX_ACCEL; // pedal to the metal, dude!
          	    else if (current_velocity > target_velocity){
          	        if (current_velocity < target_velocity + 3.0)
              	        target_accel = -1*CRUISE_ACCEL; // ease on into position
              	    else
              	        target_accel = -1*MAX_ACCEL; // pedal to the metal, dude!
          	    }
          	    else
          	        target_accel = 0.0;
          	    
          	    // Adjust acceleration if our acceleration isn't right
          	    if (current_accel < target_accel)
          	        current_accel += MAX_JERK * CAR_MVMNT_TIME;
          	    else if (current_accel > target_accel)
          	        current_accel -= MAX_JERK * CAR_MVMNT_TIME;
          	       
          	    // Advance velocity by acceleration over time
          	    current_velocity += current_accel * CAR_MVMNT_TIME;
                
                // New D was precalculated for a smooth transition
                new_d = d_transition(i);
                
                // Advance position by velocity over time
          	    new_s += current_velocity*CAR_MVMNT_TIME;
          	    
                // Convert Frenet (s,d) to Cartesian (x,y)          	    
          	    vector<double> xy = getXY(new_s, new_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          	    
          	    // WORKAROUND FOR SIMULATOR SPEED MEASUREMENT BUG 
          	    if (next_x_vals.size() > 0){
          	        // check against actual velocity we're going for
          	        double last_x = next_x_vals[next_x_vals.size()-1];
          	        double last_y = next_y_vals[next_y_vals.size()-1];
          	        
          	        double calcdist = magnitude(xy[0] - last_x, xy[1] - last_y);
          	        double disterror = current_velocity*CAR_MVMNT_TIME - calcdist;
      	            new_s += disterror;
      	            
      	            // Recalculate (x,y) given new (s,d)
      	            xy = getXY(new_s, new_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          	    }
          	    
          	    // Add onto our path
          	    next_x_vals.push_back(xy[0]);
          	    next_y_vals.push_back(xy[1]);

                // Keep track of all these intermediate values we've calculated, in case we want to reuse the path
          	    last_svec.push_back(new_s);
          	    last_dvec.push_back(new_d);
          	    last_accels.push_back(current_accel);
          	    last_vels.push_back(current_velocity);
          	}

         	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          	
 
          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	//ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
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
