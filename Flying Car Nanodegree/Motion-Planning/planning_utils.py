from enum import Enum
from queue import PriorityQueue
import numpy as np
from bresenham import bresenham
from sklearn.neighbors import KDTree
import numpy.linalg as LA
import networkx as nx

def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)

def straight_line_possible(node, goal, grid):
    #print ("node = ",node)
    #print ("goal = ",goal)
    cells = list(bresenham(node[0],node[1],goal[0],goal[1]))
    for q in cells:
        #if grid[q[1],q[0]]:
        if grid[q[0],q[1]]:
            return False
    #print("Straight line possible!")
    return True

def add_point_to_graph(g,grid,pt,n_closest=10):
    d1 = list(pt[0:2])
    nodes = np.array(g.nodes)[:,0:2]
    tree = KDTree(nodes)
    idxs = tree.query([d1], k=n_closest+1, return_distance=False)[0]
    for idx in idxs:
        d2 = list(nodes[idx][0:2])
        if np.any(d1 != d2):
            if straight_line_possible(d1, d2, grid):
                dist = LA.norm(np.array(d2) - np.array(d1))
                g.add_edge(tuple(d1), tuple(d2), weight=dist)
    return g
 

def graph_from_grid(grid,start=None,goal=None,graph_size=500,n_closest=10):
    print ("Creating graph from grid...")
    xmin,ymin,xmax,ymax = 0,0,grid.shape[0],grid.shape[1]
    xmin1,ymin1 = min(start[0],goal[0]),min(start[1],goal[1])
    xmax1,ymax1 = max(start[0],goal[0]),max(start[1],goal[1])
    xmin1 = max(xmin,xmin1 - 0.25*(xmax1 - xmin1))
    ymin1 = max(ymin,ymin1 - 0.25*(ymax1 - ymin1))
    xmax1 = min(xmax,xmax1 + 0.25*(xmax1 - xmin1))
    ymax1 = min(ymax,ymax1 + 0.25*(ymax1 - ymin1))
    #print(xmin1,ymin1,xmax1,ymax1)
    to_keep = []
    # The first half of points we'll focus in the box connecting start and goal directly
    while len(to_keep) < graph_size*0.5:
        rx = int(np.random.uniform(xmin1, xmax1))
        ry = int(np.random.uniform(ymin1, ymax1))
        if grid[rx,ry]==0:
            point = (rx,ry)
            #print(rx,ry)
            to_keep.append(point)
    # The second half of points are all across the board
    while len(to_keep) < graph_size:
        rx = int(np.random.uniform(xmin, xmax))
        ry = int(np.random.uniform(ymin, ymax))
        point = (rx,ry)
        if grid[rx,ry]==0:
            to_keep.append(point)

    tree2 = KDTree(np.array(to_keep)[:,0:2])
    g = nx.Graph()
    #tested_combos = []
    for d in to_keep:
        d1 = list(d[0:2])
        # Get n closest neighbors
        idxs = tree2.query([d1], k=n_closest+1, return_distance=False)[0]
        for idx in idxs:
            d2 = list(to_keep[idx][0:2])
            if np.any(d1 != d2):# and (d1,d2) not in tested_combos:
                if straight_line_possible(d1, d2, grid): #can_connect(d1,d2,polygons):
                    dist = LA.norm(np.array(d2) - np.array(d1))
                    #print (d1, d2, dist)
                    g.add_edge(tuple(d1), tuple(d2), weight=dist)
                #tested_combos.append((d1,d2))
                #print (tested_combos)
    print("Complete")
    return g


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)
    NORTHWEST = (-1, -1, 1*np.sqrt(2))
    NORTHEAST = (-1, 1, 1*np.sqrt(2))
    SOUTHWEST = (1, -1, 1*np.sqrt(2))
    SOUTHEAST = (1, 1, 1*np.sqrt(2))

    # We could really get crazy with this, but the "straight_line_possible" option
    #  makes these moot.  They would also just slow down our a* search, by meaning more
    #  options, which are mostly redundant.
    
    #NNORTHWEST = (-2, -1, np.sqrt(5))
    #NNORTHEAST = (-2, 1, np.sqrt(5))
    #SSOUTHWEST = (2, -1, np.sqrt(5))
    #SSOUTHEAST = (2, 1, np.sqrt(5))
    #WNORTHWEST = (-1, -2, np.sqrt(5))
    #ENORTHEAST = (-1, 2, np.sqrt(5))
    #WSOUTHWEST = (1, -2, np.sqrt(5))
    #ESOUTHEAST = (1, 2, np.sqrt(5))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle
    for a in list(Action):
        av = a.value
        if x + av[0] < 0 or y + av[1] < 0 or x + av[0] > n or y + av[1] > m or grid[x+av[0],y+av[1]]==1:
            valid_actions.remove(a)

    return valid_actions


def a_star_grid(grid, h, start, goal):
    print("Running a* grid mode...")

    if (grid[goal[0],goal[1]]==1):
        print ("That goal collides with an obstacle!")
        return None,0

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
            current_action = None
        else:              
            current_cost = branch[current_node][0]
            current_action = branch[current_node][2]
            
        slp = straight_line_possible(current_node,goal,grid)
        if current_node == goal or slp:
            if current_node != goal:
                branch[goal] = (0.0, current_node, None)
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                action_cost = action.cost
                #if action != current_action:
                #    action_cost += 1
                branch_cost = current_cost + action_cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            this_action = branch[n][2]
            path.append(branch[n][1])  # We can prune here in a computationally cheaper fashion
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost

def a_star_graph(graph, grid, heuristic, start, goal):
    print("Running a* graph mode...")

    # Add start state and goal state to graph, in case they don't exist
    graph = add_point_to_graph(graph,grid,start,n_closest=10)
    graph = add_point_to_graph(graph,grid,goal,n_closest=10)
    
    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:        
            print('Found a path.')
            found = True
            break
            
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    queue.put((new_cost, next_node))
                    
                    branch[next_node] = (new_cost, current_node)
             
    path = []
    path_cost = 0
    if found:
        
        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
        print (path)
    else:
        return None, 0
            
    return path[::-1], path_cost


def convert_graph_path_to_grid(graph_path):
    grid_path = [graph_path[0]]
    for i in range(len(graph_path)-1):
        node = graph_path[i]
        goal = graph_path[i+1]
        cells = list(bresenham(node[0],node[1],goal[0],goal[1]))
        for q in cells:
            grid_path += [(q[0],q[1])]
    return grid_path

def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def point(p):
    return np.array([p[0], p[1], 1.]).reshape(1, -1)

def colinearity_check(p1, p2, p3, epsilon=1e-6):   
    m = np.concatenate((p1, p2, p3), 0)
    det = np.linalg.det(m)
    return abs(det) < epsilon
   
   
def prune_path_colinearity(path):
    pruned_path = [p for p in path]
    last_point = path[-1]
    for i in range(len(path)-2,0,-1):
        if colinearity_check(point(last_point),point(path[i]),point(path[i-1])):
            del pruned_path[i]
        else:
            last_point = path[i]
    return pruned_path

def prune_path_bresenham(path, grid):
    pruned_path = [p for p in path]
    last_point = path[-1]
    prune_happened = False
    for i in range(len(path)-2,0,-1):
        if straight_line_possible(last_point, path[i-1], grid):
        #if colinearity_check(point(),point(path[i]),point()):
            prune_happened = True
            del pruned_path[i]
        else:
            last_point = path[i]
    return pruned_path, prune_happened

