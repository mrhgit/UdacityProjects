assignments = []
rows = 'ABCDEFGHI'
cols = '123456789'


def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s+t for s in A for t in B]

boxes = cross(rows, cols)
ul_lr_diag = [r+cols[idx] for idx, r in enumerate(rows)]
ur_ll_diag = [r+cols[-1-idx] for idx, r in enumerate(rows)]
diag_units=[ul_lr_diag, ur_ll_diag]
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
unitlist = row_units + column_units + square_units + diag_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    # Find all instances of naked twins
    naked_twins = []
    naked_twins_peer = []
    solved_values = [box for box in values.keys() if len(values[box]) == 2]
    for box in solved_values:
        twin = values[box]
        for peer in peers[box]:
            if values[peer]==twin and box not in naked_twins:
                naked_twins.append(box)
                naked_twins_peer.append(peer)

    # Eliminate the naked twins as possibilities for their peers
    new_values = values.copy()
    for idx, box in enumerate(naked_twins):
        peer_box = naked_twins_peer[idx]
        intersected_units = list(set(peers[box]) & set(peers[peer_box]))
        twin = values[box]
        for peer in intersected_units:
            if values[peer]!=twin:
                for digit in twin:
                    assign_value(new_values, peer, new_values[peer].replace(digit,''))
    #print("Old")
    #display(values)
    #print("New")
    #display(new_values)
    return new_values
    


def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    grid_dict = {}
    for idx, val in enumerate(grid):
      if (val=="."): val = "123456789"
      grid_dict[boxes[idx]] = val
    
    return grid_dict

def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return


def eliminate(values):
    """
    Go through all the boxes, and whenever there is a box with a value, eliminate this value from the values of all its peers.
    Input: A sudoku in dictionary form.
    Output: The resulting sudoku in dictionary form.
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            assign_value(values, peer, values[peer].replace(digit,''))
    return values

def only_choice(values):
    """
    Go through all the units, and whenever there is a unit with a value that only fits in one box, assign the value to this box.
    Input: A sudoku in dictionary form.
    Output: The resulting sudoku in dictionary form.
    """
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                assign_value(values,dplaces[0],digit)
    return values


def reduce_puzzle(values):
    """
    Iterate naked_twins(), eliminate() and only_choice(). If at some point, there is a box with no available values, return False.
    If the sudoku is solved, return the sudoku.
    If after an iteration of both functions, the sudoku remains the same, return the sudoku.
    Input: A sudoku in dictionary form.
    Output: The resulting sudoku in dictionary form.
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        values = naked_twins(values)
        values = eliminate(values)
        values = only_choice(values)
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    "Using depth-first search and propagation, create a search tree and solve the sudoku."
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if (values==False):
        #print ("Bad box!")
        return False
    
    # Choose one of the unfilled squares with the fewest possibilities
    chosenbox = None
    for choicelen in range(2,10):
        for box in boxes:
            if len(values[box])==choicelen:
                chosenbox = box
                break
        if chosenbox is not None:
            break
    if chosenbox is None:
        #print ("Completed Box!")
        return values
    #print("Chosenbox = "+chosenbox+" has a length of ",choicelen," and values of "+values[chosenbox])
        
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for choice in values[chosenbox]:
        #print ("Trying choice of "+choice)
        testvalues = values.copy()
        testvalues[chosenbox] = choice
        testresult = search(testvalues)
        if (testresult != False):
            #print ("Successful Test",testresult)
            #display(testresult)
            return testresult
            
    return False # If none of the choices work, we fail!


def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    return search(grid_values(grid))

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))
    exit

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except Exception as e:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
