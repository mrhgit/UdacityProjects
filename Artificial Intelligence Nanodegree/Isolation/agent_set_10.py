                        *************************                         
                             Playing Matches                              
                        *************************                         

 Match #   Opponent    AB_Improved   AB_Custom   AB_Custom_2  AB_Custom_3 
                        Won | Lost   Won | Lost   Won | Lost   Won | Lost 
    1       Random      179 |  21    189 |  11    186 |  14    186 |  14  
    2       MM_Open     106 |  94    115 |  85    125 |  75    104 |  96  
    3      MM_Center    157 |  43    172 |  28    172 |  28    167 |  33  
    4     MM_Improved   112 |  88    113 |  87    117 |  83    123 |  77  
    5       AB_Open     98  |  102   114 |  86    122 |  78    112 |  88  
    6      AB_Center    97  |  103   139 |  61    127 |  73    134 |  66  
    7     AB_Improved   99  |  101   117 |  83    115 |  85    111 |  89  
--------------------------------------------------------------------------
           Win Rate:      60.6%        68.5%        68.9%        66.9%    

"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #dismissive of opponent
    opponent = game.get_opponent(player)
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(opponent))
    if (player == game._inactive_player): # we're waiting for opp to move
        if not opp_moves: return float("inf") # and he can't, we won
        return float(own_moves/(opp_moves-1+0.0001))
    else: # it's our turn
        if not own_moves: return float("-inf") # but we don't have moves!
        return float((own_moves)/(opp_moves+0.0001))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    opponent = game.get_opponent(player)
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(opponent))
    if (player == game._inactive_player):
        if not opp_moves: return float("inf")
    else:
        if not own_moves: return float("-inf")
        
    return float(own_moves/(opp_moves+0.0001))



def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    #realist
    opponent = game.get_opponent(player)
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(opponent))
    if (player == game._inactive_player): # we're waiting for opp to move
        if not opp_moves: return float("inf") # and he can't, we won
        return float(own_moves/(opp_moves-1+0.0001))
    else: # it's our turn
        if not own_moves: return float("-inf") # but we don't have moves!
        return float((own_moves-1)/(opp_moves+0.0001))
        
    return float(own_moves/(opp_moves+0.0001))

    opponent = game.get_opponent(player)
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(opponent))
    if (player == game._inactive_player):
        if not opp_moves: return float("inf")
    else:
        if not own_moves: return float("-inf")
    
    #King of the Hill
    y1, x1 = game.get_player_location(player)
    y2, x2 = game.get_player_location(opponent)
    #w, h = game.width / 2., game.height / 2.
    #distance_to_center1 = (abs(h - y1) + abs(w - x1))
    #distance_to_center2 = (abs(h - y2) + abs(w - x2))
    distance_to_opponent = (abs(y2 - y1) + abs(x2 - x1))
    if (own_moves > opp_moves):
        return float(distance_to_opponent)
    else:
        return float(20-distance_to_opponent)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        own_moves = game.get_legal_moves(game.active_player)
        if len(own_moves)>0:
            best_move = own_moves[0]
        else:
            best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def max_value(self, game, depth, player):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        own_moves = game.get_legal_moves(player)
        if depth==0 or len(own_moves)==0:
            return self.score(game,player), (-1,-1)
        
        v = float("-inf")
        for move in own_moves:
            subgame = game.forecast_move(move)
            checkv, xxx = self.min_value(subgame, depth-1,game.get_opponent(player))
            if (v <= checkv):
                v = checkv
                best_move = move
        return v, best_move
        
    def min_value(self, game, depth, player):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        opponent_moves = game.get_legal_moves(player)
        if depth==0 or len(opponent_moves)==0:
            return self.score(game,game.get_opponent(player)), (-1,-1) # opponent of the opponent is the active player
        
        v = float("inf")
        for move in opponent_moves:
            subgame = game.forecast_move(move)
            checkv, xxx = self.max_value(subgame, depth-1,game.get_opponent(player))
            if (v >= checkv):
                v = checkv
                best_move = move
        return v, best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        v, best_move = self.max_value(game, depth, game.active_player)
        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        own_moves = game.get_legal_moves(game.active_player)
        if len(own_moves)>0:
            best_move = own_moves[0]
        else:
            best_move = (-1, -1)
        search_depth = self.search_depth

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.

            while 1: # Let the timeout exception break the loop
                best_move = self.alphabeta(game, search_depth)
                search_depth += 1

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def max_value(self, game, depth, player, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        own_moves = game.get_legal_moves(player)
        if depth==0 or len(own_moves)==0:
            return self.score(game,player), (-1,-1)
        
        v = float("-inf")
        for move in own_moves:
            subgame = game.forecast_move(move)
            checkv, xxx = self.min_value(subgame, depth-1,game.get_opponent(player), alpha, beta)
            if (checkv >= beta):  # if our lowest possible is bigger than the worst contender, give up
                return checkv, move
            if (checkv >= v):
                v = checkv
                best_move = move
            alpha = max([alpha, checkv]) # we keep track of the best contender
        return v, best_move
        
    def min_value(self, game, depth, player, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        opponent_moves = game.get_legal_moves(player)
        if depth==0 or len(opponent_moves)==0:
            return self.score(game,game.get_opponent(player)), (-1,-1) # opponent of the opponent is the active player
        
        v = float("inf")
        for move in opponent_moves:
            subgame = game.forecast_move(move)
            checkv, xxx = self.max_value(subgame, depth-1,game.get_opponent(player), alpha, beta)
            if (checkv <= alpha): # if our highest possible is lower than the best contender, give up
                return checkv, move
            if (checkv <= v):
                v = checkv
                best_move = move
            beta = min([beta, checkv]) # keep track of worst contender
        return v, best_move



    def max_valueOLD(self, game, depth, player):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        own_moves = game.get_legal_moves(player)
        if depth==0:
            return self.score(game,player)
        
        v = float("-inf")
        for move in own_moves:
            subgame = game.forecast_move(move)
            v = max([v, self.min_value(subgame, depth-1,game.get_opponent(player))])
            if v >= self.beta:
                return v
            self.alpha = max([v,self.alpha])
        return v
        
    def min_valueOLD(self, game, depth, player):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        opponent_moves = game.get_legal_moves(player)
        if depth==0:
            return self.score(game,game.get_opponent(player))
        
        v = float("inf")
        for move in opponent_moves:
            subgame = game.forecast_move(move)
            v = min([v, self.max_value(subgame, depth-1,game.get_opponent(player))])
            if v <= self.alpha:
                return v
            self.beta = min([v,self.beta])
        return v

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        v, best_move = self.max_value(game, depth, game.active_player, alpha, beta)
        return best_move
