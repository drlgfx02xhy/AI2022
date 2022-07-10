import util

"""
Data sturctures we will use are stack, queue and priority queue.

Stack: first in last out
Queue: first in first out
    collection.push(element): insert element
    element = collection.pop() get and remove element from collection

Priority queue:
    pq.update('eat', 2)
    pq.update('study', 1)
    pq.update('sleep', 3)
pq.pop() will return 'study' because it has highest priority 1.

"""

"""
problem is a object has 3 methods related to search state:

problem.getStartState()
Returns the start state for the search problem.

problem.isGoalState(state)
Returns True if and only if the state is a valid goal state.

problem.getChildren(state)
For a given state, this should return a list of tuples, (next_state,
step_cost), where 'next_state' is a child to the current state, 
and 'step_cost' is the incremental cost of expanding to that child.

"""
def myDepthFirstSearch(problem):
    # pdb.set_trace()
    visited = {}
    frontier = util.Stack()

    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]
        
        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))

    return []

def myBreadthFirstSearch(problem):
    visited = {}
    frontier = util.Queue()

    frontier.push((problem.getStartState(), None)) # (startstate, None)

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]
        
        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))

    return []

def myAStarSearch(problem, heuristic):
    # A* is a heuristic search algorithm.
    route = {}
    open_set = util.PriorityQueue()
    close_set = []
    solution = []
    step_cost = {}
    
    # pdb.set_trace()
    open_set.update(problem.getStartState(), 0)
    step_cost[problem.getStartState()] = 0
    route[problem.getStartState()] = None
    
    while not open_set.isEmpty():
        # pdb.set_trace()
        state = open_set.pop()
        close_set.append(state)

        if problem.isGoalState(state):
            # pdb.set_trace()
            while state != None:
                solution.append(state)
                state = route[state]
            return solution[::-1]
		
        # pdb.set_trace()
        for next_state, current_step_cost in problem.getChildren(state):
            if next_state not in close_set:
                new_cost = step_cost[state] + current_step_cost
                if next_state not in step_cost or new_cost < step_cost[next_state]:
                    step_cost[next_state] = new_cost
                    route[next_state] = state
                    open_set.update(next_state, new_cost + heuristic(next_state))

    return []

"""
Game state has 4 methods we can use.

state.isTerminated()
Return True if the state is terminated. We should not continue to search if the state is terminated.

state.isMe()
Return True if it's time for the desired agent to take action. We should check this function to determine whether an agent should maximum or minimum the score.

state.getChildren()
Returns a list of legal state after an agent takes an action.

state.evaluateScore()
Return the score of the state. We should maximum the score for the desired agent.

"""
class MyMinimaxAgent():

    def __init__(self, depth):
        self.depth = depth


    def max_value(self, state, depth):
        if state.isTerminated() or depth == 0:
            return None, state.evaluateScore()
        v = - float('inf')
        for child in state.getChildren():
            _, child_v = self.min_value(child, depth)
            if child_v > v:
                v = child_v
                best_state = child
        return best_state, v
    
    def min_value(self, state, depth):
        if state.isTerminated() or depth == 0:
            return None, state.evaluateScore()
        v = float('inf')
        for child in state.getChildren():
            if child.isMe():
                _, child_v = self.max_value(child, depth - 1)
            else:
                _, child_v = self.min_value(child, depth)
            if child_v < v:
                v = child_v
                best_state = child
        return best_state, v

    def minimax(self, state, depth):
        return self.max_value(state, depth)

    def getNextState(self, state):
        best_state, _ = self.minimax(state, self.depth)
        return best_state
    
class MyAlphaBetaAgent():

    def __init__(self, depth):
        self.depth = depth


    def max_value(self, state, alpha, beta, depth):
        if state.isTerminated() or depth == 0:
            return None, state.evaluateScore()
        v = - float('inf')
        for child in state.getChildren():
            _, child_v = self.min_value(child, alpha, beta, depth)
            if child_v > v:
                v = child_v
                best_state = child
            if v > beta:
                break
            alpha = max(alpha, v)
        return best_state, v
    
    def min_value(self, state, alpha, beta, depth):
        if state.isTerminated() or depth == 0:
            return None, state.evaluateScore()
        v = float('inf')
        for child in state.getChildren():
            if child.isMe():
                _, child_v = self.max_value(child, alpha, beta, depth - 1)
            else:
                _, child_v = self.min_value(child, alpha, beta, depth)
            if child_v < v:
                v = child_v
                best_state = child
            if v < alpha:
                break
            beta = min(beta, v)
        return best_state, v
            
    def alphabeta(self, state, alpha, beta, depth): # alpha：best_score的下确界， beta：best_score的上确界
        return self.max_value(state, alpha, beta, depth)

        
    def getNextState(self, state):
        best_state, _ = self.alphabeta(state, -float("inf"), float("inf"), self.depth)
        return best_state
