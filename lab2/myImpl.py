import util
import pdb
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

    def minimax(self, state, depth):
        if state.isTerminated() or depth == -1:
            return None, state.evaluateScore()

        flag = state.isMe()
        if flag == 1:
            best_state, best_score = None, -float('inf')
        else:
            best_state, best_score = None, float('inf')
        # pdb.set_trace()
        if flag == 1: # 这一步轮到人走，人希望max; 那么下一步，也就是child，是鬼；鬼希望min
            for child in state.getChildren():
                _, score = self.minimax(child, depth - 1)  # 记鬼给人的状态分数为score
                if score > best_score: # 如果下一步鬼给人的状态比当前最好的分数大，那么就更新最好的分数
                    best_state, best_score = child, score

        else: # 这一步轮到鬼走，那么下一步，也就是child，是人/鬼（可能不止一个鬼）；人/鬼希望max/min
            for child in state.getChildren():
                # pdb.set_trace()
                nextflag = child.isMe()
                if nextflag == 1:
                    _, score = self.minimax(child, depth - 1)
                else:
                    _, score = self.minimax(child, depth)
                if score < best_score: # 对鬼来说，想要分数min
                    best_state, best_score = child, score
        
        return best_state, best_score

    def getNextState(self, state):
        best_state, _ = self.minimax(state, self.depth)
        return best_state
    
"""
function alphabeta(node, depth, α, β, Player)         
    if  depth = 0 or node is a terminal node
        return the heuristic value of node
    if  Player = MaxPlayer // 极大节点
        for each child of node // 极小节点
            α := max(α, alphabeta(child, depth-1, α, β, not(Player) ))   
            if β ≤ α // 该极大节点的值>=α>=β，该极大节点后面的搜索到的值肯定会大于β，因此不会被其上层的极小节点所选用了。对于根节点，β为正无穷
                break                             (* Beta cut-off *)
        return α
    else // 极小节点
        for each child of node // 极大节点
            β := min(β, alphabeta(child, depth-1, α, β, not(Player) )) // 极小节点
            if β ≤ α // 该极大节点的值<=β<=α，该极小节点后面的搜索到的值肯定会小于α，因此不会被其上层的极大节点所选用了。对于根节点，α为负无穷
                break                             (* Alpha cut-off *)
        return β 
(* Initial call *)
alphabeta(origin, depth, -infinity, +infinity, MaxPlayer)
"""

class MyAlphaBetaAgent():

    def __init__(self, depth):
        self.depth = depth

    def alphabeta(self, state, depth, alpha, beta): # alpha：best_score的下确界， beta：best_score的上确界
        if state.isTerminated() or depth == -1:
            return None, state.evaluateScore()

        flag = state.isMe()
        best_state = None
        # pdb.set_trace()
        if flag == 1: # 这一步轮到人走，人希望max; 那么下一步，也就是child，是鬼；鬼希望min
            for child in state.getChildren():
                _, score = self.alphabeta(child, depth - 1, alpha, beta)  # 记鬼给人的状态分数为score
                if score > alpha: # 如果下一步鬼给人的状态比当前最好的分数大，那么就更新最好的分数
                    best_state, alpha = child, score
                if beta <= alpha: # 该极大节点的值>=α>=β，该极大节点后面的搜索到的值肯定会大于β，因此不会被其上层的极小节点所选用了。对于根节点，β为正无穷
                    break
            return best_state, alpha

        else: # 这一步轮到鬼走，那么下一步，也就是child，是人/鬼（可能不止一个鬼）；人/鬼希望max/min
            for child in state.getChildren():
                # pdb.set_trace()
                nextflag = child.isMe()
                if nextflag == 1:
                    _, score = self.alphabeta(child, depth - 1, alpha, beta)
                else:
                    _, score = self.alphabeta(child, depth, alpha, beta)
                if score < beta: # 对鬼来说，想要分数min
                    best_state, beta = child, score
                if beta <= alpha:
                    break
            return best_state, beta
        
    def getNextState(self, state):
        best_state, _ = self.alphabeta(state, self.depth, -float("inf"), float("inf"))
        return best_state
