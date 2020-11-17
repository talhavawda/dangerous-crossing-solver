"""
			COMP301 (ARTIFICIAL INTELLIGENCE) 2020
						ASSIGNMENT 1

					Dangerous Crossing Problem

				Developed by: Talha Vawda (218023210)

				This project has been developed using:
					Python 3.8.1
					PyCharm 2019.3.3 (Professional Edition) Build #PY-193.6494.30

				Acknowledgements:
					search.py

				Notes:
					1.	Due to the assignment's requirement of this single script being able to be run from the command line,
						I have copied the relevant code from the search.py file (and the utils.py from the AIMA GitHub repo)
						into this file instead of importing search and utils
						-	I imported them and this script is executed from the command line without
							the other 2 files present in the same directory, it will not execute
						-	The Assignment Specification has given permission for the search.py file (which also imports utils.py)
							so I assumed that structuring my code in this way is allowed
							-	I have also acknowledged in my code any classes or functions that were taken from
								search.py or utils.py

"""

import sys
from collections import deque
import functools
import heapq

"""
	Helper Functions (from utils.py)
"""


def is_in(elt, seq):
	"""Similar to (elt in seq), but compares with 'is', not '=='."""
	return any(x is elt for x in seq)


def memoize(fn, slot=None, maxsize=32):
	"""Memoize fn: make it remember the computed value for any argument list.
	If slot is specified, store result in that slot of first argument.
	If slot is false, use lru_cache for caching the values."""
	if slot:
		def memoized_fn(obj, *args):
			if hasattr(obj, slot):
				return getattr(obj, slot)
			else:
				val = fn(obj, *args)
				setattr(obj, slot, val)
				return val
	else:
		@functools.lru_cache(maxsize=maxsize)
		def memoized_fn(*args):
			return fn(*args)

	return memoized_fn

# =============================================================================
# =============================================================================


"""
	The classes Problem and Node, and the Search Algorithms have been taken from search.py
	whilst PriorityQueue has been taken from utils.py
"""


class Problem:
	"""
		The abstract class for a formal problem. You should subclass
		this and implement the methods actions and result, and possibly
		__init__, goal_test, and path_cost. Then you will create instances
		of your subclass and solve them with the various search functions.
	"""

	def __init__(self, initial, goal=None):
		"""The constructor specifies the initial state, and possibly a goal
		state, if there is a unique goal. Your subclass's constructor can add
		other arguments."""
		self.initial = initial
		self.goal = goal

	def actions(self, state):
		"""Return the actions that can be executed in the given
		state. The result would typically be a list, but if there are
		many actions, consider yielding them one at a time in an
		iterator, rather than building them all at once."""
		raise NotImplementedError

	def result(self, state, action):
		"""Return the state that results from executing the given
		action in the given state. The action must be one of
		self.actions(state)."""
		raise NotImplementedError

	def goal_test(self, state):
		"""Return True if the state is a goal. The default method compares the
		state to self.goal or checks for state in self.goal if it is a
		list, as specified in the constructor. Override this method if
		checking against a single self.goal is not enough."""
		if isinstance(self.goal, list):
			return is_in(state, self.goal)
		else:
			return state == self.goal

	def path_cost(self, c, state1, action, state2):
		"""Return the cost of a solution path that arrives at state2 from
		state1 via action, assuming cost c to get up to state1. If the problem
		is such that the path doesn't matter, this function will only look at
		state2. If the path does matter, it will consider c and maybe state1
		and action. The default method costs 1 for every step in the path."""
		return c + 1

	def value(self, state):
		"""For optimization problems, each state has a value. Hill Climbing
		and related algorithms try to maximize this value."""
		raise NotImplementedError


# =============================================================================


class Node:
	"""A node in a search tree. Contains a pointer to the parent (the node
	that this is a successor of) and to the actual state for this node. Note
	that if a state is arrived at by two paths, then there are two nodes with
	the same state. Also includes the action that got us to this state, and
	the total path_cost (also known as g) to reach the node. Other functions
	may add an f and h value; see best_first_graph_search and astar_search for
	an explanation of how the f and h values are handled. You will not need to
	subclass this class."""

	def __init__(self, state, parent=None, action=None, path_cost=0):
		"""Create a search tree Node, derived from a parent by an action."""
		self.state = state
		self.parent = parent
		self.action = action
		self.path_cost = path_cost
		self.depth = 0
		if parent:
			self.depth = parent.depth + 1

	def __repr__(self):
		return "<Node {}>".format(self.state)

	def __lt__(self, node):
		return self.state < node.state

	def expand(self, problem):
		"""List the nodes reachable in one step from this node."""
		return [self.child_node(problem, action)
				for action in problem.actions(self.state)]

	def child_node(self, problem, action):
		"""[Figure 3.10]"""
		next_state = problem.result(self.state, action)
		next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
		return next_node

	def solution(self):
		"""Return the sequence of actions to go from the root to this node."""
		return [node.action for node in self.path()[1:]]

	def path(self):
		"""Return a list of nodes forming the path from the root to this node."""
		node, path_back = self, []
		while node:
			path_back.append(node)
			node = node.parent
		return list(reversed(path_back))

	# We want for a queue of nodes in breadth_first_graph_search or
	# astar_search to have no duplicated states, so we treat nodes
	# with the same state as equal. [Problem: this may not be what you
	# want in other contexts.]

	def __eq__(self, other):
		return isinstance(other, Node) and self.state == other.state

	def __hash__(self):
		# We use the hash value of the state
		# stored in the node instead of the node
		# object itself to quickly search a node
		# with the same state in a Hash Table
		return hash(self.state)

# =============================================================================


class PriorityQueue:
	"""A Queue in which the minimum (or maximum) element (as determined by f and
	order) is returned first.
	If order is 'min', the item with minimum f(x) is
	returned first; if order is 'max', then it is the item with maximum f(x).
	Also supports dict-like lookup."""

	def __init__(self, order='min', f=lambda x: x):
		self.heap = []
		if order == 'min':
			self.f = f
		elif order == 'max':  # now item with max f(x)
			self.f = lambda x: -f(x)  # will be popped first
		else:
			raise ValueError("Order must be either 'min' or 'max'.")

	def append(self, item):
		"""Insert item at its correct position."""
		heapq.heappush(self.heap, (self.f(item), item))

	def extend(self, items):
		"""Insert each item in items at its correct position."""
		for item in items:
			self.append(item)

	def pop(self):
		"""Pop and return the item (with min or max f(x) value)
		depending on the order."""
		if self.heap:
			return heapq.heappop(self.heap)[1]
		else:
			raise Exception('Trying to pop from empty PriorityQueue.')

	def __len__(self):
		"""Return current capacity of PriorityQueue."""
		return len(self.heap)

	def __contains__(self, key):
		"""Return True if the key is in PriorityQueue."""
		return any([item == key for _, item in self.heap])

	def __getitem__(self, key):
		"""Returns the first value associated with key in PriorityQueue.
		Raises KeyError if key is not present."""
		for value, item in self.heap:
			if item == key:
				return value
		raise KeyError(str(key) + " is not in the priority queue")

	def __delitem__(self, key):
		"""Delete the first occurrence of key."""
		try:
			del self.heap[[item == key for _, item in self.heap].index(True)]
		except ValueError:
			raise KeyError(str(key) + " is not in the priority queue")
		heapq.heapify(self.heap)

# =============================================================================
# =============================================================================


"""
	Search Algorithms (taken from search.py)
"""

"""Depth-First Search"""

def depth_first_graph_search(problem):
	"""
	[Figure 3.7]
	Search the deepest nodes in the search tree first.
	Search through the successors of a problem to find a goal.
	The argument frontier should be an empty queue.
	Does not get trapped by loops.
	If two paths reach a state, only use the first one.
	"""

	frontier = [(Node(problem.initial))]  # Stack (implemented as a list)

	explored = set()
	while frontier:
		node = frontier.pop()
		if problem.goal_test(node.state):
			return node
		explored.add(node.state)
		frontier.extend(child for child in node.expand(problem)
						if child.state not in explored and child not in frontier)
	return None


"""Breadth-First Search"""

def breadth_first_graph_search(problem):


	node = Node(problem.initial)
	if problem.goal_test(node.state):
		return node
	frontier = deque([node]) #FIFO Queue (implemented as a collection.deque)
	explored = set()
	while frontier:
		node = frontier.popleft()
		explored.add(node.state)
		for child in node.expand(problem):
			if child.state not in explored and child not in frontier:
				if problem.goal_test(child.state):
					return child
				frontier.append(child)
	return None


"""Best-First Search"""

def best_first_graph_search(problem, f, display=False):
	"""Search the nodes with the lowest f scores first.
	You specify the function f(node) that you want to minimize; for example,
	if f is a heuristic estimate to the goal, then we have greedy best
	first search; if f is node.depth then we have breadth-first search.
	There is a subtlety: the line "f = memoize(f, 'f')" means that the f
	values will be cached on the nodes as they are computed. So after doing
	a best first search you can examine the f values of the path returned.
	"""
	f = memoize(f, 'f')
	node = Node(problem.initial)
	frontier = PriorityQueue('min', f)
	frontier.append(node)
	explored = set()
	while frontier:
		node = frontier.pop()
		if problem.goal_test(node.state):
			if display:
				print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
			return node
		explored.add(node.state)
		for child in node.expand(problem):
			if child.state not in explored and child not in frontier:
				frontier.append(child)
			elif child in frontier:
				if f(child) < frontier[child]:
					del frontier[child]
					frontier.append(child)
	return None


"""Greedy Best-First Search = Best-First Search with f(n) = h(n)"""


"""A-Star Search"""

def astar_search(problem, h=None, display=False):
	"""A* search is best-first graph search with f(n) = g(n)+h(n).
	You need to specify the h function when you call astar_search, or
	else in your Problem subclass.
	"""

	h = memoize(h or problem.h, 'h')
	return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)


# ==============================================================================
# ==============================================================================


class DangerousCrossing(Problem):
	"""
		The problem of moving n people across a bridge (from one side of it to the other)
		in the shortest/minimum possible time.

		Constraints:
			1. Each person has a travel time (in minutes) to cross the bridge
			2. No more than two people can cross the bridge at one time
				- Thus either 1 or 2 persons cross the bridge at a time (0 persons crossing results in the same State, so we dont count it)
			3. If two people are on the bridge together, they must travel at the pace of the slower person
			4. There is 1 flashlight and each party (of max 2 people) needs to cross the bridge with the flashlight

		This class is a subclass of the Problem class and implements the functions actions() and result()
		so that this class specifies the Dangerous Bridge Crossing Problem

		A State in the State Space is represented as a list of bits (i.e. a list whose values are restricted to {0,1})
		of size n+1 where n is the number of people crossing the bridge.
		The element (a bit) at index i represents the side of the bridge where person i is (currently) situated at
		and the element at index 0 represents the side of the bridge where the flashlight is situated at.


		The bridge has 2 sides, LEFT and RIGHT, and all n people (along with their flashlight) start off on the
		LEFT side of the bridge with the aim of crossing over to the RIGHT side of the bridge.

		Thus the Initial State of this problem can be represented as:
			state[i] = 0 for all i = {0, n}	(LEFT = 0)
		E.g. with n=4: initial = [0,0,0,0,0]

		And the (unique) Goal State of this problem can be represented as:
			state[i] = 1 for all i = {0, n}	(RIGHT = 1)
		E.g. with n=4: initial = [1,1,1,1,1]
	"""

	# Class Variables
	LEFT  = 0	# represents the left-hand side of the bridge
	RIGHT = 1	# represents the right-hand side of the bridge


	def __init__(self, n: int, crossingTime: list, minimumTime: int):
		"""
			The constructor for (to initialise) this DangerousCrossingProblem.

			In addition to specifying the Initial State and Goal State,
			it takes as parameters the number of people crossing the bridge,
			the crossing time of each, and the shortest possible time it takes
			for everyone to cross over to the other side (from the LEFT side
			to the RIGHT side).

			Since the DangerousCrossingProblem has specific Initial and Goal States,
			it does not take in these as parameters, but computes them based on n and
			passes it into the super constructor.


			Remember that in the State representation, the first element represents the flashlight so
			person i is represented by the element at index i
			-	Thus to have the indexes of a State and crossingTime linked to the same person
				(i.e. person i is represented in the State and now also in crossingTime by the element at index i),
				I've inserted a dummy value (of 0) as the first element (index 0)
				-	person i's location = state[i]
				-	person i's crossing time = crossingTime[i]

			The crossingTime list is sorted in ascending order
			-	Thus for persons i = [1, n], person 1 is the person with the shortest crossing time and
				person n is the person with the longest crossing time, and crossingTime[i-1] <= crossingTime[i] <= crossingTime[i+1]


			:param n: The number of people crossing the bridge
			:param crossingTime: A list (of positive integers) of size n denoting the time it takes for each of the n people
				to cross from one side of the bridge to the other
			:param minimumTime: The shortest possible time it takes for all the n people to cross the bridge within the Constraints
		"""

		# Specify Initial State and Goal State
		initial = []
		goal = []

		for i in range(n+1): # i traverses {0, ..., n}
			initial.append(DangerousCrossing.LEFT)
			goal.append(DangerousCrossing.RIGHT)

		# Call super constructor
		super().__init__(initial, goal)

		# Specify fields unique to this Problem
		crossingTime.sort()
		crossingTimeAdjusted = [0]
		crossingTimeAdjusted.extend(crossingTime)

		self.n = n

		if (n != len(crossingTime)): # Sanity/Validation check
			self.n = len(crossingTime)

		self.crossingTime = crossingTimeAdjusted
		self.mimimumTime = minimumTime


	def actions(self, state):
		"""
			Implmenting this abstract method from the superclass Problem

			Return the actions that can be executed in the given state.

			An action of this Dangerous Crossing Problem is either 1 or 2 people cross from the side of
			the bridge (either LEFT or RIGHT) that the flashlight is situated at, to the other side of the bridge
			(and taking the flashlight with them)

			Thus we represent a single action as a list of integers where each element (an integer) is the person(s)
			that are crossing the bridge

			Thus the list of actions that is returned is a list of lists (each action is a list)

			:param state: a State in the State Space, a list of integer bits (of size n+1)
							representing the location of the n people and the flashlight
			:return: a list of actions that can be executed/performed on this state
		"""

		possibleActions = []

		flashlightLocation = state[0]

		"""
			If a person is on the side of the flashlight, then they can cross the bridge by themselves or 
			they can cross with another person who is also on their side (the side of the flashlight).
			-	So we add an action for this person crossing by themselves, and also actions for them crossing
				with other people (each of these actions is them crossing with one of these other 
				people, making 2 of them crossing the bridge)
				
			Note that person i and person j crossing the bridge is the same action as person j and person i crossing, 
			and we only want to add this action once so when determining the people that person i can cross with 
			we look at people who come after this person i (a person j where j > i) 
		"""

		for personI in range(1, self.n+1): # exclude the flashlight - only traverse the peoples' locations
			if state[personI] == flashlightLocation: #This person can cross the bridge
				action = [personI] # This person (person i) can cross bridge on their own (with the flashlight)
				possibleActions.append(action)
				for personJ in range(personI+1, self.n+1):
					if state[personJ] == flashlightLocation:  # This person (person j) can cross the bridge
						action = [personI, personJ] # person i can cross the bridge with person j (and the flashlight)
						possibleActions.append(action)

		return possibleActions


	def result(self, state: list, action: list):
		"""
			Implmenting this abstract method from the superclass Problem

			Return the State that results from executing/performing the specified action in the specified state

			The Resultant State has the flashlight and the 1 or 2 persons moved from the side of
			the bridge (either LEFT or RIGHT) that they are currently situated/located at,
			to the other side of the bridge
			-	if they on LEFT then move to RIGHT
			-	if they on RIGHT then move to LEFT

			Remember that state is a list of integer bits (of size n+1) representing the location of the flashlight and the n people
			Remeber that an element in the action list represents a person [their number/ID] (which in a state representation, is an index)

			:param state: a State in the State Space that the action is applied to
			:param action: a list of either 1 or 2 persons (their index/number) crossing to the other side of the bridge
			:return: the resulting state from executing the action on this state
		"""

		# Initialise resultState
		resultState = state.copy() # a list of n+1 integer elements (elements = {LEFT, RIGHT})

		flashlightLocation = state[0]

		# The side of the bridge to move to
		# initialise to the RIGHT side
		#	i.e. flashlight and persons are moving from LEFT side to RIGHT side (i.e. flashlightLocation == DangerousCrossing.LEFT)
		moveToSide = DangerousCrossing.RIGHT

		#if flashlight and persons are on the RIGHT side, then they must move to LEFT side
		if flashlightLocation == DangerousCrossing.RIGHT:
			moveToSide = DangerousCrossing.LEFT

		# Move the flashlight and persons to the opposite side of the bridge
		resultState[0] = moveToSide
		for person in action:
			resultState[person] = moveToSide

		return resultState


	def action_cost(self, action):
		"""
			The cost of performing the specified action
			i.e. The time taken for 1 or 2 persons (and the flashlight) to cross from one side of the bridge to the other

			The Action Cost (Crossing Time) of this Dangerous Crossing Problem is independent of the actual/specific
			state the action is applied to and the specific resultant state, so we dont need to have the states as parameters
				- 	The Action Cost (Crossing Time) of an action is solely dependent on the action - which persons are
					crossing from one side of the bridge to the other
						-	The cost (crossing time) from crossing from LEFT to RIGHT is the same as crossing from
							RIGHT to LEFT


			Remember Problem Constraint 3:
				If two people are on the bridge together, they must travel at the pace of the slower person
					i.e. cost = MAX(crossingTime[i], crossingTime[j])


			:param action: 	The action executed/performed on the state
							i.e. A list of either 1 or 2 persons (their index) crossing to the other side of the bridge

			:return:		The cost of performing the specified action on the specified state
							i.e. The time taken for 1 or 2 persons (and the flashlight) to cross from one side
							of the bridge to the other
		"""

		if len(action) == 1:

			# Only 1 person is crossing
			personI = action[0]
			return self.crossingTime[personI]

		elif len(action) == 2:

			# 2 people are crossing
			personI = action[0]
			personJ = action[1]
			ctPersonI = self.crossingTime[personI] # the Crossing Time of the first person
			ctPersonJ = self.crossingTime[personJ] # the Crossing Time of the second person
			return max(ctPersonI, ctPersonJ)



	def path_cost(self, c, state1, action, state2):
		"""
			The path cost is the cost from starting off at the initial state and reaching the current state (state2)
				i.e. The time that has elapsed since the people started crossing the bridge till where they
				(and the flashlight) are currently located

			We are given the path cost to state1, so we just have to add to it the cost of performing the action (on state1
			to get to state2) - i.e PathCost(state2) = PathCost(state1) + ActionCost(state1 + action -> state2)
				-	The Action Cost is independent of the states (see the doc of actionCost()) but I've kept the
					state parameters in the parameter list (even though we wont need them for the Path Cost of this
					Dangerous Crossing Problem) as this function is meant to override the one in Problem

			I'm overriding this function as the super function (defined in Problem) has the step cost as 1


			:param c: 		The cost to get to state1
							i.e. The elapsed time before this action was executed

			:param state1:	A State in the State Space that the action was applied to
							i.e. the location of the flashlight and the n people
							represented as a list of integer bits (of size n+1)

			:param action: 	The action applied to state1
							i.e. (A list of) Either 1 or 2 persons crossing from one side of the bridge to the otherside

			:param state2: 	The Resultant State in the State Space from the action being applied to state 1
							i.e. the location of the flashlight and the n people after 1 or 2 persons and the flashlight
								have crossed the bridge

			:return:		The cost of the path that arrives at state2 by applying action to state1
							i.e. The time that has elapsed since the people started crossing the bridge
		"""

		return c + self.action_cost(action)



	def goal_test(self, state):
		"""
			Return True if state is the Goal State

			I'm overriding this function as the super function (defined in Problem) first checks if the
			self.goal is a list (of states) and if so, checks if the state passed in is an element of this list,
			however this Dangerous Crossing Problem defines a state as a list (and there is only 1 Goal State)
			so this check will	result in a logical error, thus we need to exclude it and just compare the state
			passed in (a list) to the Goal State (a list)

		:param state: a State in the Search Space
		:return: True if state is the Goal State otherwise False
		"""

		return self.goal == state



# ==============================================================================
# ==============================================================================
# ==============================================================================


def main():
	# n ->  number of people who wish to cross the bridge
	# Set default of n to 4 (gets applied when the user enters a non-digit)
	n = 4

	userN = input("Please enter the number of people who wish to cross the bridge: ")

	if userN.isdigit():
		n = userN

	userSS = input("Please select the Search Strategy you wish to solve the problem with:\n"
				   "\t1. Depth-First Search\n"
				   "\t2. Breadth-First Search\n"
				   "\t3. Greedy Best-First Search\n"
				   "\t4. A-Star Search\n")


	print("hello world")
	l1 = [0,1,2,3,4,5]
	l2 = []
	l2 = l1.copy()
	l2[1] = 10
	print(l1)
	print(l2)

	print()
	print("test problem")

	p = DangerousCrossing(4,[1,2,5,8],15)
	#state = p.initial
	state = [1,1,1,1,1]
	actions = p.actions(state)
	print("State:\t", state)

	for action in actions:
		print("action:\t\t", action)
		print("Action cost:\t", p.action_cost(action))
		print("new state:\t", p.result(state, action))
		print()

	#print(p.actions([0,0,1,0,1]))


main()
