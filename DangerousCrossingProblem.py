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


"""
	Helper Functions (from utils.py)
"""


def is_in(elt, seq):
	"""Similar to (elt in seq), but compares with 'is', not '=='."""
	return any(x is elt for x in seq)

# =============================================================================
# =============================================================================


"""
	class Problem and Node have been taken from search.py
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


# ==============================================================================
# ==============================================================================


class DangerousCrossing(Problem):
	"""
		The problem of moving n people across a bridge (from one side of it to the other)
		in the shortest/minimum possible time.

		Constraints:
			1. Each person has a travel time (in minutes) to cross the bridge
			2. No more than two people can cross the bridge at one time
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
	LEFT = 0	# represents the left-hand side of the bridge
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
				people, making 2 of the crossing the bridge)
				
			
			
			Note that person i and person j crossing is the same action as person j and person i crossing, and 
			we only want to add this action once so when determining the people that person i can cross with 
			we look at people who come after this person i (a person j where j > i) 
		"""
		for person in range(1, self.n+1): # exclude the flashlight - only traverse the peoples' locations
			if state[person] == flashlightLocation: #This person can cross the bridge
				action = [person] # This person can cross bridge on their own (with the flashlight)
				possibleActions.append(action)
				for person2 in range(person+1, self.n+1):
					if state[person2] == flashlightLocation:  # This person (person2) can cross the bridge
						action = [person, person2] # person can cross the bridge with person2 (and the flashlight)
						possibleActions.append(action)

		return possibleActions


	def result(self, state, action):
		pass


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

	def value(self, state):
		pass



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
	l1 = [0, 1,2,3,4,5]

	p = DangerousCrossing(4,[1,2,5,8],15)
	print(p.actions(p.initial))


main()
