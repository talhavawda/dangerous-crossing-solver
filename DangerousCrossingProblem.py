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

"""


def main():
	# n ->  number of people who wish to cross the bridge
	# Set default of n to 4 (gets applied when the user enters a non-digit)
	n = 4

	userN = input("Please enter the number of people who wish to cross the bridge: ")

	if userN.isdigit():
		n = userN

	userSS = input("Please select the Search Strategy you wish to solve the problem with:"
				   "\n1. Depth-First Search"
				   "\n2. Breadth-First Search"
				   "\n3. Greedy Best-First Search"
				   "\n4. A-Star Search")



main()
