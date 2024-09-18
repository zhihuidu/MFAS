# MFAS
This repository contains various Integer Programming (IP) and Linear Programming (LP) methods to solve the Minimum Feedback Arc Set (MFAS) problem. The input file is an edge list with the format: source, destination, weight.

Available Methods:

Linear Programming

Integer Programming

Combined Integer + Linear Programming

Integer Programming with Indicator Constraints

How to Run:
To execute the code, use the following command:

Output file:
The output is the linear order of the vertices.

python x.py test.csv 0
Here, test.csv is the input file, and 0 indicates that no initial set of removed edges is provided.
Using Initial Removed Edges:
If you have previously removed edges from other approximate solutions, they can be used as an initial value. To run the program with these edges:

python x.py test.csv 1
Ensure the file containing the removed edges is named removed.csv.

Example Files:
graph.csv: A large graph representing a real-world problem.
removed.csv: The file containing edges removed by other methods.

