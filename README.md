# MFAS

This repository contains various Integer Programming (IP) and Linear Programming (LP) methods to solve the **Minimum Feedback Arc Set (MFAS)** problem. The input file is an edge list with the format: `source, destination, weight`.

## Available Methods:
- **Linear Programming**
- **Integer Programming**
- **Combined Integer + Linear Programming**
- **Integer Programming with Indicator Constraints**

## How to Run:
To execute the code, use the following command:

```bash
python x.py test.csv 0
```

Here, test.csv is the input file, and 0 indicates that no initial set of removed edges is provided.

## Using Initial Removed Edges:
If you have previously removed edges from other approximate solutions, they can be used as an initial value. To run the program with these edges:

```bash
python x.py test.csv 1
```
Ensure the file containing the removed edges is named removed.csv.

## Warm starting:
The IP method may run for a long time for a large graph, such as graph.csv. We can split the total optimization procedure into several periods and write the model for continuous optimization. To restart an optimization from last time, we can run the code like this: 

```bash
python ip.py graph.csv 0 ipcheckpoint.sol
or
python ip-indicator.py graph.csv 0 ip-indcheckpoint.sol
```


## Output file:
The output is the linear order of the vertices.

## Feasible Solution:
Our IP method will output the latest feasible solution during its optimization. To evaluate the result of the feasible solution, we can run the code like this:

```bash
python f.py graph.csv feasible_solution.sol
```


