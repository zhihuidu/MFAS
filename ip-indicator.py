import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
from gurobipy import GRB,Model
import sys
import os
from datetime import datetime
import time
import csv
import random
from itertools import permutations
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

FileNameHead="ip-indicator"

EarlyExit=False

# Set Gurobi license information using environment variables
os.environ['GRB_WLSACCESSID'] = 'fb436391-3bb5-4b06-9a8c-66f0354b5011'
os.environ['GRB_WLSSECRET'] = '37c29f28-6ae4-4a19-913d-6b8100964563'
os.environ['GRB_LICENSEID'] = '2540055'

# Optional: Disable the local license check by unsetting GRB_LICENSE_FILE
#os.environ.pop('GRB_LICENSE_FILE', None)


removed_list=[]
complete_removed_list=[]


def generate_complete_removed_list(edge_flag,edge_weights):
    addnum=0
    sourceset=set()
    targetset=set()
    global complete_removed_list
    complete_removed_list=[]
    weight_list=[]
    for (u,v),flag in edge_flag.items():
        if flag==0:
           complete_removed_list.append((u,v))
           weight_list.append(edge_weights[(u,v)])

    tmp_list=[x for _,x in sorted(zip(weight_list, complete_removed_list))]
    complete_removed_list=tmp_list



#given a graph file in the csv format (each line is (source,destination, weight)), generate the graph data structure
def build_ArrayDataStructure(csv_file_path):
    node_list = set()
    edges = []
    with open(csv_file_path, mode='r') as csvfile:
        csvreader = csv.reader(csvfile)
        #next(csvreader)

        for row in csvreader:
            source, target, weight = row
            source = int(source)
            target = int(target)
            weight = int(weight) #only weight should be integer, vertex ID can be a string
            if source != target :
                edges.append((source, target, weight))
            node_list.add(source)
            node_list.add(target)

    node_list = list(node_list)

    #here we merge multiple edges
    merged_edges = {}
    for source, dest, weight in edges:
        if (source, dest) in merged_edges:
            merged_edges[(source, dest)] += weight
        else:
            merged_edges[(source, dest)] = weight

    in_adj={}
    out_adj={}
    for node in node_list:
        out_adj[node]=[]
        in_adj[node]=[]
    for source, target in merged_edges:
        out_adj[source].append((target, merged_edges[(source,target)]))
        in_adj[target].append((source, merged_edges[(source,target)]))

    return node_list, merged_edges, in_adj, out_adj 


def build_from_EdgeAndFlag(edge_weights,edge_flag):
    G = nx.DiGraph()
    for (u,v) in edge_weights :
         if edge_flag[(u,v)]==1:
              G.add_edge(u,v,weight=edge_weights[(u,v)])
    return G

def build_from_GraphAndFlag (G,edge_flag):
    shrunkG = nx.DiGraph()
    for u, v, data in G.edges(data=True):
         if edge_flag[(u,v)]==1:
              shrunkG.add_edge(u,v,weight=data['weight'])
    return shrunkG


def build_from_EdgeList(edge_weights):
    G = nx.DiGraph()
    for (u,v) in edge_weights :
        G.add_edge(u,v,weight=edge_weights[(u,v)])
    return G

# Relabel the vertices in the DAG
def relabel_dag(G_dag):
    topological_order = list(nx.topological_sort(G_dag))
    mapping = {node: i for i, node in enumerate(topological_order)}
    return mapping

# Write the original vertex ID and its relative order to a file
def write_relabelled_nodes_to_file(mapping, output_file):
    with open(output_file, 'w') as f:
        for node, order in mapping.items():
            f.write(f"{node},{order}\n")

def read_removed_edges(file_path,edge_flag):
    removed_weight=0
    with open(file_path,mode='r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            source = int(row[0])
            dest = int(row[1])
            weight = int(row[2])  
            if edge_flag[(source,dest)]==1:
                edge_flag[(source,dest)]=0
                removed_weight+=weight
    return removed_weight


# Write the edge flag array
def write_removed_edges(output_file,edge_flag,edge_weights):
    with open(output_file, 'w') as f:
        for u,v in edge_flag:
            if edge_flag[(u,v)]==0:
                f.write(f"{u},{v},{edge_weights[(u,v)]}\n")


num=0

#remove cycle in G and update edge_flag
def solve_ip_scc(G,edge_flag,SuperG):
    global num
    print(f"num of nodes is {G.number_of_nodes()}, num of edges is {G.number_of_edges()}\n")
    removed_weight=0
    model = gp.Model("min_feedback_arc_set")
    model.setParam('OutputFlag', 0)  # Silent mode

    # Create binary variables for each edge
    edge_vars = {}
    for u, v, data in G.edges(data=True):
        edge_vars[(u, v)] = model.addVar(vtype=GRB.BINARY, obj=data['weight'])

    cycles = nx.simple_cycles(G)
    for cycle in cycles:
        #print(f"the cycle is {cycle}")
        cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
        model.addConstr(gp.quicksum(edge_vars[edge] for edge in cycle_edges) >= 1)

    # Optimize the model
    model.optimize()

    # Get the edges to be removed
    for edge, var in edge_vars.items():
            if var.x > 0.5:
               if edge_flag[(edge[0],edge[1])]==1:
                    removed_weight+=G[edge[0]][edge[1]]['weight']
                    edge_flag[(edge[0],edge[1])]=0
                    SuperG.remove_edge(edge[0],edge[1])
                    num+=1
    return removed_weight


def solve_fas_with_weighted_lp(graph,edge_flag):
    # Initialize the Gurobi model
    model = Model("FeedbackArcSet_Weighted_LP")

    model.setParam('OutputFlag', 0)  # Silent mode
    epsilon = 1e-6  # A small constant to enforce strict inequality
    # Variables: x_uv for each edge (relaxed between 0 and 1), and p_v for each vertex v
    x = {}
    p = {}
    M = len(graph.nodes())  # Large constant, usually the number of vertices

    # Decision variables for each edge (continuous between 0 and 1)
    for u, v in graph.edges():
        x[(u, v)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"x_{u}_{v}")

    print("add x variable")
    # Position variables for each vertex (continuous)
    for v in graph.nodes():
        p[v] = model.addVar(vtype=GRB.CONTINUOUS,lb=0,ub=M-1, name=f"p_{v}")

    print("add p variable")
    # Objective: minimize the total weight of edges removed
    model.setObjective(sum(graph[u][v]['weight'] * (1 - x[(u, v)]) for u, v in graph.edges()), GRB.MINIMIZE)

    print("set objective")
    # Constraints: Linear ordering constraints (relaxed for fractional x_uv)
    for u, v in graph.edges():
        # p_u < p_v if edge (u, v) is kept (x_uv close to 1)
        model.addConstr(p[u] + 0.001 <= p[v] + M * (1 - x[(u, v)]), f"order_{u}_{v}")

    # Optimize the model
    model.optimize()

    print("after optimization")
    # Retrieve the fractional edge removal solution
    #fractional_edges = { (u, v): x[(u, v)].x for u, v in graph.edges() }
    removed_weight = sum(graph[u][v]['weight']  for u, v in graph.edges()  if x[(u, v)].x < 0.2 )

    # Retrieve the linear ordering based on the positions of vertices (p_v)
    vertex_ordering = sorted(graph.nodes(), key=lambda v: p[v].x)

    # Apply rounding to get the final optimal result (binary solution for edge removal)
    #removed_edges = [(u, v) for u, v in graph.edges() if fractional_edges[(u, v)] < 0.5]
    for u, v in graph.edges() :
        if x[(u, v)].x < 0.2:
            edge_flag[(u,v)]=0

    # Apply rounding to get the final optimal result (binary solution for edge removal)
    #removed_edges = [(u, v) for u, v in graph.edges() if fractional_edges[(u, v)] < 0.5]

    # Final vertex ordering
    #final_ordering = sorted(graph.nodes(), key=lambda v: p[v].x)

    return removed_weight



def solve_fas_with_weighted_ip(graph,edge_flag):
    # Initialize the Gurobi model
    model = Model("FeedbackArcSet_Weighted_IP")
    epsilon = 1e-6  # A small constant to enforce strict inequality
 
    #model.setParam('OutputFlag', 0)  # Silent mode
    # Variables: x_uv for each edge (binary), and p_v for each vertex (position)
    x = {}
    p = {}
    M = len(graph.nodes())  # Large constant, typically the number of vertices

    # Decision variables for each edge (binary: 0 if removed, 1 if kept)
    for u, v in graph.edges():
        x[(u, v)] = model.addVar(vtype=GRB.BINARY, name=f"x_{u}_{v}")

    # Position variables for each vertex (continuous, representing topological position)
    for v in graph.nodes():
        p[v] = model.addVar(vtype=GRB.CONTINUOUS,lb=0,ub=M-1, name=f"p_{v}")

    # Objective: minimize the total weight of removed edges
    model.setObjective(sum(graph[u][v]['weight'] * (1 - x[(u, v)]) for u, v in graph.edges()), GRB.MINIMIZE)

    # Constraints: Linear ordering constraints for cycle elimination (no cycles)
    for u, v in graph.edges():
        # If x_uv = 1 (edge is kept), then p_u must come before p_v
        model.addConstr(p[u]+ 1 <= p[v] + M * (1 - x[(u, v)]), f"order_{u}_{v}")
    '''
    # Constraints: For every bidirectional edge (u, v) and (v, u), ensure that at least one is removed
    for u, v in graph.edges():
        if (v, u) in graph.edges():
            model.addConstr(x[(u, v)] + x[(v, u)] <= 1, f"no_cycle_2_{u}_{v}")
'''
    # Optimize the model
    model.optimize()

    # Retrieve the final optimal removed edges (where x_uv = 0, meaning edge is removed)
    #removed_edges = [(u, v) for u, v in graph.edges() if x[(u, v)].x < 0.5]
    removed_weight = sum(graph[u][v]['weight']  for u, v in graph.edges()  if x[(u, v)].x < 0.5 )
    removededge=[]
    for u, v in graph.edges() :
        if x[(u, v)].x < 0.5:
            edge_flag[(u,v)]=0
            removededge.append((u,v))

    for (u,v) in removededge:
        graph.remove_edge(u,v)

    # Retrieve the linear ordering based on the positions of vertices (p_v)
    #vertex_ordering = sorted(graph.nodes(), key=lambda v: p[v].x)

    #up_bound = sum(graph[u][v]['weight'] * x[(u, v)] for u, v in graph.edges())

    return removed_weight



def solve_indicator(graph,edge_flag):
    # Initialize the Gurobi model
    model = gp.Model("MaxWeightDirectedGraph")
    model.setParam('OutputFlag', 0)  # Silent mode

    # Variables: continuous labels for each node, and binary values for each edge
    p = {}  # Continuous label for each node
    x = {}  # Binary variable for each edge (0 or 1)

    epsilon = 1e-6  # Small constant to ensure strict inequality


    M=len(graph.nodes())
    # Create continuous label variables for each node
    for v in graph.nodes():
        p[v] = model.addVar(vtype=GRB.CONTINUOUS,lb=0,ub=M-1, name=f"p_{v}")

    print(f"add p variable")
    # Create binary variables for each edge and add indicator constraints
    for u, v in graph.edges():
        x[(u, v)] = model.addVar(vtype=GRB.BINARY, name=f"x_{u}_{v}")


    for u, v in graph.edges():
        # If x_uv = 0, enforce p_u >= p_v + epsilon (i.e., p_u > p_v)
        #model.addGenConstrIndicator(x[(u, v)], False, p[u] >= p[v] , name=f"remove_edge_{u}_{v}")
        # If x_uv = 1, enforce p_u + epsilon <= p_v (i.e., p_u < p_v)
        model.addGenConstrIndicator(x[(u, v)], True, p[u] + 1  <= p[v], name=f"keep_edge_{u}_{v}")



    print(f"add edge variable and indicator")
    # Objective: maximize the total weight of the kept edges
    model.setObjective(gp.quicksum(graph[u][v]['weight'] *(1- x[(u, v)]) for u, v in graph.edges()), GRB.MINIMIZE)

    print(f"add objective")
    # Optimize the model
    model.optimize()

    print(f"optimization")
    # Retrieve results
    removed_weight=0
    removededge=[]
    for u, v in graph.edges() :
        print(f"x[({u},{v})].x is {x[(u,v)].x}")
        print(f"p[{u}] is {p[u].x}, p[{v}] is {p[v].x}")
        if x[(u, v)].x < 0.5:
            edge_flag[u,v]=0
            removed_weight+=graph[u][v]['weight']
            removededge.append((u,v))
    for (u,v) in removededge:
        graph.remove_edge(u,v)
    #node_labels = {v: p[v].x for v in graph.nodes()}

    return removed_weight


# Define a callback function
def mycallback(model, where):
    if where == GRB.Callback.MIPSOL:  # A new feasible solution is found
        # Get the current solution
        solution = model.cbGetSolution(model.getVars())

        # Write the solution to a file
        with open("feasible_solution.sol", "w") as f:
            for v in model.getVars():
                f.write(f"{v.varName} {model.cbGetSolution(v)}\n")
        print("Feasible solution written to feasible_solution.sol")


def solve_indicator_half_linear(graph,edge_flag,initial=False,checkpoint_file=None):
    global EarlyExit
    # Initialize the Gurobi model
    model = gp.Model("MaxWeightDirectedGraph")
    #model.setParam('Threads', 64)

    #model.setParam('OutputFlag', 0)  # Silent mode

    model.setParam('TimeLimit', 172800)    # Set a time limit of 3600*48 seconds
    '''
    # Set parameters to prioritize speed over optimality
    model.setParam('MIPGap', 0.1)      # Allow a 10% optimality gap
    #model.setParam('TimeLimit', 7200)    # Set a time limit of 30 seconds
    model.setParam('Presolve', 2)      # Use aggressive presolve
    model.setParam('Cuts', 1)          # Moderate cut generation, larger cuts will be slow
    model.setParam('MIPFocus', 1)      # Focus on finding feasible solutions quickly,2 optimal,3 balance
    #model.setParam('Threads', 8)       # Use 8 threads
    model.setParam('SolutionLimit', 10)  # Stop after finding 10 feasible solutions
'''


    # Variables: continuous labels for each node, and binary values for each edge
    p = {}  # Continuous label for each node
    x = {}  # Binary variable for each edge (0 or 1)

    epsilon = 1e-6  # Small constant to ensure strict inequality


    M=len(graph.nodes())
    # Create continuous label variables for each node
    for v in graph.nodes():
        p[v] = model.addVar(vtype=GRB.CONTINUOUS,lb=0,ub=M-1, name=f"p_{v}")

    print(f"add p variable")
    # Create binary variables for each edge and add indicator constraints
    for u, v in graph.edges():
        x[(u, v)] = model.addVar(vtype=GRB.BINARY,lb=0,ub=1, name=f"x_{u}_{v}")


    for u, v in graph.edges():
        # If x_uv = 0, enforce p_u >= p_v + epsilon (i.e., p_u > p_v)
        #model.addGenConstrIndicator(x[(u, v)], False, p[u] >= p[v] , name=f"remove_edge_{u}_{v}")
        # If x_uv = 1, enforce p_u + epsilon <= p_v (i.e., p_u < p_v)
        model.addGenConstrIndicator(x[(u, v)], True, p[u] + 1  <= p[v], name=f"keep_edge_{u}_{v}")



    print(f"add edge variable and indicator")
    # Objective: maximize the total weight of the kept edges
    model.setObjective(gp.quicksum(graph[u][v]['weight'] *(1- x[(u, v)]) for u, v in graph.edges()), GRB.MINIMIZE)

    print(f"add objective")

    if checkpoint_file != None:
        print(f"Update the model")
        model.update()
        print(f"Loading checkpoint from {checkpoint_file}")
        model.read(checkpoint_file)
        print(f"Starting new optimization")

    else:

        if initial:
            for u, v in graph.edges():
                x[(u, v)].start = 1  # Set initial value for the edge variable
            for (u, v) in complete_removed_list:
                if graph.has_edge(u,v):
                    x[(u, v)].start = 0  # Set initial value for the edge variable


    # Optimize the model
    model.optimize(mycallback)

    print(f"optimization")


    # Save checkpoint if optimization is interrupted
    if model.status == GRB.INTERRUPTED or model.status == GRB.TIME_LIMIT:
            print(f"write model")
            model.write('ip-indcheckpoint.sol')
            EarlyExit=True
            return 0


    # Retrieve results
    removed_weight=0
    removededge=[]
    for u, v in graph.edges() :
        print(f"x[({u},{v})].x is {x[(u,v)].X}")
        print(f"p[{u}] is {p[u].X}, p[{v}] is {p[v].X}")
        if x[(u, v)].X < 0.5:
            edge_flag[u,v]=0
            removed_weight+=graph[u][v]['weight']
            removededge.append((u,v))
    for (u,v) in removededge:
        graph.remove_edge(u,v)
    #node_labels = {v: p[v].x for v in graph.nodes()}

    return removed_weight



def process_graph(file_path,precondition=0,checkpoint_file=None):
    global EarlyExit
    print(f"read data")
    node_list, edge_weights, in_edges, out_edges= build_ArrayDataStructure(file_path)
    G=build_from_EdgeList(edge_weights)
    total=sum(edge_weights[(u,v)] for (u,v) in edge_weights)
    print(f"total number of nodes={len(node_list)}, total number of edges={len(edge_weights)}")
    print(f"sum of weight={total}")

    edge_flag={(u,v):1 for (u,v) in edge_weights }
    Init_flag=False
    removed_weight=0
    if precondition==1:
        old_edge_flag=edge_flag.copy()
        if "test.csv" in file_path:
            removed_weight=read_removed_edges("test_removed.csv",edge_flag )
        else :
            removed_weight=read_removed_edges("removed.csv",edge_flag )
        print(f"to here removed weight is {removed_weight}, percentage is {removed_weight/total*100}")
        generate_complete_removed_list(edge_flag,edge_weights)
        print(f"length of the complete removed list is {len(complete_removed_list)}")
        edge_flag=old_edge_flag
        Init_flag=True

    removed_weight=0



    shG=G.copy()


    numcheckacyclic=0
    acyclic_flag=nx.is_directed_acyclic_graph(shG)
    addback_flag=False
    while not acyclic_flag :
        scc=list(nx.strongly_connected_components(shG))
        print(f"number of scc is {len(scc)}")

        numcomponent=0
        oldnum=num
        numcheckacyclic+=1

        for component in scc:
            if len(component)==1:
                 continue

            numcomponent+=1
            print(f"{numcheckacyclic} check, handle the {numcomponent}th component with size {len(component)}")
            subnum=0
            G_sub = shG.subgraph(component).copy()
            if len(component)<1000:
                try:
                     removed_weight1=solve_fas_with_weighted_ip(G_sub,edge_flag)
                     removed_weight+=removed_weight1
                     print(f"The {numcomponent}th component, removed weight is {removed_weight1}, totally removed {removed_weight}, percentage is {removed_weight/total*100}\n")
                     acyclic_flag=nx.is_directed_acyclic_graph(G_sub)
                     if acyclic_flag :
                         print("no cycle")
                     else:
                         print("still has cycles, wrong")
                except ValueError as e:
                     print(f"Caught an error  {e}")
            else:
                try:
                     removed_weight1=0
                     if checkpoint_file==None:
                         removed_weight1=solve_indicator_half_linear(G_sub,edge_flag,Init_flag)
                     else:
                         removed_weight1=solve_indicator_half_linear(G_sub,edge_flag,False,"ip-indcheckpoint.sol")
                     if EarlyExit:
                         return 0
                     removed_weight+=removed_weight1
                     print(f"The {numcomponent}th component, removed weight is {removed_weight1}, totally removed {removed_weight}, percentage is {removed_weight/total*100}\n")
                     acyclic_flag=nx.is_directed_acyclic_graph(G_sub)
                     if acyclic_flag :
                         print("no cycle")
                     else:
                         print("still has cycles, wrong")
                except ValueError as e:
                     print(f"Caught an error  {e}")




        shG=build_from_EdgeAndFlag(edge_weights,edge_flag)
        acyclic_flag=nx.is_directed_acyclic_graph(shG)

    print(f"relabel dag")
    shG=build_from_EdgeAndFlag(edge_weights,edge_flag)
    acyclic_flag=nx.is_directed_acyclic_graph(shG)
    if acyclic_flag:
        mapping = relabel_dag(shG)
        print(f"write file")
        current_time = datetime.now()
        time_string = current_time.strftime("%Y%m%d_%H%M%S")
        output_file = f"{FileNameHead}-Relabel-{time_string}.csv"
        write_relabelled_nodes_to_file(mapping, output_file)
    else:
        print(f"not acyclic graph")


file_path = sys.argv[1]
precondition=False
checkpoint=None
if len(sys.argv)>2:
    precondition=int(sys.argv[2])
    if len(sys.argv)>3:
        checkpoint=sys.argv[3]

process_graph(file_path,precondition,checkpoint)

