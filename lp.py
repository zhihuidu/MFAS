from gurobipy import Model, GRB
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
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

FileNameHead="lp"

# Set Gurobi license information using environment variables
os.environ['GRB_WLSACCESSID'] = 'd051c581-fa20-4960-a4fe-4a10026018c5'
os.environ['GRB_WLSSECRET'] = 'b2f5f047-f697-4b38-987d-44371d5a4e5f'
os.environ['GRB_LICENSEID'] = '2540055'

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

    tmp_edges=merged_edges.copy()
    tmp_edges=dict(sorted(tmp_edges.items(), key=lambda item:item[1]))
    sorted_edges=[]
    for (u,v), w in tmp_edges.items():
        sorted_edges.append((u,v))

    return node_list, merged_edges, in_adj, out_adj,sorted_edges


def build_from_EdgeList(edge_weights):
    G = nx.DiGraph()
    for (u,v) in edge_weights :
        G.add_edge(u,v,weight=edge_weights[(u,v)])
    return G


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



def save_checkpoint(model, filename):
    """Save the current MIP start file."""
    # Save the current solution to a .mst file
    model.write(filename)



def solve_fas_with_weighted_lp(graph,initial=False,checkpoint_file=None):
    # Initialize the Gurobi model
    # Set Gurobi license information using environment variables

    model = Model("FeedbackArcSet_Weighted_LP")
    
    model.setParam('OutputFlag', 0)    # Silent mode (turn off output)
    '''
    model.setParam('TimeLimit', 216000)    # Set a time limit of less than three days
    model.setParam('CKMUTIME', 50)  # Save checkpoint every 600 seconds (10 minutes)
    # Set parameters to prioritize speed over optimality
    model.setParam('OutputFlag', 0)    # Silent mode (turn off output)
    model.setParam('MIPGap', 0.1)      # Allow a 10% optimality gap
    model.setParam('Presolve', 2)      # Use aggressive presolve
    model.setParam('Cuts', 1)          # Moderate cut generation, larger cuts will be slow
    model.setParam('MIPFocus', 1)      # Focus on finding feasible solutions quickly,2 optimal,3 balance
    #model.setParam('Threads', 8)       # Use 8 threads
    model.setParam('SolutionLimit', 10)  # Stop after finding 10 feasible solutions
'''

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
        model.addConstr(p[u] + 1 <= p[v] + M * (1 - x[(u, v)]), f"order_{u}_{v}")

    if checkpoint_file:
        print(f"Loading checkpoint from {checkpoint_file}")
        model.read(checkpoint_file)

    else:

        if initial:
            for u, v in graph.edges():
                x[(u, v)].start = 1  # Set initial value for the edge variable
            for (u, v) in complete_removed_list:
                if graph.has_edge(u,v):
                    x[(u, v)].start = 0  # Set initial value for the edge variable

    # Optimize the model
    model.optimize()

    '''
    # Save checkpoint if optimization is interrupted
    if model.status == GRB.INTERRUPTED or model.status == GRB.TIME_LIMIT:
            save_checkpoint(model, 'lpcheckpoint.ckp')
'''

    print("after optimization")

    up_bound = 0
    final_ordering = set()
    removed_edges = []
    # Check if the optimization was successful
    if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
        # Retrieve the fractional edge removal solution
        #fractional_edges = { (u, v): x[(u, v)].x for u, v in graph.edges() }
        up_bound = sum(graph[u][v]['weight'] * x[(u, v)].X for u, v in graph.edges())


        # Apply rounding to get the final optimal result (binary solution for edge removal)
        removed_edges = [(u, v) for u, v in graph.edges() if x[(u, v)].X < 0.5]

        # Final vertex ordering
        final_ordering = sorted(graph.nodes(), key=lambda v: p[v].X)
        '''
        for u,v in graph.edges():
            print(f"X[{u},{v}] is {x[(u,v)].X}")
        for v in final_ordering:
            print(f"p[{v}] is {p[v].X} ")
'''
    return removed_edges, final_ordering, up_bound



def process_graph(file_path,precondition):
    print(f"read data")
    node_list, edge_weights, in_edges, out_edges,sorted_edges = build_ArrayDataStructure(file_path)
    G=build_from_EdgeList(edge_weights)
    total=sum(edge_weights[(u,v)] for (u,v) in edge_weights)
    print(f"total number of nodes={len(node_list)}, total number of edges={len(edge_weights)}")
    print(f"sum of weight={total}")

    edge_flag={(u,v):1 for (u,v) in edge_weights }
    Init_flag=False
    if precondition==1:
        old_edge_flag=edge_flag.copy()
        removed_weight=read_removed_edges("removed.csv",edge_flag )
        print(f"to here removed weight is {removed_weight}, percentage is {removed_weight/total*100}")
        generate_complete_removed_list(edge_flag,edge_weights)
        print(f"length of the complete removed list is {len(complete_removed_list)}")
        edge_flag=old_edge_flag
        Init_flag=True

    removed_edges, vertex_ordering, up_bound=solve_fas_with_weighted_lp(G,Init_flag,None)
    #removed_edges, vertex_ordering, up_bound=solve_fas_with_weighted_lp(G,True,"lpcheckpoint.mst")

    print(f"up bound is {up_bound}")
    if up_bound != 0:
        removed_weight= sum(edge_weights[(u,v)]  for u, v in removed_edges )
        if removed_weight/total < 0.45 :
            mapping={}
            for i in range(len(vertex_ordering)):
                mapping[vertex_ordering[i]]=i

            print(f"write file")
            current_time = datetime.now()
            time_string = current_time.strftime("%Y%m%d_%H%M%S")
            output_file = f"{FileNameHead}-Relabel-{time_string}.csv"
            write_relabelled_nodes_to_file(mapping, output_file)
            output_file = f"{FileNameHead}-removed-edge-{time_string}.csv"
            with open(output_file, 'w') as f:
                for u,v in removed_edges:
                    f.write(f"{u},{v},{G[u][v]['weight']}\n")

file_path = sys.argv[1]
precondition=int(sys.argv[2])
process_graph(file_path,precondition)

