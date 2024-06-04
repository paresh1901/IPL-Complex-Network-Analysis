import networkx as nx 
import numpy as np
import pandas as pd
from lib.average import get_batting_career_avg, get_bowling_career_avg
from lib.pagerank import get_in_strengths

def get_QIB(batting_career_file, bowling_career_file, bowler_vs_batsmen_file):
    CBa = get_batting_career_avg(batting_career_file)
    CBo = get_bowling_career_avg(bowling_career_file)
    
    qib = {}
    for row in np.array(pd.read_csv(bowler_vs_batsmen_file)):
        bowler, batsman, runs, dismisses = row
        if bowler not in qib:
            qib[bowler] = {}
        
        cba = CBa[batsman]
        if CBo[bowler] == 0: cbo = 1
        else: cbo = CBo[bowler]
        d = dismisses
        
        qib[bowler][batsman] = d * (cba / cbo)
    
    return qib

def get_QIB_graph(batting_career_file, bowling_career_file, bowler_vs_batsmen_file):
    G = nx.DiGraph()
    qib = get_QIB(batting_career_file, bowling_career_file, bowler_vs_batsmen_file)
    for bowler in qib:
        for batsman in qib[bowler]:
            G.add_edge(batsman, bowler, weight = qib[bowler][batsman])

    return G

def get_one_mode_projected_from_QIB(qib):
    G_und = nx.Graph()

    for node in qib.nodes():
        out_edges = qib.out_edges(node)

        for batter, bowler_A in out_edges:
            for batter, bowler_B in out_edges:
                if bowler_A != bowler_B:
                    G_und.add_edge(bowler_A, bowler_B)
    
    in_strengths = get_in_strengths(qib)

    G = nx.DiGraph()
    for bowler_A, bowler_B in G_und.edges():
        in_str_A = in_strengths[bowler_A]
        in_str_B = in_strengths[bowler_B]
        if in_str_A < in_str_B:
            G.add_edge(bowler_A, bowler_B, weight = in_str_B - in_str_A)
        else:
            G.add_edge(bowler_B, bowler_A, weight = in_str_A - in_str_B)

    return G