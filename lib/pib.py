import networkx as nx 
import numpy as np
import pandas as pd
from lib.average import get_bowling_career_avg
from lib.pagerank import get_in_strengths

def get_PIB(bowling_career_file, batsmen_vs_bowler_file):
    CBo = get_bowling_career_avg(bowling_career_file)
    
    pib = {}
    for row in np.array(pd.read_csv(batsmen_vs_bowler_file)):
        batsman, bowler, runs, dismisses = row
        if batsman not in pib:
            pib[batsman] = {}
        
        aba = runs / max(1, dismisses)
        if CBo[bowler] == 0: cbo = 1
        else: cbo = CBo[bowler]
        pib[batsman][bowler] = aba / cbo
    
    return pib

def get_PIB_graph(bowling_career_file, batsmen_vs_bowler_file):
    G = nx.DiGraph()
    pib = get_PIB(bowling_career_file, batsmen_vs_bowler_file)
    for batsman in pib:
        for bowler in pib[batsman]:
            G.add_edge(bowler, batsman, weight = pib[batsman][bowler])

    return G

def get_one_mode_projected_from_PIB(pib):
    G_und = nx.Graph()
    
    for node in pib.nodes():
        out_edges = pib.out_edges(node)

        for bowler, batter_A in out_edges:
            for bowler, batter_B in out_edges:
                if batter_A != batter_B:
                    G_und.add_edge(batter_A, batter_B)
    
    in_strengths = get_in_strengths(pib)

    G = nx.DiGraph()
    for batter_A, batter_B in G_und.edges():
        in_str_A = in_strengths[batter_A]
        in_str_B = in_strengths[batter_B]
        if in_str_A < in_str_B:
            G.add_edge(batter_A, batter_B, weight = in_str_B - in_str_A)
        else:
            G.add_edge(batter_B, batter_A, weight = in_str_A - in_str_B)

    return G