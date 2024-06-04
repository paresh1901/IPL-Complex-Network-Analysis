import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
from lib.average import get_batting_career_avg, get_bowling_career_avg
from lib.pagerank import pageRank, get_D_inv, get_in_strengths
from lib.pib import get_PIB_graph, get_one_mode_projected_from_PIB
from lib.qib import get_QIB_graph, get_one_mode_projected_from_QIB


batting_career_file = 'data/stats/batsmen_averages.csv'
bowling_career_file = 'data/stats/bowlers_averages.csv'
batsmen_vs_bowler_file = 'data/stats/batsmen_against_bowlers_averages.csv'
bowler_vs_batsmen_file = 'data/stats/bowlers_against_batsmen_averages.csv'
results_dir = 'results'

# Check if the results directory exists, if not, create it
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

PIB_graph = get_PIB_graph(bowling_career_file, batsmen_vs_bowler_file)
PIB_proj = get_one_mode_projected_from_PIB(PIB_graph)

# this gives i->j convention
PIB_adj = nx.linalg.graphmatrix.adjacency_matrix(PIB_proj).todense()
PIB_D_inv = get_D_inv(PIB_proj)

scores, change = pageRank(PIB_adj, PIB_D_inv, 25, PIB_proj)
plt.figure()
plt.plot(change)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Difference')
plt.title('Convergence (PIB)')


QIB_graph = get_QIB_graph(batting_career_file, bowling_career_file, bowler_vs_batsmen_file)
QIB_proj = get_one_mode_projected_from_QIB(QIB_graph)

# this gives i->j convention
QIB_adj = nx.linalg.graphmatrix.adjacency_matrix(QIB_proj).todense()
QIB_D_inv = get_D_inv(QIB_proj)

scores, change = pageRank(QIB_adj, QIB_D_inv, 25, QIB_proj)

plt.plot(change)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Difference')
plt.title('Convergence (QIB)')
plt.savefig("results/convergence_plots/pib_qib_convergence_plot.png")
