import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
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
top25_results_dir= 'results/top25'
convergence_plot_dir= 'results/convergence_plots'

# Check if the results directory exists, if not, create it
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
# Check if the results directory exists, if not, create it
if not os.path.exists(top25_results_dir):
    os.makedirs(top25_results_dir)
    
# Check if the results directory exists, if not, create it
if not os.path.exists(convergence_plot_dir):
    os.makedirs(convergence_plot_dir)

QIB_graph = get_QIB_graph(batting_career_file, bowling_career_file, bowler_vs_batsmen_file)
QIB_proj = get_one_mode_projected_from_QIB(QIB_graph)

# this gives i->j convention
QIB_adj = nx.linalg.graphmatrix.adjacency_matrix(QIB_proj).todense()
QIB_D_inv = get_D_inv(QIB_proj)

scores, change = pageRank(QIB_adj, QIB_D_inv, 25, QIB_proj)
plt.figure()
plt.plot(change)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Difference')
plt.title('Convergence (QIB)')
plt.savefig("results/convergence_plots/qib_convergence_plot.png")
# plt.show()
plt.close()
    
n = len(scores)
ranks = sorted(range(n), key=lambda x: scores[x], reverse=True)
bowlers = list(QIB_proj.nodes())
CBo = get_bowling_career_avg(bowling_career_file)
CBo_in_strengths = get_in_strengths(QIB_graph)

data = []
for i in range(n):
    # score = round(float(scores[ranks[i]]), 5)
    # score = str(score).ljust(5+2)
    
    score_element = scores[ranks[i]]  # Ensure the element is a scalar
    
        # Convert to scalar if needed
    if isinstance(score_element, (list, np.ndarray)):
        score_element = score_element.item()
        
    score = round(float(score_element), 5)
    score = str(score).ljust(5 + 2)
    rank = str(i+1).zfill(2)
    bowler = bowlers[ranks[i]]
    cbo = round(CBo[bowler], 5)
    cbo_in_strength = round(CBo_in_strengths[bowler], 5)
    data.append([rank, bowler, cbo_in_strength, score, cbo])

headers = ["Rank", "Bowler", "In Strength", "PageRank Score", "Bowling Average"]

# Write the table to a CSV file
with open(os.path.join(results_dir, "bowlers_ranks_by_pagerank.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)  # Write the headers
    writer.writerows(data)  # Write the data rows


# Ranking based on in-strength scores
ranks_by_in_strength = sorted(range(n), key=lambda x: CBo_in_strengths[bowlers[x]], reverse=True)
data_in_strength = []
for i in range(n):
    rank = str(i+1).zfill(2)
    bowler = bowlers[ranks_by_in_strength[i]]
    cbo_in_strength = round(CBo_in_strengths[bowler], 5)
    score_element = scores[ranks_by_in_strength[i]]  # Ensure the element is a scalar
    
        # Convert to scalar if needed
    if isinstance(score_element, (list, np.ndarray)):
        score_element = score_element.item()
        
    score = round(float(score_element), 5)
    score = str(score).ljust(5 + 2)
#   score = round(float(scores[ranks_by_in_strength[i]]), 5)
    cbo = round(CBo[bowler], 5)
    data_in_strength.append([rank, bowler, cbo_in_strength, score, cbo])

# Write the in-strength table to a CSV file
with open(os.path.join(results_dir, "bowlers_ranks_by_in_strength.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)  # Write the headers
    writer.writerows(data_in_strength)  # Write the data rows

# Ranking based on bowling average scores (lower is better)
ranks_by_bowling_avg = sorted(range(n), key=lambda x: CBo[bowlers[x]])
data_bowling_avg = []
for i in range(n):
    rank = str(i+1).zfill(2)
    bowler = bowlers[ranks_by_bowling_avg[i]]
    cbo = round(CBo[bowler], 5)
    cbo_in_strength = round(CBo_in_strengths[bowler], 5)
    score_element = scores[ranks_by_bowling_avg[i]]  # Ensure the element is a scalar
    
        # Convert to scalar if needed
    if isinstance(score_element, (list, np.ndarray)):
        score_element = score_element.item()
        
    score = round(float(score_element), 5)
    score = str(score).ljust(5 + 2)
    # score = round(float(scores[ranks_by_bowling_avg[i]]), 5)
    data_bowling_avg.append([rank, bowler, cbo_in_strength, score, cbo])

# Write the bowling average table to a CSV file
with open(os.path.join(results_dir, "bowlers_ranks_by_bowling_avg.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)  # Write the headers
    writer.writerows(data_bowling_avg)  # Write the data rows
    
# Extracting data for the top 25 bowlers based on in-strength ranking
top25_bowlers_data = data_in_strength[:25]

# Write the top 25 bowlers data to a CSV file
top25_headers = ["Rank", "Bowler", "In Strength", "PageRank Score", "Bowling Average"]
with open(os.path.join(top25_results_dir, "top25_bowlers.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(top25_headers)  # Write the headers
    writer.writerows(top25_bowlers_data)  # Write the data rows


print("Ranking tables have been generated and saved in the results directory.")