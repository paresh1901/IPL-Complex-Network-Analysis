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
plt.savefig("results/convergence_plots/pib_convergence_plot.png")
# plt.show()
plt.close

n = len(scores)
ranks = sorted(range(n), key=lambda x: scores[x], reverse=True)
batsmen = list(PIB_proj.nodes())
CBa = get_batting_career_avg(batting_career_file)
CBa_in_strengths = get_in_strengths(PIB_graph)

data = []
for i in range(n):
    
    score_element = scores[ranks[i]]  # Ensure the element is a scalar
    
        # Convert to scalar if needed
    if isinstance(score_element, (list, np.ndarray)):
        score_element = score_element.item()
    score = round(float(score_element), 5)
    score = str(score).ljust(5 + 2)
    
    # score = round(float(scores[ranks[i]]), 5)
    # score = str(score).ljust(5+2)
    rank = str(i+1).zfill(2)
    batter = batsmen[ranks[i]]
    cba = str(round(CBa[batter], 5)).ljust(7+2)
    in_strength = round(CBa_in_strengths[batter], 5)
    data.append([rank, batter, in_strength, score, cba])

headers = ["Rank", "Batsman", "In Strength", "PageRank Score", "Batting Average"]

# Print the table
# print("Top 25 Batsman")
# print(tabulate(data, headers=headers, tablefmt="grid"))

# Write the table to a CSV file
with open(os.path.join(results_dir, "batsman_ranks_by_pagerank.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)  # Write the headers
    writer.writerows(data)  # Write the data rows
    
# Ranking based on in-strength scores
ranks_by_in_strength = sorted(range(n), key=lambda x: CBa_in_strengths[batsmen[x]], reverse=True)
data_in_strength = []
for i in range(n):
    rank = str(i+1).zfill(2)
    batter = batsmen[ranks_by_in_strength[i]]
    cba_in_strength = round(CBa_in_strengths[batter], 5)
    score_element = scores[ranks_by_in_strength[i]]  # Ensure the element is a scalar
    
        # Convert to scalar if needed
    if isinstance(score_element, (list, np.ndarray)):
        score_element = score_element.item()
        
    score = round(float(score_element), 5)
    score = str(score).ljust(5 + 2)
#   score = round(float(scores[ranks_by_in_strength[i]]), 5)
    cba = round(CBa[batter], 5)
    data_in_strength.append([rank, batter, cba_in_strength, score, cba])
    
    # Write the in-strength table to a CSV file
with open(os.path.join(results_dir, "batsman_ranks_by_in_strength.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)  # Write the headers
    writer.writerows(data_in_strength)  # Write the data rows

# Ranking based on batting average scores (lower is better)
ranks_by_batting_avg = sorted(range(n), key=lambda x: CBa[batsmen[x]], reverse= True)
data_batting_avg = []
for i in range(n):
    rank = str(i+1).zfill(2)
    batter = batsmen[ranks_by_batting_avg[i]]
    cba = round(CBa[batter], 5)
    cba_in_strength = round(CBa_in_strengths[batter], 5)
    score_element = scores[ranks_by_batting_avg[i]]  # Ensure the element is a scalar
    
        # Convert to scalar if needed
    if isinstance(score_element, (list, np.ndarray)):
        score_element = score_element.item()
        
    score = round(float(score_element), 5)
    score = str(score).ljust(5 + 2)
    # score = round(float(scores[ranks_by_batting_avg[i]]), 5)
    data_batting_avg.append([rank, batter, cba_in_strength, score, cba])

# Write the batting average table to a CSV file
with open(os.path.join(results_dir, "batsman_ranks_by_batting_avg.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)  # Write the headers
    writer.writerows(data_batting_avg)  # Write the data rows
    
# Extracting data for the top 25 batsmen based on in-strength ranking
top25_batsmen_data = data_in_strength[:25]

# Write the top 25 Batsmen data to a CSV file
top25_headers = ["Rank", "Batsman", "In Strength", "PageRank Score", "Batting Average"]
with open(os.path.join(top25_results_dir, "top25_batsmen.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(top25_headers)  # Write the headers
    writer.writerows(top25_batsmen_data)  # Write the data rows


print("Ranking tables have been generated and saved in the results directory.")
