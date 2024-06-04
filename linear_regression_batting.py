import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

# Load the data for batsmen
batsmen_ranks_by_instrength = pd.read_csv('results/batsman_ranks_by_in_strength.csv')
batsmen_ranks_by_pagerank = pd.read_csv('results/batsman_ranks_by_pagerank.csv')
batsmen_ranks_by_batting_avg = pd.read_csv('results/batsman_ranks_by_batting_avg.csv')
results_dir= 'results/linear_regression'

# Check if the results directory exists, if not, create it
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Merge data based on Batsman names
merged_data_batsmen = pd.merge(
    batsmen_ranks_by_batting_avg[['Batsman', 'Rank']].rename(columns={'Rank': 'Rank_Batting_Avg'}),
    batsmen_ranks_by_instrength[['Batsman', 'Rank']].rename(columns={'Rank': 'Rank_In_Strength'}),
    on='Batsman',
    how='inner'
).merge(
    batsmen_ranks_by_pagerank[['Batsman', 'Rank']].rename(columns={'Rank': 'Rank_PageRank'}),
    on='Batsman',
    how='inner'
)

# Ensure the merged data is not empty
if merged_data_batsmen.empty:
    print("The merged DataFrame for batsmen is empty. Please check the input CSV files for consistency.")
else:
    # A. Correlation between different ranks for batsmen
    corr_in_strength_vs_batting_avg, _ = spearmanr(merged_data_batsmen['Rank_In_Strength'], merged_data_batsmen['Rank_Batting_Avg'])
    corr_page_rank_vs_batting_avg, _ = spearmanr(merged_data_batsmen['Rank_PageRank'], merged_data_batsmen['Rank_Batting_Avg'])
    corr_in_strength_vs_page_rank_bat, _ = spearmanr(merged_data_batsmen['Rank_In_Strength'], merged_data_batsmen['Rank_PageRank'])

    print(f"Spearman correlation between In-Strength Rank and Batting Average Rank for batsmen: {corr_in_strength_vs_batting_avg:.2f}")
    print(f"Spearman correlation between PageRank Rank and Batting Average Rank for batsmen: {corr_page_rank_vs_batting_avg:.2f}")
    print(f"Spearman correlation between In-Strength Rank and PageRank Rank for batsmen: {corr_in_strength_vs_page_rank_bat:.2f}")

    # B. Linear Regression Model for In-Strength for batsmen
    X_batsmen = merged_data_batsmen[['Rank_Batting_Avg', 'Rank_PageRank']]
    y_in_strength_batsmen = merged_data_batsmen['Rank_In_Strength']
    model_in_strength_batsmen = LinearRegression().fit(X_batsmen, y_in_strength_batsmen)
    print(f'Linear Regression Coefficients for In-Strength for batsmen: {model_in_strength_batsmen.coef_}')
    print(f'Intercept for In-Strength for batsmen: {model_in_strength_batsmen.intercept_}')

    # C. Linear Regression Model for Batting Average for batsmen
    y_batting_avg_batsmen = merged_data_batsmen['Rank_Batting_Avg']
    model_batting_avg_batsmen = LinearRegression().fit(X_batsmen, y_batting_avg_batsmen)
    print(f'Linear Regression Coefficients for Batting Average for batsmen: {model_batting_avg_batsmen.coef_}')
    print(f'Intercept for Batting Average for batsmen: {model_batting_avg_batsmen.intercept_}')

    # D. Linear Regression Model for PageRank for batsmen
    y_page_rank_batsmen = merged_data_batsmen['Rank_PageRank']
    model_page_rank_batsmen = LinearRegression().fit(X_batsmen, y_page_rank_batsmen)
    print(f'Linear Regression Coefficients for PageRank for batsmen: {model_page_rank_batsmen.coef_}')
    print(f'Intercept for PageRank for batsmen: {model_page_rank_batsmen.intercept_}')

    # Create subplots for visualization
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Scatter plot for In-Strength vs. Batting Average
    sns.scatterplot(x='Rank_Batting_Avg', y='Rank_In_Strength', data=merged_data_batsmen, ax=axs[0, 0])
    axs[0, 0].plot([merged_data_batsmen['Rank_Batting_Avg'].min(), merged_data_batsmen['Rank_Batting_Avg'].max()],
                   [merged_data_batsmen['Rank_Batting_Avg'].min(), merged_data_batsmen['Rank_Batting_Avg'].max()],
                   linestyle='--', color='r')
    axs[0, 0].set_title('A')
    axs[0, 0].set_xlabel('Batting Average Rank')
    axs[0, 0].set_ylabel('In-Strength Rank')

    # Scatter plot for PageRank vs. Batting Average
    sns.scatterplot(x='Rank_Batting_Avg', y='Rank_PageRank', data=merged_data_batsmen, ax=axs[0, 1])
    axs[0, 1].plot([merged_data_batsmen['Rank_Batting_Avg'].min(), merged_data_batsmen['Rank_Batting_Avg'].max()],
                   [merged_data_batsmen['Rank_Batting_Avg'].min(), merged_data_batsmen['Rank_Batting_Avg'].max()],
                   linestyle='--', color='r')
    axs[0, 1].set_title('B')
    axs[0, 1].set_xlabel('Batting Average Rank')
    axs[0, 1].set_ylabel('PageRank Rank')

    # Scatter plot for In-Strength vs. PageRank
    sns.scatterplot(x='Rank_PageRank', y='Rank_In_Strength', data=merged_data_batsmen, ax=axs[1, 0])
    axs[1, 0].plot([merged_data_batsmen['Rank_PageRank'].min(), merged_data_batsmen['Rank_PageRank'].max()],
                   [merged_data_batsmen['Rank_PageRank'].min(), merged_data_batsmen['Rank_PageRank'].max()],
                   linestyle='--', color='r')
    axs[1, 0].set_title('C')
    axs[1, 0].set_xlabel('PageRank Rank')
    axs[1, 0].set_ylabel('In-Strength Rank')
    
    # Hide the last subplot
    fig.delaxes(axs[1, 1])

    # Show the plot
    plt.tight_layout()
    plt.savefig("results/linear_regression/scatter_batting.png")
    
    # Open a text file in write mode
with open("results/linear_regression/correlation_and_regression_batting_info.txt", "w") as f:
    # Write correlation information
    f.write(f"Spearman correlation between In-Strength Rank and Batting Average Rank: {corr_in_strength_vs_batting_avg:.2f}\n")
    f.write(f"Spearman correlation between PageRank Rank and Batting Average Rank: {corr_page_rank_vs_batting_avg:.2f}\n")
    f.write(f"Spearman correlation between In-Strength Rank and PageRank Rank: {corr_in_strength_vs_page_rank_bat:.2f}\n\n")

    # Write Linear Regression coefficients and intercepts
    f.write(f'Linear Regression Coefficients for In-Strength: {model_in_strength_batsmen.coef_}\n')
    f.write(f'Intercept for In-Strength: {model_in_strength_batsmen.intercept_}\n\n')

    f.write(f'Linear Regression Coefficients for Batting Average: {model_batting_avg_batsmen.coef_}\n')
    f.write(f'Intercept for Batting Average: {model_batting_avg_batsmen.intercept_}\n\n')

    f.write(f'Linear Regression Coefficients for PageRank: {model_page_rank_batsmen.coef_}\n')
    f.write(f'Intercept for PageRank: {model_page_rank_batsmen.intercept_}\n')

    