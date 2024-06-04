import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

# Load the data
bowlers_ranks_by_instrength = pd.read_csv('results/bowlers_ranks_by_in_strength.csv')
bowlers_ranks_by_pagerank = pd.read_csv('results/bowlers_ranks_by_pagerank.csv')
bowlers_ranks_by_bowling_avg = pd.read_csv('results/bowlers_ranks_by_bowling_avg.csv')
results_dir= 'results/linear_regression'

# Check if the results directory exists, if not, create it
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# Merge data based on Bowler names
merged_data_bowlers = pd.merge(
    bowlers_ranks_by_bowling_avg[['Bowler', 'Rank']].rename(columns={'Rank': 'Rank_Bowling_Avg'}),
    bowlers_ranks_by_instrength[['Bowler', 'Rank']].rename(columns={'Rank': 'Rank_In_Strength'}),
    on='Bowler',
    how='inner'
).merge(
    bowlers_ranks_by_pagerank[['Bowler', 'Rank']].rename(columns={'Rank': 'Rank_PageRank'}),
    on='Bowler',
    how='inner'
)

# Ensure the merged data is not empty
if merged_data_bowlers.empty:
    print("The merged DataFrame for bowlers is empty. Please check the input CSV files for consistency.")
else:
    # A. Correlation between different ranks for bowlers
    corr_in_strength_vs_bowling_avg, _ = spearmanr(merged_data_bowlers['Rank_In_Strength'], merged_data_bowlers['Rank_Bowling_Avg'])
    corr_page_rank_vs_bowling_avg, _ = spearmanr(merged_data_bowlers['Rank_PageRank'], merged_data_bowlers['Rank_Bowling_Avg'])
    corr_in_strength_vs_page_rank_bowl, _ = spearmanr(merged_data_bowlers['Rank_In_Strength'], merged_data_bowlers['Rank_PageRank'])

    print(f"Spearman correlation between In-Strength Rank and Bowling Average Rank for bowlers: {corr_in_strength_vs_bowling_avg:.2f}")
    print(f"Spearman correlation between PageRank Rank and Bowling Average Rank for bowlers: {corr_page_rank_vs_bowling_avg:.2f}")
    print(f"Spearman correlation between In-Strength Rank and PageRank Rank for bowlers: {corr_in_strength_vs_page_rank_bowl:.2f}")

    # B. Linear Regression Model for In-Strength for bowlers
    X_bowlers = merged_data_bowlers[['Rank_Bowling_Avg', 'Rank_PageRank']]
    y_in_strength_bowlers = merged_data_bowlers['Rank_In_Strength']
    model_in_strength_bowlers = LinearRegression().fit(X_bowlers, y_in_strength_bowlers)
    print(f'Linear Regression Coefficients for In-Strength for bowlers: {model_in_strength_bowlers.coef_}')
    print(f'Intercept for In-Strength for bowlers: {model_in_strength_bowlers.intercept_}')

    # C. Linear Regression Model for Bowling Average for bowlers
    y_bowling_avg_bowlers = merged_data_bowlers['Rank_Bowling_Avg']
    model_bowling_avg_bowlers = LinearRegression().fit(X_bowlers, y_bowling_avg_bowlers)
    print(f'Linear Regression Coefficients for Bowling Average for bowlers: {model_bowling_avg_bowlers.coef_}')
    print(f'Intercept for Bowling Average for bowlers: {model_bowling_avg_bowlers.intercept_}')

    # D. Linear Regression Model for PageRank for bowlers
    y_page_rank_bowlers = merged_data_bowlers['Rank_PageRank']
    model_page_rank_bowlers = LinearRegression().fit(X_bowlers, y_page_rank_bowlers)
    print(f'Linear Regression Coefficients for PageRank for bowlers: {model_page_rank_bowlers.coef_}')
    print(f'Intercept for PageRank for bowlers: {model_page_rank_bowlers.intercept_}')

    # Create subplots for visualization
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Scatter plot for In-Strength vs. Bowling Average
    sns.scatterplot(x='Rank_Bowling_Avg', y='Rank_In_Strength', data=merged_data_bowlers, ax=axs[0, 0])
    axs[0, 0].plot([merged_data_bowlers['Rank_Bowling_Avg'].min(), merged_data_bowlers['Rank_Bowling_Avg'].max()],
                   [merged_data_bowlers['Rank_Bowling_Avg'].min(), merged_data_bowlers['Rank_Bowling_Avg'].max()],
                   linestyle='--', color='r')
    axs[0, 0].set_title('A')
    axs[0, 0].set_xlabel('Bowling Average Rank')
    axs[0, 0].set_ylabel('In-Strength Rank')

    # Scatter plot for PageRank vs. Bowling Average
    sns.scatterplot(x='Rank_Bowling_Avg', y='Rank_PageRank', data=merged_data_bowlers, ax=axs[0, 1])
    axs[0, 1].plot([merged_data_bowlers['Rank_Bowling_Avg'].min(), merged_data_bowlers['Rank_Bowling_Avg'].max()],
                   [merged_data_bowlers['Rank_Bowling_Avg'].min(), merged_data_bowlers['Rank_Bowling_Avg'].max()],
                   linestyle='--', color='r')
    axs[0, 1].set_title('B')
    axs[0, 1].set_xlabel('Bowling Average Rank')
    axs[0, 1].set_ylabel('PageRank Rank')

    # Scatter plot for In-Strength vs. PageRank
    sns.scatterplot(x='Rank_PageRank', y='Rank_In_Strength', data=merged_data_bowlers, ax=axs[1, 0])
    axs[1, 0].plot([merged_data_bowlers['Rank_PageRank'].min(), merged_data_bowlers['Rank_PageRank'].max()],
                   [merged_data_bowlers['Rank_PageRank'].min(), merged_data_bowlers['Rank_PageRank'].max()],
                   linestyle='--', color='r')
    axs[1, 0].set_title('C')
    axs[1, 0].set_xlabel('PageRank Rank')
    axs[1, 0].set_ylabel('In-Strength Rank')

    # Hide the last subplot
    fig.delaxes(axs[1, 1])

    # Save the plot
    plt.tight_layout()
    plt.savefig("results/linear_regression/scatter_bowling.png")

    # Open a text file in write mode
    with open("results/linear_regression/correlation_and_regression_bowling_info.txt", "w") as f:
        # Write correlation information
        f.write(f"Spearman correlation between In-Strength Rank and Bowling Average Rank: {corr_in_strength_vs_bowling_avg:.2f}\n")
        f.write(f"Spearman correlation between PageRank Rank and Bowling Average Rank: {corr_page_rank_vs_bowling_avg:.2f}\n")
        f.write(f"Spearman correlation between In-Strength Rank and PageRank Rank: {corr_in_strength_vs_page_rank_bowl:.2f}\n\n")

        # Write Linear Regression coefficients and intercepts
        f.write(f'Linear Regression Coefficients for In-Strength: {model_in_strength_bowlers.coef_}\n')
        f.write(f'Intercept for In-Strength: {model_in_strength_bowlers.intercept_}\n\n')

        f.write(f'Linear Regression Coefficients for Bowling Average: {model_bowling_avg_bowlers.coef_}\n')
        f.write(f'Intercept for Bowling Average: {model_bowling_avg_bowlers.intercept_}\n\n')

        f.write(f'Linear Regression Coefficients for PageRank: {model_page_rank_bowlers.coef_}\n')
        f.write(f'Intercept for PageRank: {model_page_rank_bowlers.intercept_}\n')