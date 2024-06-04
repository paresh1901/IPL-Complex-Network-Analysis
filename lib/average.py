import pandas as pd
import numpy as np

def get_batting_career_avg(batting_career_file):
    bat = pd.read_csv(batting_career_file)
    
    batsmen = np.array(bat['batsman'])
    runs = np.array(bat['total_runs'])
    innings = np.array(bat['total_innings'])
    
    avg = runs / innings
    
    bat_career_avg = {}
    for i, batsman in enumerate(batsmen):
        bat_career_avg[batsman] = avg[i]

    return bat_career_avg

def get_bowling_career_avg(bowling_career_file):
    ball = pd.read_csv(bowling_career_file)
    
    bowlers = np.array(ball['bowler'])
    runs = np.array(ball['total_runs'])
    dismisses = np.array(ball['total_dismisses'])
    
    avg = runs / np.maximum(np.ones(dismisses.shape), dismisses)
    
    ball_career_avg = {}
    for i, bowler in enumerate(bowlers):
        ball_career_avg[bowler] = avg[i]

    return ball_career_avg