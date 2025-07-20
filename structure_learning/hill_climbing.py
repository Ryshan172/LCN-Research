from pgmpy.estimators import HillClimbSearch, BIC

def run_hillclimb_search(df):
    # Convert true/false columns to categorical values
    df = df.astype("category")
    
    # Feed to hillclimb
    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=BIC(df))
    return (best_model.edges())
