from pgmpy.estimators import HillClimbSearch, BIC

def run_hillclimb_search(df):
    # Convert true/false columns to categorical values
    df = df.astype("category")
    
    # Feed to hillclimb
    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=BIC(df))
    return (best_model.edges())


def run_hillclimbing_bic(df):
    # Create the scoring method
    bic = BIC(df)

    # Run hill-climbing
    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=bic)

    # Compute the BIC score of the final structure
    best_score = bic.score(best_model)

    return best_model.edges(), best_score
