from pgmpy.estimators import HillClimbSearch, BIC

def run_hillclimb_search(df):
    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=BIC(df))
    return (best_model.edges())
