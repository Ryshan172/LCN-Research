from workflows.app_lcn_gen import generate_basic_lcn, optimize_lcn_bic


def generate_initial_lcn_graph(csv_data):
    """
    Generate an initial graph structure which will then be 'filled in' 
    Input is the csv data path
    """
    initial_graph, df = generate_basic_lcn(csv_data)

    return initial_graph,df


def run_lcn_bic_optimization(initial_lcn, df):
    """
    - Use GHC + mutations (add/remove/reverse edges)
    - Score each candidate structure using BIC
    - Keep improving until convergence -> final LCN

    - Requires originally loaded csv data df as well
    """

    bic_lcn = optimize_lcn_bic(initial_lcn=initial_lcn, df=df)

    return bic_lcn



def optimize_lcn_ibic(initial_lcn):
    """
    - Use same search procedure and mutations as BIC version
    - Use Interval BIC (IBIC) instead of BIC
    - Output LCN
    """

    ibic_lcn = {}

    return ibic_lcn


def run_application_workflow(csv_data):

    """
    Run 1: initial graph -> GHC + mutations -> BIC scoring -> LCN1
    Run 2: same initial graph -> GHC + mutations -> IBIC scoring -> LCN2

    Note: Overall point is to generate a graph and improve it using different approachs
    Then test predication on the data and see which optimisation approach worked best i.e
    using the IBIC vs BIC
    """
    

    # Step 1: Load data and generate initial LCN graph 
    initial_lcn, df = generate_initial_lcn_graph(csv_data)

    # Step 2: Optimize initial graph using BIC
    bic_lcn = optimize_lcn_bic(initial_lcn, df)

    print("Initial LCN: ")
    print(initial_lcn)

    print("BIC LCN: ")
    print(bic_lcn)

    # Step 3: Optimize initial graph using IBIC
    ibic_lcn = optimize_lcn_ibic(initial_lcn)


    return 