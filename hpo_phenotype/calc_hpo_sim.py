import os
import numpy as np
import pandas as pd
import obonet
import networkx as nx

def get_graph(hpo_terms, hpo_id_as_label):
    """
    Get a graph from specific HPO terms
    
    Parameters
    ----------
    hpo_terms: list
        HPO terms to create a graph for
    hpo_id_as_label: bool
        Whether to use IDs as the labels. If false, use HPO names
        
    Returns
    ----------
    graph: networkx graph
        Graph based on input HPO terms    
    """
    graph, id_to_name = get_base_graph(hpo_id_as_label)

    for hpo in hpo_terms:
        if hpo_id_as_label == False:
            hpo = id_to_name[hpo]
        if hpo not in graph.nodes():
            for graph_node in graph.nodes(data=True):
                if 'alt_id' in graph_node[1]:
                    if hpo in graph_node[1]['alt_id']:
                        hpo = graph_node[0]
                        break
        if hpo not in graph.nodes():
            continue
        parent_nodes = list(nx.descendants(graph, hpo))
        parent_nodes.append(hpo)
        for hpo_ in parent_nodes:
            graph.nodes[hpo_]['present_in_patient'] += 1
    
    nodes_to_del = []
    
    for node in graph.nodes(data=True):
        if (node[1]['present_in_patient'] == 0):
            nodes_to_del.append(node[0])
    graph.remove_nodes_from(nodes_to_del)   
    id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
    if hpo_id_as_label == False:
        graph = nx.relabel_nodes(graph, id_to_name)
    return graph

def get_base_graph(hpo_id_as_label):
    """
    Get the basis HPO graph
    
    Parameters
    ----------
    hpo_id_as_label: bool
        Use the HPO IDs as the node names. If False, use the HPO names
      
    Returns
    -------
    graph: networkx graph
        The HPO graph
    id_to_name: dict
        Dictionary that can be used to convert HPO IDs to HPO names
    """
    url = 'http://purl.obolibrary.org/obo/hp.obo'
    graph = obonet.read_obo(url)
    nx.set_node_attributes(graph, 0, "present_in_patient")
    id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
    
    for node in graph.nodes(data=True):
        if 'alt_id' in node[1]:
            for alt_id in node[1]['alt_id']:
                id_to_name[alt_id] = node[1]['name']
    
    if hpo_id_as_label == False:
        graph = nx.relabel_nodes(graph, id_to_name)
    return graph, id_to_name

def init_calc_similarity(phenopy_data_directory, scoring_method='Resnik', sum_method='BMA'):
    """
    Initialize phenopy to load needed objects to calculate the semantic similarity later
    
    Parameters
    ----------
    scoring_method: str
        Method use to compare HPO terms, can be Lin, HRSS, etc. We use Resnik, as explained in the paper
    sum_method: str
        Method to calculate similarity of multiple terms, can be maximum, BMA, BMWA etc. We use BMA.
      
    Returns
    -------
    hpo_network: networkx graph
        The HPO graph as initiliazed by phenopy
    name_to_id: dict
        Dictionary that can be used to convert HPO names to HPO IDs
    scorer: phenopy scorer instance
        Scorer object that can be used to calculate semantic similarity between lists of HPO terms
    """
    from phenopy.build_hpo import generate_annotated_hpo_network
    from phenopy.score import Scorer
    
    # files used in building the annotated HPO network
    obo_file = os.path.join(phenopy_data_directory, 'hp.obo')
    disease_to_phenotype_file = os.path.join(phenopy_data_directory, 'phenotype.hpoa')
    
    try:
        hpo_network, alt2prim, disease_records = generate_annotated_hpo_network(obo_file,disease_to_phenotype_file,)
    except:
        obo_file = os.path.join(phenopy_data_directory, 'hp.obo')
        disease_to_phenotype_file = os.path.join(phenopy_data_directory, 'phenotype.hpoa')
        hpo_network, alt2prim, disease_records = generate_annotated_hpo_network(obo_file,disease_to_phenotype_file,)
        
    name_to_id = {data.get('name') : id_ for id_, data in hpo_network.nodes(data=True)}
    for id_, data in hpo_network.nodes(data=True):
        if 'synonyms' in data:
            for syn in data.get('synonyms'):
                name_to_id[syn] = id_
    
    def_graph, id_to_name_old = get_base_graph(True)
    temp_dict = {v: k for k, v in id_to_name_old.items()}
    
    name_to_id = {**name_to_id, **temp_dict}
    
    scorer = Scorer(hpo_network)
    scorer.scoring_method = scoring_method
    scorer.summarization_method = sum_method
    
    return hpo_network, name_to_id, scorer

def calc_similarity(terms_a, terms_b, hpo_network, name_to_id, scorer):
    """
    Use the initialized phenopy object to calculate the semantic similarity between two lists of HPO terms
    
    Parameters
    ----------
    terms_a: list
        First list of HPO terms to compare
    terms_b: list
        Second list of HPO terms to compare
    hpo_network: networkx graph
        The HPO graph as initiliazed by phenopy
    name_to_id: dict
        Dictionary that can be used to convert HPO names to HPO IDs
    scorer: phenopy scorer instance
        Scorer object that can be used to calculate semantic similarity between lists of HPO terms
      
    Returns
    -------
    The calculated semantic similarity between the two lists
    """
    terms_a_proc = []
    for term in terms_a:
        if 'HP' in term:
            if term in hpo_network.nodes():
                terms_a_proc.append(term)
        else:
            if name_to_id[term] in hpo_network.nodes():
                terms_a_proc.append(name_to_id[term])
    terms_b_proc = []
    for term in terms_b:
        if 'HP' in term:
            if term in hpo_network.nodes():
                terms_b_proc.append(term)
        else:
            if name_to_id[term] in hpo_network.nodes():
                terms_b_proc.append(name_to_id[term])
            
    return scorer.score_term_sets_basic(terms_a_proc, terms_b_proc)

def calc_full_sim_mat(X,y, hpo_network, name_to_id, scorer, mlb=None):
    """
    Calculate the full similarity matrix between a list of patients and controls
    
    Parameters
    ----------
    X: numpy array
        Array of size n x 2623: the VGG-Face feature vector and one cell with a list of the HPO IDs
    y: numpy array
        The labels (usually 0 for control and 1 for patient) 

    hpo_network: networkx graph
        The HPO graph as initiliazed by phenopy
    name_to_id: dict
        Dictionary that can be used to convert HPO names to HPO IDs
    scorer: phenopy scorer instance
        Scorer object that can be used to calculate semantic similarity between lists of HPO terms
    mlb: sklearn MultiLabelBinarizer object
        Only used when calling this function while generating LIME explanations. This is needed since LIME needs the expanded HPO list, one-hot encoded, to pertube this matrix.
        
    Returns
    -------
    sim_mat: numpy array
        The calculated similarity matrix between every combination in X
    """
    from sklearn.preprocessing import MultiLabelBinarizer
    
    sim_mat = np.ones((len(X),len(X)))
    
    if type(X[0,-1]) != list:
        hpos = mlb.inverse_transform(X[:,:])
    else:
        hpos = X[:,-1]
    
    hpos = filter_hpo_df(hpos).flatten()
    
    for i in range(len(sim_mat)):
        for z in range(len(sim_mat)):
            sim_mat[i,z] = calc_similarity(list(set(hpos[i])), list(set(hpos[z])), hpo_network, name_to_id, scorer)
    return sim_mat

def calc_sim_scores(sim_mat,train_index,test_index,y_train):
    """
    Calculate the full similarity matrix between a list of patients and controls
    
    Parameters
    ----------
    sim_mat: numpy array
        The calculated similarity matrix between every combination in X
    train_index: numpy array
        Indices of training instances in sim_mat
    test_index: numpy array
        Indices of test instances in sim_mat
    y_train: numpy array
        All training labels

    Returns
    -------
    sim_mat_train: numpy array
        A nx2 array with the average similarity score with all patients and all controls for all training instances
    sim_mat_test: numpy array
        A nx2 array with the average similarity score with all patients and all controls for all test instances    
    """
    sim_mat_train = sim_mat[:, train_index][train_index,:]
    sim_mat_test = sim_mat[:, train_index][test_index,:]
    #calculating averages from whole pairwise similarity score matrix                        
    sim_avg_pat = sim_mat_train[:,y_train == 1].mean(axis=1).reshape(-1,1)
    sim_avg_control = sim_mat_train[:,y_train == 0].mean(axis=1).reshape(-1,1)
    
    if sim_mat_test.ndim > 1:
        sim_avg_pat_test = sim_mat_test[:,y_train == 1].mean(axis=1).reshape(-1,1)
        sim_avg_control_test = sim_mat_test[:,y_train == 0].mean(axis=1).reshape(-1,1)
    else:
        sim_avg_pat_test = sim_mat_test[y_train == 1].mean().reshape(-1,1)
        sim_avg_control_test = sim_mat_test[y_train == 0].mean().reshape(-1,1)
    
    sim_mat_train = np.append(sim_avg_pat, sim_avg_control,axis=1)
    sim_mat_test = np.append(sim_avg_pat_test, sim_avg_control_test,axis=1)
    
    return sim_mat_train, sim_mat_test

def filter_hpo_df(df):
    """
    Exclude certain HPO terms and all child nodes from a dataframe, list, numpy array etc.
    
    Parameters
    ----------
    df: list or dataframe with hpo_all column
        List with lists of HPO IDs per individual. Can also be a dataframe with a column hpo_all with in each cell a list of the HPO IDs of that individual

    Returns
    -------
    df: list/dataframe
        Filtered HPO IDs
    """
    parents_to_exclude = ["HP:0000708", "HP:0000271", "HP:0011297", "HP:0031703", "HP:0012372"] #excluding behaviour, facial features, finger/toe abnormalities, ear/eye morphological abnormalities
    
    hpo_base, id_to_name = get_base_graph(True)
        
    temp_df = pd.DataFrame(hpo_base.nodes(data=True))
    for i in range(len(temp_df)):
        if 'alt_id' in temp_df.loc[i, 1]:
            for alt in temp_df.loc[i, 1]['alt_id']:
                id_to_name[alt] = temp_df.loc[i,1]['name']
    
    exclude = []
    
    for hpo in parents_to_exclude:
        for child_node in nx.algorithms.dag.ancestors(hpo_base,hpo):
            exclude.append(child_node)
        exclude.append(hpo)
        
    if type(df) == list:
        return list(set(df) - set(exclude))
    elif type(df) == np.ndarray:
         df = pd.DataFrame(df)
         df.iloc[:, -1] = [[i for i in L if i not in exclude] for L in df.iloc[:, -1]]
         return df.to_numpy()
    else:
        if type(df) == pd.Series:
            df = pd.DataFrame(df).T
        df['hpo_all'] = [[i for i in L if i not in exclude] for L in df['hpo_all']]
        return df

