from phenopy.build_hpo import generate_annotated_hpo_network
from phenopy.config import download_resource_files
from phenopy.score import Scorer
import networkx as nx
import os
import numpy as np
import pandas as pd
import copy
import obonet


class SimScorer:
    def __init__(self, scoring_method='Resnik', sum_method='BMA'):
        """
        Constructor

        Parameters
        ----------
        scoring_method: str
            Method use to compare HPO terms, can be Lin, HRSS, etc. We use Resnik, as explained in the paper
        sum_method: str
            Method to calculate similarity of multiple terms, can be maximum, BMWA, BMA etc. We use BMA.
        """
        self.hpo_network, self.name_to_id_and_reverse, self.scorer = self._init_calc_similarity(scoring_method=scoring_method,
                                                                                    sum_method=sum_method)

    def get_graph(self, hpo_terms, hpo_id_as_label):
        """
        Get a graph from specific HPO terms

        Parameters
        ----------
        hpo_terms: list
            HPO terms to create a graph for
        hpo_id_as_label: bool
            Whether to relabel the output graph with the HPO labels. If false, keep IDs

        Returns
        ----------
        graph: networkx graph
            Graph based on input HPO terms
        """
        local_copy_hpo_network = copy.deepcopy(self.hpo_network)
        nx.set_node_attributes(local_copy_hpo_network, 0, "present_in_patient")
        for hpo in hpo_terms:
            if hpo not in local_copy_hpo_network.nodes():
                for graph_node in local_copy_hpo_network.nodes(data=True):
                    if 'alt_id' in graph_node[1]:
                        if hpo in graph_node[1]['alt_id']:
                            hpo = graph_node[0]
                            break
            if hpo not in local_copy_hpo_network.nodes():
                continue
            parent_nodes = list(nx.ancestors(local_copy_hpo_network, hpo))
            parent_nodes.append(hpo)
            for hpo_ in parent_nodes:
                local_copy_hpo_network.nodes[hpo_]['present_in_patient'] += 1

        nodes_to_del = []

        for node in local_copy_hpo_network.nodes(data=True):
            if node[1]['present_in_patient'] == 0:
                nodes_to_del.append(node[0])
        local_copy_hpo_network.remove_nodes_from(nodes_to_del)
        id_to_name = {id_: data.get('name') for id_, data in local_copy_hpo_network.nodes(data=True)}
        for node in local_copy_hpo_network.nodes(data=True):
            if 'alt_id' in node[1]:
                for alt_id in node[1]['alt_id']:
                    id_to_name[alt_id] = node[1]['name']

        if hpo_id_as_label == False:
            local_copy_hpo_network = nx.relabel_nodes(local_copy_hpo_network, id_to_name)
        return local_copy_hpo_network

    def _init_calc_similarity(self, scoring_method, sum_method):
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

        # files used in building the annotated HPO network
        phenopy_data_directory = os.path.join(os.path.expanduser("~"), '.phenopy', 'data')
        obo_file = os.path.join(phenopy_data_directory, 'hp.obo')
        disease_to_phenotype_file = os.path.join(phenopy_data_directory, 'phenotype.hpoa')
        download_resource_files()
        hpo_network, alt2prim, disease_records = generate_annotated_hpo_network(obo_file,
                                                                                disease_to_phenotype_file, )

        file_path = os.path.join(os.path.expanduser("~"), '.phenopy', 'data', 'hp.obo')
        full_hpo_graph = obonet.read_obo(file_path)

        #the phenopy hpo_network does not included some terms like inheritance etc since they are not phenotypes
        #for name/id to name/id dict we need all
        name_to_id = {data.get('name'): id_ for id_, data in full_hpo_graph.nodes(data=True)}
        temp_dict = {v: k for k, v in name_to_id.items()}
        name_to_id_and_reverse = {**name_to_id, **temp_dict}

        for id_, data in full_hpo_graph.nodes(data=True):
            if 'synonyms' in data:
                for syn in data.get('synonyms'):
                    name_to_id_and_reverse[syn] = id_
            if 'alt_id' in data:
                for alt in data['alt_id']:
                    name_to_id_and_reverse[alt] = data['name']

        scorer = Scorer(hpo_network)
        scorer.scoring_method = scoring_method
        scorer.summarization_method = sum_method

        return hpo_network.reverse(), name_to_id_and_reverse, scorer  # needs reverse otherwise ancestors/desc incorrect

    def calc_similarity(self, terms_a, terms_b):
        """
        Use the initialized phenopy object to calculate the semantic similarity between two lists of HPO terms

        Parameters
        ----------
        terms_a: list
            First list of HPO terms to compare
        terms_b: list
            Second list of HPO terms to compare

        Returns
        -------
        The calculated semantic similarity between the two lists
        """
        terms_a_proc = []
        for term in terms_a:
            if 'HP' in term:
                if term in self.hpo_network.nodes():
                    terms_a_proc.append(term)
            else:
                if self.name_to_id_and_reverse[term] in self.hpo_network.nodes():
                    terms_a_proc.append(self.name_to_id_and_reverse[term])
        terms_b_proc = []
        for term in terms_b:
            if 'HP' in term:
                if term in self.hpo_network.nodes():
                    terms_b_proc.append(term)
            else:
                if self.name_to_id_and_reverse[term] in self.hpo_network.nodes():
                    terms_b_proc.append(self.name_to_id_and_reverse[term])
        result = self.scorer.score_term_sets_basic(terms_a_proc, terms_b_proc)
        if np.isnan(result):
            result = 0
        return result

    def calc_full_sim_mat(self, X, mlb=None):
        """
        Calculate the full similarity matrix between a list of patients and controls

        Parameters
        ----------
        X: numpy array
            Array: can inlcude the VGG-Face feature vector (optional) and one cell with a list of the HPO IDs
        mlb: sklearn MultiLabelBinarizer object
            Only used when calling this function while generating LIME explanations. This is needed since LIME needs the expanded HPO list, one-hot encoded, to pertube this matrix.

        Returns
        -------
        sim_mat: numpy array
            The calculated similarity matrix between every combination in X
        """
        sim_mat = np.ones((len(X), len(X)))

        if type(X[0, -1]) != list:
            hpos = mlb.inverse_transform(X[:, :])
        else:
            hpos = X[:, -1]

        hpos = self.filter_hpo_df(hpos).flatten()

        for i in range(len(sim_mat)):
            for z in range(len(sim_mat)):
                sim_mat[i, z] = self.calc_similarity(list(set(hpos[i])), list(set(hpos[z])))
        return sim_mat

    @staticmethod
    def calc_sim_scores(sim_mat, train_index, test_index, y_train):
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
        sim_mat_train = sim_mat[:, train_index][train_index, :]
        sim_mat_test = sim_mat[:, train_index][test_index, :]
        # calculating averages from whole pairwise similarity score matrix
        sim_avg_pat = sim_mat_train[:, y_train == 1].mean(axis=1).reshape(-1, 1)
        sim_avg_control = sim_mat_train[:, y_train == 0].mean(axis=1).reshape(-1, 1)

        if sim_mat_test.ndim > 1:
            sim_avg_pat_test = sim_mat_test[:, y_train == 1].mean(axis=1).reshape(-1, 1)
            sim_avg_control_test = sim_mat_test[:, y_train == 0].mean(axis=1).reshape(-1, 1)
        else:
            sim_avg_pat_test = sim_mat_test[y_train == 1].mean().reshape(-1, 1)
            sim_avg_control_test = sim_mat_test[y_train == 0].mean().reshape(-1, 1)

        sim_mat_train = np.append(sim_avg_pat, sim_avg_control, axis=1)
        sim_mat_test = np.append(sim_avg_pat_test, sim_avg_control_test, axis=1)

        return sim_mat_train, sim_mat_test

    def filter_hpo_df(self, df):
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
        parents_to_exclude = ["HP:0000708", "HP:0000271", "HP:0011297", "HP:0031703",
                              "HP:0012372"]  # excluding behaviour, facial features, finger/toe abnormalities, ear/eye morphological abnormalities

        temp_df = pd.DataFrame(self.hpo_network.nodes(data=True))
        for i in range(len(temp_df)):
            if 'alt_id' in temp_df.loc[i, 1]:
                for alt in temp_df.loc[i, 1]['alt_id']:
                    self.name_to_id_and_reverse[alt] = temp_df.loc[i, 1]['name']

        exclude = []

        for hpo in parents_to_exclude:
            for child_node in nx.algorithms.dag.descendants(self.hpo_network, hpo):
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
