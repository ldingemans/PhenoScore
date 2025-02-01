import networkx as nx
import numpy as np
import pandas as pd
import json
import mmap
import struct
from pathlib import Path
import os
import urllib.request


class SimScorer:
    def __init__(self, similarity_data_path, hpo_network_csv_path, name_to_id_json):
        """
        Constructor

        Parameters
        ----------
        similarity_data_path: str
            Path to directory containing similarity data files:
            - similarities_data.bin: Binary file with non-zero similarities
            - similarities_index_file.json: NumPy file with index for fast lookups
        hpo_network_csv_path: str
            Path to CSV file containing HPO term relationships.
            Should have columns: term, parent_term
            where terms are stored as integers
        name_to_id_json: str
            Path to JSON file that contains mapping from HPO IDs to names and vice-versa.
        """
        self.data_dir = Path(similarity_data_path)

        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Paths to the local files
        bin_path = self.data_dir / 'similarities_data.bin'
        json_path = self.data_dir / 'similarities_index_file.json'

        # Download if necessary
        if not bin_path.is_file():
            file_url_bin = "https://www.dropbox.com/scl/fi/jom6tifl9kzkuckyfhrml/similarities_data.bin?rlkey=dmdhxj9jbddtv44kovd02wvf3&st=msca38ss&dl=1"
            urllib.request.urlretrieve(file_url_bin, bin_path)

        if not json_path.is_file():
            file_url_json = "https://www.dropbox.com/scl/fi/40ru28czp1rl9m9bqgixw/similarities_index_file.json?rlkey=5a86apljd3mjwhj2d6cntvqxh&st=dvy509f9&dl=1"
            urllib.request.urlretrieve(file_url_json, json_path)

        # Initialize binary similarity lookup
        self.bin_file = open(bin_path, 'rb')
        self.mmap = mmap.mmap(self.bin_file.fileno(), 0, access=mmap.ACCESS_READ)
        with open(json_path, 'r') as f_in:
             data_index_file = json.load(f_in)

        # Convert the string keys back to integers
        self.similarity_index = {
            int(k): {int(subk): v for subk, v in subdict.items()}
            for k, subdict in data_index_file.items()
        }

        # Load name to ID mapping
        with open(name_to_id_json, "r") as f:
            self.name_to_id_and_reverse = json.load(f)

        self.valid_terms = set(self._convert_hpo_list([key for key in self.name_to_id_and_reverse.keys() if "HP:" in key]))
        # Create directed graph from HPO relationships
        self.hpo_network = nx.DiGraph()
        hpo_relations = pd.read_csv(hpo_network_csv_path)
        for _, row in hpo_relations.iterrows():
            self.hpo_network.add_edge(row['parent_term'], row['term'])

    def _get_similarity(self, term_a, term_b):
        """Get similarity score for two terms from binary storage"""
        if term_a not in self.valid_terms or term_b not in self.valid_terms:
            raise ValueError(f"Term {term_a if term_a not in self.valid_terms else term_b} not found in database")

        try:
            offset = self.similarity_index[term_a][term_b]
            self.mmap.seek(offset)
            _, _, similarity = struct.unpack('IIf', self.mmap.read(12))
            return similarity
        except KeyError:
            return 0.0  # Terms exist but similarity is zero

    def _convert_hpo_to_int(self, hpo_term):
        """Convert HP:0000001 format to integer 1"""
        if isinstance(hpo_term, (int, np.integer)):
            return int(hpo_term)
        return int(hpo_term.replace('HP:', ''))

    def _convert_hpo_list(self, hpo_terms):
        """Convert a list of HPO terms to integers"""
        return [self._convert_hpo_to_int(term) for term in hpo_terms]

    def filter_hpo_df(self, df):
        """
        Exclude certain HPO terms and all child nodes from a dataframe, list, numpy array etc.

        Parameters
        ----------
        df: list or dataframe with hpo_all column
            List with lists of HPO IDs per individual. Can also be a dataframe with a column hpo_all
            with in each cell a list of the HPO IDs of that individual

        Returns
        -------
        df: list/dataframe
            Filtered HPO IDs
        """
        # Terms to exclude (converted to integers)
        parents_to_exclude = [708, 271, 11297, 31703, 12372]

        # Get all descendants of excluded terms
        exclude = set()
        for hpo in parents_to_exclude:
            descendants = nx.algorithms.dag.descendants(self.hpo_network, hpo)
            exclude.update(descendants)
            exclude.add(hpo)

        if isinstance(df, list):
            terms = self._convert_hpo_list(df)
            return [term for term in terms if term not in exclude]

        elif isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
            df.iloc[:, -1] = [
                [term for term in self._convert_hpo_list(L) if term not in exclude]
                for L in df.iloc[:, -1]
            ]
            return df.to_numpy()

        else:
            if isinstance(df, pd.Series):
                df = pd.DataFrame(df).T

            df['hpo_all'] = [
                [term for term in self._convert_hpo_list(L) if term not in exclude]
                for L in df['hpo_all']
            ]
            return df

    def calc_similarity(self, terms_a, terms_b):
        """
        Calculate similarity between two lists of HPO terms using pre-calculated scores

        Parameters
        ----------
        terms_a: list
            First list of HPO terms (can be either HP:0000001 format or 1 format)
        terms_b: list
            Second list of HPO terms (can be either HP:0000001 format or 1 format)

        Returns
        -------
        float: The calculated semantic similarity between the two lists using BMA method
        """
        # Convert terms to integers if they're in HP:XXXXXXX format
        terms_a = self._convert_hpo_list(terms_a)
        terms_b = self._convert_hpo_list(terms_b)

        if not terms_a or not terms_b:
            return 0.0

        # Calculate similarity matrix between all terms
        sim_matrix = []
        for term_a in terms_a:
            row = []
            for term_b in terms_b:
                sim = self._get_similarity(term_a, term_b)
                row.append(sim)
            sim_matrix.append(row)

        sim_matrix = np.array(sim_matrix)

        # Calculate BMA using phenopys's method:
        # Get max values for each row and column, then average all maxima together
        max_a = np.max(sim_matrix, axis=1)  # Best match for each term in A
        max_b = np.max(sim_matrix, axis=0)  # Best match for each term in B

        # Concatenate all maxima and take the overall average
        all_maxes = np.append(max_a, max_b)
        return np.average(all_maxes)

    def calc_full_sim_mat(self, X, mlb=None):
        """
        Calculate the full similarity matrix between a list of patients

        Parameters
        ----------
        X: numpy array
            Array with one cell containing a list of the HPO IDs (as integers or HP:XXXXXXX format)
        mlb: sklearn MultiLabelBinarizer object
            Only used when calling this function while generating LIME explanations. This is needed since LIME needs the expanded HPO list, one-hot encoded, to pertube this matrix.

        Returns
        -------
        sim_mat: numpy array
            The calculated similarity matrix between every combination in X
        """
        sim_mat = np.ones((len(X), len(X)))

        if isinstance(X, list):
            X = np.array(X).reshape(-1, 1)

        if type(X[0, -1]) != list:
            hpos = mlb.inverse_transform(X[:, :])
        else:
            hpos = X[:, -1]

        hpos = self.filter_hpo_df(hpos).flatten()

        for i in range(len(sim_mat)):
            for j in range(len(sim_mat)):
                sim_mat[i, j] = self.calc_similarity(hpos[i], hpos[j])

        return sim_mat

    def calc_sim_scores(self, sim_mat, train_index, test_index, y_train):
        """
        Calculate similarity scores for training and test sets

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
            A nx2 array with average similarity scores with patients and controls for training
        sim_mat_test: numpy array
            A nx2 array with average similarity scores with patients and controls for testing
        """
        sim_mat_train = sim_mat[:, train_index][train_index, :]
        sim_mat_test = sim_mat[:, train_index][test_index, :]

        # Calculate averages from similarity matrix
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

    def __del__(self):
        """Cleanup binary file resources"""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'bin_file'):
            self.bin_file.close()

    def get_graph(self, hpo_terms, hpo_id_as_label=False):
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
        # Convert terms to integers if they're in HP:XXXXXXX format
        terms = self._convert_hpo_list(hpo_terms)

        # Create a new directed graph
        graph = nx.DiGraph()

        # For each term, find all ancestors (including the term itself)
        # and add the edges between them to the graph
        for term in terms:
            if term not in self.valid_terms:
                continue

            # Get all ancestors including the term itself
            ancestors = nx.ancestors(self.hpo_network, term)
            ancestors.add(term)

            # Add all edges between ancestors that exist in the original network
            for ancestor in ancestors:
                for successor in self.hpo_network.successors(ancestor):
                    if successor in ancestors:
                        graph.add_edge(ancestor, successor)

        # Relabel nodes with HPO terms if requested
        if hpo_id_as_label:
            mapping = {}
            for node in graph.nodes():
                # Convert integer ID back to HPO format
                hpo_id = f"HP:{node:07d}"
                # Get label from the mapping dictionary
                if hpo_id in self.name_to_id_and_reverse:
                    mapping[node] = self.name_to_id_and_reverse[hpo_id]
                else:
                    mapping[node] = hpo_id
            graph = nx.relabel_nodes(graph, mapping)

        return graph

    @staticmethod
    def build_similarity_files(output_dir: str = '.'):
        """
        Build similarity files from scratch.
        Requires the 'build' extras to be installed.
        """
        try:
            from phenoscore.hpo_phenotype import hpo_similarity_builder
        except ImportError:
            raise ImportError(
                "Building similarity files requires additional dependencies. "
                "Please install them with: pip install your_library[build]"
            )

        return hpo_similarity_builder.build_similarities(output_dir)