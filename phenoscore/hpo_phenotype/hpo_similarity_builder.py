import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import combinations, islice
import gc
import struct
import os
import numpy as np
import pandas as pd
from collections import defaultdict


def check_phenopy_installation():
    """Check if phenopy and its dependencies are installed."""
    try:
        import phenopy
        import obonet
        return True
    except ImportError:
        raise ImportError(
            "The phenopy package is required to build HPO similarities from scratch. "
            "Please install it with: pip install phenopy obonet"
        )


# Global variable for multiprocessing
_sim_scorer = None


def init_worker():
    """Initialize global SimScorer for each worker process."""
    global _sim_scorer
    _sim_scorer = SimScorer()


def extract_id(term: str) -> int:
    """Extract numeric ID from HPO term."""
    return int(term.split(':')[1])


def calc_pair(pair: tuple) -> tuple:
    """Calculate similarity between a pair of HPO terms."""
    global _sim_scorer
    node_a, node_b = pair
    terms_a = []
    terms_b = []

    if 'HP' in node_a and node_a in _sim_scorer.hpo_network.nodes():
        terms_a.append(node_a)
    elif node_a in _sim_scorer.name_to_id_and_reverse and _sim_scorer.name_to_id_and_reverse[
        node_a] in _sim_scorer.hpo_network.nodes():
        terms_a.append(_sim_scorer.name_to_id_and_reverse[node_a])

    if 'HP' in node_b and node_b in _sim_scorer.hpo_network.nodes():
        terms_b.append(node_b)
    elif node_b in _sim_scorer.name_to_id_and_reverse and _sim_scorer.name_to_id_and_reverse[
        node_b] in _sim_scorer.hpo_network.nodes():
        terms_b.append(_sim_scorer.name_to_id_and_reverse[node_b])

    assert len(terms_a) == 1
    assert len(terms_b) == 1

    if terms_a and terms_b:
        result = _sim_scorer.scorer.score_hpo_pair_hrss(terms_a[0], terms_b[0])
        return (extract_id(terms_a[0]), extract_id(terms_b[0]), 0 if np.isnan(result) else result)
    return (0, 0, 0)


def process_chunk(node_pairs: list, output_csv: str, chunk_size: int = 100000, pbar=None) -> None:
    """Process a chunk of node pairs and save results."""
    num_processes = max(1, cpu_count() - 2)

    with Pool(num_processes, initializer=init_worker) as pool:
        chunk_results = []
        for result in pool.imap(calc_pair, node_pairs, chunksize=1000):
            chunk_results.append(result)
            if len(chunk_results) >= chunk_size:
                df = pd.DataFrame(chunk_results, columns=['term_a', 'term_b', 'similarity'])
                df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv),
                          index=False)
                chunk_results.clear()
                gc.collect()

            if pbar is not None:
                pbar.update(1)

        if chunk_results:
            df = pd.DataFrame(chunk_results, columns=['term_a', 'term_b', 'similarity'])
            df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv),
                      index=False)


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
        from phenopy.build_hpo import generate_annotated_hpo_network
        from phenopy.config import download_resource_files
        from phenopy.score import Scorer
        import obonet

        self.hpo_network, self.name_to_id_and_reverse, self.scorer = self._init_calc_similarity(
            scoring_method=scoring_method,
            sum_method=sum_method,
            generate_annotated_hpo_network=generate_annotated_hpo_network,
            download_resource_files=download_resource_files,
            Scorer=Scorer,
            obonet=obonet
        )

    def _init_calc_similarity(self, scoring_method, sum_method, generate_annotated_hpo_network,
                              download_resource_files, Scorer, obonet):
        """Initialize phenopy to load needed objects to calculate the semantic similarity."""
        phenopy_data_directory = os.path.join(os.path.expanduser("~"), '.phenopy', 'data')
        obo_file = os.path.join(phenopy_data_directory, 'hp.obo')
        disease_to_phenotype_file = os.path.join(phenopy_data_directory, 'phenotype.hpoa')
        download_resource_files()
        hpo_network, alt2prim, disease_records = generate_annotated_hpo_network(
            obo_file,
            disease_to_phenotype_file,
        )

        file_path = os.path.join(os.path.expanduser("~"), '.phenopy', 'data', 'hp.obo')
        full_hpo_graph = obonet.read_obo(file_path)

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

        return hpo_network.reverse(), name_to_id_and_reverse, scorer

    def save_additional_files(self, output_dir: str):
        """Save name to ID mapping and HPO network structure"""
        output_dir = Path(output_dir)

        # Save name to ID mapping
        with open(output_dir / 'hpo_name_to_id_and_reverse.json', 'w') as f:
            json.dump(self.name_to_id_and_reverse, f)

        # Save network structure using direct edge iteration
        edges = []
        for parent, child in self.hpo_network.edges():
            edges.append({
                'term': extract_id(child),
                'parent_term': extract_id(parent)
            })
        pd.DataFrame(edges).to_csv(output_dir / 'hpo_network.csv', index=False)


class FastSimStorage:
    """Storage class for HPO similarities."""

    @staticmethod
    def collect_all_terms(csv_path):
        """Collect all unique terms from the CSV, regardless of similarity value."""
        all_terms = set()
        for chunk in pd.read_csv(csv_path, chunksize=100000):
            all_terms.update(chunk['term_a'])
            all_terms.update(chunk['term_b'])
        return sorted(list(all_terms))

    @staticmethod
    def csv_to_binary(csv_path, bin_path, terms_path):
        """Convert CSV to binary file, storing only non-zero similarities but tracking all terms."""
        # First collect ALL terms, regardless of similarity
        all_terms = FastSimStorage.collect_all_terms(csv_path)
        np.save(terms_path, all_terms)

        # Write non-zero similarities to binary file
        with open(bin_path, 'wb') as f:
            for chunk in pd.read_csv(csv_path, chunksize=100000):
                non_zero = chunk[chunk['similarity'] > 0]
                for _, row in non_zero.iterrows():
                    f.write(struct.pack('IIf',
                                        int(row['term_a']),
                                        int(row['term_b']),
                                        float(row['similarity'])))

    @staticmethod
    def create_index_file(bin_path, index_path):
        """Create an index file for binary data"""
        term_dict = defaultdict(dict)
        record_size = 12  # 4 bytes each for term_a, term_b, similarity

        with open(bin_path, 'rb') as f:
            offset = 0
            while True:
                data = f.read(record_size)
                if not data or len(data) < record_size:
                    break

                term_a, term_b, similarity = struct.unpack('IIf', data)
                if similarity > 0:  # Only index non-zero similarities
                    term_dict[term_a][term_b] = offset
                    term_dict[term_b][term_a] = offset
                offset += record_size

        np.save(index_path, dict(term_dict))


class NodePairIterator:
    def __init__(self, nodes):
        self.nodes = nodes
        self.i = 0
        self.j = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.nodes):
            raise StopIteration

        pair = (self.nodes[self.i], self.nodes[self.j + self.i])

        # Move to next column
        self.j += 1

        # If we've hit the end of the row, move to next row
        if self.j + self.i >= len(self.nodes):
            self.i += 1
            self.j = 0

        return pair


def build_similarities(output_dir: str = '.'):
    """
    Build HPO similarity files from scratch using phenopy.

    Parameters
    ----------
    output_dir: str
        Directory where the output files will be stored

    Returns
    -------
    dict
        Paths to the generated files (similarities.bin, valid_terms.npy, name_to_id.json, hpo_network.csv)
    """
    # Check for required packages first
    check_phenopy_installation()

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    csv_path = output_dir / 'hpo_similarities.csv'
    bin_path = output_dir / 'similarities.bin'
    terms_path = output_dir / 'valid_terms.npy'
    name_to_id_path = output_dir / 'hpo_name_to_id_and_reverse.json'
    network_path = output_dir / 'hpo_network.csv'
    index_path = output_dir / 'similarities_index.npy'

    # Initialize SimScorer to get nodes
    sim = SimScorer()
    all_nodes = list(sim.hpo_network.nodes())

    # Save additional files
    sim.save_additional_files(output_dir)

    del sim
    gc.collect()

    # Calculate total number of combinations
    n = len(all_nodes)
    total_combinations = (n * (n - 1)) // 2

    # Create single progress bar for entire process
    with tqdm(total=total_combinations, desc="Computing HPO similarities") as pbar:
        # Generate and process pairs in chunks
        chunk_size = 100000
        nodes_iter = NodePairIterator(all_nodes)

        while True:
            chunk = list(islice(nodes_iter, chunk_size))
            if not chunk:
                break
            process_chunk(chunk, csv_path, chunk_size, pbar)
            gc.collect()

    # Convert to binary format
    converter = FastSimStorage()
    converter.csv_to_binary(csv_path, bin_path, terms_path)
    converter.create_index_file(bin_path, index_path)
    # Clean up temporary CSV file
    os.remove(csv_path)

    return {
        'similarities_bin': bin_path,
        'valid_terms': terms_path,
        'name_to_id': name_to_id_path,
        'similarities_index' : index_path,
        'network': network_path
    }
