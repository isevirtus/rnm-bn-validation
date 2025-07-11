import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import truncnorm
import json
import os
import matplotlib.pyplot as plt
import networkx as nx


# Network states
STATES = ['VL', 'L', 'M', 'H', 'VH']  # Very Low, Low, Medium, High, Very High
class FitnessBayesianNetwork:
    """
    A calibrated Bayesian network for fitness assessment modeling.
    
    This class implements a specialized Bayesian network that uses:
    - Weighted aggregation functions (WMEAN, WMIN, WMAX, MIXMINMAX)
    - Truncated normal distributions for probability calibration
    - Automatic CPT generation from a data repository
    
    Attributes:
        model (DiscreteBayesianNetwork): The underlying Bayesian network model
        nodes (dict): Metadata about network nodes (name, states)
        evidence (dict): Currently set evidence for nodes
        config (dict): Network structure configuration
        repo (dict): Data repository for probability calibration
    """
    
    # Default network configuration
    DEFAULT_CONFIG = {
        "AC": {
            "func": "WMAX", 
            "weights": [5, 4, 5, 4, 1], 
            "variance": 0.1, 
            "parents": ["PC_VH", "PC_H", "PC_M", "PC_L", "PC_VL"]
        },
        "AT": {
            "func": "WMEAN", 
            "weights": [4, 2, 5], 
            "variance": 0.1, 
            "parents": ["Dom", "Eco", "Ling"]
        },
        "AE": {
            "func": "WMIN", 
            "weights": [3, 2], 
            "variance": 0.1, 
            "parents": ["AT", "AC"]
        },
        "PC": {
            "func": "WMIN",
            "weights": [1, 5],
            "variance": 0.1,
            "parents": ["OSF", "SLF"]
        }
    }
    
    # Mapping of aggregation function names to methods
    AGGREGATOR_MAP = {
        "WMEAN": "weighted_mean",
        "WMIN": "weighted_min",
        "WMAX": "weighted_max",
        "MIXMINMAX": "min_max_mix"
    }
    
    

    def __init__(self, config=None, repo_path='repository.json'):
        """
        Initialize the fitness Bayesian network.
        
        Args:
            config (dict, optional): Custom network configuration. Uses default if None.
            repo_path (str): Path to the data repository JSON file.
        """
        self.model = DiscreteBayesianNetwork()
        self.nodes = {}
        self.evidence = {}
        self.config = config or self.DEFAULT_CONFIG
        self.repo = self.load_repository(repo_path)
        self.build_network()
        
    def build_network(self):
        """Construct the network structure based on configuration."""
        # Create all nodes
        all_nodes = set()
        for cfg in self.config.values():
            all_nodes.update(cfg["parents"])
        all_nodes.update(self.config.keys())
        
        for node_id in all_nodes:
            self.create_node(node_id, node_id, STATES)

        
        # Add edges (dependencies)
        for child, cfg in self.config.items():
            for parent in cfg["parents"]:
                self.add_edge(parent, child)
        
        # Set uniform CPDs for input nodes (nodes without parents in config)
        for node in all_nodes - self.config.keys():
            self.set_uniform_cpd(node)
        
        # Generate and set CPDs for configured nodes
        for child, cfg in self.config.items():
            cpt = self.generate_cpt(
                cfg["parents"], 
                getattr(self, self.AGGREGATOR_MAP[cfg["func"]]),
                cfg["weights"],
                cfg["variance"]
            )
            self.set_cpd(child, cpt)
            
        # Validate the model
        self.model.check_model()

    def create_node(self, node_id, name, states):
        """
        Add a new node to the network.
        
        Args:
            node_id (str): Unique node identifier
            name (str): Human-readable node name
            states (list): Possible states for the node
        """
        self.nodes[node_id] = {"name": name, "states": states}
        self.model.add_node(node_id)

    def add_edge(self, parent_id, child_id):
        """
        Connect two nodes with a dependency edge.
        
        Args:
            parent_id (str): Parent node ID
            child_id (str): Child node ID
        """
        self.model.add_edge(parent_id, child_id)

    def set_cpd(self, node_id, cpt_values):
        """
        Set the Conditional Probability Distribution (CPD) for a node.
        
        Args:
            node_id (str): Node identifier
            cpt_values (np.array): CPT values array
        """
        parent_ids = list(self.model.get_parents(node_id))
        parent_cards = [len(self.nodes[p]["states"]) for p in parent_ids]
        node_card = len(self.nodes[node_id]["states"])
        
        # Validate CPT dimensions
        if cpt_values.shape != (node_card, np.prod(parent_cards)):
            raise ValueError(
                f"Invalid CPT shape for {node_id}. Expected "
                f"{(node_card, np.prod(parent_cards))}, got {cpt_values.shape}"
            )
        
        # Create and add CPD
        cpd = TabularCPD(
            variable=node_id,
            variable_card=node_card,
            values=cpt_values.tolist(),
            evidence=parent_ids if parent_ids else None,
            evidence_card=parent_cards if parent_cards else None,
            state_names={
                node_id: self.nodes[node_id]["states"],
                **{pid: self.nodes[pid]["states"] for pid in parent_ids}
            }
        )
        self.model.add_cpds(cpd)

    def set_uniform_cpd(self, node_id):
        """Set a uniform CPD for a node with no parents."""
        uniform_cpd = np.full((len(STATES), 1), 1/len(STATES))

        self.set_cpd(node_id, uniform_cpd)

    def set_evidence(self, node_id, state):
        """
        Set evidence for a specific node.
        
        Args:
            node_id (str): Node identifier
            state (str): Observed state
        """
        self.evidence[node_id] = state

    def predict(self, node_id):
        """
        Predict probability distribution for a node.
        
        Args:
            node_id (str): Node identifier
            
        Returns:
            list: Probability distribution for each state
        """
        inference = VariableElimination(self.model)
        return inference.query([node_id], evidence=self.evidence).values.tolist()

    def visualize_network(self, size=(10, 8)):
        """Generate a visual representation of the network structure."""
        plt.figure(figsize=size)
        pos = nx.spring_layout(self.model)
        nx.draw(
            self.model, 
            pos, 
            with_labels=True, 
            node_size=2500, 
            node_color='lightblue', 
            font_size=10, 
            font_weight='bold',
            edge_color='gray'
        )
        plt.title("Fitness Bayesian Network Structure")
        plt.show()
        
    # ===================================================================
    # Aggregation Functions (Static Methods)
    # ===================================================================
    
    @staticmethod
    def weighted_mean(*args):
        """
        Calculate weighted mean of values.
        
        Args:
            *args: Alternating weights and values (weight1, value1, weight2, value2, ...)
            
        Returns:
            float: Weighted mean
        """
        if len(args) % 2 != 0:
            raise ValueError("Invalid number of arguments (expected weight-value pairs)")
        
        total_value = 0.0
        total_weight = 0.0
        
        for i in range(0, len(args), 2):
            weight = args[i]
            value = args[i + 1]
            total_value += weight * value
            total_weight += weight
            
        if total_weight == 0:
            return None
            
        return total_value / total_weight

    @staticmethod
    def weighted_min(*args):
        """
        Calculate weighted minimum using compensation formula.
        
        Args:
            *args: Alternating weights and values
            
        Returns:
            float: Weighted minimum value
        """
        if len(args) % 2 != 0:
            raise ValueError("Invalid number of arguments (expected weight-value pairs)")
        
        n = len(args) // 2
        if n < 2:
            return None
            
        weights = [args[2 * i] for i in range(n)]
        values = [args[2 * i + 1] for i in range(n)]
        
        if sum(weights) == 0:
            return None
            
        total_value = sum(values)
        current_min = float('inf')
        
        for i in range(n):
            denominator = weights[i] + (n - 1)
            numerator = weights[i] * values[i] + (total_value - values[i])
            e_i = numerator / denominator
            current_min = np.minimum(current_min, e_i)
            
        return current_min

    @staticmethod
    def weighted_max(*args):
        """
        Calculate weighted maximum using compensation formula.
        
        Args:
            *args: Alternating weights and values
            
        Returns:
            float: Weighted maximum value
        """
        if len(args) % 2 != 0:
            return None
            
        n = len(args) // 2
        if n < 2:
            return None
            
        weights = []
        values = []
        
        for i in range(n):
            weight = args[2 * i]
            value = args[2 * i + 1]
            if weight < 0 or not (0 <= value <= 1):
                return None
            weights.append(weight)
            values.append(value)
            
        if all(w + (n - 1) == 0 for w in weights):
            return None
            
        current_max = None
        total_value = sum(values)
        
        for i in range(n):
            denominator = weights[i] + (n - 1)
            numerator = weights[i] * values[i] + (total_value - values[i])
            e_i = numerator / denominator
            current_max = e_i if current_max is None else np.maximum(current_max, e_i)
            
        return current_max

    @staticmethod
    def min_max_mix(*args):
        """
        Combine values using a weighted mix of minimum and maximum.
        
        Args:
            *args: Alternating weights and values
            
        Returns:
            float: Combined value
        """
        if len(args) % 2 != 0:
            raise ValueError("Expected weight-value pairs")
            
        n = len(args) // 2
        weights = [args[2 * i] for i in range(n)]
        values = [args[2 * i + 1] for i in range(n)]
        
        if any(w < 0 for w in weights):
            raise ValueError("Weights must be non-negative")
            
        if sum(weights) == 0:
            return None
            
        values_array = np.array(values)
        mins = np.min(values_array, axis=0)
        maxes = np.max(values_array, axis=0)
        
        return (weights[0] * mins + weights[1] * maxes) / sum(weights)

    # ===================================================================
    # CPT Generation and State Mixing
    # ===================================================================
    
    def load_repository(self, path='repository.json'):
        """
        Load the data repository for probability calibration.
        
        Args:
            path (str): Path to repository JSON file
            
        Returns:
            dict: Repository data
        """
        with open(path, 'r', encoding='utf-8') as f:
            repo = json.load(f)
            
        # Convert samples to numpy arrays
        for state in repo:
            if 'samples' in repo[state]:
                repo[state]['samples'] = np.array(repo[state]['samples'])
                
        return repo

    def mix_states(self, parent_states, weights, variance, aggregation_func):
        """
        Combine parent states using aggregation function and truncated normal transformation.
        
        Args:
            parent_states (list): States of parent nodes
            weights (list): Weights for each parent
            variance (float): Desired variance
            aggregation_func (callable): Aggregation function
            
        Returns:
            np.array: Resulting probability distribution
        """
        if not parent_states:
            raise ValueError("parent_states is empty")
            
        weights = np.array(weights, dtype=float)
        if len(weights) != len(parent_states):
            raise ValueError("Weights length doesn't match parent_states")
            
        if np.any(weights < 0) or np.sum(weights) <= 0:
            raise ValueError("Invalid weights")
            
        parent_samples = []
        
        # Retrieve samples for each parent state
        for state in parent_states:
            if state not in self.repo:
                raise KeyError(f"State {state} missing in repository")
                
            samples = self.repo[state].get('samples', [])
            if len(samples) < 10000:
                raise ValueError(f"Insufficient samples for {state} ({len(samples)} < 10000)")
                
            parent_samples.append(np.array(samples[:10000]))
        
        # calcuate the mean for each set of samples 
        parent_means = [np.mean(samples) for samples in parent_samples]
        
        # args 
        agg_args = []
        for weight, mean_val in zip(weights, parent_means):
            agg_args.append(weight)
            agg_args.append(mean_val)
            
        aggregated_value = aggregation_func(*agg_args)
        
        # validate 
        if aggregated_value < 0 or aggregated_value > 1:
            aggregated_value = np.clip(aggregated_value, 0, 1)
            print(f"⚠️ Clipped aggregated value to {aggregated_value}")
        
        
        # calculate statistics
        mean = aggregated_value
        if variance <= 0:
            variance = 0.0001  
            
        # adjustment
        max_variance = mean * (1 - mean)
        if variance > max_variance:
            variance = max(max_variance, 0.0001)
            
        std_dev = np.sqrt(variance)
        
        # truncnorm
        a, b = (0 - mean) / std_dev, (1 - mean) / std_dev
        dist = truncnorm(a, b, loc=mean, scale=std_dev)
        
        bin_edges = np.linspace(0, 1, 6)
        probabilities = np.zeros(5)
        
        for i in range(5):
            prob = dist.cdf(bin_edges[i+1]) - dist.cdf(bin_edges[i])
            probabilities[i] = max(prob, 0)  # Garantir não negativo
            
        # Normalize 
        total = np.sum(probabilities)
        if total > 0:
            probabilities /= total
            
        return np.round(probabilities, 3)

    
    def generate_cpt(self, parents, aggregation_func, weights, variance):
        """
        Generate the complete CPT for a node.
        
        Args:
            parents (list): Parent node IDs
            aggregation_func (callable): Aggregation function
            weights (list): Weights for each parent
            variance (float): Variance parameter
            
        Returns:
            np.array: Complete CPT table
        """
        num_states = len(STATES)
        num_parents = len(parents)
        
        # Criar uma grade de todas as combinações possíveis de estados dos pais
        grid = np.indices((num_states,) * num_parents)
        grid = grid.reshape(num_parents, -1).T
        
        cpt = []
    
        # Gerar distribuição para cada combinação de estados dos pais
        for combo in grid:
            state_names = [STATES[i] for i in combo]
            dist = self.mix_states(state_names, weights, variance, aggregation_func)
            cpt.append(dist)
        
        return np.array(cpt).T


