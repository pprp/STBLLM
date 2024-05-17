import torch
import json
import random


def fun1_standardize(M):
    if M.dim() == 2:
        M_mean = M.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
        M = M - M_mean
        std = M.view(M.size(0), -1).std(dim=1).view(-1, 1) + 1e-5
        M = M / std.expand_as(M)
    elif M.dim() == 1:
        std = M.std() + 1e-5
        M = (M - M.mean()) / std
    else:
        raise ValueError("Input tensor must have either 1 or 2 dimensions.")
    return M


def fun2_ria(M):
    if M.dim() == 2:
        return torch.abs(M) / (
            torch.sum(torch.abs(M), dim=0, keepdim=True) + 1e-5
        ) + torch.abs(M) / (
            torch.sum(torch.abs(M), dim=1, keepdim=True) + 1e-5
        ).expand_as(
            M
        )
    elif M.dim() == 1:
        return torch.abs(M) / (torch.sum(torch.abs(M)) + 1e-5)
    else:
        raise ValueError("Input tensor must have either 1 or 2 dimensions.")


def fun3_log_1(M):
    # Logarithm of M, adding a small constant to avoid log(0)
    return torch.log(M + 1)


def fun4_min_max_scale(M):
    if M.dim() == 2:
        M_min = M.min(dim=0, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        M_max = M.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        return (M - M_min) / (M_max - M_min + 1e-5)
    elif M.dim() == 1:
        M_min = M.min()
        M_max = M.max()
        return (M - M_min) / (M_max - M_min + 1e-5)
    else:
        raise ValueError("Input tensor must have either 1 or 2 dimensions.")


def fun5_mean(M):
    if M.dim() == 2:
        return M / M.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
    elif M.dim() == 1:
        return M / M.mean(dim=0, keepdim=True)
    else:
        raise ValueError("Input tensor must have either 1 or 2 dimensions.")


class MetricEngine:
    """
    A class for computing pruning metrics using a configurable computational graph.

    The MetricEngine class allows you to define a computational graph using a string representation,
    which consists of a series of operations applied to the weight tensor (W) and an auxiliary tensor (X).
    The class provides methods to parse the graph string, compute the pruning metric based on the defined
    operations, and generate random graphs.

    The supported operations for W and X tensors are:
    - ABS: Element-wise absolute value
    - SUM: Element-wise sum normalization
    - SQRT: Element-wise square root
    - LOG: Element-wise logarithm (with a small constant added to avoid log(0))
    - MMS: Min-max scaling
    - Z_SCALE: Z-score scaling (standardization)
    - MEAN: Element-wise division by mean
    - STANDARDIZE: Standardization (custom implementation)
    - RIA: Relative importance analysis (custom implementation)
    - LOG_PLUS_ONE: Element-wise logarithm plus one (custom implementation)
    - SIGMOID: Element-wise sigmoid
    - TANH: Element-wise hyperbolic tangent

    The supported types for the X tensor are:
    - ROW: Row-wise tensor
    - COL: Column-wise tensor
    - VAR: Variance tensor
    - COL_L1: Column-wise L1 norm tensor
    - ROW_L1: Row-wise L1 norm tensor
    - ROW_MEAN: Row-wise mean tensor
    - COL_MEAN: Column-wise mean tensor
    - ROW_STD: Row-wise standard deviation tensor
    - COL_STD: Column-wise standard deviation tensor

    Args:
        graph_string (str, optional): A string representation of the computational graph. If not provided, a random
            graph will be generated. The string format should be: 'W:(ops)-X[type]:(ops)', where 'ops' are
            comma-separated lists of operations for W and X tensors, and 'type' is the type of the X tensor.

    Attributes:
        _OPS (dict): A dictionary mapping operation names to their corresponding functions.
        _X_DICT_KEY (set): A set of supported types for the X tensor.
        _graph_structure (dict): A dictionary representing the parsed computational graph structure.
        _graph_string (str): The string representation of the computational graph.

    Methods:
        compute_metric(W, X): Applies the graph of operations to the weight tensor W and auxiliary tensor X
            to compute the pruning metric.
        from_string(graph_string): Creates a MetricEngine instance from a graph string.
        generate_random_graph(): Generates a random computational graph string.
        save_json(json_path): Saves the graph structure to a JSON file.
        load_json(json_path): Loads the graph structure from a JSON file.

    Example:
        pengine = MetricEngine("W:(ABS,SUM)-X[ROW]:(SQRT,LOG)")
        metric = pengine.compute_metric(weight_tensor, aux_tensor_dict)
    """
    def __init__(self, graph_string=None):
        self._OPS = {
            "ABS": torch.abs,
            "SUM": lambda x: x / torch.sum(x, dim=0, keepdim=True).expand_as(x),
            "SQRT": torch.sqrt,
            "LOG": lambda x: torch.log(x + 1e-9),
            "MMS": lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-9),
            "Z_SCALE": lambda x: (x - torch.mean(x)) / (torch.std(x) + 1e-9),
            "MEAN": fun5_mean,
            "STANDARDIZE": fun1_standardize,
            "RIA": fun2_ria,
            "LOG_PLUS_ONE": fun3_log_1,
            "SIGMOID": torch.sigmoid,
            "TANH": torch.tanh,
        }

        self._X_DICT_KEY = {
            "ROW",
            "COL",
            "VAR",
            "COL_L1",
            "ROW_L1",
            "ROW_MEAN",
            "COL_MEAN",
            "ROW_STD",
            "COL_STD",
        }
        # 'OUT', 'HESSIAN'}

        self._graph_structure = {
            "W_OP": None,
            "X_TYPE": None,  # Type of X
            "X_OP": None,
        }
        if graph_string is not None:
            self._graph_string = graph_string
            self._parse_graph(graph_string)
        else:
            self._graph_string = self.generate_random_graph()

    def _parse_graph(self, graph_string):
        """Parses the graph string into a structured computational graph"""
        # Example of string: 'W:(mean,abs)-X[row]:(sqrt,log)'
        layers = graph_string.split("-")
        for layer in layers:
            if ":" in layer:
                tensor_type, ops_string = layer.split(":")
                ops_string = ops_string.strip("(").strip(")")
                funcs = ops_string.split(",")
                layer_ops = [self._OPS[func] for func in funcs]
                if tensor_type.startswith("W"):
                    self._graph_structure["W_OP"] = layer_ops
                elif tensor_type.startswith("X"):
                    self._graph_structure["X_TYPE"] = tensor_type[
                        tensor_type.find("[") + 1 : tensor_type.find("]")
                    ]
                    self._graph_structure["X_OP"] = layer_ops
                else:
                    raise ValueError(f"Invalid tensor type: {tensor_type}")
            else:
                raise ValueError(f"Invalid layer format: {layer}")

    def compute_metric(self, W, X: dict):
        """Applies the graph of operations to W and X to compute the final metric"""
        assert (
            self._graph_structure["X_TYPE"] in X
        ), f"X type '{self._graph_structure['X_TYPE']}' not found in X"

        if self._graph_structure["W_OP"] is not None:
            for operation in self._graph_structure["W_OP"]:
                W = operation(W)

        if self._graph_structure["X_OP"] is not None:
            X_tensor = X[self._graph_structure["X_TYPE"]]
            for operation in self._graph_structure["X_OP"]:
                X_tensor = operation(X_tensor)
        else:
            X_tensor = X[self._graph_structure["X_TYPE"]]

        return W * X_tensor

    @staticmethod
    def from_string(graph_string):
        return MetricEngine(graph_string)

    def __str__(self):
        return self._graph_string

    def __repr__(self) -> str:
        return f"MetricEngine({self._graph_string})"

    def generate_random_graph(self):
        """Generates a random graph string"""
        layers = []
        tensor_types = ["W"] + ["X"]
        for tensor_type in tensor_types:
            if tensor_type == "X" and len(self._X_DICT_KEY) > 0:
                ops = ",".join(
                    random.choices(list(self._OPS.keys()), k=random.randint(1, 4))
                )
                tensor_type = f"{tensor_type}[{random.choice(list(self._X_DICT_KEY))}]"
            elif tensor_type == "W" and len(self._OPS) > 0:
                ops = ",".join(
                    random.choices(list(self._OPS.keys()), k=random.randint(1, 4))
                )
            else:
                raise ValueError(
                    "Cannot generate random graph: empty _OPS or _X_DICT_KEY"
                )
            layer_string = f"{tensor_type}:({ops})"
            layers.append(layer_string)
        graph_string = "-".join(layers)
        print("DEBUG:", graph_string)
        self._parse_graph(graph_string)
        return graph_string

    def save_json(self, json_path=None):
        assert json_path is not None, "Path must not be None"
        assert json_path.endswith(".json"), "Path must end with .json"
        # save self._graph_structure to a json file
        with open(json_path, "w") as f:
            json.dump(self._graph_structure, f)

    def load_json(self, json_path=None):
        assert json_path is not None, "Path must not be None"
        assert json_path.endswith(".json"), "Path must end with .json"
        # load self._graph_structure from a json file
        with open(json_path, "r") as f:
            self._graph_structure = json.load(f)
