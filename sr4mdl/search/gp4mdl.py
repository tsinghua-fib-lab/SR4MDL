import json
import time
import torch
import logging
import sklearn
import traceback
import numpy as np
import nd2py as nd
import pandas as pd
from numpy.random import RandomState, default_rng
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List, Tuple, Dict, Generator, Optional, Literal, Set
from sklearn.base import BaseEstimator, RegressorMixin
from ..env.tokenizer import Tokenizer
from ..model.mdlformer import MDLformer
from functools import reduce

_logger = logging.getLogger(__name__)


class Individual:
    def __init__(self, eqtrees: List[nd.Symbol]):
        self.eqtrees = eqtrees
        self.phi = None
        self.complexity = None
        self.accuracy = None
        self.fitness = None
        self.MDL = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            "["
            + ", ".join(str(eq) for eq in self.eqtrees)
            + "]"
            + f" (MDL={self.MDL:.0f})"
        )

    def copy(self) -> "Individual":
        copy = Individual([eqtree.copy() for eqtree in self.eqtrees])
        copy.phi = self.phi.copy() if self.phi is not None else None
        copy.complexity = self.complexity
        copy.accuracy = self.accuracy
        copy.fitness = self.fitness
        copy.MDL = self.MDL
        return copy


class GP4MDL(BaseEstimator, RegressorMixin):
    """
    Minimum Description Length (MDL)-based Genetic Programming-based Symbolic Regression.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        model: MDLformer,
        variables: List[nd.Variable],
        binary: List[nd.Symbol] = [nd.Add, nd.Sub, nd.Mul, nd.Div, nd.Max, nd.Min],
        unary: List[nd.Symbol] = [
            nd.Sqrt,
            nd.Log,
            nd.Abs,
            nd.Neg,
            nd.Inv,
            nd.Sin,
            nd.Cos,
            nd.Tan,
        ],
        max_params: int = 2,
        elitism_k: int = 10,
        population_size: int = 1000,
        tournament_size: int = 20,
        p_crossover: float = 0.9,
        p_subtree_mutation: float = 0.01,
        p_hoist_mutation: float = 0.01,
        p_point_mutation: float = 0.01,
        p_point_replace: float = 0.05,
        const_range: Tuple[float, float] = (-1.0, 1.0),
        depth_range: Tuple[int, int] = (2, 6),
        full_prob: float = 0.5,
        nettype: Optional[Literal["node", "edge", "scalar"]] = "scalar",
        n_jobs: int = None,
        log_per_iter: int = float("inf"),
        log_per_sec: float = float("inf"),
        log_detailed_speed: bool = False,
        save_path: str = None,
        random_state: Optional[int] = None,
        n_iter=100,
        use_tqdm=False,
        edge_list: Tuple[List[int], List[int]] = None,
        num_nodes: int = None,
        keep_vars=False,
        normalize_y=False,
        normalize_all=False,
        remove_abnormal=False,
        train_eval_split: float = 0.25,
        c=1.41,
        eta=0.999,
        **kwargs,
    ):
        """
        Args:
            variables: list of variables
            binary: list of binary operators
            unary: list of unary operators
            max_params: max number of parameters in the equation
            elitism_k: number of elite individuals to keep in the population
            population_size: size of the population
            tournament_size: size of the tournament
            p_crossover: probability of crossover
            p_subtree_mutation: probability of subtree mutation
            p_hoist_mutation: probability of hoist mutation
            p_point_mutation: probability of point mutation
            p_point_replace: probability of point replacement
            const_range: range of constant values
            depth_range: range of tree depth
            full_prob: probability of full tree generation
            nettype: nettype for the generated equations (node, edge, scalar)
            n_jobs: number of jobs for parallel processing (default is None)
            log_per_iter: log every n iterations (default is float('inf'))
            log_per_sec: log every n seconds (default is float('inf'))
            log_detailed_speed: log the speed of each step (default is False)
            save_path: path to save the logs (default is None)
        """
        if num_nodes is None and edge_list is not None:
            num_nodes = np.reshape(edge_list, (-1,)).max() + 1

        self.tokenizer = tokenizer
        self.model = model
        self.keep_vars = keep_vars
        self.normalize_y = normalize_y
        self.normalize_all = normalize_all
        self.remove_abnormal = remove_abnormal

        self.eqtree = None
        self.variables = variables
        self.binary = binary
        self.unary = unary
        self.max_params = max_params
        self.elitism_k = elitism_k
        self.random_state = random_state
        self._rng = default_rng(random_state)
        self.nettype = nettype
        self.train_eval_split = train_eval_split

        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.const_range = const_range
        self.depth_range = depth_range
        self.full_prob = full_prob

        self.population_size = population_size
        self.tournament_size = tournament_size
        self.method_probs = {
            "crossover": p_crossover,  # replace a subtree of parents with a subtree of another parent
            "subtree-mutation": p_subtree_mutation,  # replace a subtree of parent with a random tree
            "hoist-mutation": p_hoist_mutation,  # select a random subtree of parent
            "point-mutation": p_point_mutation,  # replace a random node of parent with a symbol of same arity
        }
        self.p_point_replace = p_point_replace
        assert sum(self.method_probs.values()) <= 1
        self.method_probs["reproduction"] = 1 - sum(self.method_probs.values())

        self.generator = nd.GPLearnGenerator(
            variables=self.variables,
            binary=self.binary,
            unary=self.unary,
            const_range=const_range,
            depth_range=depth_range,
            full_prob=full_prob,
            rng=self._rng,
            edge_list=edge_list,
            num_nodes=num_nodes,
        )

        self.n_jobs = n_jobs
        self.log_per_iter = log_per_iter
        self.log_per_sec = log_per_sec
        self.log_detailed_speed = log_detailed_speed
        self.records = []
        self.speed_timer = nd.utils.Timer()
        self.named_timer = nd.utils.NamedTimer()
        self.save_path = save_path
        self.n_iter = n_iter
        self.use_tqdm = use_tqdm
        self.edge_list = edge_list
        self.num_nodes = num_nodes
        self.eta = eta
        self.c = c

        if kwargs:
            _logger.warning(
                "Unknown args: %s", ", ".join(f"{k}={v}" for k, v in kwargs.items())
            )

    def fit(
        self,
        X: np.ndarray | pd.DataFrame | Dict[str, np.ndarray],
        y: np.ndarray | pd.Series,
    ):
        """
        Args:
            X: (n_samples, n_dims)
            y: (n_samples,)
        """
        if isinstance(X, np.ndarray):
            X = {var.name: x for var, x in zip(self.variables, X[..., :])}
        elif isinstance(X, pd.DataFrame):
            X = {col: X[col].values for i, col in enumerate(X.columns)}
        elif isinstance(X, dict):
            X = {k: np.asarray(v) for k, v in X.items()}
        else:
            raise ValueError(f"Unknown type: {type(X)}")

        self.start_time = time.time()
        population = self.init_population(X, y)
        for iter in tqdm(range(1, 1 + self.n_iter), disable=not self.use_tqdm):
            population = self.evolve(population, X, y)

            self.speed_timer.add()
            best = max(population, key=lambda x: x.fitness).copy()
            self.eqtree = best.phi
            record = dict(
                iter=iter,
                time=time.time() - self.start_time,
                fitness=best.fitness,
                complexity=best.complexity,
                mse=best.accuracy,
                r2=float(1 - best.accuracy / y.var()),
                eqtree=str(best),
                population_size=len(population),
            )
            if (
                not iter % self.log_per_iter
                or self.named_timer.total_time() > self.log_per_sec
            ):
                log = {}
                log["Iter"] = record["iter"]
                log["Fitness"] = record["fitness"]
                log["Complexity"] = record["complexity"]
                log["MSE"] = record["mse"]
                log["R2"] = record["r2"]
                log["Best equation"] = record["eqtree"]
                if self.log_detailed_speed:
                    log["Speed"] = str(self.speed_timer)
                    log["Time"] = str(self.named_timer)
                log["Population size"] = record["population_size"]
                _logger.info(
                    " | ".join(f"\033[4m{k}\033[0m: {v}" for k, v in log.items())
                )
                self.named_timer.clear()
                self.speed_timer.clear()
            record = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in record.items()
            }
            self.records.append(record)
            if self.save_path:
                with open(self.save_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            if best.accuracy < 1e-6:
                _logger.info(
                    f"Early stopping at iter {iter} with accuracy {best.accuracy} ({best})"
                )
                break
            self.named_timer.add("postprocess")
        return self

    def predict(
        self, X: np.ndarray | pd.DataFrame | Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Args:
            X: (n_samples, n_dims)
        Returns:
            y: (n_samples,)
        """
        if self.eqtree is None:
            raise ValueError("Model not fitted yet")

        if isinstance(X, np.ndarray):
            X = {f"x_{i+1}": X[:, i] for i in range(X.shape[1])}
        elif isinstance(X, pd.DataFrame):
            X = {col: X[col].values for col in X.columns}
        elif isinstance(X, dict):
            pass
        else:
            raise ValueError(f"Unknown type: {type(X)}")
        return self.eqtree.eval(vars=X)

    def evolve(
        self,
        population: List[Individual],
        X: Dict[str, np.ndarray],
        y: np.ndarray,
        children_size=None,
        elitism_k=None,
    ) -> List[Individual]:
        if children_size is None:
            children_size = self.population_size
        if elitism_k is None:
            elitism_k = self.elitism_k
        assert (
            children_size > elitism_k
        ), f"children_size {children_size} must be greater than elitism_k {elitism_k}"

        children = []

        top_k = sorted(population, key=lambda x: x.fitness, reverse=True)[:elitism_k]
        children.extend(top_k)

        for parent in tqdm(
            self.tournament(population, children_size - elitism_k),
            disable=not self.use_tqdm,
        ):
            method = list(self.method_probs.keys())[
                np.searchsorted(
                    np.cumsum(list(self.method_probs.values())), self._rng.random()
                )
            ]
            if method == "crossover":
                donor = self.tournament(population, 1)[0]
                child = self.crossover(parent, donor)
            elif method == "subtree-mutation":
                child = self.subtree_mutation(parent)
            elif method == "hoist-mutation":
                child = self.hoist_mutation(parent)
            elif method == "point-mutation":
                child = self.point_mutation(parent)
            elif method == "reproduction":
                child = parent.copy()
            else:
                raise ValueError(f"Unknown method: {method}")
            self.named_timer.add("evolve")
            self.set_fitness(child, X, y)
            children.append(child)
            self.named_timer.add("evaluate")
        return children

    def init_population(
        self, X: Dict[str, np.ndarray], y: np.ndarray
    ) -> List[Individual]:
        """Initialize: 生成初始种群"""
        # 现在的逻辑：每个个体都是相同的 (x1, x2, ..., xn)
        # 被注释掉的逻辑：生成随机公式，再按加法切分为多个单项式 (f1, f2, ..., fn)
        # 虽然后者是 GP 的标准做法，但发现好像不适合 MDL-guided 的方式，所以用了前者的逻辑
        # 不一定管用，需要进一步探索别的方案
        population = []
        for _ in range(self.population_size):
            # eqtree = self.generator.generate_eqtree(nettype=self.nettype)
            # eqtrees = eqtree.split_by_add(
            #     split_by_sub=True,
            #     expand_mul=True,
            #     expand_div=True,
            #     merge_bias=True,
            #     remove_coefficients=True,
            # )[:self.model.args.max_var-1]
            eqtrees = self.variables
            individual = Individual(eqtrees)
            self.set_fitness(individual, X, y)
            population.append(individual)
        return population

    def tournament(self, population: List[Individual], num) -> List[Individual]:
        tournaments = self._rng.choice(
            population, size=(num, self.tournament_size), replace=True
        )
        winners = [
            max(tournament, key=lambda x: x.fitness) for tournament in tournaments
        ]
        return winners

    def crossover(self, parent: Individual, donor: Individual) -> Individual:
        """Crossover: 用 donor 的某个子树替换 parent 的某个子树"""
        # 现在的逻辑：将 child.eqtrees 合并为一个树，donor.eqtrees 也合并为一个树，
        # 然后从 donor 中选择一个子树替换 child 的某个子树。
        # 再把替换后的结果按照 add 拆分成多个 eqtree。
        # 不一定管用，需要进一步探索别的方案
        child = parent.copy()
        child_tree = reduce(lambda x, y: x + y, child.eqtrees)
        donor_tree = reduce(lambda x, y: x + y, donor.eqtrees)
        removed_subtree = self.get_random_subtree(child_tree)
        donored_subtree = self.get_random_subtree(
            donor_tree, nettype=removed_subtree.replaceable_nettype()
        )
        new_tree = child_tree.replace(removed_subtree, donored_subtree)
        child.eqtrees = new_tree.split_by_add(merge_bias=True, remove_coefficients=True)[:self.model.args.max_var-1]
        return child

    def subtree_mutation(self, parent: Individual) -> Individual:
        """Subtree mutation: 用一个随机树替换某个子树"""
        # 同上，不一定管用，需要进一步探索别的方案
        child = parent.copy()
        child_tree = reduce(lambda x, y: x + y, child.eqtrees)
        subtree = self.get_random_subtree(child_tree)
        random_tree = self.generator.generate_eqtree(nettype=subtree.nettype)
        new_tree = child_tree.replace(subtree, random_tree)
        child.eqtrees = new_tree.split_by_add(merge_bias=True, remove_coefficients=True)[:self.model.args.max_var-1]
        return child

    def hoist_mutation(self, parent: Individual) -> Individual:
        """Hoist mutation: 用某个子树替换根节点"""
        # 同上，不一定管用，需要进一步探索别的方案
        child = parent.copy()
        child_tree = reduce(lambda x, y: x + y, child.eqtrees)
        subtree = self.get_random_subtree(child_tree)
        child.eqtrees = subtree.split_by_add(merge_bias=True, remove_coefficients=True)[:self.model.args.max_var-1]
        return child

    def point_mutation(self, parent: Individual) -> Individual:
        """Point mutation: 用随机符号替换 / 插入某个节点"""
        # 同上，不一定管用，需要进一步探索别的方案
        child = parent.copy()
        for idx, eqtree in enumerate(child.eqtrees):
            mutate_nodes = [
                node
                for node in eqtree.iter_postorder()  # Use postorder to ensure we mutate deeper nodes first
                if self._rng.random() < self.p_point_replace
            ]
            for node in mutate_nodes:
                if node.n_operands == 0:
                    # if self._rng.integers(0, 1):
                    sym = self.generator.generate_leaf(nettype=node.nettype)
                    # else:
                    #     sym = self._rng.choice(self.unary)(node.copy())
                elif node.n_operands == 1:
                    child_types = [op.nettype for op in node.operands]
                    nodes = [
                        sym
                        for sym in self.unary
                        if sym.map_nettype(child_types) in node.replaceable_nettype()
                    ]
                    sym = self._rng.choice(nodes)(*node.operands)
                elif node.n_operands == 2:
                    # if self._rng.integers(0, 1):
                    child_types = [op.nettype for op in node.operands]
                    nodes = [
                        sym
                        for sym in self.binary
                        if sym.map_nettype(child_types) in node.replaceable_nettype()
                    ]
                    assert (
                        len(nodes) > 0
                    ), f"No possible nettype for {node} with {child_types}"
                    sym = self._rng.choice(nodes)(*node.operands)
                    # else:
                    #     sym = self._rng.choice(self.unary)(node.copy())
                else:
                    raise ValueError(f"Unknown arity: {node.n_operands}")
                child.eqtrees[idx] = eqtree.replace(node, sym)
        return child

    def set_fitness(
        self, individual: Individual, X: Dict[str, np.ndarray], y: np.ndarray
    ) -> Individual:
        self.estimate_MDL(individual, X, y)
        self.named_timer.add("drop")

        if self.train_eval_split < 1.0:
            train_idx = np.random.rand(y.shape[0]) < self.train_eval_split
            eval_idx = ~train_idx
        else:
            train_idx = np.ones_like(y).astype(bool)
            eval_idx = train_idx

        # Calculate Z
        Z = np.zeros((y.shape[0], 1 + len(individual.eqtrees)))
        Z[:, 0] = 1.0
        for idx, eqtree in enumerate(individual.eqtrees, 1):
            try:
                Z[:, idx] = eqtree.eval(X)
            except:
                Z[:, idx] = np.nan
        Z[~np.isfinite(Z)] = 0.0

        # linear model as phi
        try:
            assert np.isfinite(Z).all()
            A, _, _, _ = np.linalg.lstsq(Z[train_idx, :], y[train_idx], rcond=None)
            A = np.round(A, 6)
            individual.accuracy = np.mean((Z[eval_idx, :] @ A - y[eval_idx])**2)
            individual.phi = nd.Number(A[0]) if A[0] != 0 else None
            for a, op in zip(A[1:], individual.eqtrees):
                if a == 0:
                    pass
                elif a == 1:
                    if individual.phi is None:
                        individual.phi = op
                    else:
                        individual.phi += op
                elif a == -1:
                    if individual.phi is None:
                        individual.phi = -op
                    else:
                        individual.phi -= op
                else:
                    if individual.phi is None:
                        individual.phi = nd.Number(a) * op
                    else:
                        individual.phi += nd.Number(a) * op
            if individual.phi is None:
                individual.phi = nd.Number(0.0)
            individual.complexity = len(individual.phi)
            individual.fitness = self.eta**individual.complexity / (2 - individual.accuracy) + self.c / individual.MDL
        except Exception as e:
            _logger.warning(traceback.format_exc())
            individual.accuracy = np.inf
            individual.complexity = np.inf
            individual.fitness = -np.inf

        # prod model as phi: y = phi(f1, f2, ...) = a0 * |f1|^a1 * |f2|^a2 * ...
        try:
            Z_ = np.log(np.abs(Z).clip(1e-10, None))
            Z_[:, 0] = 1.0
            y_ = np.log(np.abs(y).clip(1e-10, None))
            assert np.isfinite(Z_).all() and np.isfinite(y_).all()
            A, _, _, _ = np.linalg.lstsq(Z_[train_idx, :], y_[train_idx], rcond=None)
            A[0] = np.exp(A[0])
            A = np.round(A, 6)
            prod = 1
            for z, a in zip(Z[:, 1:].T, A[1:]):
                prod *= np.abs(z) ** a
            A[0] *= np.sign(
                np.median(y[train_idx] / (A[0] * prod[train_idx]).clip(1e-6))
            )
            accuracy = np.mean((y[eval_idx] - A[0] * prod[eval_idx]) ** 2)
            fitness = self.eta**individual.complexity / (2 - accuracy) + self.c / individual.MDL
            if fitness > individual.fitness:
                individual.accuracy = accuracy
                individual.fitness = fitness
                individual.phi = nd.Number(A[0]) if A[0] != 1 else None
                for idx, (a, op) in enumerate(zip(A[1:], individual.eqtrees), 1):
                    if (Z[idx] < 0).any():
                        op = nd.Abs(op)
                    if a == 0:
                        pass
                    elif a == 1:
                        if individual.phi is None:
                            individual.phi = op
                        else:
                            individual.phi *= op
                    elif a == -1:
                        if individual.phi is None:
                            individual.phi = nd.Inv(op)
                        else:
                            individual.phi /= op
                    elif a == 2:
                        if individual.phi is None:
                            individual.phi = nd.Pow2(op)
                        else:
                            individual.phi *= nd.Pow2(op)
                    elif a == -2:
                        if individual.phi is None:
                            individual.phi = nd.Inv(nd.Pow2(op))
                        else:
                            individual.phi /= nd.Pow2(op)
                    elif a == 3:
                        if individual.phi is None:
                            individual.phi = nd.Pow3(op)
                        else:
                            individual.phi *= nd.Pow3(op)
                    elif a == -3:
                        if individual.phi is None:
                            individual.phi = nd.Inv(nd.Pow3(op))
                        else:
                            individual.phi /= nd.Pow3(op)
                    elif a == 0.5:
                        if individual.phi is None:
                            individual.phi = nd.Sqrt(op)
                        else:
                            individual.phi *= nd.Sqrt(op)
                    elif a == -0.5:
                        if individual.phi is None:
                            individual.phi = nd.Inv(nd.Sqrt(op))
                        else:
                            individual.phi /= nd.Sqrt(op)
                    elif a > 0:
                        if individual.phi is None:
                            individual.phi = op ** nd.Number(a)
                        else:
                            individual.phi *= op ** nd.Number(a)
                    elif a < 0:
                        if individual.phi is None:
                            individual.phi = nd.Inv(op ** nd.Number(-a))
                        else:
                            individual.phi /= op ** nd.Number(-a)
                    else:
                        raise ValueError(f"Unknown a: {a}")
                if individual.phi is None:
                    individual.phi = nd.Number(1.0)
                individual.complexity = len(individual.phi)
        except Exception as e:
            _logger.warning(traceback.format_exc())
            # logger.warning(str(e))
            individual.accuracy = -np.inf
            individual.complexity = np.inf
            individual.fitness = 0.0

        if not np.isfinite(individual.fitness):
            individual.fitness = 0.0

    def get_random_subtree(
        self,
        tree: nd.Symbol,
        nettype: Set[Literal["node", "edge", "scalar"]] = None,
    ) -> nd.Symbol:
        """
        follow the same approach as GPlearn and Koza (1992) to choose functions 90% of the time and leaves 10% of the time.
        """
        if isinstance(nettype, str):
            nettype = {nettype}
        if nettype is None:
            nodes = [op for op in tree.iter_preorder()]
        else:
            nodes = []
            for op in tree.iter_preorder():
                if op.nettype in nettype or op.nettype == "scalar":
                    nodes.append(op)
                elif op.nettype == "edge" and "node" in nettype:
                    if nd.Aggr in self.unary:
                        nodes.append(nd.Aggr(op))
                    if nd.Rgga in self.unary:
                        nodes.append(nd.Rgga(op))
                elif op.nettype == "node" and "edge" in nettype:
                    if nd.Sour in self.unary:
                        nodes.append(nd.Sour(op))
                    if nd.Targ in self.unary:
                        nodes.append(nd.Targ(op))
        if len(nodes) == 0:
            return self.generator.generate_eqtree(nettype=nettype)

        probs = np.array([0.9 if node.n_operands > 0 else 0.1 for node in nodes])
        subtree = nodes[
            np.searchsorted(np.cumsum(probs / probs.sum()), self._rng.random())
        ]
        return subtree

    def estimate_MDL(self, individuals:Individual|List[Individual], X:Dict[str,np.ndarray], y:np.ndarray, batch_size=64) -> float:
        if isinstance(individuals, Individual): individuals = [individuals]
        batch = np.zeros((len(individuals), y.shape[0], self.model.args.max_var+1)) # (B, N_i, D_max+1,)
        for idx, indiv in enumerate(individuals):
            for i, eqtree in enumerate(indiv.eqtrees): 
                batch[idx, :, i] = eqtree.eval(X)
        if self.normalize_y: y = (y - y.mean()) / (y.std()+1e-6)
        batch[:, :, -1] = y[np.newaxis, :]
        if self.remove_abnormal: batch[((batch > -5) & (batch < 5)).all(axis=-1), :] = np.nan
        if self.normalize_all: batch = (batch - batch.mean(axis=1, keepdims=True)) / (batch.std(axis=1, keepdims=True)+1e-6)
        x = torch.from_numpy(self.tokenizer.float2index(batch, nan_to_pad=True)).to(self.model.device)
        pred = []
        for i in range(0, len(individuals), batch_size):
            pred.extend(self.model.predict(x[i:i+batch_size]).cpu().tolist())
        for indiv, c in zip(individuals, pred):
            indiv.MDL = c
        return pred if len(individuals) > 1 else pred[0]
