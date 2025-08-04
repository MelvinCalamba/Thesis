from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import time
import heapq
from math import radians, cos, sin, asin, sqrt
import os

app = Flask(__name__)
CORS(app)

# ====================== Data Loading and Preparation ======================
def load_and_prepare_data(filepath, split_data=False, test_size=0.3, random_state=42):
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError("CSV file is empty")
        # Keep only needed columns
        df = df[['start_node', 'end_node', 'speed_limit_kph', 'distance_meters', 
                'duration_seconds', 'is_signed', 'is_blocked']]
        
        # Convert is_blocked to int if it's not already
        df['is_blocked'] = df['is_blocked'].astype(int)
        
        # Calculate cost - adjust weights as needed
        df['cost'] = df['duration_seconds'] + df['distance_meters'] * 0.1 + (df['is_blocked'] * 1000)
        
        if split_data:
            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=random_state,
                stratify=df['is_blocked']
            )
            return train_df, test_df
        return df
    except Exception as e:
        app.logger.error(f"Error loading data: {str(e)}")
        raise

# ====================== Random Forest Cost Predictor ======================
class RoadCostPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.le_start = LabelEncoder()
        self.le_end = LabelEncoder()
        self.is_trained = False
        self.trained_on_test_data = False
    
    def prepare_features(self, df):
        """Prepare features for the random forest model"""
        try:
            # Encode node IDs
            all_nodes = pd.concat([df['start_node'], df['end_node']]).unique()
            self.le_start.fit(all_nodes)
            self.le_end.fit(all_nodes)
            
            # Create features
            X = pd.DataFrame({
                'start_node_enc': self.le_start.transform(df['start_node']),
                'end_node_enc': self.le_end.transform(df['end_node']),
                'distance': df['distance_meters'],
                'speed_limit': df['speed_limit_kph'],
                'is_blocked': df['is_blocked']
            })
            
            # Target variable
            y = df['cost']
            
            return X, y
        except Exception as e:
            app.logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def train(self, df):
        """Train the random forest model on the provided data"""
        try:
            X, y = self.prepare_features(df)
            self.model.fit(X, y)
            self.is_trained = True
            self.trained_on_test_data = True  # Mark that we trained on test data
            
            # Calculate training error
            preds = self.model.predict(X)
            mse = mean_squared_error(y, preds)
            app.logger.info(f"Model trained with MSE: {mse:.2f}")
            return mse
        except Exception as e:
            app.logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict_cost(self, start_node, end_node, distance, speed_limit, is_blocked=0):
        """Predict cost between two nodes"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            # Encode nodes
            start_enc = self.le_start.transform([start_node])[0]
            end_enc = self.le_end.transform([end_node])[0]
            
            # Create feature vector
            X = pd.DataFrame([{
                'start_node_enc': start_enc,
                'end_node_enc': end_enc,
                'distance': distance,
                'speed_limit': speed_limit,
                'is_blocked': is_blocked
            }])
            
            return self.model.predict(X)[0]
        except Exception as e:
            app.logger.error(f"Error predicting cost: {str(e)}")
            return float('inf')

# Initialize the predictor
cost_predictor = RoadCostPredictor()

# ====================== Graph Construction ======================
def build_graph(df, use_predicted_costs=False):
    """Optimized graph building function"""
    # Determine if graph should be directed
    is_directed = 'is_signed' in df.columns and any(df['is_signed'].str.lower() == 'oneway')
    G = nx.DiGraph() if is_directed else nx.Graph()
    
    # Pre-compute predicted costs if needed
    if use_predicted_costs and cost_predictor.is_trained:
        predicted_costs = {}
        for _, row in df.iterrows():
            key = (row['start_node'], row['end_node'])
            predicted_costs[key] = cost_predictor.predict_cost(
                row['start_node'],
                row['end_node'],
                row['distance_meters'],
                row['speed_limit_kph'],
                row['is_blocked']
            )
    
    # Build edges
    for _, row in df.iterrows():
        start = row['start_node']
        end = row['end_node']
        is_blocked = bool(row['is_blocked'])
        
        if use_predicted_costs and cost_predictor.is_trained:
            weight = predicted_costs[(start, end)]
        else:
            weight = float('inf') if is_blocked else row['cost']
        
        # Add edge with all attributes
        edge_data = {
            'weight': weight,
            'distance': row['distance_meters'],
            'duration': row['duration_seconds'],
            'is_blocked': is_blocked,
            'speed_limit': row['speed_limit_kph'],
            'original_weight': row['cost'],
            'is_oneway': (row['is_signed'].lower() == 'oneway')
        }
        
        G.add_edge(start, end, **edge_data)
        
        # Add reverse edge if needed
        if (not is_directed or row['is_signed'].lower() == 'twoway') and not is_blocked:
            if use_predicted_costs and cost_predictor.is_trained:
                weight = predicted_costs[(end, start)] if (end, start) in predicted_costs else edge_data['original_weight']
            
            G.add_edge(end, start, **edge_data)
            G.edges[end, start]['is_oneway'] = False
    
    return G

# ====================== Heuristic Function ======================
def heuristic(u, v):
    """Euclidean distance heuristic for A* and Greedy BFS"""
    try:
        u_coords = eval(u)
        v_coords = eval(v)
        return sqrt((u_coords[0]-v_coords[0])**2 + (u_coords[1]-v_coords[1])**2)
    except:
        return float('inf')

# ====================== Pathfinding Algorithms ======================
def run_dijkstra(G, start, end):
    """Optimized Dijkstra's algorithm with early termination and priority queue improvements"""
    start_time = time.perf_counter()
    
    if start == end:
        return [start], 0, time.perf_counter() - start_time
    
    try:
        # Create a priority queue using heapq
        heap = []
        heapq.heappush(heap, (0, start))
        
        # Keep track of visited nodes and their costs
        visited = {start: 0}
        # Keep track of the path
        path = {start: [start]}
        
        while heap:
            current_cost, current_node = heapq.heappop(heap)
            
            # Early termination if we've reached the end
            if current_node == end:
                exec_time = time.perf_counter() - start_time
                return path[current_node], current_cost, exec_time
            
            # Skip if we've found a better path already
            if current_cost > visited.get(current_node, float('inf')):
                continue
                
            for neighbor, edge_data in G[current_node].items():
                if edge_data.get('is_blocked', False):
                    continue
                    
                new_cost = current_cost + edge_data['weight']
                
                # Only proceed if this path is better than any existing one
                if neighbor not in visited or new_cost < visited[neighbor]:
                    visited[neighbor] = new_cost
                    path[neighbor] = path[current_node] + [neighbor]
                    heapq.heappush(heap, (new_cost, neighbor))
        
        # If we get here, no path was found
        return None, float('inf'), time.perf_counter() - start_time
        
    except Exception as e:
        app.logger.error(f"Dijkstra error: {str(e)}")
        return None, float('inf'), time.perf_counter() - start_time

def run_astar(G, start, end):
    """Optimized A* algorithm with efficient heuristics and priority queue"""
    start_time = time.perf_counter()
    
    if start == end:
        return [start], 0, time.perf_counter() - start_time
    
    try:
        # Pre-compute heuristic for end node if possible
        try:
            end_coords = eval(end)
            heuristic_cache = {}
        except:
            # Fallback if coordinates aren't available
            def heuristic(u, v): return 0
            
        def get_heuristic(node):
            if node in heuristic_cache:
                return heuristic_cache[node]
            try:
                node_coords = eval(node)
                h = sqrt((node_coords[0]-end_coords[0])**2 + (node_coords[1]-end_coords[1])**2)
                heuristic_cache[node] = h
                return h
            except:
                return 0
        
        # Priority queue: (f_score, g_score, node)
        open_set = []
        heapq.heappush(open_set, (get_heuristic(start), 0, start))
        
        # For node n, came_from[n] is the node immediately preceding it on the cheapest path from start
        came_from = {}
        
        # For node n, g_score[n] is the cost of the cheapest path from start to n currently known
        g_score = {start: 0}
        
        while open_set:
            current_f, current_g, current_node = heapq.heappop(open_set)
            
            if current_node == end:
                # Reconstruct path
                path = []
                while current_node in came_from:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.append(start)
                path.reverse()
                exec_time = time.perf_counter() - start_time
                return path, current_g, exec_time
                
            # Skip if we have a better path already
            if current_g > g_score.get(current_node, float('inf')):
                continue
                
            for neighbor, edge_data in G[current_node].items():
                if edge_data.get('is_blocked', False):
                    continue
                    
                tentative_g = current_g + edge_data['weight']
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + get_heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    
        # No path found
        return None, float('inf'), time.perf_counter() - start_time
        
    except Exception as e:
        app.logger.error(f"A* error: {str(e)}")
        return None, float('inf'), time.perf_counter() - start_time

def run_greedy_bfs(G, start, end):
    """Optimized Greedy Best-First Search with efficient heuristics"""
    start_time = time.perf_counter()
    
    if start == end:
        return [start], 0, time.perf_counter() - start_time
    
    try:
        # Pre-compute heuristic for end node if possible
        try:
            end_coords = eval(end)
            heuristic_cache = {}
        except:
            # Fallback if coordinates aren't available
            def heuristic(u, v): return 0
            
        def get_heuristic(node):
            if node in heuristic_cache:
                return heuristic_cache[node]
            try:
                node_coords = eval(node)
                h = sqrt((node_coords[0]-end_coords[0])**2 + (node_coords[1]-end_coords[1])**2)
                heuristic_cache[node] = h
                return h
            except:
                return 0
        
        # Priority queue based on heuristic only
        open_set = []
        heapq.heappush(open_set, (get_heuristic(start), start))
        
        came_from = {start: None}
        visited = set()
        
        while open_set:
            _, current_node = heapq.heappop(open_set)
            
            if current_node == end:
                # Reconstruct path
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.reverse()
                
                # Calculate actual path cost
                cost = 0
                for u, v in zip(path[:-1], path[1:]):
                    cost += G[u][v]['weight']
                    
                exec_time = time.perf_counter() - start_time
                return path, cost, exec_time
                
            if current_node in visited:
                continue
            visited.add(current_node)
            
            for neighbor, edge_data in G[current_node].items():
                if edge_data.get('is_blocked', False):
                    continue
                if neighbor not in came_from:
                    came_from[neighbor] = current_node
                    heapq.heappush(open_set, (get_heuristic(neighbor), neighbor))
                    
        # No path found
        return None, float('inf'), time.perf_counter() - start_time
        
    except Exception as e:
        app.logger.error(f"Greedy BFS error: {str(e)}")
        return None, float('inf'), time.perf_counter() - start_time

class OptimizedAntColony:
    """Optimized Ant Colony Optimization with multiple performance improvements"""

    def __init__(self, graph, n_ants=15, n_iterations=100, decay=0.5, alpha=1, beta=3,
                 elitist_factor=2, stagnation_limit=15, parallel_ants=False):
        """
        Initialize the optimized ACO algorithm

        Parameters:
        - graph: NetworkX graph
        - n_ants: Number of ants per iteration
        - n_iterations: Maximum number of iterations
        - decay: Pheromone decay rate (0-1)
        - alpha: Importance of pheromone (≥0)
        - beta: Importance of heuristic (≥0)
        - elitist_factor: Extra pheromone for best path
        - stagnation_limit: Stop if no improvement after this many iterations
        - parallel_ants: Whether to simulate parallel ant exploration (conceptual)
        """
        # Create a temporary graph without blocked edges
        self.graph = nx.DiGraph() if isinstance(graph, nx.DiGraph) else nx.Graph()
        for u, v, d in graph.edges(data=True):
            if not d.get('is_blocked', False):
                self.graph.add_edge(u, v, weight=max(0.1, d['weight']))  # Ensure positive weight

        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist_factor = elitist_factor
        self.stagnation_limit = stagnation_limit
        self.parallel_ants = parallel_ants

        # Initialize pheromones inversely proportional to edge weights
        self.pheromone = {}
        self.heuristic_cache = {}
        for u, v, d in self.graph.edges(data=True):
            self.pheromone[(u, v)] = 1 / max(0.1, d['weight'])
            self.heuristic_cache[(u, v)] = (1 / max(0.0001, d['weight'])) ** self.beta
            if not isinstance(self.graph, nx.DiGraph):
                self.pheromone[(v, u)] = self.pheromone[(u, v)]
                self.heuristic_cache[(v, u)] = self.heuristic_cache[(u, v)]

        # Initialize best path tracking
        self.best_path = None
        self.best_cost = float('inf')
        self.stagnation_count = 0
        self.iteration_stats = []

    def run(self, start, end):
        """Run the optimized ACO algorithm"""
        start_time = time.time()

        for iteration in range(self.n_iterations):
            # Generate solutions from all ants
            if self.parallel_ants:
                paths, costs = self._parallel_ant_exploration(start, end)
            else:
                paths, costs = self._sequential_ant_exploration(start, end)

            # Update best solution
            self._update_best_solution(paths, costs)

            # Early termination if stagnating
            if self.stagnation_count >= self.stagnation_limit:
                break

            # Update pheromones
            self._update_pheromones(paths, costs)

            # Adaptive parameter adjustment
            if iteration % 10 == 0:
                self._adapt_parameters(iteration)

        exec_time = time.time() - start_time
        return self.best_path, self.best_cost, exec_time

    def _parallel_ant_exploration(self, start, end):
        """Simulate parallel ant exploration (conceptual optimization)"""
        paths = []
        costs = []

        # Generate paths for all ants (conceptually parallel)
        for _ in range(self.n_ants):
            path = self._construct_path(start, end)
            if path and path[-1] == end:
                cost = sum(self.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                paths.append(path)
                costs.append(cost)

        return paths, costs

    def _sequential_ant_exploration(self, start, end):
        """Traditional sequential ant exploration"""
        return self._parallel_ant_exploration(start, end)  # Same implementation for now

    def _construct_path(self, start, end):
        """Construct a path for a single ant with optimized probability calculation"""
        path = [start]
        current = start
        visited = set([start])

        while current != end:
            neighbors = list(self.graph.neighbors(current))
            unvisited = [n for n in neighbors if n not in visited]

            if not unvisited:
                return None  # Dead end

            # Calculate probabilities using cached values
            probs = np.zeros(len(unvisited))
            total = 0.0

            for i, neighbor in enumerate(unvisited):
                edge = (current, neighbor)
                pheromone = self.pheromone.get(edge, 1e-10) ** self.alpha
                heuristic = self.heuristic_cache.get(edge, 1e-10)
                probs[i] = pheromone * heuristic
                total += probs[i]

            # Normalize probabilities
            if total <= 0:
                probs = np.ones(len(unvisited)) / len(unvisited)
            else:
                probs /= total

            # Choose next node using numpy's optimized random choice
            next_node = unvisited[np.random.choice(len(unvisited), p=probs)]
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    def _update_best_solution(self, paths, costs):
        """Update the best found solution"""
        if costs and min(costs) < self.best_cost:
            idx = np.argmin(costs)
            self.best_path = paths[idx]
            self.best_cost = costs[idx]
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

    def _update_pheromones(self, paths, costs):
        """Update pheromone trails with elitist strategy"""
        # Evaporate all pheromones
        for edge in self.pheromone:
            self.pheromone[edge] *= self.decay

        # Deposit pheromones from all ants
        for path, cost in zip(paths, costs):
            deposit = 1 / max(0.1, cost)
            for u, v in zip(path[:-1], path[1:]):
                self.pheromone[(u, v)] += deposit
                if not isinstance(self.graph, nx.DiGraph):
                    self.pheromone[(v, u)] += deposit

        # Elitist strategy - reinforce best path
        if self.best_path and self.best_cost < float('inf'):
            elite_deposit = self.elitist_factor / max(0.1, self.best_cost)
            for u, v in zip(self.best_path[:-1], self.best_path[1:]):
                self.pheromone[(u, v)] += elite_deposit
                if not isinstance(self.graph, nx.DiGraph):
                    self.pheromone[(v, u)] += elite_deposit

    def _adapt_parameters(self, iteration):
        """Adaptively adjust parameters based on performance"""
        # Gradually shift from exploration to exploitation
        progress = iteration / self.n_iterations
        self.alpha = min(2.0, 1.0 + progress)  # Increase pheromone importance
        self.beta = max(1.0, 3.0 - progress)   # Decrease heuristic importance

        # Adjust decay rate based on solution diversity
        if self.stagnation_count > self.stagnation_limit / 2:
            self.decay = max(0.3, self.decay * 0.95)  # More exploration
        else:
            self.decay = min(0.9, self.decay * 1.05)  # More exploitation

# ====================== Evaluation with Data Splitting ======================
def evaluate_with_splitting(filepath, algorithms, test_size=0.2, n_runs=5):
    """
    Evaluate algorithms on multiple train/test splits of the data
    Uses test data for calculating the 'cost' of paths found in training
    Returns average performance metrics across all runs
    """
    results = {
        'train': {name: {'success_rate': [], 'avg_cost': [], 'avg_time': []} for name in algorithms},
        'test': {name: {'success_rate': [], 'avg_cost': [], 'avg_time': []} for name in algorithms}
    }
    
    for _ in range(n_runs):
        # Load and split data
        train_df, test_df = load_and_prepare_data(filepath, split_data=True, test_size=test_size)
        
        # Build graphs - we need both the training and test graphs
        train_G = build_graph(train_df)
        test_G = build_graph(test_df)
        
        # Also build a combined graph with test data weights for cost calculation
        combined_df = pd.concat([train_df, test_df])
        combined_G = build_graph(combined_df)
        
        # Select random start and end nodes that exist in both graphs
        common_nodes = list(set(train_G.nodes()) & set(test_G.nodes()))
        if len(common_nodes) < 2:
            continue
            
        start, end = np.random.choice(common_nodes, 2, replace=False)
        
        # Evaluate on training set but calculate costs using test data
        train_results = evaluate_algorithms(train_G, start, end, algorithms)
        for algo in train_results:
            name = algo['algorithm']
            path = algo['path']
            
            # Calculate the actual cost using the test data graph
            test_cost = float('inf')
            if path:
                try:
                    # Calculate path cost using the combined graph (which includes test data)
                    test_cost = sum(combined_G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                except:
                    test_cost = float('inf')
            
            results['train'][name]['success_rate'].append(path is not None)
            if path is not None:
                results['train'][name]['avg_cost'].append(test_cost)
                results['train'][name]['avg_time'].append(algo['execution_time'])
        
        # Evaluate on test set (uses test data naturally)
        test_results = evaluate_algorithms(test_G, start, end, algorithms)
        for algo in test_results:
            name = algo['algorithm']
            results['test'][name]['success_rate'].append(algo['path_found'])
            if algo['path_found']:
                results['test'][name]['avg_cost'].append(algo['cost'])
                results['test'][name]['avg_time'].append(algo['execution_time'])
    
    # Calculate averages
    summary = {}
    for dataset in ['train', 'test']:
        summary[dataset] = {}
        for algo in algorithms:
            success_rate = np.mean(results[dataset][algo]['success_rate'])
            avg_cost = np.mean(results[dataset][algo]['avg_cost']) if results[dataset][algo]['avg_cost'] else None
            avg_time = np.mean(results[dataset][algo]['avg_time']) if results[dataset][algo]['avg_time'] else None
            
            summary[dataset][algo] = {
                'success_rate': success_rate,
                'avg_cost': avg_cost,
                'avg_time': avg_time
            }
    
    return summary

# ====================== Modified evaluate_algorithms to accept custom algorithms ======================
def evaluate_algorithms(G, start, end, algorithms=None, cost_graph=None):
    """Optimized algorithm evaluation function"""
    if algorithms is None:
        algorithms = {
            'Dijkstra': run_dijkstra,
            'A*': run_astar,
            'Greedy BFS': run_greedy_bfs,
            'Ant Colony': lambda G, s, e: OptimizedAntColony(G, n_ants=20, n_iterations=150).run(s, e)
        }
    
    results = []
    
    # If cost_graph isn't provided, use the main graph
    cost_G = cost_graph if cost_graph is not None else G
    
    # Run all algorithms and collect results
    for name, algo in algorithms.items():
        path, cost, exec_time = algo(G, start, end)
        
        # Calculate actual cost if path exists
        actual_cost = float('inf')
        if path is not None:
            try:
                actual_cost = 0
                for u, v in zip(path[:-1], path[1:]):
                    actual_cost += cost_G[u][v]['weight']
            except:
                actual_cost = float('inf')
        
        results.append({
            'algorithm': name,
            'path_length': len(path) if path else 0,
            'cost': actual_cost if actual_cost != float('inf') else -1,
            'execution_time': exec_time,
            'path_found': path is not None,
            'path': path if path else []
        })
    
    # Sort by execution time (fastest first)
    results.sort(key=lambda x: x['execution_time'])
    
    return results

# ====================== API Endpoints ======================
@app.route('/get_road_data', methods=['GET'])
def get_road_data():
    df = load_and_prepare_data("Updated_Speed_Limits_with_Duration.csv")
    road_data = df[['start_node', 'end_node', 'is_blocked']].to_dict('records')
    return jsonify(road_data)

@app.route('/train_cost_predictor', methods=['POST'])
def train_cost_predictor():
    try:
        data = request.json
        test_size = float(data.get('test_size', 0.2))
        
        # Load and split data - we'll use the test portion for training
        train_df, test_df = load_and_prepare_data(
            "Updated_Speed_Limits_with_Duration.csv",
            split_data=True,
            test_size=test_size
        )
        
        # Train the model on TEST data (not train data)
        train_mse = cost_predictor.train(test_df)  # Using test_df instead of train_df
        
        # Evaluate on training set (reverse of normal approach)
        X_train, y_train = cost_predictor.prepare_features(train_df)
        preds = cost_predictor.model.predict(X_train)
        test_mse = mean_squared_error(y_train, preds)
        
        return jsonify({
            'status': 'success',
            'train_mse': train_mse,  # Actually test data performance
            'test_mse': test_mse,    # Actually train data performance
            'feature_importances': dict(zip(X_train.columns, cost_predictor.model.feature_importances_)),
            'note': 'Model trained on test data and evaluated on training data'
        })
    
    except Exception as e:
        app.logger.error(f"Error training cost predictor: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_path_cost', methods=['POST'])
def predict_path_cost():
    try:
        data = request.json
        path = data.get('path', [])
        use_predicted = data.get('use_predicted', False)
        
        if not path or len(path) < 2:
            return jsonify({'error': 'Invalid path'}), 400
        
        if use_predicted and not cost_predictor.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        # Load the full dataset to get edge attributes
        df = load_and_prepare_data("Updated_Speed_Limits_with_Duration.csv")
        full_G = build_graph(df)
        
        total_cost = 0
        edge_details = []
        
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            
            try:
                edge_data = full_G[u][v]
                
                if use_predicted:
                    cost = cost_predictor.predict_cost(
                        u, v,
                        edge_data['distance'],
                        edge_data['speed_limit'],
                        edge_data['is_blocked']
                    )
                else:
                    cost = edge_data['weight']
                
                total_cost += cost
                edge_details.append({
                    'from': u,
                    'to': v,
                    'distance': edge_data['distance'],
                    'speed_limit': edge_data['speed_limit'],
                    'is_blocked': edge_data['is_blocked'],
                    'cost': cost,
                    'cost_type': 'predicted (from test data)' if use_predicted else 'actual'
                })
            except KeyError:
                return jsonify({'error': f'Edge {u}->{v} not found'}), 400
        
        return jsonify({
            'total_cost': total_cost,
            'edge_details': edge_details,
            'used_predicted': use_predicted,
            'trained_on_test_data': cost_predictor.trained_on_test_data if use_predicted else None
        })
    
    except Exception as e:
        app.logger.error(f"Error predicting path cost: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        data = request.json
        
        # Safely get parameters with defaults
        start_node = data.get('start_node', '').strip()
        end_node = data.get('end_node', '').strip()
        blocked_edges = data.get('blocked_edges', [])
        use_predicted_costs = data.get('use_predicted_costs', False)
        
        if not start_node or not end_node:
            return jsonify({'error': 'Missing start or end node'}), 400

        # Load and prepare data
        df = load_and_prepare_data("Updated_Speed_Limits_with_Duration.csv")
        df['is_blocked'] = 0  # Reset all edges

        # Helper function to parse coordinates safely
        def parse_coord(coord_str):
            try:
                if not coord_str or not isinstance(coord_str, str):
                    return None
                # Remove parentheses and split
                clean_str = coord_str.strip().strip('()')
                lat, lon = map(float, clean_str.split(','))
                return (lat, lon)
            except:
                return None

        # Apply blockages
        for edge in blocked_edges:
            if not isinstance(edge, dict):
                continue
                
            blocked_start = parse_coord(edge.get('start_node'))
            blocked_end = parse_coord(edge.get('end_node'))
            
            if not blocked_start or not blocked_end:
                continue

            for i, row in df.iterrows():
                row_start = parse_coord(row['start_node'])
                row_end = parse_coord(row['end_node'])
                
                if not row_start or not row_end:
                    continue

                # Check both directions
                if ((blocked_start == row_start and blocked_end == row_end) or
                    (blocked_start == row_end and blocked_end == row_start)):
                    df.at[i, 'is_blocked'] = 1

        # Rebuild graph (with option to use predicted costs)
        Gh = build_graph(df, use_predicted_costs=use_predicted_costs)
        results = evaluate_algorithms(Gh, start_node, end_node)
        
        return jsonify({
            'results': results,
            'used_predicted_costs': use_predicted_costs and cost_predictor.is_trained,
            'trained_on_test_data': cost_predictor.trained_on_test_data if use_predicted_costs else None
        })

    except Exception as e:
        app.logger.error(f"Error in simulation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_with_splits', methods=['POST'])
def evaluate_with_splits():
    try:
        data = request.json
        test_size = float(data.get('test_size', 0.2))
        n_runs = int(data.get('n_runs', 5))
        
        # Define which algorithms to evaluate
        algorithms = {
            'Dijkstra': run_dijkstra,
            'A*': run_astar,
            'Greedy BFS': run_greedy_bfs,
            'Ant Colony': lambda G, s, e: OptimizedAntColony(G, n_ants=20, n_iterations=150).run(s, e)
        }
        
        results = evaluate_with_splitting(
            "Updated_Speed_Limits_with_Duration.csv",
            algorithms,
            test_size=test_size,
            n_runs=n_runs
        )
        
        return jsonify(results)
    
    except Exception as e:
        app.logger.error(f"Error in split evaluation: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
