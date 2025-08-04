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
from math import radians, cos, sin, asin, sqrt
import os

app = Flask(__name__)
CORS(app)

# ====================== Data Loading and Preparation ======================
def load_and_prepare_data(filepath, split_data=False, test_size=0.2, random_state=42):
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError("CSV file is empty")
        df = df.drop(columns=['paths', 'route'], errors='ignore')
        if 'is_blocked' not in df.columns:
            df['is_blocked'] = 0
        df['cost'] = df['duration_seconds'] + df['distance_meters'] + (df['is_blocked'] * 1000)
        
        if split_data:
            # Split the data into training and test sets
            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=random_state,
                stratify=df['is_blocked']  # Maintain same proportion of blocked roads
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
    is_directed = 'oneway' in df.columns and any(df['oneway'].str.lower() == 'oneway')
    G = nx.DiGraph() if is_directed else nx.Graph()

    for _, row in df.iterrows():
        # Use predicted cost if enabled and model is trained
        if use_predicted_costs and cost_predictor.is_trained:
            weight = cost_predictor.predict_cost(
                row['start_node'],
                row['end_node'],
                row['distance_meters'],
                row['speed_limit_kph'],
                row['is_blocked']
            )
        else:
            weight = float('inf') if row['is_blocked'] else row['cost']
            
        G.add_edge(
            row['start_node'],
            row['end_node'],
            weight=weight,
            distance=row['distance_meters'],
            duration=row['duration_seconds'],
            is_blocked=bool(row['is_blocked']),
            speed_limit=row['speed_limit_kph'],
            original_weight=row['cost']
        )

        # Add reverse edge for undirected graphs if not blocked
        if not is_directed and not row['is_blocked']:
            if use_predicted_costs and cost_predictor.is_trained:
                weight = cost_predictor.predict_cost(
                    row['end_node'],
                    row['start_node'],
                    row['distance_meters'],
                    row['speed_limit_kph'],
                    row['is_blocked']
                )
            
            G.add_edge(
                row['end_node'],
                row['start_node'],
                weight=weight,
                distance=row['distance_meters'],
                duration=row['duration_seconds'],
                is_blocked=bool(row['is_blocked']),
                speed_limit=row['speed_limit_kph'],
                original_weight=row['cost']
            )
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
    """Dijkstra's algorithm implementation that respects blocked edges"""
    start_time = time.perf_counter()
    try:
        # Create a temporary graph where blocked edges have infinite weight
        temp_G = nx.DiGraph() if isinstance(G, nx.DiGraph) else nx.Graph()
        for u, v, d in G.edges(data=True):
            weight = float('inf') if d.get('is_blocked', False) else d['weight']
            temp_G.add_edge(u, v, weight=weight)

        path = nx.shortest_path(temp_G, source=start, target=end, weight='weight')
        cost = nx.shortest_path_length(temp_G, source=start, target=end, weight='weight')
        exec_time = time.perf_counter() - start_time
        return path, cost, exec_time
    except:
        return None, float('inf'), time.perf_counter() - start_time

def run_astar(G, start, end):
    """A* algorithm implementation that respects blocked edges"""
    start_time = time.perf_counter()
    try:
        # Create a temporary graph where blocked edges have infinite weight
        temp_G = nx.DiGraph() if isinstance(G, nx.DiGraph) else nx.Graph()
        for u, v, d in G.edges(data=True):
            weight = float('inf') if d.get('is_blocked', False) else d['weight']
            temp_G.add_edge(u, v, weight=weight)

        path = nx.astar_path(temp_G, source=start, target=end, heuristic=heuristic, weight='weight')
        cost = sum(temp_G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        exec_time = time.perf_counter() - start_time
        return path, cost, exec_time
    except:
        return None, float('inf'), time.perf_counter() - start_time

def run_greedy_bfs(G, start, end):
    """Greedy Best-First Search implementation that respects blocked edges"""
    start_time = time.perf_counter()
    try:
        # Create a temporary graph where blocked edges are removed
        temp_G = nx.DiGraph() if isinstance(G, nx.DiGraph) else nx.Graph()
        for u, v, d in G.edges(data=True):
            if not d.get('is_blocked', False):
                temp_G.add_edge(u, v)

        path = nx.astar_path(temp_G, source=start, target=end, heuristic=heuristic, weight=None)
        cost = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:])) if path else float('inf')
        exec_time = time.perf_counter() - start_time
        return path, cost, exec_time
    except:
        return None, float('inf'), time.perf_counter() - start_time

class AntColony:
    """Optimized Ant Colony Optimization implementation"""
    def __init__(self, graph, n_ants=15, n_iterations=100, decay=0.5, alpha=1, beta=3, 
                 elitist_factor=2, stagnation_limit=10):
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
        
        # Initialize pheromones inversely proportional to edge weights
        self.pheromone = {}
        for u, v, d in self.graph.edges(data=True):
            self.pheromone[(u, v)] = 1 / max(0.1, d['weight'])
            if not isinstance(self.graph, nx.DiGraph):
                self.pheromone[(v, u)] = self.pheromone[(u, v)]

    def run(self, start, end):
        start_time = time.perf_counter()
        best_path = None
        best_cost = float('inf')
        stagnation_count = 0
        iteration_stats = []

        for iteration in range(self.n_iterations):
            paths = []
            costs = []
            
            # Generate paths for all ants
            for _ in range(self.n_ants):
                path = self._construct_path(start, end)
                if path and path[-1] == end:
                    cost = sum(self.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                    paths.append(path)
                    costs.append(cost)
                    
                    if cost < best_cost:
                        best_path = path
                        best_cost = cost
                        stagnation_count = 0
                    else:
                        stagnation_count += 1

            # Early termination if no improvement
            if stagnation_count >= self.stagnation_limit:
                break

            # Update pheromones
            self._update_pheromones(paths, costs, best_path, best_cost)

            # Adaptive parameter adjustment
            if iteration > 0 and iteration % 10 == 0:
                self._adapt_parameters(iteration_stats)

            iteration_stats.append({
                'iteration': iteration,
                'best_cost': best_cost,
                'avg_cost': np.mean(costs) if costs else float('inf')
            })

        exec_time = time.perf_counter() - start_time
        return best_path, best_cost, exec_time

    def _construct_path(self, start, end):
        path = [start]
        current = start
        visited = set([start])
        
        while current != end:
            neighbors = list(self.graph.neighbors(current))
            unvisited = [n for n in neighbors if n not in visited]
            
            if not unvisited:
                return None  # Dead end
            
            # Calculate probabilities with epsilon to avoid division by zero
            probs = []
            total = 0
            epsilon = 1e-10
            
            for neighbor in unvisited:
                edge = (current, neighbor)
                pheromone = self.pheromone.get(edge, epsilon) ** self.alpha
                heuristic = (1 / max(epsilon, self.graph[current][neighbor]['weight'])) ** self.beta
                prob = pheromone * heuristic
                probs.append(prob)
                total += prob
            
            # Normalize probabilities
            if total <= 0:
                probs = [1/len(unvisited)] * len(unvisited)
            else:
                probs = [p/total for p in probs]
            
            next_node = np.random.choice(unvisited, p=probs)
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        return path

    def _update_pheromones(self, paths, costs, best_path, best_cost):
        # Evaporate pheromones
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
        if best_path:
            elite_deposit = self.elitist_factor / max(0.1, best_cost)
            for u, v in zip(best_path[:-1], best_path[1:]):
                self.pheromone[(u, v)] += elite_deposit
                if not isinstance(self.graph, nx.DiGraph):
                    self.pheromone[(v, u)] += elite_deposit

    def _adapt_parameters(self, iteration_stats):
        """Adapt parameters based on performance"""
        if len(iteration_stats) < 2:
            return
            
        # If improvements are slowing down, increase exploration
        last_improvement = iteration_stats[-1]['best_cost'] - iteration_stats[-2]['best_cost']
        if last_improvement > -0.01:  # Small or no improvement
            self.alpha = max(0.5, self.alpha * 0.95)  # Reduce pheromone influence
            self.beta = min(5, self.beta * 1.05)  # Increase heuristic influence
        else:
            # Reset to default values if we're making good progress
            self.alpha = 1
            self.beta = 3

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
    """
    Evaluate algorithms on a graph, optionally using a different graph for cost calculation
    """
    if algorithms is None:
        algorithms = {
            'Dijkstra': run_dijkstra,
            'A*': run_astar,
            'Greedy BFS': run_greedy_bfs,
            'Ant Colony': lambda G, s, e: AntColony(G, n_ants=20, n_iterations=150).run(s, e)
        }
    
    results = []
    
    # Run all algorithms and collect results
    for name, algo in algorithms.items():
        path, cost, exec_time = algo(G, start, end)
        
        # If a separate cost graph is provided, calculate the actual cost using that
        actual_cost = cost
        if cost_graph is not None and path is not None:
            try:
                actual_cost = sum(cost_graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
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
            'Ant Colony': lambda G, s, e: AntColony(G, n_ants=20, n_iterations=150).run(s, e)
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
