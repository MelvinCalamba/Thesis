from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import networkx as nx
import numpy as np
import time
from math import radians, cos, sin, asin, sqrt
import os

app = Flask(__name__)
CORS(app)

# ====================== Data Loading and Preparation ======================
def load_and_prepare_data(filepath):
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError("CSV file is empty")
        df = df.drop(columns=['paths', 'route'], errors='ignore')
        if 'is_blocked' not in df.columns:
            df['is_blocked'] = 0
        df['cost'] = df['duration_seconds'] + df['distance_meters'] + (df['is_blocked'] * 1000)
        return df
    except Exception as e:
        app.logger.error(f"Error loading data: {str(e)}")
        raise

# ====================== Graph Construction ======================
def build_graph(df):
    is_directed = 'oneway' in df.columns and any(df['oneway'].str.lower() == 'oneway')
    G = nx.DiGraph() if is_directed else nx.Graph()

    for _, row in df.iterrows():
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
    """Ant Colony Optimization implementation that respects blocked edges"""
    def __init__(self, graph, n_ants=10, n_iterations=50, decay=0.5, alpha=1, beta=2):
        # Create a temporary graph without blocked edges
        self.graph = nx.DiGraph() if isinstance(graph, nx.DiGraph) else nx.Graph()
        for u, v, d in graph.edges(data=True):
            if not d.get('is_blocked', False):
                self.graph.add_edge(u, v, weight=d['weight'])

        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.pheromone = {(u, v): 1 for u, v in self.graph.edges()}

    def run(self, start, end):
        start_time = time.perf_counter()
        best_path = None
        best_cost = float('inf')

        for _ in range(self.n_iterations):
            paths = []
            costs = []

            for _ in range(self.n_ants):
                path = [start]
                current = start
                visited = set([start])
                dead_end = False

                while current != end and not dead_end:
                    neighbors = list(self.graph.neighbors(current))
                    unvisited = [n for n in neighbors if n not in visited]

                    if not unvisited:
                        dead_end = True
                        break

                    probs = []
                    for neighbor in unvisited:
                        edge = (current, neighbor)
                        if edge not in self.pheromone:
                            self.pheromone[edge] = 1
                        pheromone = self.pheromone[edge] ** self.alpha
                        heuristic_val = (1 / max(0.0001, self.graph[current][neighbor]['weight'])) ** self.beta
                        probs.append(pheromone * heuristic_val)

                    total = sum(probs)
                    if total == 0:
                        probs = [1/len(unvisited)] * len(unvisited)
                    else:
                        probs = [p/total for p in probs]

                    next_node = np.random.choice(unvisited, p=probs)
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

                if not dead_end and path[-1] == end:
                    cost = sum(self.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                    paths.append(path)
                    costs.append(cost)

                    if cost < best_cost:
                        best_path = path
                        best_cost = cost

            # Evaporate pheromones
            for edge in self.pheromone:
                self.pheromone[edge] *= self.decay

            # Deposit pheromones
            for path, cost in zip(paths, costs):
                for u, v in zip(path[:-1], path[1:]):
                    edge = (u, v)
                    if edge in self.pheromone:
                        self.pheromone[edge] += 1 / max(0.0001, cost)

        exec_time = time.perf_counter() - start_time
        return best_path, best_cost, exec_time

# ====================== Simulation ======================
def evaluate_algorithms(G, start, end):
    algorithms = {
        'Dijkstra': run_dijkstra,
        'A*': run_astar,
        'Greedy BFS': run_greedy_bfs,
        'Ant Colony': lambda G, s, e: AntColony(G).run(s, e)
    }
    results = []
    for name, algo in algorithms.items():
        path, cost, exec_time = algo(G, start, end)
        results.append({
            'algorithm': name,
            'path_length': len(path) if path else 0,
            'cost': cost if cost != float('inf') else -1,
            'execution_time': exec_time,
            'path_found': path is not None,
            'path': path if path else []
        })
    return results

# ====================== Flask API ======================
@app.route('/get_road_data', methods=['GET'])
def get_road_data():
    df = load_and_prepare_data("Updated_Speed_Limits_with_Duration.csv")
    road_data = df[['start_node', 'end_node', 'is_blocked']].to_dict('records')
    return jsonify(road_data)

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        data = request.json
        
        # Safely get parameters with defaults
        start_node = data.get('start_node', '').strip()
        end_node = data.get('end_node', '').strip()
        blocked_edges = data.get('blocked_edges', [])  # This is the critical fix
        
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

        # Rebuild graph and evaluate
        Gh = build_graph(df)
        results = evaluate_algorithms(Gh, start_node, end_node)
        
        return jsonify({'results': results})

    except Exception as e:
        app.logger.error(f"Error in simulation: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
