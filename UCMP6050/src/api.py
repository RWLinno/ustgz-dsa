"""
Flask API for CTIS Backend
Provides endpoints for:
- Multi-variate time series forecasting
- Route optimization
- Interactive Q&A with LLM
- Real-time predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import json
import os
from typing import Dict, List

from models import create_model
from rag_module import CTISRAGSystem
from rl_module import RLOptimizer


app = Flask(__name__)
# Enable CORS for all origins and methods
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Global variables for models
model = None
rag_system = None
rl_optimizer = None
config = None
device = None


def load_models():
    """Load trained models"""
    global model, rag_system, rl_optimizer, config, device
    
    # Load config
    config_path = os.environ.get('CONFIG_PATH', 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            'n_nodes': 100,
            'input_dim': 10,
            'hidden_dim': 64,
            'output_dim': 10,
            'num_layers': 3,
            'node_feature_dim': 64,
            'window_size': 12,
            'horizon': 12
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load forecasting model
    model = create_model(config).to(device)
    checkpoint_path = os.environ.get('MODEL_PATH', 'checkpoints/best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, using random weights")
    
    # Load RAG system
    rag_system = CTISRAGSystem(config)
    
    # Load RL optimizer
    rl_optimizer = RLOptimizer(config)
    
    print("All models loaded successfully")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Input:
        {
            "node_id": int,
            "historical_data": [[...]], // (T, F)
            "horizon": int
        }
    Output:
        {
            "predictions": [[...]], // (T_out, F)
            "timestamp": str
        }
    """
    try:
        data = request.json
        node_id = data.get('node_id')
        historical_data = np.array(data.get('historical_data'))
        horizon = data.get('horizon', config['horizon'])
        
        # Prepare input
        # Expand to batch format: (1, N, T, F)
        n_nodes = config['n_nodes']
        x = np.zeros((1, n_nodes, historical_data.shape[0], historical_data.shape[1]))
        x[0, node_id, :, :] = historical_data
        
        # Create dummy adjacency and node features
        adj = np.eye(n_nodes)
        node_features = np.random.randn(n_nodes, config['node_feature_dim'])
        
        # Convert to tensors
        x_tensor = torch.FloatTensor(x).to(device)
        adj_tensor = torch.FloatTensor(adj).to(device)
        node_features_tensor = torch.FloatTensor(node_features).to(device)
        
        # Prediction
        with torch.no_grad():
            pred = model(x_tensor, adj_tensor, node_features_tensor)
        
        # Extract prediction for requested node
        pred_np = pred[0, node_id, :horizon, :].cpu().numpy()
        
        return jsonify({
            'predictions': pred_np.tolist(),
            'node_id': node_id,
            'horizon': horizon
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/predict_all', methods=['POST'])
def predict_all():
    """
    Predict for all nodes
    Input:
        {
            "historical_data": [[[...]]], // (N, T, F)
            "horizon": int
        }
    Output:
        {
            "predictions": [[[...]]], // (N, T_out, F)
            "node_ids": [...]
        }
    """
    try:
        data = request.json
        historical_data = np.array(data.get('historical_data'))  # (N, T, F)
        horizon = data.get('horizon', config['horizon'])
        
        # Add batch dimension: (1, N, T, F)
        x = np.expand_dims(historical_data, axis=0)
        
        # Load adjacency and node features (in practice, load from database)
        n_nodes = x.shape[1]
        adj = np.eye(n_nodes)
        node_features = np.random.randn(n_nodes, config['node_feature_dim'])
        
        # Convert to tensors
        x_tensor = torch.FloatTensor(x).to(device)
        adj_tensor = torch.FloatTensor(adj).to(device)
        node_features_tensor = torch.FloatTensor(node_features).to(device)
        
        # Prediction
        with torch.no_grad():
            pred = model(x_tensor, adj_tensor, node_features_tensor)
        
        # Extract predictions
        pred_np = pred[0, :, :horizon, :].cpu().numpy()  # (N, T_out, F)
        
        return jsonify({
            'predictions': pred_np.tolist(),
            'node_ids': list(range(n_nodes)),
            'horizon': horizon
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/optimize_route', methods=['POST'])
def optimize_route():
    """
    Route optimization using RL
    Input:
        {
            "current_locations": [...],
            "demand_prediction": [[...]],  // (N, T)
            "num_vehicles": int
        }
    Output:
        {
            "optimized_routes": [[...]], // List of routes for each vehicle
            "estimated_time": float,
            "estimated_deliveries": int
        }
    """
    try:
        data = request.json
        current_locations = data.get('current_locations', [0] * config.get('n_vehicles', 10))
        demand_prediction = np.array(data.get('demand_prediction'))
        
        # Build current state
        current_state = {
            'locations': current_locations,
            'traffic': np.random.rand(config['n_nodes'])  # In practice, get real traffic data
        }
        
        # Optimize routes
        routes = rl_optimizer.optimize_routes(current_state, demand_prediction)
        
        # Calculate estimated metrics
        estimated_time = len(routes[0]) * 2.5  # Simplified estimation
        estimated_deliveries = int(demand_prediction.sum())
        
        return jsonify({
            'optimized_routes': routes,
            'estimated_time': estimated_time,
            'estimated_deliveries': estimated_deliveries
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Interactive chat with LLM agent
    Input:
        {
            "query": str,
            "context": {...},
            "role": str  // 'driver', 'dispatcher', 'customer', 'analyst'
        }
    Output:
        {
            "answer": str,
            "role": str
        }
    """
    try:
        data = request.json
        query = data.get('query')
        context = data.get('context', {})
        role = data.get('role', 'analyst')
        
        # Get answer from LLM agent
        answer = rag_system.interactive_query(query, context, role)
        
        return jsonify({
            'answer': answer,
            'role': role
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/node_info', methods=['GET'])
def get_node_info():
    """
    Get information about a specific node
    Query params: node_id
    """
    try:
        node_id = int(request.args.get('node_id'))
        
        # In practice, load from database
        node_info = {
            'node_id': node_id,
            'name': f'Delivery Point {node_id}',
            'location': {
                'lat': 37.5665 + np.random.randn() * 0.01,
                'lng': 126.9780 + np.random.randn() * 0.01
            },
            'type': 'pickup' if node_id % 2 == 0 else 'delivery',
            'capacity': np.random.randint(10, 50),
            'current_demand': np.random.randint(0, 20)
        }
        
        return jsonify(node_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/system_status', methods=['GET'])
def get_system_status():
    """
    Get overall system status
    """
    try:
        # Generate system status
        status = {
            'total_nodes': config['n_nodes'],
            'active_vehicles': config.get('n_vehicles', 10),
            'total_deliveries_today': np.random.randint(500, 1000),
            'average_delivery_time': np.random.uniform(15, 30),
            'system_load': np.random.uniform(0.3, 0.9),
            'timestamp': '2025-01-01T12:00:00Z'
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """
    Get system statistics
    Query params: time_range (day, week, month)
    """
    try:
        time_range = request.args.get('time_range', 'day')
        
        # Generate statistics
        stats = {
            'time_range': time_range,
            'total_deliveries': np.random.randint(1000, 5000),
            'success_rate': np.random.uniform(0.90, 0.98),
            'average_wait_time': np.random.uniform(10, 25),
            'peak_hours': [9, 12, 18],
            'busiest_locations': list(np.random.choice(config['n_nodes'], 5, replace=False).astype(int))
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/data/pickup', methods=['GET'])
def get_pickup_data():
    """Get all pickup points for demo"""
    try:
        data_path = 'demo/data/pickup_points.json'
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/data/delivery', methods=['GET'])
def get_delivery_data():
    """Get all delivery points for demo"""
    try:
        data_path = 'demo/data/delivery_points.json'
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/route', methods=['POST'])
def calculate_route():
    """
    Calculate route between two points using routing API
    This is a simplified version - in production use OSRM or Google Directions API
    """
    try:
        data = request.json
        start_lat = data.get('start_lat')
        start_lng = data.get('start_lng')
        end_lat = data.get('end_lat')
        end_lng = data.get('end_lng')
        
        # Simulate route points (in production, call OSRM API)
        # For demo, create curved path between two points
        num_points = 20
        route_points = []
        
        for i in range(num_points + 1):
            t = i / num_points
            # Bezier curve for more realistic route
            lat = start_lat + (end_lat - start_lat) * t + np.sin(t * np.pi) * 0.002
            lng = start_lng + (end_lng - start_lng) * t + np.cos(t * np.pi) * 0.002
            route_points.append({'lat': lat, 'lng': lng})
        
        # Calculate estimated distance and time
        distance = np.sqrt((end_lat - start_lat)**2 + (end_lng - start_lng)**2) * 111
        time = distance * 3  # Assume 20km/h
        
        return jsonify({
            'route': route_points,
            'distance': float(distance),
            'time': float(time)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Load models before starting server
    load_models()
    
    # Start server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

