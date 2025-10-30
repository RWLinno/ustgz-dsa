"""
Simple script to generate demo data for CTIS
No external dependencies required
"""

import json
import os
from datetime import datetime, timedelta
import random
import math

def generate_demo_data():
    """Generate simulated LaDe-style data for demo"""
    
    # Shanghai base coordinates
    base_lat, base_lng = 31.2304, 121.4737
    n_points = 100
    
    # Define role types and colors
    role_types = [
        {'name': 'Hub', 'color': '#e74c3c'},        # 红色 - 配送中心
        {'name': 'Warehouse', 'color': '#3498db'},  # 蓝色 - 仓库
        {'name': 'Station', 'color': '#2ecc71'},    # 绿色 - 配送站
        {'name': 'Residential', 'color': '#f39c12'}, # 橙色 - 住宅区
        {'name': 'Commercial', 'color': '#9b59b6'},  # 紫色 - 商业区
        {'name': 'Office', 'color': '#1abc9c'}       # 青色 - 办公区
    ]
    
    # Create demo directory
    os.makedirs('demo/data', exist_ok=True)
    
    # Generate pickup points with role types
    pickup_points = []
    for i in range(n_points):
        lat = base_lat + (random.random() - 0.5) * 0.1
        lng = base_lng + (random.random() - 0.5) * 0.1
        
        # Assign role type
        role = random.choice(role_types)
        
        # Generate hourly demand (normalized distribution)
        hourly_demand = []
        for hour in range(24):
            # Peak hours: 9-11, 18-20
            base_demand = 5
            if 9 <= hour <= 11 or 18 <= hour <= 20:
                base_demand = 15
            elif 12 <= hour <= 14:
                base_demand = 12
            demand = base_demand + random.randint(-3, 3)
            hourly_demand.append(max(0, demand))
        
        pickup_points.append({
            'id': f'PKG_{i:06d}',
            'type': 'pickup',
            'lat': lat,
            'lng': lng,
            'role': role['name'],
            'role_color': role['color'],
            'aoi_type': role['name'],
            'accept_time': (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
            'courier_id': f'C{random.randint(1, 20):03d}',
            'demand': random.randint(5, 25),
            'hourly_demand': hourly_demand
        })
    
    with open('demo/data/pickup_points.json', 'w') as f:
        json.dump(pickup_points, f, indent=2)
    print(f'✓ Generated {len(pickup_points)} pickup points')
    
    # Generate delivery points with role types
    delivery_points = []
    for i in range(n_points):
        lat = base_lat + (random.random() - 0.5) * 0.1
        lng = base_lng + (random.random() - 0.5) * 0.1
        
        # Assign role type
        role = random.choice(role_types)
        
        # Generate hourly demand
        hourly_demand = []
        for hour in range(24):
            base_demand = 5
            if 9 <= hour <= 11 or 18 <= hour <= 20:
                base_demand = 15
            elif 12 <= hour <= 14:
                base_demand = 12
            demand = base_demand + random.randint(-3, 3)
            hourly_demand.append(max(0, demand))
        
        delivery_points.append({
            'id': f'DEL_{i:06d}',
            'type': 'delivery',
            'lat': lat,
            'lng': lng,
            'role': role['name'],
            'role_color': role['color'],
            'aoi_type': role['name'],
            'delivery_time': (datetime.now() + timedelta(hours=random.randint(1, 24))).isoformat(),
            'courier_id': f'C{random.randint(1, 20):03d}',
            'demand': random.randint(3, 18),
            'hourly_demand': hourly_demand
        })
    
    with open('demo/data/delivery_points.json', 'w') as f:
        json.dump(delivery_points, f, indent=2)
    print(f'✓ Generated {len(delivery_points)} delivery points')
    
    # Generate courier trajectories
    trajectories = {}
    n_couriers = 10
    
    for courier_idx in range(n_couriers):
        courier_id = f'C{courier_idx:03d}'
        trajectory = []
        
        # Start position
        current_lat = base_lat + (random.random() - 0.5) * 0.05
        current_lng = base_lng + (random.random() - 0.5) * 0.05
        
        # Generate 50 GPS points (random walk)
        for point_idx in range(50):
            current_lat += (random.random() - 0.5) * 0.002
            current_lng += (random.random() - 0.5) * 0.002
            
            time_offset = timedelta(minutes=point_idx * 15)
            gps_time = (datetime.now() + time_offset).strftime('%H:%M:%S')
            
            trajectory.append({
                'time': gps_time,
                'lat': current_lat,
                'lng': current_lng
            })
        
        trajectories[courier_id] = trajectory
    
    with open('demo/data/trajectories.json', 'w') as f:
        json.dump(trajectories, f, indent=2)
    print(f'✓ Generated trajectories for {len(trajectories)} couriers')
    
    # Generate sample predictions for time series
    predictions = {}
    for i in range(min(20, n_points)):
        node_id = f'PKG_{i:06d}'
        
        # Historical data (24 hours)
        historical = [random.randint(5, 20) for _ in range(24)]
        
        # Predicted data (24 hours)
        predicted = [random.randint(5, 20) for _ in range(24)]
        
        predictions[node_id] = {
            'historical': historical,
            'predicted': predicted
        }
    
    with open('demo/data/predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f'✓ Generated predictions for {len(predictions)} nodes')
    
    print('\n✅ Demo data generation complete!')
    print(f'   Data saved to: demo/data/')
    print(f'   - pickup_points.json')
    print(f'   - delivery_points.json')
    print(f'   - trajectories.json')
    print(f'   - predictions.json')

if __name__ == '__main__':
    generate_demo_data()

