"""
LaDe Dataset Loader
Dataset source: https://huggingface.co/datasets/Cainiao-AI/LaDe
Citation: Wu et al., "LaDe: The First Comprehensive Last-mile Delivery Dataset from Industry", 2023
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta


class LaDeDataLoader:
    """
    Loader for LaDe (Last-mile Delivery) Dataset
    Supports both LaDe-P (Pickup) and LaDe-D (Delivery) data
    """
    
    def __init__(self, data_dir: str = 'data/lade'):
        self.data_dir = data_dir
        self.pickup_data = None
        self.delivery_data = None
        self.trajectory_data = None
        
    def download_from_huggingface(self):
        """
        Download LaDe dataset from Hugging Face
        Dataset: https://huggingface.co/datasets/Cainiao-AI/LaDe
        """
        try:
            from datasets import load_dataset
            
            print("Downloading LaDe dataset from Hugging Face...")
            dataset = load_dataset("Cainiao-AI/LaDe")
            
            # Save to local directory
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Process and save different splits
            for split in dataset.keys():
                df = pd.DataFrame(dataset[split])
                output_path = os.path.join(self.data_dir, f"{split}.csv")
                df.to_csv(output_path, index=False)
                print(f"Saved {split} split to {output_path}")
                
            print("Dataset download complete!")
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please download manually from https://huggingface.co/datasets/Cainiao-AI/LaDe")
    
    def load_pickup_data(self, city: str = 'sh') -> pd.DataFrame:
        """
        Load LaDe-P (Pickup) data
        Fields: package_id, time_window_start, time_window_end, lng, lat, city, 
                region_id, aoi_id, aoi_type, courier_id, accept_time, 
                accept_gps_time, accept_gps_lng, accept_gps_lat, pickup_time,
                pickup_gps_time, pickup_gps_lng, pickup_gps_lat, ds
        """
        file_path = os.path.join(self.data_dir, 'pickup', f'pickup_{city}.csv')
        
        if not os.path.exists(file_path):
            print(f"Pickup data not found at {file_path}")
            print("Generating simulated pickup data...")
            return self._generate_simulated_pickup_data()
        
        df = pd.read_csv(file_path)
        self.pickup_data = df
        return df
    
    def load_delivery_data(self, city: str = 'sh') -> pd.DataFrame:
        """
        Load LaDe-D (Delivery) data
        Fields: package_id, lng, lat, city, region_id, aoi_id, aoi_type,
                courier_id, accept_time, accept_gps_time, accept_gps_lng,
                accept_gps_lat, delivery_time, delivery_gps_time, 
                delivery_gps_lng, delivery_gps_lat, ds
        """
        file_path = os.path.join(self.data_dir, 'delivery', f'delivery_{city}.csv')
        
        if not os.path.exists(file_path):
            print(f"Delivery data not found at {file_path}")
            print("Generating simulated delivery data...")
            return self._generate_simulated_delivery_data()
        
        df = pd.read_csv(file_path)
        self.delivery_data = df
        return df
    
    def load_trajectory_data(self, city: str = 'sh') -> pd.DataFrame:
        """
        Load courier trajectory data
        Fields: ds, postman_id, gps_time, lat, lng
        """
        file_path = os.path.join(self.data_dir, 'trajectory', f'trajectory_{city}.csv')
        
        if not os.path.exists(file_path):
            print(f"Trajectory data not found at {file_path}")
            print("Generating simulated trajectory data...")
            return self._generate_simulated_trajectory_data()
        
        df = pd.read_csv(file_path)
        self.trajectory_data = df
        return df
    
    def _generate_simulated_pickup_data(self, n_points: int = 100) -> pd.DataFrame:
        """Generate simulated pickup data for demo"""
        # Shanghai coordinates: ~31.2304° N, 121.4737° E
        base_lat, base_lng = 31.2304, 121.4737
        
        data = []
        start_date = datetime(2024, 1, 1)
        
        for i in range(n_points):
            lat = base_lat + np.random.randn() * 0.05
            lng = base_lng + np.random.randn() * 0.05
            
            date_offset = np.random.randint(0, 30)
            accept_time = start_date + timedelta(days=date_offset, hours=np.random.randint(8, 20))
            pickup_time = accept_time + timedelta(minutes=np.random.randint(10, 60))
            
            data.append({
                'package_id': f'PKG_{i:06d}',
                'time_window_start': (accept_time + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
                'time_window_end': (accept_time + timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S'),
                'lng': lng,
                'lat': lat,
                'city': 'Shanghai',
                'region_id': f'R{np.random.randint(1, 10)}',
                'aoi_id': f'AOI_{np.random.randint(1, 50)}',
                'aoi_type': np.random.choice(['Residential', 'Commercial', 'Office']),
                'courier_id': f'C{np.random.randint(1, 20):03d}',
                'accept_time': accept_time.strftime('%Y-%m-%d %H:%M:%S'),
                'accept_gps_time': accept_time.strftime('%Y-%m-%d %H:%M:%S'),
                'accept_gps_lng': lng + np.random.randn() * 0.001,
                'accept_gps_lat': lat + np.random.randn() * 0.001,
                'pickup_time': pickup_time.strftime('%Y-%m-%d %H:%M:%S'),
                'pickup_gps_time': pickup_time.strftime('%Y-%m-%d %H:%M:%S'),
                'pickup_gps_lng': lng + np.random.randn() * 0.001,
                'pickup_gps_lat': lat + np.random.randn() * 0.001,
                'ds': accept_time.strftime('%Y%m%d')
            })
        
        df = pd.DataFrame(data)
        
        # Save to file
        os.makedirs(os.path.join(self.data_dir, 'pickup'), exist_ok=True)
        df.to_csv(os.path.join(self.data_dir, 'pickup', 'pickup_sh.csv'), index=False)
        
        return df
    
    def _generate_simulated_delivery_data(self, n_points: int = 100) -> pd.DataFrame:
        """Generate simulated delivery data for demo"""
        base_lat, base_lng = 31.2304, 121.4737
        
        data = []
        start_date = datetime(2024, 1, 1)
        
        for i in range(n_points):
            lat = base_lat + np.random.randn() * 0.05
            lng = base_lng + np.random.randn() * 0.05
            
            date_offset = np.random.randint(0, 30)
            accept_time = start_date + timedelta(days=date_offset, hours=np.random.randint(8, 20))
            delivery_time = accept_time + timedelta(minutes=np.random.randint(30, 120))
            
            data.append({
                'package_id': f'PKG_{i:06d}',
                'lng': lng,
                'lat': lat,
                'city': 'Shanghai',
                'region_id': f'R{np.random.randint(1, 10)}',
                'aoi_id': f'AOI_{np.random.randint(1, 50)}',
                'aoi_type': np.random.choice(['Residential', 'Commercial', 'Office']),
                'courier_id': f'C{np.random.randint(1, 20):03d}',
                'accept_time': accept_time.strftime('%Y-%m-%d %H:%M:%S'),
                'accept_gps_time': accept_time.strftime('%Y-%m-%d %H:%M:%S'),
                'accept_gps_lng': lng + np.random.randn() * 0.001,
                'accept_gps_lat': lat + np.random.randn() * 0.001,
                'delivery_time': delivery_time.strftime('%Y-%m-%d %H:%M:%S'),
                'delivery_gps_time': delivery_time.strftime('%Y-%m-%d %H:%M:%S'),
                'delivery_gps_lng': lng + np.random.randn() * 0.002,
                'delivery_gps_lat': lat + np.random.randn() * 0.002,
                'ds': accept_time.strftime('%Y%m%d')
            })
        
        df = pd.DataFrame(data)
        
        # Save to file
        os.makedirs(os.path.join(self.data_dir, 'delivery'), exist_ok=True)
        df.to_csv(os.path.join(self.data_dir, 'delivery', 'delivery_sh.csv'), index=False)
        
        return df
    
    def _generate_simulated_trajectory_data(self, n_couriers: int = 10, points_per_courier: int = 50) -> pd.DataFrame:
        """Generate simulated trajectory data for demo"""
        base_lat, base_lng = 31.2304, 121.4737
        
        data = []
        start_date = datetime(2024, 1, 1, 8, 0, 0)
        
        for courier_idx in range(n_couriers):
            courier_id = f'C{courier_idx:03d}'
            current_lat = base_lat + np.random.randn() * 0.03
            current_lng = base_lng + np.random.randn() * 0.03
            
            for point_idx in range(points_per_courier):
                # Random walk
                current_lat += np.random.randn() * 0.001
                current_lng += np.random.randn() * 0.001
                
                gps_time = start_date + timedelta(minutes=point_idx * 15)
                
                data.append({
                    'ds': int(gps_time.strftime('%j')),  # Day of year
                    'postman_id': courier_id,
                    'gps_time': gps_time.strftime('%m-%d %H:%M:%S'),
                    'lat': current_lat,
                    'lng': current_lng
                })
        
        df = pd.DataFrame(data)
        
        # Save to file
        os.makedirs(os.path.join(self.data_dir, 'trajectory'), exist_ok=True)
        df.to_csv(os.path.join(self.data_dir, 'trajectory', 'trajectory_sh.csv'), index=False)
        
        return df
    
    def prepare_for_model(self, time_window: int = 12, horizon: int = 12) -> Dict:
        """
        Prepare data for CTIS model training
        Returns: dict with temporal_data, adjacency, node_features
        """
        # Load all data
        pickup_df = self.load_pickup_data()
        delivery_df = self.load_delivery_data()
        
        # Aggregate by location and time
        delivery_df['hour'] = pd.to_datetime(delivery_df['delivery_time']).dt.hour
        delivery_df['demand'] = 1  # Count deliveries
        
        # Group by location (rounded coordinates) and hour
        delivery_df['loc_id'] = (
            (delivery_df['lat'] * 100).astype(int).astype(str) + '_' +
            (delivery_df['lng'] * 100).astype(int).astype(str)
        )
        
        # Create time series for each location
        demand_series = delivery_df.groupby(['loc_id', 'hour'])['demand'].sum().unstack(fill_value=0)
        
        # Get unique locations
        unique_locs = delivery_df.groupby('loc_id')[['lat', 'lng']].mean()
        
        # Create node features
        n_nodes = len(unique_locs)
        node_features = np.random.randn(n_nodes, 64)  # Placeholder features
        
        # Create adjacency matrix (distance-based)
        from scipy.spatial.distance import cdist
        coords = unique_locs[['lat', 'lng']].values
        dist_matrix = cdist(coords, coords)
        adjacency = (dist_matrix < 0.01).astype(float)  # Threshold: ~1km
        np.fill_diagonal(adjacency, 0)
        
        # Create temporal data
        temporal_data = demand_series.values  # Shape: (n_nodes, T)
        
        return {
            'temporal_data': temporal_data,
            'adjacency': adjacency,
            'node_features': node_features,
            'locations': unique_locs.reset_index(),
            'n_nodes': n_nodes
        }
    
    def export_for_demo(self, output_dir: str = 'demo/data'):
        """Export data in format suitable for web demo"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        pickup_df = self.load_pickup_data()
        delivery_df = self.load_delivery_data()
        trajectory_df = self.load_trajectory_data()
        
        # Export pickup points
        pickup_points = []
        for _, row in pickup_df.iterrows():
            pickup_points.append({
                'id': row['package_id'],
                'type': 'pickup',
                'lat': float(row['lat']),
                'lng': float(row['lng']),
                'aoi_type': row['aoi_type'],
                'accept_time': row['accept_time'],
                'courier_id': row['courier_id']
            })
        
        with open(os.path.join(output_dir, 'pickup_points.json'), 'w') as f:
            json.dump(pickup_points, f, indent=2)
        
        # Export delivery points
        delivery_points = []
        for _, row in delivery_df.iterrows():
            delivery_points.append({
                'id': row['package_id'],
                'type': 'delivery',
                'lat': float(row['lat']),
                'lng': float(row['lng']),
                'aoi_type': row['aoi_type'],
                'delivery_time': row['delivery_time'],
                'courier_id': row['courier_id']
            })
        
        with open(os.path.join(output_dir, 'delivery_points.json'), 'w') as f:
            json.dump(delivery_points, f, indent=2)
        
        # Export trajectories
        trajectories = {}
        for courier_id in trajectory_df['postman_id'].unique():
            courier_traj = trajectory_df[trajectory_df['postman_id'] == courier_id]
            trajectories[courier_id] = [
                {
                    'time': row['gps_time'],
                    'lat': float(row['lat']),
                    'lng': float(row['lng'])
                }
                for _, row in courier_traj.iterrows()
            ]
        
        with open(os.path.join(output_dir, 'trajectories.json'), 'w') as f:
            json.dump(trajectories, f, indent=2)
        
        print(f"Demo data exported to {output_dir}")
        print(f"- Pickup points: {len(pickup_points)}")
        print(f"- Delivery points: {len(delivery_points)}")
        print(f"- Courier trajectories: {len(trajectories)}")


if __name__ == '__main__':
    # Usage example
    loader = LaDeDataLoader('data/lade')
    
    # Try to download from Hugging Face (optional)
    # loader.download_from_huggingface()
    
    # Generate simulated data for demo
    pickup_df = loader.load_pickup_data()
    delivery_df = loader.load_delivery_data()
    trajectory_df = loader.load_trajectory_data()
    
    print(f"Loaded {len(pickup_df)} pickup records")
    print(f"Loaded {len(delivery_df)} delivery records")
    print(f"Loaded {len(trajectory_df)} trajectory points")
    
    # Export for demo
    loader.export_for_demo()
    
    # Prepare for model
    model_data = loader.prepare_for_model()
    print(f"Prepared data for model: {model_data['n_nodes']} nodes")

