# CTIS: Connected Transportation Information System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-LaDe-success.svg)](https://huggingface.co/datasets/Cainiao-AI/LaDe)

> **Connected Transportation Information System for Last-Mile Delivery Optimization**
> 
> Integrating Spatio-Temporal Forecasting, Retrieval-Augmented Generation (RAG), Reinforcement Learning (RL), and Interactive LLM Agents

---

## ⚡ Quick Start (3 Steps)

```bash
# 1. Generate demo data
python generate_demo_data.py

# 2. Start complete system (API + Web)
./start_all.sh

# 3. Open browser
# http://localhost:8000/test.html  (Test page)
# http://localhost:8000/index.html (Full demo)
```

---

## 🌟 Key Features

### 🎨 Multi-Role Visualization
- **6 Role Types** with unique colors:
  - 🏢 Hub (Red) - Distribution centers
  - 🏭 Warehouse (Blue) - Storage facilities  
  - 📍 Station (Green) - Delivery stations
  - 🏠 Residential (Orange) - Residential areas
  - 🏪 Commercial (Purple) - Commercial zones
  - 🏢 Office (Cyan) - Office buildings

### 🗺️ Interactive Map
- **18px markers** (enlarged for better visibility)
- Hover to scale 1.3x
- Click for detailed popup
- Select 2 points for route planning

### 🛣️ Advanced Routing
- **Curved path** generation (Bezier curves)
- **Animated routes** (flowing dash animation)
- Distance and time estimation
- API-based routing with fallback

### 📊 Hexagon Heatmap
- Demand-based color coding
- Opacity represents intensity
- Toggle on/off with button
- 200+ hexagonal cells

### 📈 Fixed-Size Charts
- Hourly demand: **280px fixed height**
- Y-axis: 0-100 range
- Color-coded: Red (peak), Orange (medium), Green (low)
- Regional performance pie chart

### 💬 AI Assistant
- Role-based responses (Driver/Dispatcher/Customer/Analyst)
- Context-aware Q&A
- Quick question buttons
- Real-time interaction

### 🎛️ System Menu
- **Export Data**: Download JSON
- **Settings**: View configuration
- **Help**: Usage guide
- **About**: Project information

---

## 📊 LaDe Dataset

We use the **LaDe Dataset** from [Hugging Face](https://huggingface.co/datasets/Cainiao-AI/LaDe):

```bibtex
@misc{wu2023lade,
  title={LaDe: The First Comprehensive Last-mile Delivery Dataset from Industry}, 
  author={Wu, Lixia and Wen, Haomin and Hu, Haoyuan and others},
  year={2023},
  eprint={2306.10675},
  archivePrefix={arXiv}
}
```

---

## 🏗️ Architecture

```
CTIS Framework
├── Data Layer (LaDe Dataset)
├── Feature Extraction
├── Core Models
│   ├── ST-Forecasting (WaveTS + Graph Conv)
│   ├── RAG Module (RAST)
│   ├── RL Optimizer (DQN)
│   └── LLM Agent
└── Web Demo
    ├── Interactive Map (Leaflet)
    ├── Hexagon Heatmap
    ├── Route Planner
    └── AI Chat Interface
```

---

## 🔌 API Endpoints

```
# CORS-enabled endpoints
GET  /api/health              - Health check
GET  /api/system_status       - System metrics
GET  /api/data/pickup         - Pickup points
GET  /api/data/delivery       - Delivery points
POST /api/route               - Route calculation
POST /api/predict             - Demand forecasting
POST /api/optimize_route      - RL optimization
POST /api/chat                - LLM interaction
```

---

## 💻 Usage Guide

### Start Complete System

```bash
./start_all.sh
```

This starts:
- ✅ Flask API server (port 5000)
- ✅ Web demo server (port 8000)
- ✅ Auto data generation

### Test New Features

1. **Route Planning**
   - Click marker → "Route" button
   - Select 2 points
   - See curved path with animation

2. **Hexagon Heatmap**
   - Click "Toggle Heatmap" button
   - See hexagonal cells
   - Color = demand intensity

3. **Time Control**
   - Use slider in top-left
   - Filter data by hour

4. **System Menu**
   - Click "Menu" button
   - Try all 4 options

---

## 👥 Team Contributions

- **Weilin Ruan**: RAST (RAG) - [arXiv:2508.16623](https://arxiv.org/pdf/2508.16623)
- **Ziyu Zhou**: WaveTS (TS) - [arXiv:2505.11781](https://arxiv.org/pdf/2505.11781)
- **Songxin**: RL Routing - Spatio-temporal optimization
- **Yiming Huang**: LLM Security - [arXiv:2505.15386](https://arxiv.org/pdf/2505.15386)

---

## 📈 Results

### Forecasting Performance
| Model | MAE ↓ | RMSE ↓ |
|-------|-------|--------|
| LSTM  | 3.42  | 4.85   |
| STGCN | 2.93  | 4.31   |
| RAST  | 2.68  | 3.98   |
| **CTIS** | **2.45** | **3.72** |

### Route Optimization
| Method | Time ↓ | Success ↑ |
|--------|--------|-----------|
| Greedy | 45.3 min | 82% |
| **DQN** | **32.1 min** | **94%** |

---

## 🔧 Troubleshooting

### CORS Error
```bash
# Make sure API is running
python src/api.py
```

### Controls Not Visible
Fixed! z-index set to 1000.
Refresh browser to see changes.

### Route Button Not Working
Start API server first:
```bash
python src/api.py
```

### Heatmap Not Showing
Click "Toggle Heatmap" button.
Hexagons will appear over markers.

---

## 📝 Citation

```bibtex
@inproceedings{ctis2025,
  title={CTIS: Connected Transportation Information System},
  author={Zou, Xingchen and Ruan, Weilin and Zhong, Siru and Hu, Yuehong},
  year={2025}
}
```

---

## 🔗 References

1. [LaDe Dataset](https://huggingface.co/datasets/Cainiao-AI/LaDe)
2. [RAST Framework](https://arxiv.org/pdf/2508.16623)
3. [WaveTS](https://arxiv.org/pdf/2505.11781)
4. [SeoUHI Demo](https://github.com/RWLinno/DeepUHI)

---

## 📄 License

MIT License

---

## 🤝 Contact

- **Weilin Ruan**: wruan792@connect.hkust-gz.edu.cn
- **GitHub**: [Project Repository]

---

**🚀 Built with ❤️ by the CTIS Team**

*Version 1.0.0 - Last Updated: 2025-10-29*
