# SeoUHI - Seoul Urban Heat Island Management System

## Overview
This is the offical code of web system part of our KDD25 accepted paper "Fine-grained Urban Heat Island Effect Forecasting: A Context-aware Thermodynamic Modeling Framework". SeoUHI is a web-based visualization system for monitoring and managing Urban Heat Island (UHI) effects in Seoul. The system provides real-time temperature heatmap visualization and predictive analytics for urban heat patterns.

## Features
- Interactive heatmap visualization of Seoul's temperature distribution
- Real-time temperature monitoring and updates
- Location-based temperature querying
- Responsive web interface with modern design
- Time-based temperature prediction display

## Technical Stack
- **Frontend**:
  - Mapbox GL JS (v2.14.1) for map visualization
  - D3.js (v7) for data visualization
  - Chart.js for graphical representations
  - PapaParse for data parsing
  - Font Awesome for UI icons

## Project Structure
```
.
├── index.html          # Main HTML file
├── js/                 # JavaScript source files
│   ├── main.js        # Core application logic
│   ├── heatmap.js     # Heatmap visualization
│   ├── data.js        # Data management
│   ├── tool.js        # Utility functions
│   ├── style.js       # UI styling
│   └── interpolation.js# Temperature interpolation
├── css/               # Stylesheet files
└── nginx.conf         # Nginx server configuration
```

## Installation & Setup

### Prerequisites
- Linux server with sudo privileges
- Nginx web server
- Modern web browser with JavaScript enabled
- Mapbox API access token
- SSH access to the server

### Detailed Deployment Steps

#### 1. Server Preparation
```bash
# Update system packages
sudo apt update

# Install Nginx
sudo apt install nginx

# Create project directory
sudo mkdir -p /www/SeoUHI
```

#### 2. Project Deployment
From your local development machine:
```bash
# Upload project files to server (replace username with your server username)
scp -r * username@YOUR_SERVER_IP:/www/SeoUHI/
```

#### 3. Server Configuration
On the server:
```bash
# Set proper permissions
sudo chown -R www-data:www-data /www/SeoUHI
sudo chmod -R 755 /www/SeoUHI

# Configure Nginx
sudo cp /www/SeoUHI/nginx.conf /etc/nginx/sites-available/seouhi
sudo ln -s /etc/nginx/sites-available/seouhi /etc/nginx/sites-enabled/

# Test and restart Nginx
sudo nginx -t
sudo systemctl restart nginx
```

#### 4. Verification Steps
```bash
# Check Nginx status
sudo systemctl status nginx

# Verify port listening status
sudo netstat -tulpn | grep 1027

# Check firewall status
sudo ufw status

# Test connectivity
telnet YOUR_SERVER_IP 1027
# or
curl http://YOUR_SERVER_IP:1027
```

### Nginx Configuration
The system uses the following Nginx configuration (nginx.conf):
```nginx
server {
    listen 1027;
    server_name your_server_ip;
    root /www/SeoUHI;
    index index.html;
    
    # CORS and security headers
    add_header Access-Control-Allow-Origin *;
    add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS';
    add_header Access-Control-Allow-Headers 'DNT,X-Mx-ReqToken,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Authorization';
}
```

### Accessing the Application
After successful deployment:
1. Configure your Mapbox API token in `main.js`
2. Access the application through `http://your_server_ip:1027`
3. Verify that the heatmap and all interactive features are working correctly

## Usage
1. The main interface displays a map of Seoul with temperature heatmap overlay
2. Use the top navigation bar to:
   - Toggle heatmap visualization
   - Access additional menu options
   - View current time and temperature predictions
3. Interactive features include:
   - Click on map locations for detailed temperature data
   - Time-based temperature visualization
   - Search functionality for specific locations

## Development
The application is built with a modular structure:
- `main.js`: Core application initialization and setup
- `heatmap.js`: Handles temperature visualization layers
- `data.js`: Manages data processing and storage
- `style.js`: Contains UI styling configurations
- `interpolation.js`: Implements temperature interpolation algorithms

## Future Work
- Implementation of machine learning predictions
- Enhanced user interaction features
- Mobile optimization
- Additional data visualization options

## Related Project
This web demo is part of the larger DeepUHI project. For the core model and research implementation, please visit:
[DeepUHI GitHub Repository](https://github.com/RWLinno/DeepUHI)

## Citation
If you use this system in your research, please cite:
