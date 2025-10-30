// CTIS Enhanced Demo Application
// Integrates LaDe Dataset with Advanced Visualizations

const API_BASE_URL = 'http://localhost:5000/api';
// Get Mapbox token from: https://account.mapbox.com/access-tokens/
const MAPBOX_TOKEN = 'pk.eyJ1IjoiY3Rpc2RlbW8iLCJhIjoiY2x6YWJjZGVmMGFiMDJqczhqYnR6ZnlrZiJ9.demo'; // Replace with your token

// Global State
const state = {
    map: null,
    markers: {},
    routes: [],
    selectedPoint: null,
    selectedForRoute: [], // 用于路径规划的选中点
    routeLayer: null,     // 当前显示的路径
    currentTime: 12,
    activeLayer: 'all',
    charts: {},
    pickupData: [],
    deliveryData: [],
    trajectoryData: {},
    isPanelVisible: true
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 CTIS Demo initializing...');
    
    setTimeout(() => {
        const loadingEl = document.getElementById('loadingOverlay');
        if (loadingEl) {
            loadingEl.classList.remove('active');
            console.log('✅ Loading overlay removed');
        }
    }, 500);
    
    try {
        console.log('Step 1: Initializing map...');
        initializeMap();
        
        console.log('Step 2: Loading data...');
        await loadData();
        console.log('✅ Data loaded');
        
        console.log('Step 3: Setting up event listeners...');
        setupEventListeners();
        
        console.log('Step 4: Updating time...');
        updateTime();
        
        console.log('Step 5: Initializing charts...');
        initializeCharts();
        
        console.log('✅ CTIS Demo initialized successfully');
    } catch (error) {
        console.error('❌ Error initializing CTIS:', error);
        // 即使出错也要移除loading
        const loadingEl = document.getElementById('loadingOverlay');
        if (loadingEl) {
            loadingEl.classList.remove('active');
        }
    }
});

// Map Initialization
function initializeMap() {
    // Check if Mapbox token is valid
    if (!MAPBOX_TOKEN || MAPBOX_TOKEN.includes('YOUR_') || MAPBOX_TOKEN.includes('demo')) {
        console.warn('⚠️ Mapbox token not configured. Using OpenStreetMap fallback.');
        // Fallback: Use Leaflet with OpenStreetMap (no token required)
        initializeLeafletMap();
        return;
    }
    
    mapboxgl.accessToken = MAPBOX_TOKEN;
    
    // Default to Shanghai coordinates
    state.map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/mapbox/light-v11',
        center: [121.4737, 31.2304],
        zoom: 11,
        pitch: 0,
        bearing: 0
    });
    
    state.map.on('load', () => {
        console.log('Map loaded successfully');
        addMapControls();
        renderMapData();
    });
}

// Fallback map using Leaflet (no token required)
function initializeLeafletMap() {
    console.log('Initializing Leaflet map (OpenStreetMap)');
    
    // Load Leaflet if not already loaded
    if (typeof L === 'undefined') {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
        document.head.appendChild(link);
        
        const script = document.createElement('script');
        script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
        script.onload = () => setupLeafletMap();
        document.head.appendChild(script);
    } else {
        setupLeafletMap();
    }
}

function setupLeafletMap() {
    const map = L.map('map').setView([31.2304, 121.4737], 11);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(map);
    
    // Store in state with compatibility layer
    state.map = {
        _leaflet: map,
        isLeaflet: true,
        flyTo: (opts) => map.flyTo([opts.center[1], opts.center[0]], opts.zoom || 14),
        getLayer: () => null,
        getSource: () => null,
        addSource: () => {},
        addLayer: () => {},
        removeLayer: () => {},
        removeSource: () => {},
        setLayoutProperty: () => {},
        getPitch: () => 0,
        easeTo: () => {},
        addControl: () => {}
    };
    
    // 触发数据渲染
    setTimeout(() => {
        if (state.pickupData.length > 0 || state.deliveryData.length > 0) {
            renderMapData();
        }
    }, 500);
}

// Add Map Controls
function addMapControls() {
    state.map.addControl(new mapboxgl.NavigationControl(), 'top-left');
    state.map.addControl(new mapboxgl.FullscreenControl(), 'top-left');
    
    // Add geocoder
    const geocoder = new MapboxGeocoder({
        accessToken: mapboxgl.accessToken,
        mapboxgl: mapboxgl,
        marker: false
    });
    state.map.addControl(geocoder, 'top-left');
}

// Load Data from Files or API
async function loadData() {
    console.log('Loading data...');
    
    try {
        // 直接从本地文件加载,避免API延迟
        await loadLocalData();
        
        // Update stats
        updateSystemStats();
        console.log('✅ Data loaded successfully');
        
    } catch (error) {
        console.error('Error loading data:', error);
        console.log('Generating demo data as fallback');
        generateDemoData();
        updateSystemStats();
    }
}

// Load from local JSON files
async function loadLocalData() {
    try {
        console.log('Fetching pickup data...');
        const pickupResponse = await fetch('data/pickup_points.json');
        const pickup = await pickupResponse.json();
        state.pickupData = pickup;
        console.log(`✅ Loaded ${pickup.length} pickup points`);
        
        console.log('Fetching delivery data...');
        const deliveryResponse = await fetch('data/delivery_points.json');
        const delivery = await deliveryResponse.json();
        state.deliveryData = delivery;
        console.log(`✅ Loaded ${delivery.length} delivery points`);
        
        console.log('Fetching trajectories...');
        const trajResponse = await fetch('data/trajectories.json');
        const trajectories = await trajResponse.json();
        state.trajectoryData = trajectories;
        console.log(`✅ Loaded trajectories`);
        
    } catch (error) {
        console.log('Local data files not found, generating demo data');
        throw error; // 让上层处理
    }
}

// Generate Demo Data
function generateDemoData() {
    const base = [121.4737, 31.2304];
    
    // Generate pickup points
    state.pickupData = Array.from({ length: 50 }, (_, i) => ({
        id: `PKG_${i.toString().padStart(6, '0')}`,
        type: 'pickup',
        lat: base[1] + (Math.random() - 0.5) * 0.1,
        lng: base[0] + (Math.random() - 0.5) * 0.1,
        aoi_type: ['Residential', 'Commercial', 'Office'][Math.floor(Math.random() * 3)],
        accept_time: new Date(Date.now() - Math.random() * 86400000).toISOString(),
        courier_id: `C${Math.floor(Math.random() * 20).toString().padStart(3, '0')}`,
        demand: Math.floor(Math.random() * 20) + 5
    }));
    
    // Generate delivery points
    state.deliveryData = Array.from({ length: 50 }, (_, i) => ({
        id: `PKG_${i.toString().padStart(6, '0')}`,
        type: 'delivery',
        lat: base[1] + (Math.random() - 0.5) * 0.1,
        lng: base[0] + (Math.random() - 0.5) * 0.1,
        aoi_type: ['Residential', 'Commercial', 'Office'][Math.floor(Math.random() * 3)],
        delivery_time: new Date(Date.now() + Math.random() * 86400000).toISOString(),
        courier_id: `C${Math.floor(Math.random() * 20).toString().padStart(3, '0')}`,
        demand: Math.floor(Math.random() * 15) + 3
    }));
}

// Render Map Data (优化性能 - 分批渲染)
function renderMapData() {
    clearMarkers();
    
    const layerFilter = state.activeLayer;
    
    // 分批渲染以避免卡顿
    const batchSize = 20;
    
    // Render pickup points
    if (layerFilter === 'all' || layerFilter === 'pickup') {
        renderMarkersInBatches(state.pickupData, 'pickup', batchSize);
    }
    
    // Render delivery points
    if (layerFilter === 'all' || layerFilter === 'delivery') {
        renderMarkersInBatches(state.deliveryData, 'delivery', batchSize);
    }
    
    // Render routes if active
    if (layerFilter === 'routes') {
        renderRoutes();
    }
    
    // Add heatmap layer
    requestAnimationFrame(() => {
        addHeatmapLayer();
    });
}

// 分批渲染标记以优化性能
function renderMarkersInBatches(points, type, batchSize) {
    let index = 0;
    
    function renderBatch() {
        const end = Math.min(index + batchSize, points.length);
        
        for (let i = index; i < end; i++) {
            addMarker(points[i], type);
        }
        
        index = end;
        
        // 继续下一批
        if (index < points.length) {
            requestAnimationFrame(renderBatch);
        } else {
            console.log(`✅ Rendered ${points.length} ${type} markers`);
        }
    }
    
    renderBatch();
}

// Add Marker to Map (支持Leaflet和Mapbox,增加角色颜色)
function addMarker(point, type) {
    // 使用角色颜色,如果没有则使用默认颜色
    const markerColor = point.role_color || (type === 'pickup' ? '#4299e1' : '#48bb78');
    const markerSize = 18; // 增大标记点
    
    if (state.map.isLeaflet) {
        // Leaflet marker - 增大尺寸
        const icon = L.divIcon({
            className: 'custom-marker',
            html: `<div style="width: ${markerSize}px; height: ${markerSize}px; border-radius: 50%; 
                   background: ${markerColor}; 
                   border: 3px solid white; box-shadow: 0 3px 6px rgba(0,0,0,0.4);
                   transition: transform 0.2s;"></div>`,
            iconSize: [markerSize + 6, markerSize + 6]
        });
        
        const marker = L.marker([point.lat, point.lng], { icon: icon })
            .addTo(state.map._leaflet);
        
        const roleIcon = point.role === 'Hub' ? '🏢' : 
                        point.role === 'Warehouse' ? '🏭' :
                        point.role === 'Station' ? '📍' :
                        point.role === 'Residential' ? '🏠' :
                        point.role === 'Commercial' ? '🏪' : '🏢';
        
        const popupContent = `
            <div style="padding: 12px; min-width: 220px;">
                <h4 style="margin: 0 0 10px 0; color: #1a202c;">
                    ${roleIcon} ${point.id}
                </h4>
                <div style="font-size: 0.9em; line-height: 1.8;">
                    <div><strong>Role:</strong> <span style="color: ${markerColor};">●</span> ${point.role || point.aoi_type}</div>
                    <div><strong>Type:</strong> ${type === 'pickup' ? '📦 Pickup' : '🏠 Delivery'}</div>
                    <div><strong>Courier:</strong> ${point.courier_id}</div>
                    <div><strong>Demand:</strong> ${point.demand || 'N/A'}</div>
                </div>
                <div style="margin-top: 10px; display: flex; gap: 5px;">
                    <button onclick="showPointDetails('${point.id}', '${type}')" 
                            style="flex: 1; padding: 6px 10px; background: #667eea; 
                                   color: white; border: none; border-radius: 5px; cursor: pointer;">
                        Details
                    </button>
                    <button onclick="selectPointForRoute('${point.id}', '${type}')" 
                            style="flex: 1; padding: 6px 10px; background: #48bb78; 
                                   color: white; border: none; border-radius: 5px; cursor: pointer;">
                        Route
                    </button>
                </div>
            </div>
        `;
        
        marker.bindPopup(popupContent);
        marker.on('click', () => showPointDetails(point.id, type));
        
        state.markers[point.id] = { marker, type, isLeaflet: true };
    } else {
        // Mapbox marker - 增大尺寸
        const el = document.createElement('div');
        el.className = `custom-marker ${type}-marker`;
        el.style.width = markerSize + 'px';
        el.style.height = markerSize + 'px';
        el.style.borderRadius = '50%';
        el.style.cursor = 'pointer';
        el.style.border = '3px solid white';
        el.style.boxShadow = '0 3px 6px rgba(0,0,0,0.4)';
        el.style.backgroundColor = markerColor;
        el.style.transition = 'transform 0.2s';
        el.onmouseenter = () => el.style.transform = 'scale(1.3)';
        el.onmouseleave = () => el.style.transform = 'scale(1)';
        
        const marker = new mapboxgl.Marker(el)
            .setLngLat([point.lng, point.lat])
            .addTo(state.map);
        
        const popup = new mapboxgl.Popup({ offset: 25 })
            .setHTML(`
                <div style="padding: 10px;">
                    <h4 style="margin: 0 0 10px 0; color: #1a202c;">
                        ${type === 'pickup' ? '📦' : '🏠'} ${point.id}
                    </h4>
                    <div style="font-size: 0.9em;">
                        <strong>Type:</strong> ${point.aoi_type}<br>
                        <strong>Courier:</strong> ${point.courier_id}<br>
                        <strong>Demand:</strong> ${point.demand || 'N/A'}
                    </div>
                    <button onclick="showPointDetails('${point.id}', '${type}')" 
                            style="margin-top: 10px; padding: 5px 10px; background: #667eea; 
                                   color: white; border: none; border-radius: 5px; cursor: pointer;">
                        View Details
                    </button>
                </div>
            `);
        
        marker.setPopup(popup);
        el.addEventListener('click', () => showPointDetails(point.id, type));
        
        state.markers[point.id] = { marker, popup, type };
    }
}

// Clear All Markers
function clearMarkers() {
    Object.values(state.markers).forEach(({ marker, isLeaflet }) => {
        if (isLeaflet && state.map._leaflet) {
            state.map._leaflet.removeLayer(marker);
        } else if (marker.remove) {
            marker.remove();
        }
    });
    state.markers = {};
}

// Add Heatmap Layer
function addHeatmapLayer() {
    // Skip heatmap for Leaflet (可选功能)
    if (state.map.isLeaflet) {
        console.log('Heatmap not available in Leaflet mode');
        return;
    }
    
    // Remove existing heatmap layer
    if (state.map.getLayer('heatmap-layer')) {
        state.map.removeLayer('heatmap-layer');
    }
    if (state.map.getSource('heatmap-source')) {
        state.map.removeSource('heatmap-source');
    }
    
    // Create GeoJSON from data
    const features = [...state.pickupData, ...state.deliveryData].map(point => ({
        type: 'Feature',
        properties: {
            demand: point.demand || 10
        },
        geometry: {
            type: 'Point',
            coordinates: [point.lng, point.lat]
        }
    }));
    
    state.map.addSource('heatmap-source', {
        type: 'geojson',
        data: {
            type: 'FeatureCollection',
            features: features
        }
    });
    
    state.map.addLayer({
        id: 'heatmap-layer',
        type: 'heatmap',
        source: 'heatmap-source',
        paint: {
            'heatmap-weight': ['get', 'demand'],
            'heatmap-intensity': 0.5,
            'heatmap-color': [
                'interpolate',
                ['linear'],
                ['heatmap-density'],
                0, 'rgba(72, 187, 120, 0)',
                0.2, 'rgba(72, 187, 120, 0.5)',
                0.4, 'rgba(237, 137, 54, 0.6)',
                0.6, 'rgba(245, 101, 101, 0.7)',
                1, 'rgba(180, 65, 65, 0.8)'
            ],
            'heatmap-radius': 30,
            'heatmap-opacity': 0.6
        }
    }, 'waterway-label');
}

// Render Routes
function renderRoutes() {
    // Generate sample routes
    const numRoutes = 5;
    const colors = ['#e53e3e', '#dd6b20', '#d69e2e', '#38a169', '#3182ce'];
    
    for (let i = 0; i < numRoutes; i++) {
        const route = generateRoute(i);
        addRouteToMap(route, colors[i], i);
    }
}

// Generate Route
function generateRoute(routeId) {
    const numPoints = Math.floor(Math.random() * 5) + 5;
    const allPoints = [...state.pickupData, ...state.deliveryData];
    const selectedPoints = [];
    
    for (let i = 0; i < numPoints; i++) {
        const point = allPoints[Math.floor(Math.random() * allPoints.length)];
        selectedPoints.push([point.lng, point.lat]);
    }
    
    return selectedPoints;
}

// Add Route to Map
function addRouteToMap(coordinates, color, id) {
    const sourceId = `route-${id}`;
    const layerId = `route-layer-${id}`;
    
    if (state.map.getLayer(layerId)) {
        state.map.removeLayer(layerId);
    }
    if (state.map.getSource(sourceId)) {
        state.map.removeSource(sourceId);
    }
    
    state.map.addSource(sourceId, {
        type: 'geojson',
        data: {
            type: 'Feature',
            properties: {},
            geometry: {
                type: 'LineString',
                coordinates: coordinates
            }
        }
    });
    
    state.map.addLayer({
        id: layerId,
        type: 'line',
        source: sourceId,
        layout: {
            'line-join': 'round',
            'line-cap': 'round'
        },
        paint: {
            'line-color': color,
            'line-width': 3,
            'line-opacity': 0.8
        }
    });
}

// Show Point Details
window.showPointDetails = function(pointId, type) {
    const data = type === 'pickup' ? state.pickupData : state.deliveryData;
    const point = data.find(p => p.id === pointId);
    
    if (!point) return;
    
    state.selectedPoint = { ...point, type };
    
    // Switch to details tab
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.querySelector('[data-tab="details"]').classList.add('active');
    document.getElementById('details-tab').classList.add('active');
    
    // Update details panel
    const detailsHtml = `
        <div class="detail-row">
            <span class="detail-label">ID:</span>
            <span class="detail-value">${point.id}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Type:</span>
            <span class="detail-value">${type === 'pickup' ? '📦 Pickup' : '🏠 Delivery'}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Location:</span>
            <span class="detail-value">${point.lat.toFixed(4)}, ${point.lng.toFixed(4)}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">AOI Type:</span>
            <span class="detail-value">${point.aoi_type}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Courier:</span>
            <span class="detail-value">${point.courier_id}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Demand:</span>
            <span class="detail-value">${point.demand || 'N/A'}</span>
        </div>
    `;
    
    document.getElementById('pointDetails').innerHTML = detailsHtml;
    
    // Show and update time series chart
    document.getElementById('timeSeriesContainer').style.display = 'block';
    document.getElementById('spatialContext').style.display = 'block';
    updateTimeSeriesChart(point);
    
    // Fly to point
    state.map.flyTo({
        center: [point.lng, point.lat],
        zoom: 14,
        duration: 1000
    });
};

// Update Time Series Chart
function updateTimeSeriesChart(point) {
    const ctx = document.getElementById('timeSeriesChart');
    if (!ctx) return;
    
    // Generate time series data
    const historical = Array.from({ length: 24 }, () => Math.random() * 15 + 5);
    const predicted = Array.from({ length: 24 }, () => Math.random() * 15 + 5);
    const labels = Array.from({ length: 48 }, (_, i) => `${i}:00`);
    
    if (state.charts.timeSeries) {
        state.charts.timeSeries.destroy();
    }
    
    state.charts.timeSeries = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Historical Demand',
                    data: [...historical, ...Array(24).fill(null)],
                    borderColor: '#4299e1',
                    backgroundColor: 'rgba(66, 153, 225, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Predicted Demand',
                    data: [...Array(24).fill(null), ...predicted],
                    borderColor: '#48bb78',
                    backgroundColor: 'rgba(72, 187, 120, 0.1)',
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Demand'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time (hours)'
                    }
                }
            }
        }
    });
}

// Initialize Charts
function initializeCharts() {
    console.log('Initializing charts...');
    
    // Demand Distribution Chart - 归一化数据
    const demandCtx = document.getElementById('demandDistChart');
    if (!demandCtx) {
        console.warn('demandDistChart canvas not found');
        return;
    }
    
    try {
        // 生成符合实际分布的需求数据 (峰值时段)
        const hourlyData = Array.from({ length: 24 }, (_, hour) => {
            let base = 30;
            if (hour >= 9 && hour <= 11) base = 80;      // 早高峰
            if (hour >= 12 && hour <= 14) base = 60;     // 午间
            if (hour >= 18 && hour <= 20) base = 90;     // 晚高峰
            if (hour >= 0 && hour <= 6) base = 10;       // 凌晨
            return base + Math.random() * 15 - 7.5;
        });
        
        state.charts.demandDist = new Chart(demandCtx, {
            type: 'bar',
            data: {
                labels: Array.from({ length: 24 }, (_, i) => `${i}:00`),
                datasets: [{
                    label: 'Hourly Demand',
                    data: hourlyData,
                    backgroundColor: hourlyData.map(v => 
                        v > 80 ? 'rgba(245, 101, 101, 0.7)' :   // 高峰 - 红色
                        v > 60 ? 'rgba(237, 137, 54, 0.7)' :    // 中等 - 橙色
                        'rgba(72, 187, 120, 0.7)'                // 低峰 - 绿色
                    ),
                    borderColor: '#667eea',
                    borderWidth: 1,
                    barThickness: 'flex',
                    maxBarThickness: 20
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,  // 关键: 不保持宽高比
                aspectRatio: 2,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Demand: ${context.parsed.y.toFixed(0)} orders`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,  // 固定Y轴最大值
                        ticks: {
                            stepSize: 20
                        },
                        title: {
                            display: true,
                            text: 'Demand (orders)',
                            font: {
                                size: 11
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Hour of Day',
                            font: {
                                size: 11
                            }
                        },
                        ticks: {
                            font: {
                                size: 9
                            }
                        }
                    }
                },
                layout: {
                    padding: 10
                }
            }
        });
        console.log('✅ Demand chart created');
    } catch (error) {
        console.error('Error creating demand chart:', error);
    }
    
    // Regional Chart
    const regionalCtx = document.getElementById('regionalChart');
    if (!regionalCtx) {
        console.warn('regionalChart canvas not found');
        return;
    }
    
    try {
        state.charts.regional = new Chart(regionalCtx, {
            type: 'doughnut',
            data: {
                labels: ['Hub', 'Warehouse', 'Station', 'Residential', 'Commercial', 'Office'],
                datasets: [{
                    data: [15, 20, 25, 18, 12, 10],
                    backgroundColor: [
                        '#e74c3c',
                        '#3498db',
                        '#2ecc71',
                        '#f39c12',
                        '#9b59b6',
                        '#1abc9c'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 8,
                            font: {
                                size: 10
                            }
                        }
                    }
                },
                cutout: '50%'
            }
        });
        console.log('✅ Regional chart created');
    } catch (error) {
        console.error('Error creating regional chart:', error);
    }
}

// Update System Stats
function updateSystemStats() {
    document.getElementById('totalPackages').textContent = 
        (state.pickupData.length + state.deliveryData.length).toString();
    document.getElementById('activeVehicles').textContent = '15';
    document.getElementById('avgDeliveryTime').textContent = '28.5';
    document.getElementById('successRate').textContent = '94%';
}

// Update Time Display
function updateTime() {
    const now = new Date();
    document.getElementById('current-time').textContent = 
        now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    
    setInterval(() => {
        const now = new Date();
        document.getElementById('current-time').textContent = 
            now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    }, 1000);
}

// Setup Event Listeners
function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    try {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tabName = btn.dataset.tab;
                switchTab(tabName);
            });
        });
    
    // Layer buttons
    document.querySelectorAll('.layer-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.activeLayer = btn.dataset.layer;
            renderMapData();
        });
    });
    
    // Time slider
    document.getElementById('timeSlider').addEventListener('input', (e) => {
        state.currentTime = parseInt(e.target.value);
        document.getElementById('timeDisplay').textContent = `${state.currentTime}:00`;
        updateVisualizationForTime(state.currentTime);
    });
    
    // Toggle heatmap - 使用六边形热力图
    document.getElementById('toggleHeatmapBtn').addEventListener('click', () => {
        toggleHexagonHeatmap();
    });
    
    // Toggle 3D - 对Leaflet使用不同实现
    document.getElementById('toggle3DBtn').addEventListener('click', () => {
        if (state.map.isLeaflet) {
            // Leaflet不支持3D,改为缩放效果
            const currentZoom = state.map._leaflet.getZoom();
            state.map._leaflet.setZoom(currentZoom === 11 ? 13 : 11, {
                animate: true,
                duration: 1
            });
            alert('ℹ️ 3D view is not available in Leaflet mode. Zooming instead.');
        } else {
            const currentPitch = state.map.getPitch();
            state.map.easeTo({
                pitch: currentPitch === 0 ? 60 : 0,
                bearing: currentPitch === 0 ? 20 : 0,
                duration: 1000
            });
        }
    });
    
    // Menu button
    document.getElementById('menuBtn').addEventListener('click', () => {
        document.getElementById('menuModal').style.display = 'block';
    });
    
    // Close modal
    document.querySelector('.modal-close').addEventListener('click', () => {
        document.getElementById('menuModal').style.display = 'none';
    });
    
    // Toggle panel
    document.getElementById('togglePanelBtn').addEventListener('click', () => {
        const panel = document.getElementById('sidePanel');
        panel.classList.toggle('collapsed');
        state.isPanelVisible = !state.isPanelVisible;
    });
    
    // Chat
    document.getElementById('sendBtn').addEventListener('click', sendMessage);
    document.getElementById('chatInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    // Quick questions
    document.querySelectorAll('.quick-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('chatInput').value = btn.textContent;
            sendMessage();
        });
    });
    
    // Generate forecast
    const forecastBtn = document.getElementById('generateForecast');
    if (forecastBtn) {
        forecastBtn.addEventListener('click', generateForecast);
    }
    
    // Initialize menu handlers
    initMenuHandlers();
    
    } catch (error) {
        console.error('Error setting up event listeners:', error);
    }
}

// Switch Tab
function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`${tabName}-tab`).classList.add('active');
}

// Send Message
function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    addChatMessage('user', message);
    input.value = '';
    
    // Simulate AI response
    setTimeout(() => {
        const response = generateAIResponse(message);
        addChatMessage('assistant', response);
    }, 1000);
}

// Add Chat Message
function addChatMessage(sender, content) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const role = sender === 'user' ? 'You' : document.getElementById('roleSelect').selectedOptions[0].text;
    messageDiv.innerHTML = `
        <div class="message-role">${role}</div>
        <div>${content}</div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Generate AI Response
function generateAIResponse(query) {
    const responses = {
        peak: "Peak delivery hours are typically between 12:00 PM and 2:00 PM, with highest demand in downtown areas.",
        busy: "The busiest area is currently Region A (Downtown Shanghai) with 30% of total deliveries.",
        success: "Current success rate is 94%, which is above our 90% target. Great job!",
        default: "Based on our CTIS analysis combining spatio-temporal forecasting and RL optimization, I can help you with route planning, demand predictions, and system insights."
    };
    
    const lower = query.toLowerCase();
    if (lower.includes('peak') || lower.includes('hour')) return responses.peak;
    if (lower.includes('busy') || lower.includes('area')) return responses.busy;
    if (lower.includes('success') || lower.includes('rate')) return responses.success;
    
    return responses.default;
}

// Generate Forecast
function generateForecast() {
    const horizon = parseInt(document.getElementById('horizonSelect').value);
    const ctx = document.getElementById('forecastChart');
    
    if (state.charts.forecast) {
        state.charts.forecast.destroy();
    }
    
    const labels = Array.from({ length: horizon }, (_, i) => `T+${i+1}`);
    const forecast = Array.from({ length: horizon }, () => Math.random() * 20 + 10);
    const lower = forecast.map(v => v * 0.9);
    const upper = forecast.map(v => v * 1.1);
    
    state.charts.forecast = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Forecast',
                    data: forecast,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Upper Bound',
                    data: upper,
                    borderColor: '#48bb78',
                    borderDash: [5, 5],
                    fill: false
                },
                {
                    label: 'Lower Bound',
                    data: lower,
                    borderColor: '#f56565',
                    borderDash: [5, 5],
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Update Visualization for Time
function updateVisualizationForTime(hour) {
    console.log(`Updating visualization for hour: ${hour}`);
    // Filter and update data based on time
    // This would filter actual time-based data in production
}

// 路径规划功能
window.selectPointForRoute = function(pointId, type) {
    const data = type === 'pickup' ? state.pickupData : state.deliveryData;
    const point = data.find(p => p.id === pointId);
    
    if (!point) return;
    
    // 添加到选中列表
    if (state.selectedForRoute.length >= 2) {
        alert('最多选择2个点。清除当前选择...');
        clearRouteSelection();
    }
    
    state.selectedForRoute.push({ ...point, type });
    
    // 高亮选中的标记
    highlightMarker(pointId, true);
    
    console.log(`Selected point ${pointId} for routing. Total: ${state.selectedForRoute.length}/2`);
    
    // 如果已选择2个点,计算并显示路径
    if (state.selectedForRoute.length === 2) {
        calculateAndShowRoute();
    }
};

function highlightMarker(pointId, highlight) {
    const markerData = state.markers[pointId];
    if (!markerData) return;
    
    if (markerData.isLeaflet) {
        // Leaflet: 修改marker样式
        const markerEl = markerData.marker.getElement();
        if (markerEl) {
            const div = markerEl.querySelector('div');
            if (highlight) {
                div.style.transform = 'scale(1.5)';
                div.style.borderWidth = '4px';
                div.style.borderColor = '#fbbf24';
            } else {
                div.style.transform = 'scale(1)';
                div.style.borderWidth = '3px';
                div.style.borderColor = 'white';
            }
        }
    }
}

function clearRouteSelection() {
    // 取消高亮
    state.selectedForRoute.forEach(point => {
        highlightMarker(point.id, false);
    });
    
    // 移除路径图层
    if (state.routeLayer && state.map.isLeaflet) {
        state.map._leaflet.removeLayer(state.routeLayer);
        state.routeLayer = null;
    }
    
    state.selectedForRoute = [];
}

async function calculateAndShowRoute() {
    const [point1, point2] = state.selectedForRoute;
    
    console.log(`Calculating route from ${point1.id} to ${point2.id}`);
    
    // 移除旧路径
    if (state.routeLayer && state.map.isLeaflet) {
        state.map._leaflet.removeLayer(state.routeLayer);
    }
    
    try {
        // 调用API获取路径
        const response = await fetch(`${API_BASE_URL}/route`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                start_lat: point1.lat,
                start_lng: point1.lng,
                end_lat: point2.lat,
                end_lng: point2.lng
            })
        });
        
        const routeData = await response.json();
        const routePoints = routeData.route;
        const distance = routeData.distance;
        const time = routeData.time;
        
        // 使用API返回的路径点绘制曲线
        if (state.map.isLeaflet) {
            const latlngs = routePoints.map(p => [p.lat, p.lng]);
            
            state.routeLayer = L.polyline(latlngs, {
                color: '#667eea',
                weight: 5,
                opacity: 0.9,
                lineJoin: 'round',
                lineCap: 'round'
            }).addTo(state.map._leaflet);
            
            // 添加动画效果 - 流动的虚线
            animateRoute(state.routeLayer);
            
            // 显示路径信息popup
            showRouteInfo(point1, point2, distance, time);
            
            // 缩放到路径范围
            state.map._leaflet.fitBounds(latlngs, { padding: [50, 50] });
        }
        
    } catch (error) {
        console.log('API route failed, using fallback');
        // Fallback: 使用贝塞尔曲线
        const latlngs = generateCurvedRoute(
            [point1.lat, point1.lng],
            [point2.lat, point2.lng]
        );
        
        if (state.map.isLeaflet) {
            state.routeLayer = L.polyline(latlngs, {
                color: '#667eea',
                weight: 5,
                opacity: 0.9,
                lineJoin: 'round',
                lineCap: 'round'
            }).addTo(state.map._leaflet);
            
            animateRoute(state.routeLayer);
            
            const distance = calculateDistance(point1, point2);
            const time = distance * 3;
        
            // 显示路径信息popup
            showRouteInfo(point1, point2, distance, time);
            
            // 缩放到路径范围
            state.map._leaflet.fitBounds(latlngs, { padding: [50, 50] });
        }
    }
}

// 生成曲线路径 (贝塞尔曲线)
function generateCurvedRoute(start, end) {
    const points = [];
    const numPoints = 30;
    
    // 控制点(曲线的弯曲度)
    const midLat = (start[0] + end[0]) / 2;
    const midLng = (start[1] + end[1]) / 2;
    const offsetLat = (end[0] - start[0]) * 0.2;
    const offsetLng = (end[1] - start[1]) * 0.2;
    
    const controlLat = midLat + offsetLng;
    const controlLng = midLng - offsetLat;
    
    // 二次贝塞尔曲线
    for (let i = 0; i <= numPoints; i++) {
        const t = i / numPoints;
        const lat = (1-t)*(1-t)*start[0] + 2*(1-t)*t*controlLat + t*t*end[0];
        const lng = (1-t)*(1-t)*start[1] + 2*(1-t)*t*controlLng + t*t*end[1];
        points.push([lat, lng]);
    }
    
    return points;
}

// 计算两点距离
function calculateDistance(point1, point2) {
    return Math.sqrt(
        Math.pow(point1.lat - point2.lat, 2) + 
        Math.pow(point1.lng - point2.lng, 2)
    ) * 111;
}

// 动画路径效果
function animateRoute(polyline) {
    let offset = 0;
    
    function animate() {
        offset = (offset + 1) % 20;
        polyline.setStyle({
            dashArray: `10, 10`,
            dashOffset: -offset
        });
        requestAnimationFrame(animate);
    }
    
    animate();
}

// 显示路径信息
function showRouteInfo(point1, point2, distance, time) {
        
    const routeInfo = `
        <div style="padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
            <h4 style="margin: 0 0 10px 0; color: #667eea;">🚚 Route Calculated</h4>
            <div style="font-size: 0.9em; line-height: 1.8;">
                <div><strong>From:</strong> ${point1.id} (${point1.role})</div>
                <div><strong>To:</strong> ${point2.id} (${point2.role})</div>
                <div><strong>Distance:</strong> ~${distance.toFixed(2)} km</div>
                <div><strong>Est. Time:</strong> ~${time.toFixed(0)} min</div>
            </div>
            <button onclick="clearRouteSelection()" 
                    style="margin-top: 10px; width: 100%; padding: 6px; 
                           background: #f56565; color: white; border: none; 
                           border-radius: 5px; cursor: pointer;">
                Clear Route
            </button>
        </div>
    `;
    
    state.routeLayer.bindPopup(routeInfo).openPopup();
    
    // 更新通知栏
    const notification = document.getElementById('notification-text');
    if (notification) {
        notification.textContent = `🚚 Route from ${point1.role} to ${point2.role} - Distance: ${distance.toFixed(2)}km, Time: ~${time.toFixed(0)} min`;
    }
}

// 六边形热力图
let hexagonLayers = [];
let heatmapVisible = false;

function toggleHexagonHeatmap() {
    if (heatmapVisible) {
        // 移除热力图
        hexagonLayers.forEach(layer => {
            if (state.map.isLeaflet) {
                state.map._leaflet.removeLayer(layer);
            }
        });
        hexagonLayers = [];
        heatmapVisible = false;
        console.log('Heatmap hidden');
    } else {
        // 显示热力图
        generateHexagonHeatmap();
        heatmapVisible = true;
        console.log('Heatmap shown');
    }
}

function generateHexagonHeatmap() {
    if (!state.map.isLeaflet) return;
    
    // 合并所有点数据
    const allPoints = [...state.pickupData, ...state.deliveryData];
    
    // 为每个点创建六边形
    allPoints.forEach(point => {
        const demand = point.demand || 10;
        
        // 根据demand计算颜色和透明度
        const opacity = Math.min(demand / 25, 0.6);
        const color = demand > 20 ? '#e74c3c' :
                     demand > 15 ? '#f39c12' :
                     demand > 10 ? '#f1c40f' :
                     '#2ecc71';
        
        // 生成六边形坐标
        const hexCoords = generateHexagon(point.lat, point.lng, 0.003);
        
        const hexagon = L.polygon(hexCoords, {
            color: color,
            fillColor: color,
            fillOpacity: opacity,
            weight: 1,
            opacity: 0.8
        }).addTo(state.map._leaflet);
        
        hexagon.bindTooltip(`Demand: ${demand}`, {
            permanent: false,
            direction: 'top'
        });
        
        hexagonLayers.push(hexagon);
    });
}

function generateHexagon(centerLat, centerLng, radius) {
    const coords = [];
    for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 3) * i;
        const lat = centerLat + radius * Math.sin(angle);
        const lng = centerLng + radius * Math.cos(angle);
        coords.push([lat, lng]);
    }
    return coords;
}
// Menu功能 - 在setupEventListeners中初始化
function initMenuHandlers() {
    const menuOptions = document.querySelectorAll('.menu-option');
    if (menuOptions.length > 0) {
        menuOptions.forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.dataset.action;
                handleMenuAction(action);
            });
        });
    }
}

function handleMenuAction(action) {
    const modal = document.getElementById('menuModal');
    
    switch(action) {
        case 'export':
            exportData();
            break;
        case 'settings':
            showSettings();
            break;
        case 'help':
            showHelp();
            break;
        case 'about':
            showAbout();
            break;
    }
    
    modal.style.display = 'none';
}

function exportData() {
    const data = {
        pickup: state.pickupData,
        delivery: state.deliveryData,
        timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ctis_data_export.json';
    a.click();
    
    alert('✅ Data exported successfully!');
}

function showSettings() {
    alert(`⚙️ Settings
    
Current Configuration:
- Nodes: ${state.pickupData.length + state.deliveryData.length}
- Map: Leaflet (OpenStreetMap)
- Auto-refresh: Enabled
- Animation: Enabled`);
}

function showHelp() {
    alert(`❓ Help & Usage Guide

📍 Map Controls:
- Click markers for details
- Use "Route" button to plan paths
- Time slider filters data by hour
- Layer buttons toggle visibility

📊 Tabs:
- Overview: System statistics
- Details: Point information
- Forecast: Predictions
- Chat: AI assistant

🗺️ Features:
- Multi-role color coding
- Route planning
- Demand heatmap
- Real-time updates`);
}

function showAbout() {
    alert(`ℹ️ About CTIS

Connected Transportation Information System
Version 1.0.0

Team Members:
- Weilin Ruan
- Ziyu Zhou
- Songxin
- Yiming Huang

Dataset: LaDe (Last-mile Delivery)
Source: https://huggingface.co/datasets/Cainiao-AI/LaDe

© 2025 HKUST(GZ)`);
}

console.log('CTIS Enhanced Demo Initialized');

