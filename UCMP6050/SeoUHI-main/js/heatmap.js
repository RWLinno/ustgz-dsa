// Define latitude and longitude range for Seoul and surrounding areas
const latRange = [36.5, 38.5];  // Latitude range: 36.5°N - 38.5°N
const lngRange = [126.0, 128.0];  // Longitude range: 126.0°E - 128.0°E

// 初始化格网大小和格网数据数组
let gridSize = 0.03;
const gridData = [];

// 添加全局变量
let timeIndex = 0;  
const maxTimeSteps = 48; 
window.isHeatmapVisible = true;

// Fetch the interpolated data and use it to generate grid data
async function interpolateGridData() {
    try {
        console.log('Starting to load interpolated data, grid size:', gridSize);
        let InterpolatedData;

        // Fetch the interpolated data
        const response = await fetch('./data/temperature_data.json');
        InterpolatedData = await response.json();

        if (!InterpolatedData) {
            console.error('No interpolated data received');
            return;
        }
        
        gridData.length = 0; // 清空旧数据
        
        // 将对象转换为数组并生成网格数据
        Object.entries(InterpolatedData).forEach(([id, item]) => {
            gridData.push({
                type: 'Feature',
                geometry: {
                    type: 'Polygon',
                    coordinates: [
                        [
                            [item.grid_longitude - gridSize / 2, item.grid_latitude - gridSize / 2],
                            [item.grid_longitude + gridSize / 2, item.grid_latitude - gridSize / 2],
                            [item.grid_longitude + gridSize / 2, item.grid_latitude + gridSize / 2],
                            [item.grid_longitude - gridSize / 2, item.grid_latitude + gridSize / 2],
                            [item.grid_longitude - gridSize / 2, item.grid_latitude - gridSize / 2]
                        ]
                    ]
                },
                opacity: 0.1, //??
                properties: {
                    avgtemp: item.avg_temp,
                    temp: item.temp[timeIndex]  // 使用当前时间步的温度
                }
            });
        });

        // 更新地图源时使用当前时间步的温度数据
        map.getSource('avgtemp').setData({
            type: 'FeatureCollection',
            features: gridData
        });
        
        // 添加数据验证
        if (gridData.length === 0) {
            console.error('Failed to generate grid data');
            return;
        }

        console.log('Received interpolated data:', InterpolatedData);
        console.log('Generated grid data count:', gridData.length);

    } catch (error) {
        console.error('Error:', error);
    }
}

// 创建 GeoJSON 数据源
async function createGeoJSONSource() {
    if (!map.getSource('avgtemp')) {  // 添加源是否存在的检查
        map.addSource('avgtemp', {
            type: 'geojson',
            data: {
                type: 'FeatureCollection',
                features: gridData
            }
        });
    }
}

function addHeatmapLayer() {
    interpolateGridData();
    console.log('interpolateGridData function completed:');
    createGeoJSONSource();
    console.log('createGeoJSONSource function completed:');
    
    map.addLayer({
        id: 'avgtemp-fill',
        type: 'fill',
        source: 'avgtemp',
        paint: {
            'fill-color': [
                'interpolate', ['linear'], ['get', 'temp'],
                // Using soft green to red color gradient
                1, '#e8f6e3',    // Very light green - Very cold
                4, '#c7e9c0',    // Light green - Very cold
                6, '#a1d99b',    // Pale green - Cold
                8, '#74c476',    // Light green - Cool
                9, '#fdbe85',    // Light orange - Slightly warm
                10, '#fd8d3c',   // Orange - Warm
                12, '#f03b20',   // Light red - Warm
                15, '#e31a1c',   // Red - Hot
                18, '#bd0026',   // Deep red - Very hot
                25, '#800026'    // Dark red - Extremely hot
            ],
            'fill-opacity': 0.1,  // 稍微降低不透明度使颜色更柔和
            'fill-outline-color': 'rgba(0,0,0,0.1)',
        },
        layout: {
            'visibility': window.isHeatmapVisible ? 'visible' : 'none'
        }
    });

    // 添加可见性切换函数
    window.toggleHeatmap = function() {
        window.isHeatmapVisible = !window.isHeatmapVisible;
        map.setLayoutProperty(
            'avgtemp-fill',
            'visibility',
            window.isHeatmapVisible ? 'visible' : 'none'
        );
    };
    
    // 修改定时器，每0.5秒更新一次数据并递增 timeIndex
    setInterval(async () => {
        timeIndex = (timeIndex + 1) % maxTimeSteps;  // 循环更新时间步
        await interpolateGridData();
    }, 1000);
}