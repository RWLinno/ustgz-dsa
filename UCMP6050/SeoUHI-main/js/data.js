// 全局变量声明
let weatherData = null;
let stationData = null;
let charts = { temp: null, humidity: null };
let currentPopup = null;

// Function to convert CSV to GeoJSON using Fetch API and PapaParse library
function csvToGeoJSON(fileUrl, callback) {
    const geojson = {
        type: 'FeatureCollection',
        features: []
    };

    // Fetch the CSV file using Fetch API
    fetch(fileUrl)
        .then(response => response.text())
        .then(csvData => {
            // Parse CSV data using PapaParse
            Papa.parse(csvData, {
                header: true,
                skipEmptyLines: true,
                complete: function (results) {
                    results.data.forEach(row => {
                        const feature = {
                            type: 'Feature',
                            geometry: {
                                type: 'Point',
                                coordinates: [parseFloat(row.Longitude), parseFloat(row.Latitude)]
                            },
                            properties: row
                        };
                        geojson.features.push(feature);
                    });
                    callback(null, geojson);
                },
                error: function (error) {
                    callback(error, null);
                }
            });
        })
        .catch(error => {
            callback(error, null);
        });
}

// Function to add a stations layer to the map
function addStationsLayer() {
    console.log('Loading station data...');

    (async () => {
        try {
            const response = await fetch('./data/slot_position.csv');
            const csvText = await response.text();
            const csvData = d3.csvParse(csvText);

            console.log('CSV data loaded:', csvData.length, 'records');
            
            const validData = csvData.filter(row => {
                const lng = parseFloat(row.longitude);
                const lat = parseFloat(row.latitude);
                return !isNaN(lng) && !isNaN(lat);
            });

            const geojsonData = {
                type: 'FeatureCollection',
                features: validData.map(row => ({
                    type: 'Feature',
                    properties: {
                        No: row.No,
                        sequence: row.sequence,
                        address: row.address,
                        code: row.code,
                        remarks: row.remarks || '',
                        avg: parseFloat(row.avg || 0)
                    },
                    geometry: {
                        type: 'Point',
                        coordinates: [parseFloat(row.longitude), parseFloat(row.latitude)]
                    }
                }))
            };

            // 添加或更新数据源
            if (map.getSource('stations')) {
                map.getSource('stations').setData(geojsonData);
            } else {
                map.addSource('stations', {
                    type: 'geojson',
                    data: geojsonData
                });

                // 添加图层
                map.addLayer({
                    id: 'unclustered-point',
                    type: 'circle',
                    source: 'stations',
                    paint: {
                        'circle-color': [
                            'interpolate', ['linear'], ['get', 'avg'],
                            1, '#e8f6e3',    // 非常浅的绿色 - 非常冷
                            4, '#c7e9c0',    // 浅绿色 - 很冷
                            6, '#a1d99b',    // 淡绿色 - 冷
                            8, '#74c476',    // 浅绿 - 凉爽
                            9, '#fdbe85',    // 浅橙 - 稍暖
                            10, '#fd8d3c',   // 橙色 - 温暖
                            12, '#f03b20',   // 浅红 - 暖
                            15, '#e31a1c',   // 红色 - 热
                            18, '#bd0026',   // 深红 - 很热
                            21, '#800026'    // 暗红 - 非常热
                        ],
                        'circle-radius': 5,
                        'circle-stroke-width': 1,
                        'circle-stroke-color': '#666',
                        'circle-opacity': 0.5
                    }
                });
            }

            // 添加鼠标交互
            addMouseInteractions();
            console.log('Map layers added successfully');
        } catch (error) {
            console.error('Error loading or processing data:', error);
        }
    })();
}

// 确保函数可以被其他文件访问
window.addStationsLayer = addStationsLayer;

// 加载站点数据
async function loadStationData() {
    console.log('Loading station data...');
    try {
        const response = await fetch('./data/slot_position.csv');
        const csvText = await response.text();
        stationData = Papa.parse(csvText, {
            header: true,
            skipEmptyLines: true
        }).data;
        console.log('CSV data loaded:', stationData.length, 'records');
        return stationData;
    } catch (error) {
        console.error('Error loading station data:', error);
        throw error;
    }
}

// 加载气象数据
async function loadWeatherData() {
    try {
        const response = await fetch('./data/seoul_comparison.csv');
        const csvText = await response.text();
        weatherData = Papa.parse(csvText, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true
        }).data;
        console.log('Weather data loaded successfully:', weatherData.length, 'records');
    } catch (error) {
        console.error('Failed to load weather data:', error);
    }
}

// 初始化函数
async function initialize() {
    try {
        await Promise.all([
            loadWeatherData(),
            loadStationData()
        ]);
        addMouseInteractions();
    } catch (error) {
        console.error('Initialization failed:', error);
    }
}

document.addEventListener('DOMContentLoaded', initialize);

function addMouseInteractions() {
    map.on('mouseenter', 'unclustered-point', (e) => {
        if (currentPopup) return;
        console.log("mouseenter");
        map.getCanvas().style.cursor = 'pointer';
        
        if (e.features.length > 0) {
            const feature = e.features[0];
            const coordinates = feature.geometry.coordinates.slice();
            const stationId = feature.properties.sequence;
            const index = feature.properties.No % 64;
            const currentWeather = weatherData[index];
            const last24Hours = weatherData.slice(index, index + 24);
            
            // 保存弹窗内容到全局变量
            window.lastPopupContent = `
                <div class="popup-content">
                    <div class="popup-left">
                        <div class="popup-header">
                            <h3>Station Details</h3>
                            <button class="popup-close-button" aria-label="Close"></button>
                        </div>
                        
                        <!-- Basic Information -->
                        <div class="basic-info">
                            <h4>Basic Information</h4>
                            <p><strong>No:</strong> ${feature.properties.No}</p>
                            <p><strong>Station ID:</strong> ${stationId}</p>
                            <p><strong>Address:</strong> ${feature.properties.address}</p>
                            <p><strong>Coordinates:</strong> ${coordinates[0].toFixed(6)}°E, ${coordinates[1].toFixed(6)}°N</p>
                            <p><strong>Station Code:</strong> ${feature.properties.code}</p>
                            ${feature.properties.remarks ? `<p><strong>Remarks:</strong> ${feature.properties.remarks}</p>` : ''}
                        </div>

                        <!-- Weather Data Grid Layout -->
                        <div class="weather-info">
                            <h4>Real-time Weather Data</h4>
                            <div class="weather-grid-three-columns">
                                <!-- Temperature Related -->
                                <div class="weather-column">
                                    <p><strong>Temperature:</strong> ${currentWeather.temp}°C</p>
                                    <p><strong>Humidity:</strong> ${currentWeather.humidity}%</p>
                                    <p><strong>Feels Like:</strong> ${currentWeather.feelslike}°C</p>
                                    <p><strong>Dew Point:</strong> ${currentWeather.dew}°C</p>
                                    <p><strong>Pressure:</strong> ${currentWeather.sealevelpressure} hPa</p>
                                </div>
                                <!-- Wind and Humidity Related -->
                                <div class="weather-column">
                                    <p><strong>Wind Speed:</strong> ${currentWeather.windspeed} m/s</p>
                                    <p><strong>Wind Direction:</strong> ${currentWeather.winddir}°</p>
                                    <p><strong>Wind Gust:</strong> ${currentWeather.windgust} m/s</p>
                                    <p><strong>Cloud Cover:</strong> ${currentWeather.cloudcover}%</p>
                                    <p><strong>Visibility:</strong> ${currentWeather.visibility} km</p>
                                </div>
                            </div>
                        </div>

                        <!-- Chart Container -->
                        <div class="charts-container">
                            <div class="chart-wrapper">
                                <canvas id="predictionChart" width="400" height="200"></canvas>
                            </div>
                        </div>
                    </div>

                    <!-- Right Side Images -->
                    <div class="popup-right">
                        <div class="image-section">
                            <div class="image-grid">
                                <div class="image-wrapper">
                                    <h5>Street View</h5>
                                    <img src="./data/Image_500m/${stationId}_streets.png" 
                                         alt="Street View" 
                                         class="station-image"
                                         onerror="this.onerror=null; this.src='./images/no-image.png';">
                                </div>
                                <div class="image-wrapper">
                                    <h5>Satellite View</h5>
                                    <img src="./data/Image_500m/${stationId}_satellite.png" 
                                         alt="Satellite View"
                                         class="station-image"
                                         onerror="this.onerror=null; this.src='./images/no-image.png';">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            currentPopup = new mapboxgl.Popup({
                closeButton: false,
                closeOnClick: true,
                maxWidth: '1000px',
                className: 'custom-popup'
            })
            .setLngLat(coordinates)
            .setHTML(window.lastPopupContent)
            .addTo(map);

            // 在弹窗创建后添加关闭按钮事件监听
            const closeButton = document.querySelector('.popup-close-button');
            if (closeButton) {
                closeButton.addEventListener('click', () => {
                    if (currentPopup) {
                        currentPopup.remove();
                        currentPopup = null;
                        // 清理图表
                        if (charts.temp) {
                            charts.temp.destroy();
                            charts.temp = null;
                        }
                        if (charts.humidity) {
                            charts.humidity.destroy();
                            charts.humidity = null;
                        }
                    }
                });
            }

            // 在弹窗创建后初始化图表
            currentPopup.on('open', () => {
                // 获取最近24小时的数据
                const last24Hours = weatherData.slice(0, 24);
                console.log("last24Hours", last24Hours);
                // 创建温度图表
                if (charts.temp) {
                    charts.temp.destroy();
                }
                charts.temp = createTemperatureChart(last24Hours);
                
                // 创建湿度图表
                if (charts.humidity) {
                    charts.humidity.destroy();
                }
                charts.humidity = createHumidityChart(last24Hours);

                // 创建侧边温度图表
                const sideTemp = createSideTemperatureChart(last24Hours);
                if (sideTemp) {
                    charts.sideTemp = sideTemp;
                }
            });

            // 划分两条曲线，分别为真实数据和预测数据

            const groundtruth = weatherData.slice(0, 120).map(d => d.temp);
            window.groundtruth_simulated = predictNextSteps(groundtruth) //模拟一下不同点的真实值
            window.predictions = predictNextSteps(groundtruth);
            
            // 创建预测图表
            createPredictionChart(window.groundtruth_simulated, window.predictions);
        }
    });

    map.on('mouseleave', 'unclustered-point', () => {
        map.getCanvas().style.cursor = '';
        if (currentPopup) {
            currentPopup.remove();
            currentPopup = null;
        }
    });

    // 添加点击事件监听器来处理弹窗关闭
    map.on('click', () => {
        if (currentPopup) {
            currentPopup.remove();
            currentPopup = null;
            // 当弹窗关闭时重置图表状态
            if (charts.temp) {
                charts.temp.destroy();
                charts.temp = null;
            }
            if (charts.humidity) {
                charts.humidity.destroy();
                charts.humidity = null;
            }
        }
    });
}

// Function to generate HTML content for the popup, including a Chart.js chart
function generatePopupContent(lnglat) {
    // Create a unique ID for the canvas
    const uniqueCanvasId = 'chart-' + Math.random().toString(36).substr(2, 9);
    // Store the clicked data for use when the popup is open
    window.currentPopupData = {
        id: uniqueCanvasId,
    };

    getNearbyPlaces(lnglat)
    // Return the HTML for the popup with the unique canvas ID
    return `
        <div style="width:1200px; max-width:100%;">
            <canvas id="${uniqueCanvasId}" style="width:100%; height:200px;"></canvas>
        </div>
        <p align="center" style="font-size: 18px; margin-bottom: 10px;">
            <strong>Coordinate (${lnglat.lng.toFixed(3)}&deg;E, ${lnglat.lat.toFixed(3)}&deg;N)</strong>
        </p>
        <p align="center" style="font-size: 14px; margin-top: 10px;">
            <u id="placesList" style="margin-left: 10px; text-decoration: none; color: #007bff;">Popular POIs</u>
        </p>
        `;
}

function getNearbyPlaces(lnglat) {
    var mapboxAccessToken = 'pk.eyJ1Ijoic2lydXpob25nIiwiYSI6ImNsamJpNXdvcTFoc24zZG14NWU5azdqcjMifQ.W_2t66prRsaq8lZMSdfKzg'; // 替换为你的Mapbox Access Token
    var types = 'poi'; // 指定你想搜索的类型为兴趣点（POI）
    var limit = 3; // 限制返回结果数量
    var url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${lnglat.lng},${lnglat.lat}.json?` +
        `access_token=${mapboxAccessToken}&limit=${limit}&types=${types}`;

    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.features && Array.isArray(data.features)) {
                var places = data.features.map(feature => feature.text).slice(0, 3);
                updatePopupContent(places);
            } else {
                console.error('Received data is not an array:', data);
            }
        })
        .catch(error => console.error('Error:', error));
}

function updatePopupContent(places) {
    if (Array.isArray(places)) {
        document.querySelector('#placesList').innerHTML = '<i class="fas fa-info-circle"></i>' + ' ' + '<strong><u>Popular POIs</u></strong>: ' + places.join(', ');
    } else {
        console.error('places is not an array:', places);
    }
}


// Function to generate time labels for a 12-hour period every 15 minutes
function generateTimeLabels() {
    const labels = [];
    for (let hour = 0; hour < 12; hour++) {
        for (let minute = 0; minute < 60; minute += 15) {
            labels.push(`${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`);
        }
    }
    return labels;
}

// Function to initialize the Chart.js chart, called after the popup is shown
// Reference: https://www.chartjs.org/docs/latest/
function initChart(canvasId, trueData, predData) {
    trueArr = trueData.slice(1, -1).split(',').map(Number)
    predArr = predData.slice(1, -1).split(',').map(Number)

    new Chart(
        document.getElementById(canvasId),
        {
            type: 'line',
            data: {
                labels: generateTimeLabels(),
                datasets: [{
                    label: 'True',
                    data: trueArr,
                    borderColor: 'red',
                    fill: false
                }, {
                    label: 'Predicted',
                    data: predArr,
                    borderColor: 'blue',
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        ticks: {
                            autoSkip: false,
                            maxRotation: 90,
                            minRotation: 90
                        }
                    },
                    y: {
                        beginAtZero: true, // 根据你的数据可能需要调整
                        ticks: {
                            // 你可能需要自定义 tick 配置
                        }
                    }
                },
                plugins: {
                    annotation: {
                        annotations: {
                            line6AM: { // 注解的ID
                                type: 'line',
                                mode: 'vertical',
                                xMin: '06:00',
                                xMax: '06:00',
                                borderColor: 'black',
                                borderWidth: 2,
                                borderDash: [6, 6], // 这会创建一个6px的虚线和6px的间隙
                                label: {
                                    enabled: true,
                                    content: '6:00 AM',
                                    position: "start"
                                }
                            }
                        }
                    }
                }
            }
        }
    );
}

function getValueColor(value) {
    if (value <= 50) return '#00ff00';
    if (value <= 100) return '#ffff00';
    if (value <= 150) return '#ff9900';
    if (value <= 200) return '#ff0000';
    return '#990066';
}

// 初始化图表
function initializeCharts(stationId) {
    // 这里添加图表初始化代码
    // 可以使用 echarts 或其他图表库
}

// 创建温度图表
function createTemperatureChart(data, canvasId) {
    const ctx = document.getElementById('tempChart');
    console.log("creating tempChart");
    if (!ctx) return null;

    // 获取最近24小时的数据
    const recentData = weatherData.slice(0, 24);
    
    // 准备图表数据
    const chartData = {
        labels: recentData.map(d => new Date(d.datetime).toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit'
        })),
        datasets: [{
            label: 'Temperature',
            data: recentData.map(d => d.temp),
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.1)',
            tension: 0.1,
            fill: true
        }]
    };

    return new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '24-hour Temperature Trend'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `Temperature: ${context.parsed.y}°C`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Temperature (°C)'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        }
    });
}

// 创建湿度图表
function createHumidityChart(data, canvasId) {
    const ctx = document.getElementById('humidityChart');
    if (!ctx) return null;

    // 获取最近120个时间点的数据
    const recentData = weatherData.slice(0, 120);

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: recentData.map(d => new Date(d.datetime).toLocaleTimeString()),
            datasets: [{
                label: 'Humidity',
                data: recentData.map(d => d.humidity),
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Humidity Trend'
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

function createSideTemperatureChart(data) {
    const ctx = document.getElementById('sideTemperatureChart');
    console.log("sideTemperatureChart");
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => new Date(d.datetime).toLocaleTimeString()),
            datasets: [{
                label: 'Temperature (°C)',
                data: data.map(d => d.temp),
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '24-hour Temperature Trend'
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// 在 mouseenter 事件中添加图片加载监听
const images = document.querySelectorAll('.station-image');
images.forEach(img => {
    img.addEventListener('load', () => {
        console.log('图片加载成功:', img.src);
    });
    img.addEventListener('error', () => {
        console.log('图片加载失败:', img.src);
        img.src = './images/no-image.png';
    });
});

// 从CSV读取数据并处理
function prepareTimeSeriesData(data, lookback=96, horizon=24) {
    const temps = data.map(d => parseFloat(d.temp));
    const X = [];
    const y = [];
    
    for(let i = 0; i < temps.length - lookback - horizon; i++) {
        X.push(temps.slice(i, i + lookback));
        y.push(temps.slice(i + lookback, i + lookback + horizon));
    }
    
    return { X, y };
}

function predictNextSteps(historicalData, horizon=120) {
    const lastKnownTemp = historicalData[historicalData.length - 1];
    const predictions = [];
    for(let i = 0; i < horizon; i++) {
        const prevTemp = i === 0 ? lastKnownTemp : predictions[i-1];      
        const randomNoise = (Math.random() - 0.5) * 0.5;
        const seasonalTrend = Math.sin(2 * Math.PI * i / 24) * 0.5; // 24小时周期
        let predictedTemp = prevTemp + randomNoise + seasonalTrend;
        predictedTemp = Math.max(-10, Math.min(40, predictedTemp));
        predictions.push(Number(predictedTemp.toFixed(1)));
    }
    
    return predictions;
}

function createPredictionChart(historicalData, predictions) {
    const ctx = document.getElementById('predictionChart');
    if (!ctx) return null;
    
    // 创建时间标签
    const labels = [];
    for(let i = 0; i < historicalData.length; i++) {
        labels.push(i);
    }
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Historical Data',
                    data: historicalData,
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                },
                {
                    label: 'Predicted Data',
                    data: predictions.slice(0, historicalData.length), // 只取与历史数据等长的部分
                    borderColor: 'rgb(54, 162, 235)',
                    tension: 0.1,
                    borderDash: [2, 2]
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Temperature Prediction Comparison'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time Step'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Temperature (°C)'
                    }
                }
            }
        }
    });
}
