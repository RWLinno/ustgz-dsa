// 设置Mapbox的访问令牌
mapboxgl.accessToken = 'pk.eyJ1Ijoic2lydXpob25nIiwiYSI6ImNsamJpNXdvcTFoc24zZG14NWU5azdqcjMifQ.W_2t66prRsaq8lZMSdfKzg';

// 创建地图
const map = new mapboxgl.Map({
    container: 'map', // 地图容器的ID
    style: 'mapbox://styles/siruzhong/clmr3ruds027p01pj91ajfoif', // 地图样式的URL
    center: [126.9780, 37.5665],
    zoom: 11,
    minZoom: 9,
    maxZoom: 18,
    projection: 'mercator' // 初始投影方式
});

// 添加全局变量
let isClickEnabled = true; // 控制是否启用点击功能
let currentTimeIndex = 0; // 当前显示的时间点索引
let temperatureData = null; // 存储温度数据
let heatmapInterval; // 存储定时器
let lastPopupContent = '';  
let isHeatmapVisible = true; // 控制热力图层显示状态

// 添加切换点击功能的方法
function toggleClickFunction() {
    isClickEnabled = !isClickEnabled;
    console.log('点击功能已' + (isClickEnabled ? '启用' : '禁用'));
}

// 检查坐标是否在首尔市范围内
function isWithinBounds(lngLat) {
    // 首尔市的大致边界范围
    const seoulBounds = {
        west: 126.7650,  // 最西经度
        east: 127.1830,  // 最东经度
        south: 37.4280,  // 最南纬度
        north: 37.7010   // 最北纬度
    };

    return lngLat.lng >= seoulBounds.west &&
           lngLat.lng <= seoulBounds.east &&
           lngLat.lat >= seoulBounds.south &&
           lngLat.lat <= seoulBounds.north;
}

// 当地图加载完成时执行
map.on('load', function () {
    addHeatmapLayer();  // 添加热力图层 
    // 输出所有图层信息以供调试
    const style = map.getStyle();
    console.log('地图所有图层：', style.layers.map(layer => layer.id));
    console.log('热力图层加载完成!');
    addStationsLayer(); // 添加站点数据层
    console.log('站点数据层加载完成!');
});

// 更新时间的函数
function updateCurrentTime() {
    const now = new Date();
    const year = now.getFullYear();
    const month = now.getMonth() + 1;
    const day = now.getDate();
    const hours = now.getHours();
    const minutes = now.getMinutes();
    
    // 格式化时间显示
    const timeString = `${year}Y-${month}M-${day}D-${hours}:${minutes < 10 ? '0' + minutes : minutes}`;
    document.getElementById('current-time').textContent = timeString;
}

// 页面加载时立即更新一次时间
updateCurrentTime();

// 每分钟更新一次时间
setInterval(updateCurrentTime, 60000);

// 添加清理函数（在适当的时候调用，比如页面卸载时）
function cleanup() {
    if (heatmapInterval) {
        clearInterval(heatmapInterval);
    }
}

// 创建弹出窗口
var popup = new mapboxgl.Popup({
    closeOnClick: false,
    closeButton: false,
});

// 当鼠标悬停在站点上时显示数据
map.on('mouseenter', 'rwl_stations', function (e) {
    const clickedData = e.features[0].properties;
    const popupContent = generatePopupContent(clickedData, e.lngLat);
    
    popup.setLngLat(e.lngLat)
        .setHTML(popupContent)
        .addTo(map);

    map.getCanvas().style.cursor = 'pointer';
});

// 当鼠标离开站点时移除数据
map.on('mouseleave', 'rwl_stations', function () {
    map.getCanvas().style.cursor = '';
    popup.remove();
});

// 添加点击事件监听器
map.on('click', function (e) {
    if (isClickEnabled) {
        // 获取点击的经纬度
        var lngLat = e.lngLat;

        // 检查点击的经纬度是否在指定的范围内
        if (isWithinBounds(lngLat)) {
            // 使用经纬度查询数据
            // fetchDataForLocation(lngLat, function (data) {
            //     // 创建一个信息窗口
            //     new mapboxgl.Popup()
            //         .setLngLat(lngLat)
            //         .setHTML(generatePopupContent(data, lngLat))
            //         .addTo(map);
            // });
            console.log('test!');   
        }
    }
});

// 等待 DOM 加载完成
window.onload = function() {
    // 菜单按钮点击事件
    const menuBtn = document.getElementById('menuBtn');
    if (menuBtn) {
        menuBtn.addEventListener('click', function() {
            console.log('Menu button clicked'); // 调试用
            console.log('Last popup content:', window.lastPopupContent); // 调试用
            if (window.lastPopupContent) {
                document.getElementById('modalContent').innerHTML = window.lastPopupContent;
                createPredictionChart(window.groundtruth_simulated, window.predictions);
                document.getElementById('menuModal').style.display = 'block';
            }
        });
    }

    // 添加热力图切换按钮事件
    const toggleHeatmapBtn = document.getElementById('toggleHeatmapBtn');
    if (toggleHeatmapBtn) {
        toggleHeatmapBtn.addEventListener('click', function() {
            window.toggleHeatmap();
            toggleHeatmapBtn.textContent = window.isHeatmapVisible ? 'Hide heatmap': 'Display heatmap';
        });
    }

    // 关闭按钮事件
    const closeBtn = document.querySelector('.modal-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            document.getElementById('menuModal').style.display = 'none';
        });
    }
};


