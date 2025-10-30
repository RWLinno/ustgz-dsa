// 获取所有的样式切换链接
const styleLinks = document.querySelectorAll('.submenu__item a[data-style]');

// 为每个链接添加点击事件监听器
for (const link of styleLinks) {
    link.addEventListener('click', function (e) {
        e.preventDefault(); // 阻止默认的链接点击行为
        const styleId = this.getAttribute('data-style');
        map.setStyle('mapbox://styles/mapbox/' + styleId);
        map.once('style.load', addStationsLayer); // 当样式加载完成后，重新添加站点数据层
        map.once('style.load', addHeatmapLayer); // 当样式加载完成后，重新添加热力图层
    });
}
