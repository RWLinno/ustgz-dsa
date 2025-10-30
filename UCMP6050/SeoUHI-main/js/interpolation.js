// 创建一个标志来跟踪是否启用了点击事件
let isClickEnabled = false;

// 获取 interpolation 元素
const interpolationToggle = document.getElementById('interpolation');

// 初始状态下显示 "Enable Interpolation"
interpolationToggle.textContent = 'Enable Interpolation';

// 点击 interpolation 标签来切换点击事件的启用状态
interpolationToggle.addEventListener('click', function (e) {
    e.preventDefault(); // 阻止<a>标签的默认行为
    isClickEnabled = !isClickEnabled; // 切换标志状态

    // 根据 isClickEnabled 的状态更新文本内容
    interpolationToggle.textContent = isClickEnabled ? 'Disable Interpolation' : 'Enable Interpolation';
});