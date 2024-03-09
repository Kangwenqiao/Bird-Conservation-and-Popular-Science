
document.addEventListener('DOMContentLoaded', function() {
    var currentIndex = 1;
    var totalImages = 6;
    var isFading = false;

    function changeBackground() {
        if (isFading) return;

        var header = document.getElementById('panel');
        var nextImageIndex = (currentIndex % totalImages) + 1;
        var nextImage = 'url(images/image' + nextImageIndex + '.jpg)';

        // 创建一个新的背景层
        var newBackground = document.createElement('div');
        newBackground.style.backgroundImage = nextImage;
        newBackground.className = 'background-image fade-in';
        header.appendChild(newBackground);

        // 移除旧的背景层
        setTimeout(function() {
            var backgrounds = document.getElementsByClassName('background-image');
            if (backgrounds.length > 1) {
                header.removeChild(backgrounds[0]);
            }
        }, 1000); // 等待淡入动画完成

        currentIndex = nextImageIndex;
        setTimeout(changeBackground, 5000); // 下一次变换
    }

    changeBackground();
});


function toggleAudioAndAnimation() {
    var audioPlayer = document.getElementById('audioPlayer');
    var profileImage = document.getElementById('profileImage');

    if (audioPlayer.paused) {
        audioPlayer.play();
        profileImage.classList.add('rotate');
    } else {
        audioPlayer.pause();
        profileImage.classList.remove('rotate');
    }
}
