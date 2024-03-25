
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(error => {
        console.error('Error accessing webcam:', error);
    });

video.addEventListener('play', () => {
    const width = video.videoWidth;
    const height = video.videoHeight;

    const drawFrame = () => {
        if (video.paused || video.ended) {
            return;
        }
        context.drawImage(video, 0, 0, width, height);

        requestAnimationFrame(drawFrame);
    };

    drawFrame();
});
