
document.addEventListener('DOMContentLoaded', function() {
    const needle = document.getElementById('tachometer-needle');
    const scoreDisplay = document.getElementById('tachometer-score-display');
    if (needle && scoreDisplay) {
        let score = parseFloat(scoreDisplay.textContent);
        if (isNaN(score)) score = 5.0;
        // Convert score (0-10) to an angle: -90deg for 0, +90deg for 10.
        const angle = Math.min(Math.max(score, 0), 10) * 18 - 90;
        setTimeout(() => {
            needle.style.transform = `rotate(${angle}deg)`;
        }, 300);
    }
});
