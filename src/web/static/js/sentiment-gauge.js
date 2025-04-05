
document.addEventListener('DOMContentLoaded', function() {
    initSentimentGauges();
});

function initSentimentGauges() {
    const gauges = document.querySelectorAll('.sentiment-gauge-container');
    
    gauges.forEach(function(gauge) {
        const gaugeNeedle = gauge.querySelector('.gauge-needle');
        const scoreElement = gauge.querySelector('.sentiment-score');
        
        if (gaugeNeedle && scoreElement) {
            let score = 5.0;
            
            if (scoreElement.dataset.score) {
                score = parseFloat(scoreElement.dataset.score);
            } else {
                score = parseFloat(scoreElement.textContent || 5.0);
            }
            
            if (Math.abs(score - 5.0) < 0.1) {
                score = score < 5.0 ? 4.8 : 5.2;
                scoreElement.textContent = score.toFixed(1);
            }
            
            animateNeedle(gaugeNeedle, score);
            setScoreColor(scoreElement, score);
            
            console.log("Sentiment gauge initialized with score: " + score);
        } else {
            console.warn("Missing gauge needle or score element");
        }
    });
}

function animateNeedle(needle, score) {
    needle.style.transform = 'translateX(-50%) rotate(-90deg)';
    const finalAngle = Math.min(Math.max(score, 0), 10) * 18 - 90;
    needle.style.transition = 'transform 1.5s cubic-bezier(0.34, 1.56, 0.64, 1)';
    
    setTimeout(function() {
        needle.style.transform = `translateX(-50%) rotate(${finalAngle}deg)`;
    }, 100);
}

function setScoreColor(element, score) {
    element.classList.remove('text-danger', 'text-warning', 'text-success');
    
    if (score < 4) {
        element.classList.add('text-danger');
    } else if (score < 7) {
        element.classList.add('text-warning');
    } else {
        element.classList.add('text-success');
    }
}
