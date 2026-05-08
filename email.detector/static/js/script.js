document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('spam-form');
    const mainPanel = document.querySelector('.main-panel');
    const loadingState = document.getElementById('loading-state');
    const resultPanel = document.getElementById('result-panel');
    const errorPanel = document.getElementById('error-panel');
    
    // Result elements
    const statusIcon = document.getElementById('status-icon');
    const statusText = document.getElementById('status-text');
    const confidenceScore = document.getElementById('confidence-score');
    const reasoningText = document.getElementById('reasoning-text');
    
    // Action buttons
    const resetBtn = document.getElementById('reset-btn');
    const errorResetBtn = document.getElementById('error-reset-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const emailContent = document.getElementById('email_content').value;

        if (!emailContent.trim()) return;

        // UI State: Loading
        mainPanel.classList.add('hidden');
        errorPanel.classList.add('hidden');
        resultPanel.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            const response = await fetch('/api/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email_content: emailContent })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Server error occurred');
            }

            // Process results
            displayResult(data);

        } catch (error) {
            console.error('Detection Error:', error);
            showError(error.message);
        }
    });

    function displayResult(data) {
        // UI State: Show Results
        loadingState.classList.add('hidden');
        resultPanel.classList.remove('hidden');
        
        // Remove previous status classes
        resultPanel.classList.remove('status-spam', 'status-safe');

        // Update content based on classification
        const isSpam = data.classification === 'spam';
        const percentage = Math.round(data.confidence_score * 100);

        if (isSpam) {
            resultPanel.classList.add('status-spam');
            statusIcon.innerHTML = '<i class="ph-bold ph-shield-warning"></i>';
            statusText.textContent = 'Threat Detected: Spam';
        } else {
            resultPanel.classList.add('status-safe');
            statusIcon.innerHTML = '<i class="ph-bold ph-shield-check"></i>';
            statusText.textContent = 'Message Safe';
        }

        confidenceScore.textContent = `${percentage}%`;
        reasoningText.textContent = data.reasoning;
    }

    function showError(message) {
        loadingState.classList.add('hidden');
        errorPanel.classList.remove('hidden');
        document.getElementById('error-text').textContent = message;
    }

    // Reset handlers
    const resetUI = () => {
        document.getElementById('email_content').value = '';
        resultPanel.classList.add('hidden');
        errorPanel.classList.add('hidden');
        loadingState.classList.add('hidden');
        mainPanel.classList.remove('hidden');
    };

    resetBtn.addEventListener('click', resetUI);
    errorResetBtn.addEventListener('click', resetUI);
});
