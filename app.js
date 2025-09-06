// Updated for CL1 Neural DOOM Controller Dashboard

class NeuralDoomDashboard {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.isRunning = false;
        this.charts = {};
        
        this.gameState = {
            player: { health: 100, armor: 0, ammo: 50 },
            armor_visible: false,
            enemy_visible: false
        };
        
        this.neuralActivity = { navigation: 0, combat: 0, tactical: 0 };
        this.stats = { score: 0, armor_pickups: 0, enemy_kills: 0 };
        this.currentAction = 'idle';
        this.runtime = 0;
        this.cl1Connected = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupCharts();
        this.initializeSocket();
    }
    
    initializeSocket() {
        this.socket = io('http://localhost:5000');
        
        this.socket.on('connect', () => {
            console.log('🎮 Connected to CL1 Neural Controller');
            this.updateStatus('Ready to connect CL1', 'ready');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from CL1 backend');
            this.updateStatus('Disconnected', 'disconnected');
            this.isConnected = false;
            this.isRunning = false;
        });
        
        // Real-time CL1 updates
        this.socket.on('update', (data) => {
            this.gameState = data.gameState;
            this.neuralActivity = data.neuralActivity;
            this.currentAction = data.currentAction;
            this.stats = data.stats;
            this.runtime = data.runtime;
            this.cl1Connected = data.cl1_connected || false;
            
            this.updateDisplay();
        });
        
        // Episode completed
        this.socket.on('episode_finished', (finalStats) => {
            this.showEpisodeResults(finalStats);
        });
    }
    
    setupEventListeners() {
        document.getElementById('connectBtn').addEventListener('click', () => this.connectCL1());
        document.getElementById('startBtn').addEventListener('click', () => this.startExperiment());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopExperiment());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetGame());
        document.getElementById('emergencyBtn').addEventListener('click', () => this.emergencyStop());
    }
    
    setupCharts() {
        // Neural Activity Chart
        const neuralCtx = document.getElementById('neuralChart').getContext('2d');
        this.charts.neural = new Chart(neuralCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { 
                        label: 'Navigation Network', 
                        data: [], 
                        borderColor: '#00ff88', 
                        backgroundColor: 'rgba(0,255,136,0.1)',
                        tension: 0.4
                    },
                    { 
                        label: 'Combat Network', 
                        data: [], 
                        borderColor: '#ff4444', 
                        backgroundColor: 'rgba(255,68,68,0.1)',
                        tension: 0.4
                    },
                    { 
                        label: 'Tactical Network', 
                        data: [], 
                        borderColor: '#4488ff', 
                        backgroundColor: 'rgba(68,136,255,0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { 
                    y: { beginAtZero: true, max: 100 },
                    x: { display: true }
                },
                animation: { duration: 0 },
                interaction: { intersect: false },
                plugins: {
                    legend: { position: 'top' }
                }
            }
        });
        
        // VizDoom Performance Chart
        const perfCtx = document.getElementById('performanceChart').getContext('2d');
        this.charts.performance = new Chart(perfCtx, {
            type: 'doughnut',
            data: {
                labels: ['Score', 'Kills', 'Armor'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#ffaa00', '#ff4444', '#00ff88'],
                    borderWidth: 2,
                    borderColor: '#333'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }
    
    async connectCL1() {
        try {
            const response = await fetch('/api/connect', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                this.isConnected = true;
                this.updateStatus('🧠 CL1 Connected - VizDoom Ready', 'connected');
                document.getElementById('connectBtn').textContent = 'CL1 Connected';
                document.getElementById('connectBtn').disabled = true;
                document.getElementById('startBtn').disabled = false;
                
                // Update CL1 status
                const deviceType = this.cl1Connected ? 'Real CL1' : 'Mock CL1';
                document.getElementById('cl1DeviceType').textContent = deviceType;
                document.getElementById('cl1DeviceType').className = this.cl1Connected ? 'device-type real-cl1' : 'device-type mock-cl1';
                document.getElementById('cl1Status').textContent = `🧠 CL1 Status: ${deviceType} Connected`;
                document.getElementById('cl1Status').className = this.cl1Connected ? 'status-indicator cl1-real' : 'status-indicator cl1-mock';
                
                this.addAchievement(`🔗 ${deviceType} device connected successfully`);
            } else {
                this.updateStatus('❌ CL1 Connection Failed', 'error');
            }
        } catch (error) {
            console.error('Connection error:', error);
            this.updateStatus('❌ Connection Error', 'error');
        }
    }
    
    async startExperiment() {
        if (!this.isConnected) return;
        
        try {
            const response = await fetch('/api/start', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                this.isRunning = true;
                this.updateStatus('🎮 VizDoom Neural Experiment Running', 'running');
                document.getElementById('startBtn').textContent = '🧠 Neural Networks Active';
                document.getElementById('startBtn').disabled = true;
                document.getElementById('startBtn').classList.add('running');
                document.getElementById('stopBtn').disabled = false;
                
                this.showNotification('🧠 Biological neural networks now controlling DOOM!', 'cl1');
                this.addAchievement('🚀 VizDoom neural experiment started');
            }
        } catch (error) {
            console.error('Start error:', error);
            this.showNotification('❌ Failed to start experiment', 'error');
        }
    }
    
    async stopExperiment() {
        try {
            const response = await fetch('/api/stop', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                this.isRunning = false;
                this.updateStatus('⏹️ VizDoom Experiment Stopped', 'stopped');
                document.getElementById('startBtn').textContent = 'Start VizDoom Experiment';
                document.getElementById('startBtn').disabled = false;
                document.getElementById('startBtn').classList.remove('running');
                document.getElementById('stopBtn').disabled = true;
            }
        } catch (error) {
            console.error('Stop error:', error);
        }
    }
    
    async resetGame() {
        try {
            const response = await fetch('/api/reset', { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                this.stats = { score: 0, armor_pickups: 0, enemy_kills: 0 };
                this.updatePerformanceStats();
                this.showNotification('🔄 New VizDoom episode started', 'info');
                this.addAchievement('🔄 Fresh DOOM episode initialized');
            }
        } catch (error) {
            console.error('Reset error:', error);
        }
    }
    
    emergencyStop() {
        this.stopExperiment();
        this.updateStatus('🚨 Emergency Stop - Neural Stimulation Halted', 'error');
        this.showNotification('🚨 Emergency stop activated! All neural stimulation stopped.', 'error');
    }
    
    updateStatus(text, status) {
        const statusElement = document.getElementById('connectionText');
        const indicatorElement = document.getElementById('connectionStatus');
        
        statusElement.textContent = text;
        indicatorElement.className = `status-indicator ${status}`;
    }
    
    updateDisplay() {
        this.updateNeuralActivity();
        this.updateGameInfo();
        this.updatePerformanceStats();
        this.updateCharts();
        this.updateGameStatus();
    }
    
    updateNeuralActivity() {
        // Update activity meters with CL1 data
        document.getElementById('navActivity').textContent = this.neuralActivity.navigation.toFixed(1);
        document.getElementById('combatActivity').textContent = this.neuralActivity.combat.toFixed(1);
        document.getElementById('tacticalActivity').textContent = this.neuralActivity.tactical.toFixed(1);
        
        // Update meter bars with animation
        this.animateMeterBar('navMeter', this.neuralActivity.navigation);
        this.animateMeterBar('combatMeter', this.neuralActivity.combat);
        this.animateMeterBar('tacticalMeter', this.neuralActivity.tactical);
    }
    
    updateGameStatus() {
        // Update game status indicators
        if (this.stats.enemy_kills > 0) {
            document.getElementById('enemyIndicator').textContent = '💥 Combat Network Active';
            document.getElementById('enemyIndicator').className = 'status-indicator status-active';
        }
        
        if (this.gameState.armor_visible) {
            document.getElementById('armorStatus').textContent = '🛡️ Armor Detected - Navigation Active';
            document.getElementById('armorStatus').className = 'status-indicator status-detected';
        }
        
        if (this.gameState.enemy_visible) {
            document.getElementById('enemyIndicator').textContent = '👹 Enemy in Range - Combat Engaged';
            document.getElementById('enemyIndicator').className = 'status-indicator status-danger';
        } else if (!this.stats.enemy_kills) {
            document.getElementById('enemyIndicator').textContent = '👹 Scanning for enemies...';
            document.getElementById('enemyIndicator').className = 'status-indicator';
        }
    }
    
    animateMeterBar(meterId, value) {
        const bar = document.querySelector(`#${meterId} .meter-fill`);
        if (bar) {
            const percentage = Math.min(100, Math.max(0, value));
            bar.style.width = percentage + '%';
            
            // Color coding based on activity level
            if (percentage > 70) {
                bar.className = 'meter-fill high-activity';
            } else if (percentage > 30) {
                bar.className = 'meter-fill medium-activity';
            } else {
                bar.className = 'meter-fill low-activity';
            }
        }
    }
    
    updateGameInfo() {
        const player = this.gameState.player;
        document.getElementById('playerHealth').textContent = player.health || 100;
        document.getElementById('playerArmor').textContent = player.armor || 0;
        document.getElementById('playerAmmo').textContent = player.ammo || 0;
        document.getElementById('currentAction').textContent = this.currentAction;
        document.getElementById('runtime').textContent = `${this.runtime.toFixed(1)}s`;
        
        // Health bar color coding
        const healthBar = document.getElementById('healthBar');
        if (healthBar) {
            const healthPercent = (player.health / 100) * 100;
            healthBar.style.width = healthPercent + '%';
            if (player.health < 30) {
                healthBar.className = 'health-bar danger';
            } else if (player.health < 60) {
                healthBar.className = 'health-bar warning';
            } else {
                healthBar.className = 'health-bar healthy';
            }
        }
    }
    
    updatePerformanceStats() {
        document.getElementById('score').textContent = this.stats.score || 0;
        document.getElementById('armorPickups').textContent = this.stats.armor_pickups || 0;
        document.getElementById('enemyKills').textContent = this.stats.enemy_kills || 0;
        
        // Add achievements for milestones
        if (this.stats.armor_pickups > 0) {
            this.addAchievement('🛡️ Navigation network successfully collected armor');
        }
        if (this.stats.enemy_kills > 0) {
            this.addAchievement('💥 Combat network eliminated enemy target');
        }
    }
    
    updateCharts() {
        const currentTime = this.runtime.toFixed(1);
        const neural = this.charts.neural;
        
        // Keep last 60 seconds of data
        if (neural.data.labels.length > 60) {
            neural.data.labels.shift();
            neural.data.datasets[0].data.shift();
            neural.data.datasets[1].data.shift();
            neural.data.datasets[2].data.shift();
        }
        
        neural.data.labels.push(currentTime);
        neural.data.datasets[0].data.push(this.neuralActivity.navigation);
        neural.data.datasets[1].data.push(this.neuralActivity.combat);
        neural.data.datasets[2].data.push(this.neuralActivity.tactical);
        neural.update('none');
        
        // Update performance doughnut chart
        const perf = this.charts.performance;
        perf.data.datasets[0].data = [
            Math.max(1, this.stats.score),
            Math.max(1, this.stats.enemy_kills * 20),
            Math.max(1, this.stats.armor_pickups * 15)
        ];
        perf.update('none');
    }
    
    showEpisodeResults(finalStats) {
        const message = `
        🏁 VizDoom Episode Complete!
        
        🏆 Final Score: ${finalStats.score}
        💥 Enemies Eliminated: ${finalStats.enemy_kills}  
        🛡️ Armor Collected: ${finalStats.armor_pickups}
        ⏱️ Runtime: ${this.runtime.toFixed(1)}s
        
        🧠 ${this.cl1Connected ? 'Real CL1' : 'Mock'} neural networks demonstrated biological intelligence!
        `;
        
        this.showNotification(message, 'success', 8000);
        this.addAchievement('🏁 DOOM episode completed by neural networks');
        this.isRunning = false;
        document.getElementById('startBtn').textContent = 'Start New Episode';
        document.getElementById('startBtn').disabled = false;
        document.getElementById('startBtn').classList.remove('running');
        document.getElementById('stopBtn').disabled = true;
    }
    
    addAchievement(text) {
        const list = document.getElementById('achievementList');
        const item = document.createElement('div');
        item.className = 'achievement-item new';
        item.textContent = text;
        list.insertBefore(item, list.firstChild);
        
        // Remove animation class after animation
        setTimeout(() => item.classList.remove('new'), 2000);
    }
    
    showNotification(message, type = 'info', duration = 4000) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span>${message}</span>
                <button class="notification-close">&times;</button>
            </div>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Auto remove
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, duration);
        
        // Manual close
        notification.querySelector('.notification-close').addEventListener('click', () => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        });
    }
}

// Initialize CL1 dashboard
document.addEventListener('DOMContentLoaded', () => {
    console.log('🧠 Initializing Neural DOOM Controller (CL1 Edition)');
    window.dashboard = new NeuralDoomDashboard();
});
