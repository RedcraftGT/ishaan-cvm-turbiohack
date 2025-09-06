# Neural DOOM Controller with Real CL1 SDK Integration
# Real DOOM gameplay controlled by biological neural networks via CL1

import os
import json
import time
import logging
import threading
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import cv2
from PIL import Image
try:
    import cl_sdk
except ImportError:
    cl_sdk = None
try:
    import vizdoom as vzd
except ImportError:
    vzd = None

from flask import Flask, jsonify
from flask_socketio import SocketIO

# CL1 SDK Integration
if cl_sdk is not None:
    CL1_AVAILABLE = True
    print("✅ CL1 SDK imported successfully!")
else:
    print("⚠️  CL1 SDK not found - using mock device")
    CL1_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GameState:
    """Represents current DOOM game state from VizDoom"""
    armor_visible: bool = False
    armor_position: Tuple[float, float] = (0.0, 0.0)
    enemy_visible: bool = False
    enemy_position: Tuple[float, float] = (0.0, 0.0)
    player_position: Tuple[float, float] = (0.0, 0.0)
    player_health: int = 100
    player_armor: int = 0
    facing_direction: float = 0.0
    ammo: int = 50
    
@dataclass
class SpikeEvent:
    """Individual spike event from CL1"""
    electrode_id: int
    timestamp: float
    amplitude: float

class CL1Device:
    """Real CL1 device interface using cl_sdk"""
    
    def __init__(self, num_electrodes: int = 600):
        self.num_electrodes = num_electrodes
        self.is_connected = False
        self.device = None
        self.spike_buffer = deque(maxlen=10000)
        self.stimulation_history = []
        
    def connect(self) -> bool:
        """Connect to real CL1 device"""
        if not CL1_AVAILABLE:
            return False
            
        try:
            # Initialize CL1 device using real SDK
            self.device = cl_sdk.CL1Device()
            success = self.device.connect()
            if success:
                self.is_connected = True
                logger.info("🧠 Real CL1 device connected!")
                return True
            else:
                logger.error("❌ Failed to connect to CL1 device")
                return False
        except Exception as e:
            logger.error(f"❌ CL1 connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from CL1 device"""
        if self.device and self.is_connected:
            try:
                self.device.disconnect()
                self.is_connected = False
                logger.info("🔌 CL1 device disconnected")
            except Exception as e:
                logger.error(f"Disconnect error: {e}")
    
    def stimulate_electrodes(self, electrodes: List[int], amplitude: float, 
                           frequency: float, duration: int):
        """Apply electrical stimulation to specified electrodes"""
        if not self.is_connected:
            raise RuntimeError("CL1 device not connected")
            
        try:
            # Use real CL1 SDK for stimulation
            self.device.stimulate_electrodes(
                electrodes=electrodes,
                amplitude=amplitude,  # microamps
                frequency=frequency,   # Hz
                duration=duration     # ms
            )
            
            # Log stimulation for analysis
            stimulation = {
                'electrodes': electrodes, 'amplitude': amplitude,
                'frequency': frequency, 'duration': duration, 'timestamp': time.time()
            }
            self.stimulation_history.append(stimulation)
            
        except Exception as e:
            logger.error(f"Stimulation error: {e}")
    
    def read_spikes(self, duration_ms: int = 50) -> List[SpikeEvent]:
        """Read spike events from CL1 device"""
        if not self.is_connected:
            raise RuntimeError("CL1 device not connected")
            
        try:
            # Read real spikes from CL1
            spike_data = self.device.read_spikes(duration=duration_ms)
            
            # Convert to SpikeEvent objects
            spikes = []
            for spike in spike_data:
                spike_event = SpikeEvent(
                    electrode_id=spike['electrode_id'],
                    timestamp=spike['timestamp'],
                    amplitude=spike['amplitude']
                )
                spikes.append(spike_event)
                
            return spikes
            
        except Exception as e:
            logger.error(f"Spike reading error: {e}")
            return []

class CL1MockDevice:
    """Mock CL1 device for development and testing when real CL1 not available"""
    
    def __init__(self, num_electrodes: int = 600):
        self.num_electrodes = num_electrodes
        self.is_connected = False
        self.spike_buffer = deque(maxlen=10000)
        self.stimulation_history = []
        self.background_noise_level = 0.15
        
    def connect(self) -> bool:
        self.is_connected = True
        logger.info("🤖 Mock CL1 device connected (for testing)")
        return True
    
    def disconnect(self):
        self.is_connected = False
        logger.info("🔌 Mock CL1 device disconnected")
    
    def stimulate_electrodes(self, electrodes: List[int], amplitude: float, 
                           frequency: float, duration: int):
        if not self.is_connected:
            raise RuntimeError("CL1 device not connected")
            
        stimulation = {
            'electrodes': electrodes, 'amplitude': amplitude,
            'frequency': frequency, 'duration': duration, 'timestamp': time.time()
        }
        self.stimulation_history.append(stimulation)
        
        # Simulate neural activity from stimulation
        for electrode in electrodes:
            spike_prob = min(0.9, amplitude * frequency / 80.0)
            if np.random.random() < spike_prob:
                spike = SpikeEvent(
                    electrode_id=electrode, timestamp=time.time(),
                    amplitude=np.random.normal(0.6, 0.15)
                )
                self.spike_buffer.append(spike)
    
    def read_spikes(self, duration_ms: int = 50) -> List[SpikeEvent]:
        if not self.is_connected:
            raise RuntimeError("CL1 device not connected")
            
        current_time = time.time()
        cutoff_time = current_time - (duration_ms / 1000.0)
        
        recent_spikes = [spike for spike in self.spike_buffer 
                        if spike.timestamp >= cutoff_time]
        
        # Add background neural noise
        noise_spikes = []
        for electrode in range(self.num_electrodes):
            if np.random.random() < self.background_noise_level / 1000.0 * duration_ms:
                noise_spike = SpikeEvent(
                    electrode_id=electrode,
                    timestamp=current_time - np.random.random() * (duration_ms/1000.0),
                    amplitude=np.random.normal(0.25, 0.08)
                )
                noise_spikes.append(noise_spike)
        
        return recent_spikes + noise_spikes

def create_cl1_device(num_electrodes: int = 600):
    """Factory function to create appropriate CL1 device"""
    if CL1_AVAILABLE:
        return CL1Device(num_electrodes)
    else:
        return CL1MockDevice(num_electrodes)

class SubnetworkTrainer:
    """Handles training of specialized neural subnetworks"""
    
    def __init__(self, cl1_device):
        self.cl1 = cl1_device
        self.training_params = {
            'navigation': {'amplitude': 0.12, 'frequency': 45, 'duration': 120},
            'combat': {'amplitude': 0.18, 'frequency': 70, 'duration': 180},
            'tactical': {'amplitude': 0.10, 'frequency': 35, 'duration': 220}
        }
        
    def train_navigation_network(self, electrodes: List[int], reward: float):
        if reward > 0:
            params = self.training_params['navigation']
            scaled_amplitude = params['amplitude'] * min(2.5, reward / 4.0)
            self.cl1.stimulate_electrodes(
                electrodes=electrodes, amplitude=scaled_amplitude,
                frequency=params['frequency'], duration=params['duration']
            )
    
    def train_combat_network(self, electrodes: List[int], reward: float):
        if reward > 0:
            params = self.training_params['combat']
            scaled_amplitude = params['amplitude'] * min(2.5, reward / 6.0)
            self.cl1.stimulate_electrodes(
                electrodes=electrodes, amplitude=scaled_amplitude,
                frequency=params['frequency'], duration=params['duration']
            )
    
    def train_tactical_network(self, electrodes: List[int], reward: float):
        if reward > 0:
            params = self.training_params['tactical']
            scaled_amplitude = params['amplitude'] * min(2.5, reward / 8.0)
            self.cl1.stimulate_electrodes(
                electrodes=electrodes, amplitude=scaled_amplitude,
                frequency=params['frequency'], duration=params['duration']
            )

class SubnetworkDecoder:
    """Decodes spike patterns from specialized subnetworks"""
    
    def __init__(self, cl1_device):
        self.cl1 = cl1_device
        self.spike_history = {'navigation': deque(maxlen=1200), 
                             'combat': deque(maxlen=1200), 
                             'tactical': deque(maxlen=1200)}
        
    def read_subnetwork_activity(self, nav_electrodes: List[int], 
                               combat_electrodes: List[int], 
                               tactical_electrodes: List[int]) -> Dict[str, float]:
        spikes = self.cl1.read_spikes(duration_ms=60)
        
        nav_spikes = [s for s in spikes if s.electrode_id in nav_electrodes]
        combat_spikes = [s for s in spikes if s.electrode_id in combat_electrodes]
        tactical_spikes = [s for s in spikes if s.electrode_id in tactical_electrodes]
        
        self.spike_history['navigation'].extend(nav_spikes)
        self.spike_history['combat'].extend(combat_spikes)
        self.spike_history['tactical'].extend(tactical_spikes)
        
        return {
            'navigation': self._calculate_network_strength(nav_spikes),
            'combat': self._calculate_network_strength(combat_spikes),
            'tactical': self._calculate_network_strength(tactical_spikes)
        }
    
    def _calculate_network_strength(self, spikes: List[SpikeEvent]) -> float:
        if not spikes:
            return 0.0
            
        spike_rate = len(spikes) / 0.06  # 60ms window
        
        if len(spikes) > 1:
            timestamps = [s.timestamp for s in spikes]
            time_diffs = np.diff(timestamps)
            synchrony = 1.0 / (1.0 + np.std(time_diffs)) if len(time_diffs) > 0 else 0
        else:
            synchrony = 0
            
        avg_amplitude = np.mean([s.amplitude for s in spikes])
        strength = spike_rate * (1.2 + synchrony) * avg_amplitude
        return min(100.0, strength)

class ActionArbiter:
    """Arbitrates between competing actions from different subnetworks"""
    
    def __init__(self, decoder: SubnetworkDecoder):
        self.decoder = decoder
        self.action_thresholds = {
            'move_forward': 10.0, 'turn_left': 8.0, 'turn_right': 8.0,
            'shoot': 15.0, 'strafe_left': 12.0, 'strafe_right': 12.0,
            'find_armor': 18.0, 'find_weapon': 16.0
        }
        self.last_action = None
        self.action_history = deque(maxlen=200)
    
    def decide_action(self, activities: Dict[str, float], game_state: GameState) -> str:
        nav_action = self._navigation_vote(activities['navigation'], game_state)
        combat_action = self._combat_vote(activities['combat'], game_state)
        tactical_action = self._tactical_vote(activities['tactical'], game_state)
        
        final_action = self._arbitrate_actions([nav_action, combat_action, tactical_action])
        self.last_action = final_action
        self.action_history.append({
            'action': final_action, 'timestamp': time.time(),
            'activities': activities.copy(), 'game_state': game_state
        })
        return final_action
    
    def _navigation_vote(self, strength: float, game_state: GameState) -> Dict:
        if game_state.armor_visible and strength > self.action_thresholds['find_armor']:
            return {'action': 'move_to_armor', 'confidence': strength, 'priority': 3}
        elif strength > self.action_thresholds['move_forward']:
            if game_state.enemy_visible and strength > 15.0:
                return {'action': 'strafe_left', 'confidence': strength, 'priority': 2}
            else:
                return {'action': 'move_forward', 'confidence': strength, 'priority': 1}
        elif strength > self.action_thresholds['turn_left']:
            return {'action': 'turn_left', 'confidence': strength * 0.7, 'priority': 1}
        return {'action': 'none', 'confidence': 0, 'priority': 0}
    
    def _combat_vote(self, strength: float, game_state: GameState) -> Dict:
        if game_state.enemy_visible and strength > self.action_thresholds['shoot']:
            return {'action': 'shoot', 'confidence': strength, 'priority': 4}
        elif game_state.enemy_visible and game_state.player_health < 40:
            return {'action': 'retreat', 'confidence': strength * 0.9, 'priority': 3}
        return {'action': 'none', 'confidence': 0, 'priority': 0}
    
    def _tactical_vote(self, strength: float, game_state: GameState) -> Dict:
        if (game_state.player_health < 60 and game_state.armor_visible and strength > 8.0):
            return {'action': 'prioritize_armor', 'confidence': strength * 1.4, 'priority': 5}
        if (game_state.ammo < 10 and strength > 10.0):
            return {'action': 'find_weapon', 'confidence': strength, 'priority': 4}
        if (game_state.enemy_visible and game_state.player_health < 30):
            return {'action': 'tactical_retreat', 'confidence': strength, 'priority': 4}
        return {'action': 'none', 'confidence': 0, 'priority': 0}
    
    def _arbitrate_actions(self, proposals: List[Dict]) -> str:
        valid_proposals = [p for p in proposals if p['action'] != 'none']
        if not valid_proposals:
            return 'idle'
        max_priority = max(p['priority'] for p in valid_proposals)
        high_priority_actions = [p for p in valid_proposals if p['priority'] == max_priority]
        chosen = max(high_priority_actions, key=lambda x: x['confidence'])
        return chosen['action']

class VizDoomInterface:
    """VizDoom game interface for real DOOM gameplay"""
    
    def __init__(self):
        self.game = vzd.DoomGame()
        self.game_state = GameState()
        self.last_update = time.time()
        self.score = 0
        self.kill_count = 0
        self.armor_pickup_count = 0
        self.setup_game()
        
    def setup_game(self):
        """Configure VizDoom for neural network control"""
        try:
            self.game.set_doom_scenario_path(vzd.scenarios_path + "/basic.wad")
            self.game.set_doom_map("map01")
        except:
            logger.warning("Using default VizDoom scenario")
            
        # Visual settings
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_render_hud(True)
        self.game.set_render_minimal_hud(True)
        self.game.set_render_crosshair(True)
        
        # Game settings
        self.game.set_episode_timeout(2100)  # 35 seconds at 60fps
        self.game.set_episode_start_time(10)
        self.game.set_window_visible(False)  # Headless for server
        
        # Available actions
        self.game.add_available_button(vzd.Button.MOVE_FORWARD)
        self.game.add_available_button(vzd.Button.TURN_LEFT) 
        self.game.add_available_button(vzd.Button.TURN_RIGHT)
        self.game.add_available_button(vzd.Button.ATTACK)
        self.game.add_available_button(vzd.Button.MOVE_LEFT)
        self.game.add_available_button(vzd.Button.MOVE_RIGHT)
        
        # Game variables to track
        self.game.add_available_game_variable(vzd.GameVariable.HEALTH)
        self.game.add_available_game_variable(vzd.GameVariable.ARMOR)
        self.game.add_available_game_variable(vzd.GameVariable.AMMO2)
        self.game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
        
        # Initialize game
        self.game.set_mode(vzd.Mode.ASYNC_PLAYER)
        self.game.init()
        
    def start_episode(self):
        """Start a new DOOM episode"""
        self.game.new_episode()
        self.score = 0
        self.kill_count = 0
        self.armor_pickup_count = 0
        
    def get_current_state(self) -> GameState:
        """Get current game state from VizDoom"""
        if self.game.is_episode_finished():
            return self.game_state
            
        state = self.game.get_state()
        if state is None:
            return self.game_state
            
        # Update game state from VizDoom
        self.game_state.player_health = int(self.game.get_game_variable(vzd.GameVariable.HEALTH))
        self.game_state.player_armor = int(self.game.get_game_variable(vzd.GameVariable.ARMOR))
        self.game_state.ammo = int(self.game.get_game_variable(vzd.GameVariable.AMMO2))
        
        # Simple enemy/item detection
        screen = state.screen_buffer
        if screen is not None:
            self.game_state.enemy_visible = self._detect_enemy(screen)
            self.game_state.armor_visible = self._detect_armor(screen)
            
        return self.game_state
    
    def _detect_enemy(self, screen) -> bool:
        """Simple enemy detection based on color patterns"""
        if screen is None:
            return False
        try:
            screen_hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
            enemy_mask = cv2.inRange(screen_hsv, (5, 50, 50), (15, 255, 255))
            return np.sum(enemy_mask) > 1000
        except:
            return False
    
    def _detect_armor(self, screen) -> bool:
        """Simple armor detection based on green colors"""
        if screen is None:
            return False
        try:
            screen_hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
            armor_mask = cv2.inRange(screen_hsv, (40, 50, 50), (80, 255, 255))
            return np.sum(armor_mask) > 500
        except:
            return False
    
    def execute_action(self, action: str) -> Dict[str, float]:
        """Execute neural network action in VizDoom"""
        if self.game.is_episode_finished():
            return {'movement_reward': 0, 'combat_reward': 0, 'tactical_reward': 0}
            
        rewards = {'movement_reward': 0, 'combat_reward': 0, 'tactical_reward': 0}
        
        # Map neural actions to VizDoom actions
        action_map = {
            'move_forward': [1, 0, 0, 0, 0, 0],
            'turn_left': [0, 1, 0, 0, 0, 0],
            'turn_right': [0, 0, 1, 0, 0, 0],
            'shoot': [0, 0, 0, 1, 0, 0],
            'strafe_left': [0, 0, 0, 0, 1, 0],
            'strafe_right': [0, 0, 0, 0, 0, 1],
            'move_to_armor': [1, 0, 0, 0, 0, 0],
            'prioritize_armor': [1, 0, 0, 0, 0, 0],
            'retreat': [0, 0, 0, 0, 1, 0],
            'tactical_retreat': [0, 0, 0, 0, 1, 0]
        }
        
        doom_action = action_map.get(action, [0, 0, 0, 0, 0, 0])
        
        # Execute action in VizDoom
        prev_health = self.game_state.player_health
        prev_armor = self.game_state.player_armor
        prev_kills = int(self.game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        
        reward = self.game.make_action(doom_action, 4)
        
        # Calculate rewards based on game changes
        current_health = int(self.game.get_game_variable(vzd.GameVariable.HEALTH))
        current_armor = int(self.game.get_game_variable(vzd.GameVariable.ARMOR))
        current_kills = int(self.game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        
        # Movement rewards
        if action in ['move_forward', 'move_to_armor', 'prioritize_armor']:
            rewards['movement_reward'] = 1.5
            if current_armor > prev_armor:
                rewards['movement_reward'] += 15.0
                rewards['tactical_reward'] += 20.0
                self.armor_pickup_count += 1
                
        # Combat rewards
        if current_kills > prev_kills:
            rewards['combat_reward'] = 12.0
            rewards['tactical_reward'] = 8.0
            self.kill_count += 1
            
        # Survival rewards/penalties
        if current_health < prev_health:
            rewards['tactical_reward'] -= 2.0
        elif action in ['retreat', 'tactical_retreat'] and current_health == prev_health:
            rewards['tactical_reward'] += 1.0
            
        # Base VizDoom reward
        if reward > 0:
            rewards['combat_reward'] += reward / 10.0
            
        self.score += reward
        return rewards
    
    def get_game_stats(self) -> Dict:
        """Return current game statistics"""
        return {
            'score': self.score,
            'armor_pickups': self.armor_pickup_count,
            'enemy_kills': self.kill_count,
            'player_health': self.game_state.player_health,
            'player_armor': self.game_state.player_armor,
            'ammo': self.game_state.ammo,
            'episode_finished': self.game.is_episode_finished()
        }
    
    def close(self):
        """Close VizDoom game"""
        self.game.close()

# Flask Web Server with SocketIO
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'neural-doom-cl1-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

neural_controller = None

class NeuralDOOMController:
    """CL1-enabled Neural DOOM Controller"""
    
    def __init__(self, socketio):
        self.cl1 = create_cl1_device(num_electrodes=600)  # Auto-detects real vs mock
        self.doom = VizDoomInterface()
        self.trainer = SubnetworkTrainer(self.cl1)
        self.decoder = SubnetworkDecoder(self.cl1)
        self.arbiter = ActionArbiter(self.decoder)
        self.socketio = socketio
        
        # Electrode assignments for specialized subnetworks
        self.navigation_electrodes = list(range(0, 200))    # Electrodes 0-199
        self.combat_electrodes = list(range(200, 400))      # Electrodes 200-399
        self.tactical_electrodes = list(range(400, 600))    # Electrodes 400-599
        
        self.running = False
        self.connected = False
        self.start_time = None
        
    def connect_cl1(self):
        """Connect to CL1 device (real or mock)"""
        self.connected = self.cl1.connect()
        if self.connected:
            device_type = "Real CL1" if CL1_AVAILABLE else "Mock CL1"
            logger.info(f"🧠 {device_type} device connected successfully!")
        return self.connected
        
    def start_experiment(self):
        """Start the neural DOOM experiment"""
        if not self.connected:
            return False
        self.running = True
        self.start_time = time.time()
        self.doom.start_episode()
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        logger.info("🎮 Neural DOOM experiment started!")
        return True
        
    def stop_experiment(self):
        """Stop the experiment"""
        self.running = False
        if hasattr(self, 'control_thread'):
            self.control_thread.join(timeout=1.0)
        logger.info("⏹️  Neural DOOM experiment stopped")
            
    def reset_game(self):
        """Reset the game"""
        self.doom.start_episode()
        logger.info("🔄 New DOOM episode started")
        
    def _control_loop(self):
        """Main control loop - biological neural networks control DOOM"""
        frame_count = 0
        while self.running and not self.doom.game.is_episode_finished():
            try:
                # Get current game state
                game_state = self.doom.get_current_state()
                
                # Read neural activities from all three subnetworks
                activities = self.decoder.read_subnetwork_activity(
                    self.navigation_electrodes, self.combat_electrodes, self.tactical_electrodes
                )
                
                # Biological decision making - neural networks vote on actions
                chosen_action = self.arbiter.decide_action(activities, game_state)
                
                # Execute action in VizDoom
                rewards = self.doom.execute_action(chosen_action)
                
                # Train networks based on outcomes (reward-based learning)
                self._train_networks(rewards)
                
                # Send real-time data to dashboard
                data = {
                    'gameState': {
                        'player': {
                            'health': game_state.player_health,
                            'armor': game_state.player_armor,
                            'ammo': game_state.ammo
                        },
                        'armor_visible': game_state.armor_visible,
                        'enemy_visible': game_state.enemy_visible
                    },
                    'neuralActivity': activities,
                    'currentAction': chosen_action,
                    'rewards': rewards,
                    'stats': self.doom.get_game_stats(),
                    'runtime': time.time() - self.start_time if self.start_time else 0,
                    'cl1_connected': CL1_AVAILABLE and isinstance(self.cl1, CL1Device)
                }
                
                self.socketio.emit('update', data)
                frame_count += 1
                
                # Log important events
                if rewards['movement_reward'] > 10:
                    logger.info("🛡️  ARMOR COLLECTED by neural networks!")
                if rewards['combat_reward'] > 10:
                    logger.info("💥 ENEMY ELIMINATED by combat network!")
                
            except Exception as e:
                logging.error(f"Error in control loop: {e}")
                
            time.sleep(1.0 / 15.0)  # 15 FPS
            
        # Episode finished
        if self.doom.game.is_episode_finished():
            logger.info("🏁 DOOM Episode completed by biological neural networks!")
            self.socketio.emit('episode_finished', self.doom.get_game_stats())
                
    def _train_networks(self, rewards):
        """Train the three specialized neural subnetworks"""
        self.trainer.train_navigation_network(self.navigation_electrodes, rewards['movement_reward'])
        self.trainer.train_combat_network(self.combat_electrodes, rewards['combat_reward'])
        self.trainer.train_tactical_network(self.tactical_electrodes, rewards['tactical_reward'])
    
    def close(self):
        """Clean up resources"""
        self.doom.close()
        self.cl1.disconnect()

# Flask routes
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/connect', methods=['POST'])
def connect():
    global neural_controller
    neural_controller = NeuralDOOMController(socketio)
    success = neural_controller.connect_cl1()
    return jsonify({'success': success})

@app.route('/api/start', methods=['POST'])
def start():
    global neural_controller
    if neural_controller:
        success = neural_controller.start_experiment()
        return jsonify({'success': success})
    return jsonify({'success': False})

@app.route('/api/stop', methods=['POST'])
def stop():
    global neural_controller
    if neural_controller:
        neural_controller.stop_experiment()
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/api/reset', methods=['POST'])
def reset():
    global neural_controller
    if neural_controller:
        neural_controller.reset_game()
        return jsonify({'success': True})
    return jsonify({'success': False})

if __name__ == '__main__':
    print("🧠 Neural DOOM Controller with CL1 SDK Integration Starting...")
    print(f"📡 CL1 SDK Available: {CL1_AVAILABLE}")
    print("🎮 Real DOOM gameplay controlled by biological neural networks!")
    print("📊 Dashboard: http://localhost:5000")
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    finally:
        if neural_controller:
            neural_controller.close()
