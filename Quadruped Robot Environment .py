import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque
import cv2
import logging
from typing import Tuple, Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuadrupedRobotEnv(gym.Env):
    """
    Enhanced simulation environment for a quadruped robot (built on Ant-v4)
    Modified version: v1.1 (Improved Warmup & Reward Scaling)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    # --- Improvement 1: Adjusted constants for reward balancing ---
    SURVIVAL_REWARD = 0.5
    DISTANCE_REWARD_SCALE = 1.0
    TERMINATION_PENALTY = -10.0  # (Modified) Reduced penalty to encourage risk-taking and movement
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        max_energy: float = 1000.0,
        energy_decay_rate: float = 0.01,
        energy_cost_per_action: float = 0.1,
        exploration_bonus_scale: float = 1.0,
        visited_memory: int = 1000,
        image_size: Tuple[int, int] = (96, 96)
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Create base env (MuJoCo Ant)
        try:
            self.base_env = gym.make('Ant-v4', render_mode=render_mode)
            logger.info("✅ Successfully loaded Ant-v4 environment")
        except Exception as e:
            logger.error(f"❌ Failed to load Ant-v4: {e}")
            raise
        
        # Energy system
        self.max_energy = max_energy
        self.energy = self.max_energy
        self.energy_decay_rate = energy_decay_rate
        self.energy_cost_per_action = energy_cost_per_action
        
        # Curiosity / exploration
        self.visited_deque = deque(maxlen=visited_memory)
        self.visited_set = set()
        self.exploration_bonus_scale = exploration_bonus_scale
        
        # Action space passthrough
        self.num_joints = (
            getattr(self.base_env.action_space, "shape", ())[0] 
            if isinstance(self.base_env.action_space, spaces.Box) 
            else None
        )
        self.action_space = self.base_env.action_space
        
        # Observation space: image + state
        self.image_size = image_size
        base_obs_shape = self.base_env.observation_space.shape[0]
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255, 
                shape=(image_size[0], image_size[1], 3), 
                dtype=np.uint8
            ),
            'state': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(base_obs_shape + 1,),  # +1 for energy
                dtype=np.float32
            )
        })
        
        # Helpers
        self.last_position = np.array([0.0, 0.0], dtype=np.float32)
    
    def _safe_render(self) -> np.ndarray:
        """Attempt to get an image from the environment using multiple methods."""
        blank = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        
        if self.render_mode is None:
            return blank
        
        # 1) Direct attempt
        try:
            img = self.base_env.render()
            if img is not None and isinstance(img, np.ndarray) and img.ndim == 3:
                return cv2.resize(img, (self.image_size[1], self.image_size[0]))
        except Exception:
            pass
        
        # 2) Explicit attempt with render mode parameter
        try:
            img = self.base_env.render(mode="rgb_array")
            if img is not None and isinstance(img, np.ndarray) and img.ndim == 3:
                return cv2.resize(img, (self.image_size[1], self.image_size[0]))
        except Exception:
            pass
        
        # 3) Attempt to access MuJoCo interface directly
        try:
            if hasattr(self.base_env, "sim") and hasattr(self.base_env.sim, "render"):
                img = self.base_env.sim.render(
                    width=self.image_size[1], 
                    height=self.image_size[0]
                )
                if img is not None:
                    return img
        except Exception:
            pass
        
        return blank
    
    def _get_camera_image(self) -> np.ndarray:
        """Get camera image."""
        return self._safe_render()
    
    def _get_observation(self, base_obs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Build observation: state (base_obs + normalized energy) and image.
        
        Args:
            base_obs: Observation from base environment
            
        Returns:
            Dict containing 'image' and 'state'
        """
        # Normalize energy to [0,1]
        energy_normalized = np.array([self.energy / self.max_energy], dtype=np.float32)
        
        if isinstance(base_obs, np.ndarray):
            state = np.concatenate([base_obs.ravel(), energy_normalized], axis=0)
        else:
            state = np.concatenate([
                np.array(base_obs, dtype=np.float32).ravel(), 
                energy_normalized
            ], axis=0)
        
        image = self._get_camera_image()
        
        return {
            'image': image,
            'state': state
        }
    
    def _get_robot_position(self) -> np.ndarray:
        """
        Safe attempt to get robot position (x, y).
        
        Returns:
            np.ndarray: Position [x, y] or [0, 0] on failure
        """
        try:
            unwrapped = getattr(self.base_env, "unwrapped", None)
            if unwrapped is not None and hasattr(unwrapped, 'data'):
                qpos = getattr(unwrapped.data, "qpos", None)
                if qpos is not None and len(qpos) >= 2:
                    return np.array(qpos[:2], dtype=np.float32).copy()
        except Exception:
            pass
        return np.array([0.0, 0.0], dtype=np.float32)
    
    def _update_visited_positions(self, grid_pos: Tuple[float, float]) -> bool:
        """
        Update visited positions efficiently.
        
        Args:
            grid_pos: Grid-rounded position
            
        Returns:
            bool: True if it's a new position
        """
        if grid_pos in self.visited_set:
            return False
        
        # If we've reached max capacity, remove the oldest from set
        if len(self.visited_deque) == self.visited_deque.maxlen and len(self.visited_deque) > 0:
            oldest = self.visited_deque.popleft()  # Using popleft is more appropriate for deque
            self.visited_set.discard(oldest)
        
        self.visited_set.add(grid_pos)
        self.visited_deque.append(grid_pos)
        
        return True
    
    def _consume_energy(self, action: np.ndarray) -> None:
        """
        Consume energy with protection against negative values.
        
        Args:
            action: Executed action
        """
        # Fixed consumption
        self.energy -= self.energy_decay_rate
        
        try:
            # Approximate movement cost
            action_cost = float(np.sum(np.abs(action))) * self.energy_cost_per_action
            self.energy -= action_cost
        except Exception:
            pass
        
        # Protection against negative values
        self.energy = max(0.0, self.energy)
    
    def _calculate_reward(self, action: np.ndarray, base_reward: float) -> float:
        """
        Calculate total reward in an organized and optimized manner.
        
        Args:
            action: Executed action
            base_reward: Reward from base environment
            
        Returns:
            float: Total reward
        """
        reward = 0.0
        
        # 1. Base reward from environment
        reward += float(base_reward)
        
        # 2. Survival reward
        reward += self.SURVIVAL_REWARD
        
        # 3. Curiosity reward (optimized)
        current_pos = self._get_robot_position()
        grid_pos = (
            round(float(current_pos[0]), 1), 
            round(float(current_pos[1]), 1)
        )
        
        if self._update_visited_positions(grid_pos):
            reward += self.exploration_bonus_scale
        
        # 4. Distance moved reward
        distance_moved = float(np.linalg.norm(current_pos - self.last_position))
        reward += distance_moved * self.DISTANCE_REWARD_SCALE
        
        # Update last_position
        self.last_position = current_pos.copy()
        
        return reward
    
    def _is_terminated(self) -> Tuple[bool, float]:
        """
        Check termination conditions.
        
        Returns:
            Tuple[bool, float]: (terminated, penalty)
        """
        if self.energy <= 0:
            return True, self.TERMINATION_PENALTY
        return False, 0.0
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment with improved warmup.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple[observation, info]
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.energy = self.max_energy
        self.visited_deque.clear()
        self.visited_set.clear()
        
        # Reset base environment
        base_obs, base_info = self.base_env.reset(
            seed=seed, 
            options=options if options is not None else {}
        )
        
        # --- Improvement 2: More stable warmup loop ---
        try:
            if isinstance(self.action_space, spaces.Box):
                zero_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
            else:
                zero_action = self.action_space.sample()
            
            # Execute 5 steps to stabilize physics before actual start
            for _ in range(5):
                warmup_obs, _, _, _, _ = self.base_env.step(zero_action)
                base_obs = warmup_obs  # Update observation to latest state
                
        except Exception as e:
            logger.warning(f"Warmup loop failed: {e}")
        
        # Get initial position after stabilization
        self.last_position = self._get_robot_position()
        
        # --- Improvement 3: Register starting point ---
        start_grid_pos = (
            round(float(self.last_position[0]), 1), 
            round(float(self.last_position[1]), 1)
        )
        self._update_visited_positions(start_grid_pos)
        
        observation = self._get_observation(base_obs)
        
        info = {
            'energy': self.energy,
            'step': self.current_step,
            'visited_count': len(self.visited_set),
            **(base_info if isinstance(base_info, dict) else {})
        }
        
        return observation, info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple[observation, reward, terminated, truncated, info]
        """
        # Pass action to base environment
        base_obs, base_reward, base_terminated, base_truncated, base_info = \
            self.base_env.step(action)
        
        # Consume energy (optimized with protection)
        self._consume_energy(action)
        
        # Calculate reward
        reward = self._calculate_reward(action, base_reward)
        
        # Check for termination due to energy
        terminated_by_energy, termination_penalty = self._is_terminated()
        reward += termination_penalty
        
        # Merge termination conditions
        terminated = bool(base_terminated or terminated_by_energy)
        
        self.current_step += 1
        truncated = bool(self.current_step >= self.max_steps or base_truncated)
        
        observation = self._get_observation(base_obs)
        
        info = {
            'energy': self.energy,
            'step': self.current_step,
            'base_reward': float(base_reward),
            'visited_count': len(self.visited_set),
            'terminated_by_energy': terminated_by_energy,
            **(base_info if isinstance(base_info, dict) else {})
        }
        
        return observation, float(reward), terminated, truncated, info
    
    def close(self) -> None:
        """Close the environment safely."""
        try:
            self.base_env.close()
            logger.info("✅ Environment closed successfully")
        except Exception as e:
            logger.error(f"❌ Error while closing environment: {e}")


# ===== Test code to verify modifications =====
if __name__ == "__main__":
    logger.info("🤖 Running Genesis Mind (Ant) environment v1.1...")
    
    try:
        env = QuadrupedRobotEnv(
            render_mode="rgb_array", 
            max_steps=200,
            energy_decay_rate=0.01,
            exploration_bonus_scale=1.0
        )
        
        obs, info = env.reset()
        logger.info("✅ Reset successful with 5 warmup steps")
        logger.info(f"📍 Starting point registered? (Visited Count): {info['visited_count']} (expected 1)")
        logger.info(f"📊 Energy: {info['energy']}")
        
        # Quick test to verify reduced penalty
        logger.info("--- Testing energy consumption ---")
        env.energy = 0.5  # Deliberately reduce energy
        # Random action to deplete energy
        action = env.action_space.sample()
        _, reward, term, _, info = env.step(action)
        
        if term and info['terminated_by_energy']:
            logger.info(f"💀 Robot died due to energy. Final reward: {reward:.2f}")
            logger.info("💡 Note: the value should be close to -10, not -100")

        env.close()
        
    except Exception as e:
        logger.error(f"❌ An error occurred: {e}", exc_info=True)