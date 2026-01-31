import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import pybullet as p
import pybullet_data
import time
import math

# ---------------------------------------------------------
# 1. كلاس العالم (The Environment Class)
# ---------------------------------------------------------
class GenesisWorldEnv(gym.Env):
    """
    بيئة Genesis Mind: عالم ثلاثي الأبعاد يحتوي على روبوت (Ant)
    وموارد طاقة (مكعبات خضراء) يجب تجميعها للبقاء.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, image_size=(96, 96)):
        super().__init__()
        
        self.render_mode = render_mode
        self.image_size = image_size
        
        # الاتصال بمحرك الفيزياء
        if render_mode == "human":
            self.client = p.connect(p.GUI) # شاشة للمشاهدة
        else:
            self.client = p.connect(p.DIRECT) # بدون شاشة للتدريب
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # تعريف الأكشن (8 موتورات للروبوت Ant)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        
        # تعريف الملاحظات (صورة + بيانات داخلية)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8),
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32) 
        })

        # متغيرات اللعبة
        self.max_energy = 1000.0
        self.energy = self.max_energy
        self.energy_decay = 0.5
        self.movement_cost = 0.1
        self.visited_set = set()
        self.food_ids = [] 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        
        # تحميل الأرض
        p.loadURDF("plane.urdf")
        
        # تحميل الروبوت (Ant)
        start_pos = [0, 0, 0.75]
        try:
            self.robot_id = p.loadURDF("ant.urdf", start_pos, useFixedBase=False)
        except:
            # لو ملف ant.urdf مش موجود، نستخدم R2D2 كمثال بديل مؤقت لتجنب الخطأ
            print("Warning: Ant model not found, loading R2D2 instead.")
            self.robot_id = p.loadURDF("r2d2.urdf", start_pos, useFixedBase=False)

        # توزيع الطعام
        self._spawn_resources(count=5)
        
        self.energy = self.max_energy
        self.visited_set.clear()
        
        # انتظار استقرار الفيزياء
        for _ in range(10):
            p.stepSimulation()
            
        return self._get_obs(), {}

    def step(self, action):
        # تطبيق الحركة
        p.setJointMotorControlArray(
            self.robot_id,
            range(8), 
            p.TORQUE_CONTROL,
            forces=action * 30
        )
        
        p.stepSimulation()
        
        # حساب الطاقة والموارد
        self._consume_energy(action)
        reward_resource = self._check_resources()
        
        # المكافآت
        survival_reward = 0.1
        curiosity_reward = self._calculate_curiosity()
        total_reward = survival_reward + curiosity_reward + reward_resource
        
        # شروط الانتهاء
        terminated = False
        if self.energy <= 0:
            terminated = True
            total_reward -= 10
            
        self._update_camera()
        
        return self._get_obs(), total_reward, terminated, False, {'energy': self.energy}

    def _spawn_resources(self, count):
        self.food_ids = []
        for _ in range(count):
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(-5, 5)
            # مكعب طعام
            food_id = p.loadURDF("cube.urdf", [x, y, 0.5], globalScaling=0.5)
            p.changeVisualShape(food_id, -1, rgbaColor=[0, 1, 0, 1]) # أخضر
            self.food_ids.append(food_id)

    def _check_resources(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        reward = 0
        for food_id in self.food_ids[:]:
            food_pos, _ = p.getBasePositionAndOrientation(food_id)
            distance = math.sqrt((robot_pos[0]-food_pos[0])**2 + (robot_pos[1]-food_pos[1])**2)
            if distance < 0.8:
                self.energy = min(self.max_energy, self.energy + 300)
                reward += 5.0
                p.removeBody(food_id)
                self.food_ids.remove(food_id)
                self._spawn_resources(1)
        return reward

    def _consume_energy(self, action):
        motor_effort = np.sum(np.abs(action))
        self.energy -= (self.energy_decay + motor_effort * self.movement_cost)

    def _calculate_curiosity(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        grid_pos = (round(pos[0], 1), round(pos[1], 1))
        if grid_pos not in self.visited_set:
            self.visited_set.add(grid_pos)
            return 0.5
        return 0

    def _update_camera(self):
        if self.render_mode == "human":
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=pos)

    def _get_obs(self):
        # الرؤية
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(pos, 2.0, 0, -30, 0, 2)
        proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)
        _, _, px, _, _ = p.getCameraImage(self.image_size[0], self.image_size[1], view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER)
        image = np.array(px, dtype=np.uint8).reshape((self.image_size[1], self.image_size[0], 4))[:, :, :3]
        
        # الحالة الداخلية
        joint_states = p.getJointStates(self.robot_id, range(8))
        joint_positions = [s[0] for s in joint_states] if joint_states else [0]*8
        
        state = np.concatenate([joint_positions, [self.energy / self.max_energy], pos, ori])
        # ملء المصفوفة لتصل للحجم المطلوب 29
        state = np.pad(state, (0, max(0, 29 - len(state))), 'constant')[:29]
        
        return {'image': image, 'state': state.astype(np.float32)}

    def close(self):
        p.disconnect(self.client)


# ---------------------------------------------------------
# 2. كود التشغيل (Main Execution)
# ---------------------------------------------------------
if __name__ == "__main__":
    # تشغيل البيئة
    print("⏳ Starting Genesis World...")
    try:
        env = GenesisWorldEnv(render_mode="human")
        obs, _ = env.reset()
        
        print("🌍 Genesis World Created Successfully!")
        print("🕷️ Robot Spawned.")
        print("🍏 Green Cubes = Food.")
        print("اضغط Ctrl+C للإيقاف")
        
        step_count = 0
        while True:
            # حركة عشوائية للتجربة (بدون ذكاء لسه)
            action = env.action_space.sample()
            
            obs, reward, terminated, _, info = env.step(action)
            step_count += 1
            
            if step_count % 30 == 0: # طباعة كل ثانية تقريباً
                print(f"Step: {step_count} | Energy: {info['energy']:.1f} | Reward: {reward:.2f}")
            
            if terminated:
                print("💀 Robot Died (No Energy)!")
                env.reset()
                
            time.sleep(1/30) # تبطيء المحاكاة لتناسب العين البشرية
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation Stopped.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        try:
            env.close()
        except:
            pass