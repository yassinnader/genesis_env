# استدعي الكلاس اللي فوق (افترض ان اسمه saved in genesis_env.py)
# from genesis_env import GenesisWorldEnv 
# أو انسخ الكلاس هنا مباشرة

if __name__ == "__main__":
    # 1. تشغيل البيئة في وضع الـ GUI عشان تشوف بعينك
    env = GenesisWorldEnv(render_mode="human")
    obs, _ = env.reset()
    
    print("🌍 Genesis World Created!")
    print("🕷️ Robot is ready.")
    print("🍏 Green cubes are FOOD.")
    
    try:
        for i in range(1000):
            # حركة عشوائية للتجربة
            action = env.action_space.sample()
            
            obs, reward, terminated, _, info = env.step(action)
            
            if i % 10 == 0:
                print(f"Step: {i}, Energy: {info['energy']:.1f}, Reward: {reward:.2f}")
            
            if terminated:
                print("💀 Robot Died (Ran out of energy)!")
                env.reset()
                
            time.sleep(1/30) # تبطيء عشان تلحق تشوف
            
    except KeyboardInterrupt:
        print("🛑 Exiting...")
    
    env.close()