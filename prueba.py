import numpy as np
from stable_baselines3 import PPO
from env.gym_env_rl import MultiplicationEnv
import time

def run_random_inference(model_path, n_tests=20, bit_width=4):
    # 1. Cargar el cerebro entrenado
    model = PPO.load(model_path)
    
    # 2. Crear el entorno de hardware (limpio)
    env = MultiplicationEnv(bit_width=bit_width, max_cycles=24)
    
    success_count = 0
    print(f"{'TEST':<5} | {'OPERACIÓN':<10} | {'RESULTADO':<10} | {'PASOS':<6} | {'STATUS'}")
    print("-" * 60)

    for i in range(n_tests):
        # Generar datos aleatorios
        a = np.random.randint(0, 2**bit_width)
        b = np.random.randint(0, 2**bit_width)
        expected = a * b
        
        obs, info = env.reset(options={'a': a, 'b': b})
        done = False
        steps = 0
        
        while not done:
            # Pedir al modelo la mejor acción (deterministic=True)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
        actual = info['actual']
        status = "✅ OK" if info['correct'] else "❌ FAIL"
        if info['correct']: success_count += 1
        
        print(f"{i+1:<5} | {a:2d} x {b:2d} = {expected:3d} | Actual: {actual:3d} | {steps:5d} | {status}")

    print("-" * 60)
    print(f"PRECISIÓN FINAL: {(success_count/n_tests)*100:.2f}%")

# Uso:
run_random_inference("agents/models/PPO_4bit_20260128_141525_final", n_tests=50)