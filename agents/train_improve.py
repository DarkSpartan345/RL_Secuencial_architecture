# train_improved.py

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from env.gym_env_rl import MultiplicationEnv
from env.reward_logic import RewardCalculator

class BetterRewardCalculator(RewardCalculator):
    """Recompensa mejorada con mejor guidance"""
    
    def __init__(self, bit_width: int = 4, max_cycles: int = 20):
        super().__init__(bit_width, max_cycles)
        # Ajustar pesos
        self.success_bonus = 1000.0         # ↑↑ Mucho más por éxito
        self.correctness_weight = 100.0     # ↑↑ Más por acercarse
        self.efficiency_weight = 2.0        # ↑ Penalizar más los ciclos
        self.timeout_penalty = -200.0       # ↓↓ Penalizar fuerte el timeout
        self.invalid_op_penalty = -10.0     # ↓ Penalizar más operaciones inútiles
        self.progress_bonus = 5.0           # ↑ Más por operaciones productivas
        
    def calculate_reward(self, datapath, a, b, controls):
        reward = 0.0
        state = datapath.get_state()
        
        # 1. Recompensa por corrección (la más importante)
        expected = a * b
        actual = state['reg_p']
        error = abs(expected - actual)
        
        if error == 0:
            # ¡Perfecto! Gran recompensa
            reward += self.correctness_weight
        else:
            # Recompensa proporcional a proximidad
            max_error = (1 << (self.bit_width * 2)) - 1
            proximity = 1.0 - (error / max_error)
            reward += self.correctness_weight * proximity * 0.5
        
        # 2. Penalización por ciclos (queremos eficiencia)
        cycle_penalty = -self.efficiency_weight * (state['cycle_count'] / self.max_cycles)
        reward += cycle_penalty
        
        # 3. Bonus/penalización por operaciones
        if self._is_productive_operation(controls):
            reward += self.progress_bonus
        
        if self._is_invalid_operation(controls, state):
            reward += self.invalid_op_penalty
        
        # 4. Bonus extra por patrones conocidos de multiplicación
        # Si LSB de B es 1 y sumamos A a P
        if state['flag_lsb'] == 1 and controls.get('alu_op') == 0:
            if controls.get('alu_src_a') in [0, 1] and controls.get('write_p') == 1:
                reward += 10.0  # Bonus por suma condicional
        
        # Si hacemos shift en A (multiplicar por 2)
        if controls.get('shift_op') == 1 and controls.get('shift_target') == 1:
            reward += 3.0
        
        # Si hacemos shift en B (dividir por 2)
        if controls.get('shift_op') == 2 and controls.get('shift_target') == 2:
            reward += 3.0
        
        return reward

def create_env(bit_width=4, max_cycles=20):
    env = MultiplicationEnv(
        bit_width=bit_width,
        max_cycles=max_cycles,
        num_states=16
    )
    # Usar recompensa mejorada
    env.reward_calc = BetterRewardCalculator(bit_width, max_cycles)
    return Monitor(env)

def main():
    print("="*60)
    print("🚀 ENTRENAMIENTO MEJORADO - 4 bits")
    print("="*60)
    
    # Crear entornos vectorizados
    vec_env = make_vec_env(
        lambda: create_env(bit_width=4, max_cycles=20),
        n_envs=8
    )
    
    # Entorno de evaluación
    eval_env = make_vec_env(
        lambda: create_env(bit_width=4, max_cycles=20),
        n_envs=1
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models_improved/',
        log_path='./logs_improved/',
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=20
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models_improved/',
        name_prefix='ckpt'
    )
    
    # Crear modelo PPO con hiperparámetros mejorados
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.5,           # ↑ MÁS exploración
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log='./logs_improved/'
    )
    
    print("\n📚 Entrenando por 2M timesteps...")
    print("💡 Tip: Ejecuta 'tensorboard --logdir ./logs_improved/' en otra terminal")
    print()
    
    # Entrenar
    model.learn(
        total_timesteps=2_000_000,
        callback=[eval_callback, checkpoint_callback]
    )
    
    # Guardar modelo final
    model.save("model_improved_4bit_final")
    print("\n✅ Entrenamiento completado!")
    print("📁 Modelo guardado: model_improved_4bit_final.zip")
    
    # Evaluación final
    print("\n" + "="*60)
    print("📊 EVALUACIÓN FINAL")
    print("="*60)
    
    test_env = create_env(bit_width=4, max_cycles=20)
    
    correct = 0
    total = 100
    rewards = []
    cycles = []
    
    for i in range(total):
        obs, info = test_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        if info['correct']:
            correct += 1
        
        rewards.append(episode_reward)
        cycles.append(info['cycle_count'])
        
        if i < 10:  # Mostrar primeros 10
            status = "✓" if info['correct'] else "✗"
            print(f"Test {i+1}: {info['a']} × {info['b']} = {info['actual']} "
                  f"(esperado: {info['expected']}) [{info['cycle_count']} ciclos] {status}")
    
    print("\n" + "="*60)
    print(f"Precisión: {correct/total*100:.1f}% ({correct}/{total})")
    print(f"Recompensa promedio: {np.mean(rewards):.2f}")
    print(f"Ciclos promedio: {np.mean(cycles):.1f}")
    print(f"Ciclos min/max: {min(cycles)}/{max(cycles)}")
    print("="*60)

if __name__ == '__main__':
    main()