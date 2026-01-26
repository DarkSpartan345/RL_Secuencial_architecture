"""
train.py - Script principal para entrenar agentes de RL
Soporta PPO y DQN usando Stable-Baselines3
"""
import argparse
import os
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime

# RL Libraries
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from env.gym_env_rl import MultiplicationEnv
from env.reward_logic import RewardCalculator, SparseRewardCalculator, ShapedRewardCalculator


class TrainingConfig:
    """Configuración de entrenamiento"""
    
    def __init__(self, config_path: str = None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self.default_config()
        
        # Hardware parameters
        self.bit_width = config.get('bit_width', 8)
        self.max_cycles = config.get('max_cycles', 32)
        self.num_states = config.get('num_states', 16)
        
        # Training parameters
        self.algorithm = config.get('algorithm', 'PPO')
        self.total_timesteps = config.get('total_timesteps', 1_000_000)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.n_envs = config.get('n_envs', 4)
        
        # Reward type
        self.reward_type = config.get('reward_type', 'standard')
        
        # Paths
        self.models_dir = Path(config.get('models_dir', 'agents/models'))
        self.logs_dir = Path(config.get('logs_dir', 'logs'))
    
    @staticmethod
    def default_config():
        """Configuración por defecto"""
        return {
            'bit_width': 8,
            'max_cycles': 32,
            'num_states': 16,
            'algorithm': 'PPO',
            'total_timesteps': 1_000_000,
            'learning_rate': 3e-4,
            'n_envs': 4,
            'reward_type': 'standard',
            'models_dir': 'agents/models',
            'logs_dir': 'logs'
        }


def create_env(config: TrainingConfig):
    """Crea el entorno de entrenamiento"""
    
    def _init():
        # Configurar el calculador de recompensa según el tipo
        if config.reward_type == 'sparse':
            reward_calc = SparseRewardCalculator(config.bit_width, config.max_cycles)
        elif config.reward_type == 'shaped':
            reward_calc = ShapedRewardCalculator(config.bit_width, config.max_cycles)
        else:
            reward_calc = RewardCalculator(config.bit_width, config.max_cycles)
        
        env = MultiplicationEnv(
            bit_width=config.bit_width,
            max_cycles=config.max_cycles,
            num_states=config.num_states
        )
        
        # Reemplazar el reward calculator
        env.reward_calc = reward_calc
        
        return Monitor(env)
    
    return _init


def train_agent(config: TrainingConfig, resume_path: str = None):
    """
    Entrena un agente de RL
    
    Args:
        config: Configuración de entrenamiento
        resume_path: Path para continuar entrenamiento desde checkpoint
    """
    
    # Crear directorios
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.algorithm}_{config.bit_width}bit_{timestamp}"
    
    print(f"🚀 Iniciando entrenamiento: {run_name}")
    print(f"   Algoritmo: {config.algorithm}")
    print(f"   Bit width: {config.bit_width}")
    print(f"   Max cycles: {config.max_cycles}")
    print(f"   Total timesteps: {config.total_timesteps:,}")
    print(f"   Reward type: {config.reward_type}")
    
    # Crear entornos vectorizados
    vec_env = make_vec_env(
        create_env(config),
        n_envs=config.n_envs,
        seed=0
    )
    
    # Crear entorno de evaluación
    eval_env = make_vec_env(
        create_env(config),
        n_envs=1,
        seed=100
    )
    
    # Configurar callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // config.n_envs, 1),
        save_path=str(config.models_dir / run_name),
        name_prefix="ckpt"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(config.models_dir / run_name),
        log_path=str(config.logs_dir / run_name),
        eval_freq=max(5_000 // config.n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    # Crear o cargar modelo
    if resume_path and os.path.exists(resume_path):
        print(f"📂 Cargando modelo desde: {resume_path}")
        if config.algorithm == 'PPO':
            model = PPO.load(resume_path, env=vec_env)
        else:
            model = DQN.load(resume_path, env=vec_env)
    else:
        print("🆕 Creando nuevo modelo")
        if config.algorithm == 'PPO':
            model = PPO(
                'MlpPolicy',
                vec_env,
                learning_rate=config.learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log=str(config.logs_dir)
            )
        else:  # DQN
            model = DQN(
                'MlpPolicy',
                vec_env,
                learning_rate=config.learning_rate,
                buffer_size=50_000,
                learning_starts=1000,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=0.3,
                exploration_final_eps=0.05,
                verbose=1,
                tensorboard_log=str(config.logs_dir)
            )
    
    # Entrenar
    print("\n🎓 Iniciando entrenamiento...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=run_name
    )
    
    # Guardar modelo final
    final_path = config.models_dir / f"{run_name}_final.zip"
    model.save(str(final_path))
    print(f"\n✅ Entrenamiento completado!")
    print(f"   Modelo guardado en: {final_path}")
    
    return model, final_path


def evaluate_agent(model_path: str, config: TrainingConfig, n_episodes: int = 100):
    """
    Evalúa un agente entrenado
    
    Args:
        model_path: Path al modelo entrenado
        config: Configuración
        n_episodes: Número de episodios de evaluación
    """
    print(f"\n📊 Evaluando modelo: {model_path}")
    
    # Cargar modelo
    if config.algorithm == 'PPO':
        model = PPO.load(model_path)
    else:
        model = DQN.load(model_path)
    
    # Crear entorno
    env = MultiplicationEnv(
        bit_width=config.bit_width,
        max_cycles=config.max_cycles,
        num_states=config.num_states
    )
    
    # Estadísticas
    correct_count = 0
    total_rewards = []
    cycle_counts = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        if info['correct']:
            correct_count += 1
        
        total_rewards.append(episode_reward)
        cycle_counts.append(info['cycle_count'])
    
    # Resultados
    accuracy = correct_count / n_episodes
    avg_reward = np.mean(total_rewards)
    avg_cycles = np.mean(cycle_counts)
    
    print(f"\n📈 Resultados de Evaluación ({n_episodes} episodios):")
    print(f"   Precisión: {accuracy*100:.1f}% ({correct_count}/{n_episodes})")
    print(f"   Recompensa promedio: {avg_reward:.2f}")
    print(f"   Ciclos promedio: {avg_cycles:.1f}")
    print(f"   Ciclos min/max: {min(cycle_counts)}/{max(cycle_counts)}")
    
    return {
        'accuracy': accuracy,
        'avg_reward': avg_reward,
        'avg_cycles': avg_cycles,
        'correct_count': correct_count,
        'total_episodes': n_episodes
    }


def main():
    parser = argparse.ArgumentParser(description='Entrenar agente de multiplicación con RL')
    parser.add_argument('--config', type=str, help='Path al archivo de configuración YAML')
    parser.add_argument('--algorithm', type=str, choices=['PPO', 'DQN'], default='PPO')
    parser.add_argument('--bit-width', type=int, default=8)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--reward-type', type=str, choices=['standard', 'sparse', 'shaped'], 
                       default='standard')
    parser.add_argument('--resume', type=str, help='Path para continuar entrenamiento')
    parser.add_argument('--eval-only', type=str, help='Solo evaluar modelo existente')
    parser.add_argument('--n-eval', type=int, default=100, help='Episodios para evaluación')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = TrainingConfig(args.config)
    
    # Sobrescribir con argumentos CLI
    if args.algorithm:
        config.algorithm = args.algorithm
    if args.bit_width:
        config.bit_width = args.bit_width
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.reward_type:
        config.reward_type = args.reward_type
    
    # Modo evaluación
    if args.eval_only:
        evaluate_agent(args.eval_only, config, args.n_eval)
        return
    
    # Entrenar
    model, model_path = train_agent(config, args.resume)
    
    # Evaluar modelo entrenado
    print("\n" + "="*60)
    evaluate_agent(str(model_path), config, args.n_eval)


if __name__ == '__main__':
    main()
