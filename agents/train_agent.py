"""
train_agent.py - Script principal para entrenar agentes de RL con SimpleCPUEnv
Soporta PPO (recomendado para MultiDiscrete)
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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Local imports
import sys
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Importar el nuevo entorno
try:
    from env.env_rl_gym import SimpleCPUEnv
except ImportError:
    # Fallback si se ejecuta desde agents/
    sys.path.insert(0, str(Path(project_root) / "env"))
    from env.env_rl_gym import SimpleCPUEnv


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
        self.max_cycles = config.get('max_cycles', 50)
        self.target_value = config.get('target_value', 120) # Por defecto 120
        
        # Training parameters
        self.algorithm = config.get('algorithm', 'PPO')
        self.total_timesteps = config.get('total_timesteps', 1_000_000)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.n_envs = config.get('n_envs', 4)
        
        # Paths
        self.models_dir = Path(config.get('models_dir', 'agents/models'))
        self.logs_dir = Path(config.get('logs_dir', 'logs'))
    
    @staticmethod
    def default_config():
        """Configuración por defecto"""
        return {
            'bit_width': 8,
            'max_cycles': 50,
            'target_value': 120,
            'algorithm': 'PPO',
            'total_timesteps': 1_000_000,
            'learning_rate': 3e-4,
            'n_envs': 4,
            'models_dir': 'agents/models',
            'logs_dir': 'logs'
        }


def create_env(config: TrainingConfig):
    """Crea el entorno de entrenamiento"""
    
    def _init():
        env = SimpleCPUEnv(
            target_value=config.target_value,
            max_cycles=config.max_cycles,
            render_mode=None
        )
        return Monitor(env)
    
    return _init


def train_agent(config: TrainingConfig, resume_path: str = None):
    """
    Entrena un agente de RL para generar programas de SimpleCPU
    """
    
    # Crear directorios
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.algorithm}_SimpleCPU_{timestamp}"
    
    print(f"🚀 Iniciando entrenamiento: {run_name}")
    print(f"   Algoritmo: {config.algorithm}")
    print(f"   Target Value: {config.target_value}")
    
    # Crear entornos vectorizados
    # Usamos DummyVecEnv porque env.reset() puede ser rapido, Subproc a veces tiene overhead
    vec_env = make_vec_env(
        create_env(config),
        n_envs=config.n_envs,
        seed=0,
        vec_env_cls=DummyVecEnv 
    )
    
    # Crear entorno de evaluación
    eval_env = make_vec_env(
        create_env(config),
        n_envs=1,
        seed=100
    )
    
    # Configurar callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // config.n_envs, 1),
        save_path=str(config.models_dir / run_name),
        name_prefix="ckpt"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(config.models_dir / run_name),
        log_path=str(config.logs_dir / run_name),
        eval_freq=max(10_000 // config.n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Crear o cargar modelo
    if resume_path and os.path.exists(resume_path):
        print(f"📂 Cargando modelo desde: {resume_path}")
        if config.algorithm == 'PPO':
            model = PPO.load(resume_path, env=vec_env)
        else:
            raise ValueError("DQN no está completamente soportado para este entorno MultiDiscrete.")
    else:
        print("🆕 Creando nuevo modelo")
        if config.algorithm == 'PPO':
            # USAMOS MultiInputPolicy PORQUE EL OBS SPACE ES UN DICT (registers, pc)
            model = PPO(
                'MultiInputPolicy',
                vec_env,
                learning_rate=config.learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.05,        # Exploración
                verbose=1,
                tensorboard_log=str(config.logs_dir)
            ) 
        else:
             raise ValueError("Por favor usa PPO para este entorno (action space MultiDiscrete).")
    
    # Entrenar
    print("\n🎓 Iniciando entrenamiento...")
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name=run_name
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    # Guardar modelo final
    final_path = config.models_dir / f"{run_name}_final.zip"
    model.save(str(final_path))
    print(f"\n✅ Entrenamiento completado (o interrumpido)!")
    print(f"   Modelo guardado en: {final_path}")
    
    return model, final_path


def evaluate_agent(model_path: str, config: TrainingConfig, n_episodes: int = 10):
    """
    Evalúa un agente entrenado usando SimpleCPUEnv y guarda las soluciones
    """
    print(f"\n📊 Evaluando modelo: {model_path}")
    
    model = PPO.load(model_path)
    
    # Crear entorno
    env = SimpleCPUEnv(
        target_value=config.target_value,
        max_cycles=config.max_cycles,
        render_mode=None
    )
    
    correct_count = 0
    total_rewards = []
    best_cycles = float('inf')
    best_solution_path = None
    
    solutions_dir = Path(project_root) / "hdl" / "solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        program_trace = [] # Lista para guardar las instrucciones de este episodio
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Reconstruct instruction from action manually to save it
            # action is typically numpy array. env.step converts it.
            # We want to store exactly what was executed.
            program_trace.append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        # Verificar éxito
        success = False
        if config.target_value in env.cpu.registers[env.cpu.NUM_CONSTANT_REGISTERS:]:
            success = True
            correct_count += 1
            cycles = env.cpu.cycle_count
            
            # Guardar solución
            filename = f"solution_ep{ep+1}_cycles{cycles}.mem"
            save_path = solutions_dir / filename
            
            # Convert numpy actions to normal list of ints/enums for generator
            formatted_program = []
            for act in program_trace:
                # act is [Op, A, B, D1, D2]
                formatted_program.append(act.tolist())
                
            env.generate_verilog_mem(formatted_program, str(save_path))
            print(f"   💾 Solución guardada: {filename}")
            
            # Actualizar mejor solución
            if cycles < best_cycles:
                best_cycles = cycles
                best_solution_path = save_path
                # Sobrescribir el program.mem principal
                main_mem_path = Path(project_root) / "hdl" / "program.mem"
                env.generate_verilog_mem(formatted_program, str(main_mem_path))
                print(f"   🏆 Nueva mejor solución! Actualizado hdl/program.mem")
            
        print(f"Episodio {ep+1}: Reward={episode_reward:.1f}, Success={success}, Cycles={env.cpu.cycle_count}")
        if not success and env.cpu.cycle_count <= 1:
             print("   ⚠️ El agente terminó inmediatamente (HALT). No aprendió nada aún.")
        
        total_rewards.append(episode_reward)
    
    accuracy = correct_count / n_episodes
    avg_reward = np.mean(total_rewards)
    
    print(f"\n📈 Resultados de Evaluación ({n_episodes} episodios):")
    print(f"   Precisión: {accuracy*100:.1f}%")
    print(f"   Recompensa promedio: {avg_reward:.2f}")
    
    if best_solution_path:
        print(f"   Mejor solución: {best_cycles} ciclos ({best_solution_path.name})")
    else:
        print("\n❌ NO SE GENERARON ARCHIVOS DE MEMORIA.")
        print("   El agente no logró resolver el problema en ningún episodio.")
        print("   Razón más probable: El agente está atrapado en un óptimo local (siempre hace HALT).")
        print("   Solución: Entrenar por más tiempo o ajustar la función de recompensa.")


def main():
    parser = argparse.ArgumentParser(description='Entrenar agente para SimpleCPU')
    parser.add_argument('--config', type=str, help='Path al archivo de configuración YAML')
    parser.add_argument('--algorithm', type=str, default='PPO')
    parser.add_argument('--timesteps', type=int, default=100_000)
    parser.add_argument('--resume', type=str, help='Path para continuar entrenamiento')
    parser.add_argument('--eval-only', type=str, help='Solo evaluar modelo existente')
    parser.add_argument('--target', type=int, default=120, help='Valor objetivo a encontrar')
    parser.add_argument('--n-eval', type=int, default=10, help='Número de episodios de evaluación')
    
    args = parser.parse_args()
    
    config = TrainingConfig(args.config)
    
    # ClI overrides
    if args.algorithm:
        config.algorithm = args.algorithm
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.target:
        config.target_value = args.target

    if args.eval_only:
        evaluate_agent(args.eval_only, config, n_episodes=args.n_eval)
        return
    
    model, model_path = train_agent(config, args.resume)
    
    # Evaluar modelo entrenado
    print("\n" + "="*60)
    evaluate_agent(str(model_path), config, n_episodes=args.n_eval)
    

if __name__ == '__main__':
    main()
