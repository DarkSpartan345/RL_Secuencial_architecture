"""
gym_env.py - Entorno Gymnasium para entrenar agentes RL
Envuelve el datapath y define observaciones, acciones y recompensas
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any
from env.datapath import Datapath
from env.reward_logic import RewardCalculator

class MultiplicationEnv(gym.Env):
    """
    Entorno para aprender algoritmos de multiplicación
    
    Observation Space: Estado del datapath + PC
    Action Space: Señales de control del hardware (Multi-Discrete)
    """
    
    metadata = {'render_modes': []}
    
    def __init__(self, 
                 bit_width: int = 8,
                 max_cycles: int = 32,
                 num_states: int = 16):
        super().__init__()
        
        self.bit_width = bit_width
        self.max_cycles = max_cycles
        self.num_states = num_states
        
        # Inicializar datapath
        self.datapath = Datapath(bit_width)
        self.reward_calc = RewardCalculator(bit_width, max_cycles)
        
        # Estado del programa
        self.pc = 0  # Program Counter (0-15 para 16 estados)
        self.current_a = 0
        self.current_b = 0
        self.done = False
        
        # Definir espacios
        self._setup_spaces()
        
    def _setup_spaces(self):
        """Configura los espacios de observación y acción"""
        
        # Observation space: registros + flags + PC
        # Normalizamos todo a [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),  # reg_a, reg_b, reg_p, reg_temp, flag_z, flag_lsb, flag_c
            dtype=np.float32
        )
        
        # Action space: Señales de control (Multi-Discrete)
        # Formato: [alu_op, alu_src_a, alu_src_b, shift_op, shift_target, write_p, write_temp, write_b, next_state]
        self.action_space = spaces.MultiDiscrete([
            7,  # alu_op: 0-6 (ADD, SUB, AND, OR, XOR, PASS_A, PASS_B)
            3,  # alu_src_a: 0-2 (reg_a, reg_p, reg_temp)
            4,  # alu_src_b: 0-3 (reg_b, reg_p, reg_temp, 0)
            5,  # shift_op: 0-4 (NONE, SHL, SHR, ROL, ROR)
            4,  # shift_target: 0-3 (NONE, A, B, P)
            2,  # write_p: 0-1
            2,  # write_temp: 0-1
            2,  # write_b: 0-1
            16  # next_state: 0-15 (siguiente estado de la FSM)
        ])
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Resetea el entorno con nuevos valores de multiplicación"""
        super().reset(seed=seed)
        
        # Generar nuevos valores aleatorios para multiplicar
        if options and 'a' in options and 'b' in options:
            self.current_a = options['a']
            self.current_b = options['b']
        else:
            self.current_a = self.np_random.integers(0, 2**self.bit_width)
            self.current_b = self.np_random.integers(0, 2**self.bit_width)
        
        # Resetear datapath
        self.datapath.reset(self.current_a, self.current_b)
        
        # Resetear estado de control
        self.pc = 0
        self.done = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Ejecuta una acción en el entorno
        
        Args:
            action: Array con las señales de control
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")
        
        # Decodificar acción
        controls = self._decode_action(action)
        
        # Ejecutar en el datapath
        state = self.datapath.execute_operation(controls)
        
        # Actualizar PC
        self.pc = controls['next_state']
        
        # Calcular recompensa
        reward = self.reward_calc.calculate_reward(
            self.datapath,
            self.current_a,
            self.current_b,
            controls
        )
        
        # Verificar terminación
        terminated = False
        truncated = False
        
        # Terminación exitosa: llegó al estado especial de HALT (estado 15)
        if self.pc == 15:
            terminated = True
            # Bonus si la respuesta es correcta
            if self.datapath.verify_multiplication(self.current_a, self.current_b):
                reward += self.reward_calc.success_bonus
        
        # Truncamiento: excedió máximo de ciclos
        if self.datapath.cycle_count >= self.max_cycles:
            truncated = True
            reward += self.reward_calc.timeout_penalty
        
        self.done = terminated or truncated
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _decode_action(self, action: np.ndarray) -> Dict[str, int]:
        """Convierte el vector de acción en señales de control"""
        return {
            'alu_op': int(action[0]),
            'alu_src_a': int(action[1]),
            'alu_src_b': int(action[2]),
            'shift_op': int(action[3]),
            'shift_target': int(action[4]),
            'write_p': int(action[5]),
            'write_temp': int(action[6]),
            'write_b': int(action[7]),
            'next_state': int(action[8])
        }
    
    def _get_observation(self) -> np.ndarray:
        """Construye el vector de observación normalizado"""
        state = self.datapath.get_state()
        
        max_val = 2**self.bit_width - 1
        
        obs = np.array([
            state['reg_a'] / max_val,
            state['reg_b'] / max_val,
            state['reg_p'] / max_val,
            state['reg_temp'] / max_val,
            float(state['flag_z']),
            float(state['flag_lsb']),
            float(state['flag_c'])
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Información adicional para debugging"""
        state = self.datapath.get_state()
        return {
            'pc': self.pc,
            'cycle_count': state['cycle_count'],
            'a': self.current_a,
            'b': self.current_b,
            'expected': self.current_a * self.current_b,
            'actual': state['reg_p'],
            'correct': self.datapath.verify_multiplication(self.current_a, self.current_b)
        }
    
    def render(self):
        """Renderiza el estado actual (opcional)"""
        state = self.datapath.get_state()
        print(f"PC={self.pc} | A={state['reg_a']:3d} B={state['reg_b']:3d} "
              f"P={state['reg_p']:3d} TEMP={state['reg_temp']:3d} | "
              f"Z={state['flag_z']} LSB={state['flag_lsb']} C={state['flag_c']} | "
              f"Cycle={state['cycle_count']}")
