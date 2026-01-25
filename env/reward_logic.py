"""
reward_logic.py - Definición de la función de recompensa
Incentiva eficiencia, corrección y descubrimiento de algoritmos válidos
"""
from typing import Dict
from env.datapath import Datapath

class RewardCalculator:
    """Calcula recompensas para guiar el aprendizaje"""
    
    def __init__(self, 
                 bit_width: int = 8,
                 max_cycles: int = 32):
        self.bit_width = bit_width
        self.max_cycles = max_cycles
        
        # Parámetros de recompensa
        self.success_bonus = 100.0          # Bonus por multiplicación correcta
        self.correctness_weight = 10.0      # Peso para proximidad al resultado
        self.efficiency_weight = 1.0        # Peso para eficiencia (menos ciclos)
        self.timeout_penalty = -50.0        # Penalización por timeout
        self.invalid_op_penalty = -1.0      # Penalización por operaciones inútiles
        self.progress_bonus = 0.5           # Bonus por acercarse al resultado
        
    def calculate_reward(self,
                        datapath: Datapath,
                        a: int,
                        b: int,
                        controls: Dict[str, int]) -> float:
        """
        Calcula la recompensa para un paso dado
        
        Args:
            datapath: Estado actual del datapath
            a: Multiplicando
            b: Multiplicador
            controls: Señales de control ejecutadas
            
        Returns:
            Recompensa escalar
        """
        reward = 0.0
        state = datapath.get_state()
        
        # 1. Recompensa por corrección
        expected = (a * b) & ((1 << self.bit_width) - 1)
        actual = state['reg_p']
        
        # Error absoluto
        error = abs(expected - actual)
        max_error = (1 << self.bit_width) - 1
        
        # Recompensa inversamente proporcional al error
        if error == 0:
            # Multiplicación perfecta
            correctness_reward = self.correctness_weight
        else:
            # Recompensa proporcional a qué tan cerca está
            correctness_reward = self.correctness_weight * (1.0 - error / max_error)
        
        reward += correctness_reward
        
        # 2. Recompensa por eficiencia (penalizar ciclos innecesarios)
        # Menos ciclos = mejor (pero no penalizar demasiado al principio)
        if state['cycle_count'] > 0:
            efficiency_reward = -self.efficiency_weight * (state['cycle_count'] / self.max_cycles)
            reward += efficiency_reward
        
        # 3. Penalización por operaciones inválidas o inútiles
        if self._is_invalid_operation(controls, state):
            reward += self.invalid_op_penalty
        
        # 4. Bonus por progreso (mejorar el resultado)
        # Esto requiere tracking del estado anterior, lo implementaremos simple
        if self._is_productive_operation(controls):
            reward += self.progress_bonus
        
        return reward
    
    def _is_invalid_operation(self, controls: Dict[str, int], state: Dict[str, int]) -> bool:
        """
        Detecta operaciones claramente inválidas o inútiles
        
        Por ejemplo:
        - Escribir en múltiples destinos simultáneamente sin sentido
        - Operaciones que no modifican estado
        """
        # Operación NOP (no operation): no escribe a ningún registro
        if (controls['write_p'] == 0 and 
            controls['write_temp'] == 0 and 
            controls['write_b'] == 0 and
            controls['shift_op'] == 0):
            return True
        
        # Shift sin target
        if controls['shift_op'] > 0 and controls['shift_target'] == 0:
            return True
        
        return False
    
    def _is_productive_operation(self, controls: Dict[str, int]) -> bool:
        """
        Detecta si una operación es potencialmente productiva
        
        Operaciones productivas típicamente:
        - Usan ADD para acumular
        - Hacen shift para multiplicación por 2
        - Decrementan contadores
        """
        # ADD operation es típicamente útil
        if controls['alu_op'] == 0 and controls['write_p'] == 1:
            return True
        
        # Shift operations son útiles para multiplicación
        if controls['shift_op'] in [1, 2] and controls['shift_target'] > 0:
            return True
        
        # SUB para decrementar contadores
        if controls['alu_op'] == 1 and controls['write_b'] == 1:
            return True
        
        return False


class SparseRewardCalculator(RewardCalculator):
    """
    Variante con recompensa sparse - solo al final
    Útil para algoritmos más avanzados como PPO
    """
    
    def calculate_reward(self,
                        datapath: Datapath,
                        a: int,
                        b: int,
                        controls: Dict[str, int]) -> float:
        """Solo da recompensa significativa al terminar"""
        
        # Pequeñas penalizaciones por ciclos
        reward = -0.1
        
        # Penalización fuerte por operaciones inválidas
        if self._is_invalid_operation(controls, datapath.get_state()):
            reward -= 1.0
        
        return reward


class ShapedRewardCalculator(RewardCalculator):
    """
    Variante con reward shaping más agresivo
    Guía al agente con más señales intermedias
    """
    
    def __init__(self, bit_width: int = 8, max_cycles: int = 32):
        super().__init__(bit_width, max_cycles)
        self.previous_error = None
        
    def calculate_reward(self,
                        datapath: Datapath,
                        a: int,
                        b: int,
                        controls: Dict[str, int]) -> float:
        """Recompensa con shaping agresivo"""
        
        reward = super().calculate_reward(datapath, a, b, controls)
        
        state = datapath.get_state()
        expected = (a * b) & ((1 << self.bit_width) - 1)
        current_error = abs(expected - state['reg_p'])
        
        # Recompensa por reducir el error
        if self.previous_error is not None:
            if current_error < self.previous_error:
                reward += 5.0  # Bonus por mejorar
            elif current_error > self.previous_error:
                reward -= 2.0  # Penalización por empeorar
        
        self.previous_error = current_error
        
        # Bonus extra por patrones conocidos de multiplicación
        # Ej: si B es impar y se suma A a P
        if state['flag_lsb'] == 1 and controls['alu_op'] == 0:  # ADD
            reward += 1.0
        
        # Bonus por hacer shift después de procesar un bit
        if controls['shift_op'] == 2 and controls['shift_target'] == 2:  # SHR en B
            reward += 0.5
        
        return reward
    
    def reset(self):
        """Resetea el estado entre episodios"""
        self.previous_error = None
