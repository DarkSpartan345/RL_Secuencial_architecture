"""
extract_fsm.py - Extrae la FSM aprendida y genera ROM para Verilog
Este es el script CRUCIAL que materializa la investigación
"""
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

from stable_baselines3 import PPO, DQN

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from env.gym_env import MultiplicationEnv


class FSMExtractor:
    """Extrae la FSM de un agente entrenado"""
    
    def __init__(self, 
                 model_path: str,
                 algorithm: str = 'PPO',
                 bit_width: int = 8,
                 num_states: int = 16):
        
        self.bit_width = bit_width
        self.num_states = num_states
        
        # Cargar modelo
        print(f"📦 Cargando modelo desde: {model_path}")
        if algorithm.upper() == 'PPO':
            self.model = PPO.load(model_path)
        else:
            self.model = DQN.load(model_path)
        
        # Crear entorno para simulación
        self.env = MultiplicationEnv(
            bit_width=bit_width,
            max_cycles=64,
            num_states=num_states
        )
        
        # FSM extraída: dict[state] -> dict[flags] -> action
        self.fsm_table = {}
    
    def extract_state_transitions(self, num_samples: int = 1000):
        """
        Extrae las transiciones de estado explorando el espacio
        
        Estrategia:
        - Para cada estado (0-15)
        - Para cada combinación relevante de flags
        - Ejecutar el agente y ver qué acción toma
        """
        print(f"\n🔍 Extrayendo transiciones de estado...")
        print(f"   Estados: {self.num_states}")
        print(f"   Muestras por estado: {num_samples // self.num_states}")
        
        samples_per_state = max(1, num_samples // self.num_states)
        
        for state in range(self.num_states):
            print(f"   Procesando estado {state}...")
            
            state_actions = []
            
            for _ in range(samples_per_state):
                # Generar valores aleatorios para multiplicar
                a = np.random.randint(0, 2**self.bit_width)
                b = np.random.randint(0, 2**self.bit_width)
                
                # Resetear entorno
                self.env.reset(options={'a': a, 'b': b})
                
                # Simular hasta llegar al estado deseado
                # (En práctica, necesitamos una forma de forzar estados)
                obs = self._simulate_to_state(state, a, b)
                
                if obs is not None:
                    # Obtener acción del modelo
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # Guardar el contexto y la acción
                    state_actions.append({
                        'observation': obs,
                        'action': action,
                        'a': a,
                        'b': b
                    })
            
            # Analizar las acciones para este estado
            if state_actions:
                self.fsm_table[state] = self._analyze_state_actions(state, state_actions)
    
    def _simulate_to_state(self, target_state: int, a: int, b: int) -> np.ndarray:
        """
        Simula hasta llegar a un estado específico
        Retorna la observación en ese estado
        """
        self.env.reset(options={'a': a, 'b': b})
        obs = self.env._get_observation()
        
        # Para simplificar, ejecutamos pasos hasta llegar al estado
        # En producción, esto requeriría control más fino
        max_steps = 32
        
        for _ in range(max_steps):
            if self.env.pc == target_state:
                return obs
            
            # Ejecutar una acción
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
            
            if terminated or truncated:
                break
        
        # Si llegamos aquí sin alcanzar el estado, retornar None
        if self.env.pc == target_state:
            return obs
        return None
    
    def _analyze_state_actions(self, state: int, actions_list: List[Dict]) -> Dict:
        """
        Analiza las acciones tomadas en un estado y extrae el patrón
        """
        if not actions_list:
            return self._default_action()
        
        # Estrategia simple: acción más común
        actions = np.array([a['action'] for a in actions_list])
        
        # Para cada componente de la acción, tomar la moda
        action_components = []
        for i in range(actions.shape[1]):
            values, counts = np.unique(actions[:, i], return_counts=True)
            most_common = values[np.argmax(counts)]
            action_components.append(int(most_common))
        
        return {
            'action': action_components,
            'confidence': np.max(counts) / len(actions_list),
            'samples': len(actions_list)
        }
    
    def _default_action(self) -> Dict:
        """Acción por defecto para estados no visitados"""
        return {
            'action': [0, 0, 0, 0, 0, 0, 0, 0, 0],  # NOP + stay in state
            'confidence': 0.0,
            'samples': 0
        }
    
    def generate_verilog_rom(self, output_path: str):
        """
        Genera el archivo ROM para Verilog
        
        Formato: cada línea es una instrucción de 23 bits en hexadecimal
        """
        print(f"\n📝 Generando ROM de Verilog...")
        
        rom_lines = []
        
        for state in range(self.num_states):
            if state in self.fsm_table:
                action = self.fsm_table[state]['action']
                confidence = self.fsm_table[state]['confidence']
            else:
                action = self._default_action()['action']
                confidence = 0.0
            
            # Codificar acción en bits
            instruction = self._encode_instruction(action)
            
            # Convertir a hexadecimal
            hex_instr = f"{instruction:06X}"
            
            # Comentario con la decodificación
            comment = self._decode_instruction_comment(action, confidence)
            
            rom_lines.append(f"{hex_instr}  // State {state:2d}: {comment}")
        
        # Escribir archivo
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("// ROM generada automáticamente por FSM Extractor\n")
            f.write(f"// Bit width: {self.bit_width}, States: {self.num_states}\n")
            f.write("// Formato: [next_state(4) | write_b | write_temp | write_p | shift_target(2) | shift_op(3) | alu_src_b(2) | alu_src_a(2) | alu_op(3)]\n\n")
            for line in rom_lines:
                f.write(line + '\n')
        
        print(f"   ✅ ROM guardada en: {output_path}")
    
    def _encode_instruction(self, action: List[int]) -> int:
        """
        Codifica una acción en un número de 23 bits
        
        Formato (LSB primero):
        - alu_op: 3 bits (0-2)
        - alu_src_a: 2 bits (3-4)
        - alu_src_b: 2 bits (5-6)
        - shift_op: 3 bits (7-9)
        - shift_target: 2 bits (10-11)
        - write_p: 1 bit (12)
        - write_temp: 1 bit (13)
        - write_b: 1 bit (14)
        - next_state: 4 bits (15-18)
        """
        instruction = 0
        instruction |= (action[0] & 0x7)        # alu_op [2:0]
        instruction |= (action[1] & 0x3) << 3   # alu_src_a [4:3]
        instruction |= (action[2] & 0x3) << 5   # alu_src_b [6:5]
        instruction |= (action[3] & 0x7) << 7   # shift_op [9:7]
        instruction |= (action[4] & 0x3) << 10  # shift_target [11:10]
        instruction |= (action[5] & 0x1) << 12  # write_p [12]
        instruction |= (action[6] & 0x1) << 13  # write_temp [13]
        instruction |= (action[7] & 0x1) << 14  # write_b [14]
        instruction |= (action[8] & 0xF) << 15  # next_state [18:15]
        
        return instruction
    
    def _decode_instruction_comment(self, action: List[int], confidence: float) -> str:
        """Genera un comentario legible de la instrucción"""
        alu_ops = ['ADD', 'SUB', 'AND', 'OR', 'XOR', 'PASS_A', 'PASS_B']
        shift_ops = ['NONE', 'SHL', 'SHR', 'ROL', 'ROR']
        
        parts = []
        
        # ALU operation
        if action[5] or action[6] or action[7]:  # Si escribe algo
            alu_op = alu_ops[action[0]] if action[0] < len(alu_ops) else f"OP{action[0]}"
            parts.append(f"{alu_op}")
        
        # Shift operation
        if action[3] > 0 and action[4] > 0:
            shift_op = shift_ops[action[3]] if action[3] < len(shift_ops) else f"SH{action[3]}"
            targets = ['', 'A', 'B', 'P']
            parts.append(f"{shift_op}_{targets[action[4]]}")
        
        # Next state
        parts.append(f"→{action[8]}")
        
        # Confidence
        parts.append(f"[{confidence:.0%}]")
        
        return ' '.join(parts)
    
    def generate_analysis_report(self, output_path: str):
        """Genera un reporte de análisis de la FSM extraída"""
        report = {
            'bit_width': self.bit_width,
            'num_states': self.num_states,
            'states': {}
        }
        
        for state in range(self.num_states):
            if state in self.fsm_table:
                report['states'][state] = {
                    'action': self.fsm_table[state]['action'],
                    'confidence': float(self.fsm_table[state]['confidence']),
                    'samples': self.fsm_table[state]['samples']
                }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✅ Reporte guardado en: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extraer FSM de agente entrenado')
    parser.add_argument('model_path', type=str, help='Path al modelo entrenado (.zip)')
    parser.add_argument('--algorithm', type=str, choices=['PPO', 'DQN'], default='PPO')
    parser.add_argument('--bit-width', type=int, default=8)
    parser.add_argument('--num-states', type=int, default=16)
    parser.add_argument('--samples', type=int, default=1000, 
                       help='Número de muestras para extracción')
    parser.add_argument('--output', type=str, default='hdl/rom_data.mem',
                       help='Path de salida para ROM')
    parser.add_argument('--report', type=str, default='hdl/fsm_report.json',
                       help='Path para reporte de análisis')
    
    args = parser.parse_args()
    
    # Crear extractor
    extractor = FSMExtractor(
        model_path=args.model_path,
        algorithm=args.algorithm,
        bit_width=args.bit_width,
        num_states=args.num_states
    )
    
    # Extraer FSM
    extractor.extract_state_transitions(num_samples=args.samples)
    
    # Generar outputs
    extractor.generate_verilog_rom(args.output)
    extractor.generate_analysis_report(args.report)
    
    print("\n✨ Extracción completada exitosamente!")


if __name__ == '__main__':
    main()
