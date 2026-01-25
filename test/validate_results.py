"""
validate_results.py - Valida que Python y Verilog produzcan los mismos resultados
Compara la simulación Python con los resultados de la síntesis de Verilog
"""
import argparse
import numpy as np
import subprocess
from pathlib import Path
import re
from typing import List, Tuple, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.datapath import Datapath
from stable_baselines3 import PPO, DQN


class VerilogSimulator:
    """Wrapper para ejecutar simulaciones de Verilog"""
    
    def __init__(self, testbench_dir: Path):
        self.testbench_dir = testbench_dir
        self.vcd_file = testbench_dir / "multiplier_tb.vcd"
        
    def run_simulation(self) -> bool:
        """Ejecuta la simulación de Verilog con iverilog"""
        print("🔧 Compilando Verilog...")
        
        verilog_files = [
            self.testbench_dir / "../hdl/datapath.v",
            self.testbench_dir / "../hdl/control_unit.v",
            self.testbench_dir / "../hdl/top.v",
            self.testbench_dir / "test_bench.v"
        ]
        
        # Verificar que los archivos existan
        for f in verilog_files:
            if not f.exists():
                print(f"❌ Error: No se encuentra {f}")
                return False
        
        # Compilar con iverilog
        try:
            compile_cmd = ["iverilog", "-o", "sim.vvp"] + [str(f) for f in verilog_files]
            result = subprocess.run(
                compile_cmd,
                cwd=self.testbench_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print("❌ Error de compilación:")
                print(result.stderr)
                return False
            
            print("✅ Compilación exitosa")
            
            # Ejecutar simulación
            print("🚀 Ejecutando simulación...")
            result = subprocess.run(
                ["vvp", "sim.vvp"],
                cwd=self.testbench_dir,
                capture_output=True,
                text=True
            )
            
            print(result.stdout)
            
            if result.returncode != 0:
                print("❌ Error de simulación:")
                print(result.stderr)
                return False
            
            return True
            
        except FileNotFoundError:
            print("❌ Error: iverilog no está instalado")
            print("   Instalar con: sudo apt-get install iverilog")
            return False
    
    def parse_simulation_results(self, output: str) -> List[Dict]:
        """Parsea los resultados de la simulación"""
        results = []
        
        # Buscar líneas de resultados
        pattern = r"Test #(\d+): (\d+) \* (\d+) = (\d+).*Result=(\d+).*cycles: (\d+)"
        
        for match in re.finditer(pattern, output):
            results.append({
                'test_num': int(match.group(1)),
                'a': int(match.group(2)),
                'b': int(match.group(3)),
                'expected': int(match.group(4)),
                'actual': int(match.group(5)),
                'cycles': int(match.group(6))
            })
        
        return results


class PythonSimulator:
    """Simula multiplicaciones usando el modelo Python"""
    
    def __init__(self, model_path: str, algorithm: str = 'PPO', bit_width: int = 8):
        self.bit_width = bit_width
        self.datapath = Datapath(bit_width)
        
        # Cargar modelo RL
        if algorithm.upper() == 'PPO':
            self.model = PPO.load(model_path)
        else:
            self.model = DQN.load(model_path)
        
        print(f"✅ Modelo cargado desde {model_path}")
    
    def simulate_multiplication(self, a: int, b: int, max_cycles: int = 64) -> Tuple[int, int, bool]:
        """
        Simula una multiplicación
        
        Returns:
            (resultado, ciclos, éxito)
        """
        self.datapath.reset(a, b)
        
        obs = self._get_observation()
        
        for cycle in range(max_cycles):
            # Obtener acción del modelo
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Decodificar y ejecutar
            controls = self._decode_action(action)
            self.datapath.execute_operation(controls)
            
            # Verificar si terminó (estado HALT = 15)
            if controls['next_state'] == 15:
                return self.datapath.reg_p, cycle + 1, True
            
            obs = self._get_observation()
        
        # Timeout
        return self.datapath.reg_p, max_cycles, False
    
    def _get_observation(self) -> np.ndarray:
        """Construye observación para el modelo"""
        state = self.datapath.get_state()
        max_val = 2**self.bit_width - 1
        
        return np.array([
            state['reg_a'] / max_val,
            state['reg_b'] / max_val,
            state['reg_p'] / max_val,
            state['reg_temp'] / max_val,
            float(state['flag_z']),
            float(state['flag_lsb']),
            float(state['flag_c'])
        ], dtype=np.float32)
    
    def _decode_action(self, action: np.ndarray) -> Dict[str, int]:
        """Decodifica acción"""
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


class ResultValidator:
    """Valida que Python y Verilog produzcan resultados consistentes"""
    
    def __init__(self, 
                 python_sim: PythonSimulator,
                 test_cases: List[Tuple[int, int]]):
        self.python_sim = python_sim
        self.test_cases = test_cases
    
    def run_validation(self) -> Dict:
        """Ejecuta validación completa"""
        print("\n" + "="*60)
        print("🔍 VALIDACIÓN: Python vs Resultados Esperados")
        print("="*60)
        
        results = {
            'total': len(self.test_cases),
            'correct': 0,
            'incorrect': 0,
            'timeouts': 0,
            'details': []
        }
        
        for i, (a, b) in enumerate(self.test_cases, 1):
            expected = (a * b) & ((1 << self.python_sim.bit_width) - 1)
            actual, cycles, success = self.python_sim.simulate_multiplication(a, b)
            
            is_correct = (actual == expected)
            
            if not success:
                results['timeouts'] += 1
                status = "TIMEOUT"
            elif is_correct:
                results['correct'] += 1
                status = "✓"
            else:
                results['incorrect'] += 1
                status = "✗"
            
            results['details'].append({
                'a': a,
                'b': b,
                'expected': expected,
                'actual': actual,
                'cycles': cycles,
                'correct': is_correct,
                'success': success
            })
            
            if i <= 10 or not is_correct:  # Mostrar primeros 10 o errores
                print(f"Test {i:3d}: {a:3d} * {b:3d} = {actual:5d} "
                      f"(expected {expected:5d}) [{cycles:2d} cycles] {status}")
        
        # Resumen
        print("\n" + "="*60)
        print("📊 RESUMEN DE VALIDACIÓN")
        print("="*60)
        print(f"Total de pruebas: {results['total']}")
        print(f"Correctas: {results['correct']} ({results['correct']/results['total']*100:.1f}%)")
        print(f"Incorrectas: {results['incorrect']}")
        print(f"Timeouts: {results['timeouts']}")
        
        # Estadísticas de ciclos
        successful_tests = [d for d in results['details'] if d['success']]
        if successful_tests:
            cycles_list = [d['cycles'] for d in successful_tests]
            print(f"\nCiclos: min={min(cycles_list)}, "
                  f"max={max(cycles_list)}, "
                  f"promedio={np.mean(cycles_list):.1f}")
        
        return results
    
    def generate_report(self, results: Dict, output_path: str):
        """Genera un reporte detallado"""
        with open(output_path, 'w') as f:
            f.write("# Reporte de Validación\n\n")
            f.write(f"Total de pruebas: {results['total']}\n")
            f.write(f"Correctas: {results['correct']}\n")
            f.write(f"Incorrectas: {results['incorrect']}\n")
            f.write(f"Timeouts: {results['timeouts']}\n\n")
            
            f.write("## Casos Incorrectos\n\n")
            for detail in results['details']:
                if not detail['correct'] or not detail['success']:
                    f.write(f"- {detail['a']} * {detail['b']}: "
                           f"esperado={detail['expected']}, "
                           f"obtenido={detail['actual']}, "
                           f"ciclos={detail['cycles']}\n")
        
        print(f"\n📝 Reporte guardado en: {output_path}")


def generate_test_cases(bit_width: int, num_random: int = 50) -> List[Tuple[int, int]]:
    """Genera casos de prueba"""
    cases = []
    
    # Casos básicos
    cases.extend([
        (0, 0), (1, 1), (0, 5), (5, 0),
        (1, 10), (10, 1), (2, 3), (3, 2)
    ])
    
    # Potencias de 2
    for p in range(min(5, bit_width)):
        cases.append((2**p, 3))
        cases.append((3, 2**p))
    
    # Números medianos
    max_val = 2**bit_width - 1
    for i in range(10, min(100, max_val), 10):
        cases.append((i, 5))
    
    # Casos aleatorios
    np.random.seed(42)
    for _ in range(num_random):
        a = np.random.randint(0, 2**bit_width)
        b = np.random.randint(0, 2**bit_width)
        cases.append((a, b))
    
    return cases


def main():
    parser = argparse.ArgumentParser(description='Validar Python vs Verilog')
    parser.add_argument('model_path', type=str, help='Path al modelo entrenado')
    parser.add_argument('--algorithm', type=str, choices=['PPO', 'DQN'], default='PPO')
    parser.add_argument('--bit-width', type=int, default=8)
    parser.add_argument('--num-tests', type=int, default=100)
    parser.add_argument('--run-verilog', action='store_true',
                       help='Ejecutar simulación de Verilog')
    parser.add_argument('--report', type=str, default='validation_report.txt')
    
    args = parser.parse_args()
    
    # Crear simulador Python
    python_sim = PythonSimulator(
        args.model_path,
        args.algorithm,
        args.bit_width
    )
    
    # Generar casos de prueba
    test_cases = generate_test_cases(args.bit_width, args.num_tests)
    
    # Validar
    validator = ResultValidator(python_sim, test_cases)
    results = validator.run_validation()
    validator.generate_report(results, args.report)
    
    # Opcionalmente ejecutar Verilog
    if args.run_verilog:
        print("\n" + "="*60)
        print("🔧 Ejecutando simulación de Verilog")
        print("="*60)
        
        verilog_sim = VerilogSimulator(Path(__file__).parent)
        if verilog_sim.run_simulation():
            print("✅ Simulación de Verilog completada")
        else:
            print("❌ Error en simulación de Verilog")


if __name__ == '__main__':
    main()
