import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
try:
    from env.simple_cpu import SimpleCPU, OpCode
except ImportError:
    from simple_cpu import SimpleCPU, OpCode
class SimpleCPUEnv(gym.Env):
    """
    Entorno de Gymnasium para la arquitectura SimpleCPU.
    El agente genera una instrucción de 5 elementos en cada paso.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, target_value: int = 42, max_cycles: int = 50, render_mode: Optional[str] = None):
        super(SimpleCPUEnv, self).__init__()
        
        # Configuración del CPU
        self.cpu = SimpleCPU(num_registers=16, bit_width=8)
        self.max_cycles = max_cycles
        self.target_value = target_value
        self.render_mode = render_mode

        # --- Espacio de Acciones ---
        # El agente debe generar: [OpCode, Reg_A, Reg_B, Dest_1, Dest_2]
        # OpCode: 0-7
        # Reg_A, Reg_B, Dest_1: 0-15 (registros)
        # Dest_2: 0-max_cycles (puntero de estado/PC)
        self.action_space = spaces.MultiDiscrete([
            8,                  # OpCode (0-7)
            self.cpu.num_registers, # Reg_A
            self.cpu.num_registers, # Reg_B
            self.cpu.num_registers, # Dest_1
            max_cycles          # Dest_2 (Siguiente PC sugerido)
        ])

        # --- Espacio de Observación ---
        # El agente observa el estado actual de los registros y el PC
        # Usamos Box para los registros (0-255 debido a 8 bits)
        self.observation_space = spaces.Dict({
            "registers": spaces.Box(low=0, high=255, shape=(self.cpu.num_registers,), dtype=np.int32),
            "pc": spaces.Discrete(max_cycles + 1)
        })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.cpu.reset()
        
        # Opcional: Podrías inicializar un registro con un valor aleatorio como entrada
        # self.cpu._write_register(8, self.np_random.integers(1, 10))

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self):
        return {
            "registers": np.array(self.cpu.registers, dtype=np.int32),
            "pc": self.cpu.pc
        }

    def _get_info(self):
        return {
            "cycle_count": self.cpu.cycle_count,
            "halted": self.cpu.halted
        }

    def step(self, action):
        
        # Guardamos la última acción para el renderizado
        self.last_action = action 
        
        # ... lógica de recompensas ...
        
        # 1. Convertir la acción del agente (numpy array) a una lista de instrucción
        instruction = action.tolist()
        
        # 2. Ejecutar la instrucción en el CPU
        self.cpu.execute(instruction)
        
        # 3. Calcular Recompensa
        reward = 0
        terminated = False
        
        # Distancia mínima al objetivo en cualquier registro
        # Ignoramos registros de solo lectura (0-7) si es necesario, 
        # pero SimpleCPU no distingue mucho en hardware real salvo R0=0
        current_vals = self.cpu.registers
        # Buscamos el valor más cercano al target
        min_distance = np.min(np.abs(np.array(current_vals) - self.target_value))
        
        # Recompensa Densa (Shaped): Incentivar acercarse al valor
        # Damos hasta 1.0 punto extra si está muy cerca
        shaped_reward = 10.0 / (min_distance + 1.0)
        reward += shaped_reward
        
        # Recompensa por Éxito TOTAL
        if min_distance == 0:
            reward += 100
            terminated = True
        
        # Penalización por ciclo (existencia)
        reward -= 1.0
        
        # 4. Verificar condiciones de parada
        if self.cpu.halted:
            if terminated: # Éxito (target reached)
                 pass 
            else: # Falla (se rindió demasiado pronto)
                 reward -= 50 # Penalización fuerte
                 terminated = True
            
        truncated = self.cpu.cycle_count >= self.max_cycles
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            # Si no ha habido acción (primer paso), mostramos estado inicial
            act_str = self.decode_instruction(self.last_action) if hasattr(self, 'last_action') else "N/A"
            
            print("-" * 70)
            print(f" CICLO: {self.cpu.cycle_count:03} | PC ACTUAL: {self.cpu.pc:03}")
            print(f" INSTRUCCIÓN EJECUTADA: {act_str}")
            print(f" REGISTROS: {self.cpu.registers}")
            if self.cpu.halted:
                print(" >>> ESTADO: HALTED <<<")

    def decode_instruction(self,action: np.ndarray) -> str:
        """Traduce un array de acción a lenguaje ensamblador legible."""
        opcode_val, reg_a, reg_b, dest_1, dest_2 = action
        
        try:
            name = OpCode(opcode_val).name
        except ValueError:
            name = f"UNKNOWN({opcode_val})"

        if name in ["ADD", "SUB", "SHL", "SHR"]:
            return f"{name:4} R{dest_1:02} <- R{reg_a:02}, R{reg_b:02} | Next PC: {dest_2}"
        
        elif name == "MOV":
            return f"MOV  R{dest_1:02} <- R{reg_a:02}      | Next PC: {dest_2}"
        
        elif name in ["BGT", "BEQ"]:
            symbol = ">" if name == "BGT" else "=="
            return f"IF R{reg_a:02} {symbol} R{reg_b:02} THEN GOTO {dest_1} ELSE GOTO {dest_2}"
        
        elif name == "HALT":
            return "HALT"
        
        return f"OP:{opcode_val} A:{reg_a} B:{reg_b} D1:{dest_1} D2:{dest_2}"

    def generate_verilog_mem(self, program: list, output_path: str = "hdl/program.mem", memory_size: int = 16):
        """
        Genera un archivo .mem para Verilog compatible con $readmemh.
        Args:
            program: Lista de instrucciones [Op, A, B, D1, D2]
            output_path: Ruta de salida (por defecto 'hdl/program.mem')
            memory_size: Tamaño de la memoria a rellenar (por defecto 16)
        """
        print(f"Generando archivo de memoria desde Env: {output_path}")
        
        try:
            with open(output_path, 'w') as f:
                for i, instruction in enumerate(program):
                    if i >= memory_size:
                        print(f"Warning: Program is longer than memory size ({memory_size})")
                        break
                        
                    opcode, reg_a, reg_b, dest_1, dest_2 = instruction
                    
                    # Ensure opcode is int
                    if hasattr(opcode, 'value'):
                        opcode = opcode.value
                    
                    # Encode: Op(3) | A(4) | B(4) | D1(4) | D2(4)
                    val = (opcode << 16) | (reg_a << 12) | (reg_b << 8) | (dest_1 << 4) | dest_2
                    
                    hex_str = f"{val:05X}"
                    f.write(f"{hex_str}\n")
                    
                # Pad with HALT instructions
                remaining = memory_size - len(program)
                if remaining > 0:
                    halt_val = (7 << 16)
                    halt_hex = f"{halt_val:05X}"
                    for _ in range(remaining):
                        f.write(f"{halt_hex}\n")
                        
            print("Archivo generado exitosamente.")
        except IOError as e:
            print(f"Error escribiendo archivo: {e}")

# --- Ejecución de Prueba (Shift-and-Add) ---
if __name__ == "__main__":
    try:
        from env.shift_and_add_demo import SHIFT_AND_ADD_PROGRAM
    except ImportError:
        try:
            from shift_and_add_demo import SHIFT_AND_ADD_PROGRAM
        except ImportError:
            # Fallback si no se encuentra (para evitar errores si archivos movidos)
            print("No se pudo importar SHIFT_AND_ADD_PROGRAM. Usando programa dummy.")
            SHIFT_AND_ADD_PROGRAM = []

    print("\n=== Ejecutando Shift-and-Add en Environment ===")
    
    # 1. Crear entorno
    env = SimpleCPUEnv(target_value=120, max_cycles=100, render_mode="human")
    obs, info = env.reset()
    
    # 2. Configurar entradas (12 * 10)
    # A=12 (R8), B=10 (R9)
    env.cpu._write_register(8, 12)
    env.cpu._write_register(9, 10)
    env.cpu._write_register(10, 0) # P=0
    
    print(f"Estado Inicial: A(R8)={env.cpu.registers[8]}, B(R9)={env.cpu.registers[9]}")
    
    # 3. Ejecutar algoritmo paso a paso
    total_reward = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        pc = env.cpu.pc
        
        # Verificar si PC está dentro del programa
        if pc < len(SHIFT_AND_ADD_PROGRAM):
            # Fetch instrucción del programa definido
            instruction = SHIFT_AND_ADD_PROGRAM[pc]
            
            # Convertir a numpy array para el step (si es necesario)
            # Nota: step() convierte a list, pero action_space espera algo compatible
            # En este caso pasamos la lista directamente (SimpleCPU maneja OpCode enums)
            action = np.array(instruction, dtype=object) 
            # Nota: dtype=object para permitir OpCode enums si existen
            # Sin embargo, step() llama tolist() y luego execute().
            # Para mayor seguridad, convertimos OpCodes a int si están en enum
            processed_action = []
            for x in instruction:
                processed_action.append(x.value if hasattr(x, 'value') else x)
            
            action_array = np.array(processed_action, dtype=np.int32)
            
            print(f"> Fetch Addr {pc}: {instruction}")
            
            # Step del entorno
            obs, reward, terminated, truncated, info = env.step(action_array)
            env.render()
            total_reward += reward
            
        else:
            print(f"PC {pc} fuera de rango de programa. Terminando.")
            break
            
        if env.cpu.halted:
            print("CPU Halted.")
            break
            
    print(f"Ejecucion Finalizada. Resultado P(R10) = {env.cpu.registers[10]}")
    
    # 4. Generar archivo para Verilog
    print("\n=== Generando Implementación HDL ===")
    # Aseguramos que la ruta es correcta para la estructura del proyecto
    import os
    output_mem = "hdl/program.mem" 
    # Si ejecutamos desde env/, subir un nivel
    if os.path.basename(os.getcwd()) == "env":
        output_mem = "../hdl/program.mem"
        
    env.generate_verilog_mem(SHIFT_AND_ADD_PROGRAM, output_path=output_mem)