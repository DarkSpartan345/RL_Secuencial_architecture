import sys
from pathlib import Path

# Fix import path to run from 'env' or root
try:
    from simple_cpu import SimpleCPU, OpCode
except ImportError:
    # If running from root
    sys.path.insert(0, str(Path(__file__).parent))
    from simple_cpu import SimpleCPU, OpCode

def decode_instruction(instruction) -> str:
    """Traduce una instrucción a lenguaje ensamblador legible."""
    opcode_val, reg_a, reg_b, dest_1, dest_2 = instruction
    
    try:
        # Si ya es un enum, usarlo directamente, sino convertir
        if isinstance(opcode_val, OpCode):
            name = opcode_val.name
        else:
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

def main():
    print("=== Demostracion de Algoritmo Shift-and-Add en SimpleCPU ===")
    
    # 1. Inicializar CPU
    cpu = SimpleCPU(num_registers=16, bit_width=8)
    
    # 2. Configurar Valores de Entrada (Multiplicación 12 * 10 = 120)
    # R8 = A (Multiplicando) = 12
    # R9 = B (Multiplicador) = 10
    # R10 = P (Producto) = 0 (Inicialmente)
    # R11 = T (Temporal para chequeo de LSB)
    
    cpu._write_register(8, 12)
    cpu._write_register(9, 10)
    cpu._write_register(10, 0) # Asegurar que P es 0
    
    print(f"Estado Inicial: A(R8)={cpu.registers[8]}, B(R9)={cpu.registers[9]}, P(R10)={cpu.registers[10]}")
    
# 3. Definir el Programa (Algoritmo Shift-and-Add)
# Pseudo-código:
# While B != 0:
#    If (B % 2) != 0: (Check LSB)
#       P = P + A
#    A = A << 1
#    B = B >> 1

SHIFT_AND_ADD_PROGRAM = [
    # 0: Verificar si B (R9) == 0. Si es 0, ir a HALT (Addr 8). Si no, seguir a 1.
    # [BEQ, R9, R0, Dest_True=8, Dest_False=1] (R0 es constante 0)
    [OpCode.BEQ, 9, 0, 8, 1],
    
    # --- Check LSB de B ---
    # No hay instruccion AND, asi que usamos truco: (B >> 1) << 1. Si es igual a B, LSB era 0.
    
    # 1: Copiar B a T (R11)
    # [MOV, R9, 0, R11, NextPC=2]
    [OpCode.MOV, 9, 0, 11, 2],
    
    # 2: Shift Right T (T = T >> 1)
    # [SHR, R11, R1, R11, NextPC=3] (R1 es constante 1)
    [OpCode.SHR, 11, 1, 11, 3],
    
    # 3: Shift Left T (T = T << 1)
    # [SHL, R11, R1, R11, NextPC=4]
    [OpCode.SHL, 11, 1, 11, 4],
    
    # 4: Comparar T con B. 
    # Si T == B (LSB es 0), saltar suma (ir a 6).
    # Si T != B (LSB es 1), ejecutar suma (ir a 5).
    # [BEQ, R11, R9, 6, 5]
    [OpCode.BEQ, 11, 9, 6, 5],
    
    # 5: Sumar A a P (P = P + A)
    # [ADD, R10, R8, R10, NextPC=6]
    [OpCode.ADD, 10, 8, 10, 6],
    
    # --- Preparar siguiente iteración ---
    
    # 6: Shift Left A (A = A << 1)
    # [SHL, R8, R1, R8, NextPC=7]
    [OpCode.SHL, 8, 1, 8, 7],
    
    # 7: Shift Right B (B = B >> 1) y volver al inicio (NextPC=0)
    # [SHR, R9, R1, R9, NextPC=0]
    [OpCode.SHR, 9, 1, 9, 0],
    
    # 8: HALT
    [OpCode.HALT, 0, 0, 0, 0]
]

def main():
    print("=== Demostracion de Algoritmo Shift-and-Add en SimpleCPU ===")
    
    # 1. Inicializar CPU
    cpu = SimpleCPU(num_registers=16, bit_width=8)
    
    # 2. Configurar Valores de Entrada (Multiplicación 12 * 10 = 120)
    # R8 = A (Multiplicando) = 12
    # R9 = B (Multiplicador) = 10
    # R10 = P (Producto) = 0 (Inicialmente)
    # R11 = T (Temporal para chequeo de LSB)
    
    cpu._write_register(8, 12)
    cpu._write_register(9, 10)
    cpu._write_register(10, 0) # Asegurar que P es 0
    
    print(f"Estado Inicial: A(R8)={cpu.registers[8]}, B(R9)={cpu.registers[9]}, P(R10)={cpu.registers[10]}")
    
    program = SHIFT_AND_ADD_PROGRAM
    
    # 4. Ejecutar (Manual loop para evitar reset() de cpu.run())
    cpu.pc = 0
    cpu.halted = False
    cpu.cycle_count = 0
    
    max_cycles = 100
    while not cpu.halted and cpu.cycle_count < max_cycles:
        if cpu.pc < 0 or cpu.pc >= len(program):
            print(f"Error: PC {cpu.pc} fuera de rango")
            break
            
        instruction = program[cpu.pc]
        decoded = decode_instruction(instruction)
        
        # Debug print para ver traza paso a paso
        print(f"Cycle:{cpu.cycle_count:03} | PC:{cpu.pc} | {decoded:<45} | A(R8):{cpu.registers[8]:<3} B(R9):{cpu.registers[9]:<3} P(R10):{cpu.registers[10]:<3} T(R11):{cpu.registers[11]:<3}")
        
        cpu.execute(instruction)
    
    cycles = cpu.cycle_count
    halted = cpu.halted
    
    print("\n=== Ejecución Completada ===")
    print(f"Ciclos: {cycles}")
    print(f"Estado Final: A(R8)={cpu.registers[8]}, B(R9)={cpu.registers[9]}, P(R10)={cpu.registers[10]}")
    
    # Verificación
    expected = 12 * 10
    actual = cpu.registers[10]
    if actual == expected:
        print(f"EXITO: Resultado correcto ({actual})")
    else:
        print(f"FALLO: Esperaba {expected}, obtuvo {actual}")

if __name__ == "__main__":
    main()
