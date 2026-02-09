"""
SimpleCPU - Arquitectura de Procesador Simplificada para Agente RL

Formato de Instrucción: [OpCode, Reg_A, Reg_B, Dest_1, Dest_2]

OpCodes:
    0: ADD  - R[Dest1] = R[A] + R[B], siguiente estado = Dest_2
    1: SUB  - R[Dest1] = R[A] - R[B], siguiente estado = Dest_2
    2: SHL  - R[Dest1] = R[A] << R[B], siguiente estado = Dest_2
    3: SHR  - R[Dest1] = R[A] >> R[B], siguiente estado = Dest_2
    4: MOV  - R[Dest1] = R[A], siguiente estado = Dest_2
    5: BGT  - if R[A] > R[B]: Dest_1, else: Dest_2
    6: BEQ  - if R[A] == R[B]: Dest_1, else: Dest_2
    7: HALT - Detener ejecución
"""

from typing import List, Tuple, Optional
from enum import IntEnum


class OpCode(IntEnum):
    ADD = 0
    SUB = 1
    SHL = 2
    SHR = 3
    MOV = 4
    BGT = 5
    BEQ = 6
    HALT = 7


class SimpleCPU:
    """Intérprete de microinstrucciones para arquitectura simplificada."""
    
    # Número de registros constantes (solo lectura para el agente)
    NUM_CONSTANT_REGISTERS = 8
    
    def __init__(self, num_registers: int = 16, bit_width: int = 8):
        """
        Inicializa el procesador.
        
        Args:
            num_registers: Número total de registros en el banco
            bit_width: Ancho de bits para enmascarar resultados
        """
        self.num_registers = num_registers
        self.bit_width = bit_width
        self.max_value = (1 << bit_width) - 1  # Máscara para overflow
        
        self.registers = [0] * num_registers
        self.pc = 0  # Program Counter
        self.halted = False
        self.cycle_count = 0
        
        # Inicializar registros constantes
        self._init_constant_registers()
        
    def _init_constant_registers(self):
        """Inicializa registros constantes (R0=0, R1=1, R2=2, R3=4, ...)."""
        # R0 = 0, R1 = 1, R2 = 2, R3 = 4, R4 = 8, ...
        for i in range(min(self.NUM_CONSTANT_REGISTERS, self.num_registers)):
            if i == 0:
                self.registers[i] = 0
            elif i == 1:
                self.registers[i] = 1
            elif i == 2:
                self.registers[i] = 2
            else:
                self.registers[i] = 1 << (i - 1)  # Potencias de 2
                
    def reset(self):
        """Reinicia el estado del procesador."""
        self.registers = [0] * self.num_registers
        self.pc = 0
        self.halted = False
        self.cycle_count = 0
        self._init_constant_registers()
        
    def _is_writable(self, reg_index: int) -> bool:
        """Verifica si un registro es escribible (no constante)."""
        return reg_index >= self.NUM_CONSTANT_REGISTERS
    
    def _write_register(self, reg_index: int, value: int) -> bool:
        """
        Escribe un valor en un registro si es escribible.
        
        Returns:
            True si la escritura fue exitosa, False si el registro es constante
        """
        if self._is_writable(reg_index):
            self.registers[reg_index] = value & self.max_value
            return True
        return False
    
    def execute(self, instruction: List[int]) -> int:
        """
        Ejecuta una instrucción y retorna el siguiente estado (PC).
        
        Args:
            instruction: [OpCode, Reg_A, Reg_B, Dest_1, Dest_2]
            
        Returns:
            Siguiente valor de PC
        """
        if len(instruction) != 5:
            raise ValueError(f"Instrucción inválida: {instruction}")
            
        opcode, reg_a, reg_b, dest_1, dest_2 = instruction
        
        # Obtener valores de registros fuente
        val_a = self.registers[reg_a] if reg_a < self.num_registers else 0
        val_b = self.registers[reg_b] if reg_b < self.num_registers else 0
        
        next_pc = dest_2  # Default para operaciones de datos
        
        if opcode == OpCode.ADD:
            result = val_a + val_b
            self._write_register(dest_1, result)
            
        elif opcode == OpCode.SUB:
            result = val_a - val_b
            self._write_register(dest_1, result)
            
        elif opcode == OpCode.SHL:
            # Shift left por el VALOR en R[B]
            shift_amount = val_b & 0x7  # Limitar a 7 bits de shift
            result = val_a << shift_amount
            self._write_register(dest_1, result)
            
        elif opcode == OpCode.SHR:
            # Shift right por el VALOR en R[B]
            shift_amount = val_b & 0x7
            result = val_a >> shift_amount
            self._write_register(dest_1, result)
            
        elif opcode == OpCode.MOV:
            self._write_register(dest_1, val_a)
            
        elif opcode == OpCode.BGT:
            # Branch if Greater Than
            if val_a > val_b:
                next_pc = dest_1
            else:
                next_pc = dest_2
                
        elif opcode == OpCode.BEQ:
            # Branch if Equal
            if val_a == val_b:
                next_pc = dest_1
            else:
                next_pc = dest_2
                
        elif opcode == OpCode.HALT:
            self.halted = True
            next_pc = self.pc  # Quedarse en el mismo estado
            
        else:
            raise ValueError(f"OpCode desconocido: {opcode}")
            
        self.cycle_count += 1
        self.pc = next_pc
        return next_pc
    
    def run(self, program: List[List[int]], max_cycles: int = 1000) -> Tuple[int, bool]:
        """
        Ejecuta un programa completo hasta HALT o máximo de ciclos.
        
        Args:
            program: Lista de instrucciones (cada una es [OpCode, A, B, D1, D2])
            max_cycles: Máximo de ciclos antes de abortar
            
        Returns:
            Tuple de (ciclos ejecutados, terminó por HALT)
        """
        self.reset()
        
        while not self.halted and self.cycle_count < max_cycles:
            if self.pc < 0 or self.pc >= len(program):
                break
            instruction = program[self.pc]
            self.execute(instruction)
            
        return self.cycle_count, self.halted
    
    def get_state(self) -> dict:
        """Retorna el estado actual del procesador."""
        return {
            'pc': self.pc,
            'halted': self.halted,
            'cycle_count': self.cycle_count,
            'registers': self.registers.copy()
        }
    
    def __repr__(self):
        return (
            f"SimpleCPU(pc={self.pc}, halted={self.halted}, "
            f"cycles={self.cycle_count}, regs={self.registers})"
        )
