"""
Datapath.py - Digital Twin bit-accurate del hardware
Simula la ALU, registros y shifter sin usar multiplicadores nativos
"""
import numpy as np
from typing import Dict, Tuple

class Datapath:
    """Simulador bit-accurate del datapath de hardware"""
    
    def __init__(self, bit_width: int = 8):
        self.bit_width = bit_width
        self.mask = (1 << bit_width) - 1  # Máscara para limitar bits
        
        # Registros principales
        self.reg_a = 0      # Multiplicando
        self.reg_b = 0      # Multiplicador
        self.reg_p = 0      # Producto (parcial/final)
        self.reg_temp = 0   # Registro temporal
        
        # Flags
        self.flag_z = 0     # Zero flag
        self.flag_lsb = 0   # Least Significant Bit de B
        self.flag_c = 0     # Carry flag
        
        # Contador de ciclos
        self.cycle_count = 0
        
    def reset(self, a: int, b: int):
        """Resetea el datapath con nuevos valores de entrada"""
        self.reg_a = a & self.mask
        self.reg_b = b & self.mask
        self.reg_p = 0
        self.reg_temp = 0
        self.flag_z = 0
        self.flag_lsb = (b & 1)
        self.flag_c = 0
        self.cycle_count = 0
        
    def execute_operation(self, controls: Dict[str, int]) -> Dict[str, int]:
        """
        Ejecuta un ciclo de operación según las señales de control
        
        Args:
            controls: Diccionario con señales de control:
                - alu_op: Operación de la ALU (0=ADD, 1=SUB, 2=AND, 3=OR, 4=XOR, 5=PASS_A, 6=PASS_B)
                - alu_src_a: Fuente A (0=reg_a, 1=reg_p, 2=reg_temp)
                - alu_src_b: Fuente B (0=reg_b, 1=reg_p, 2=reg_temp, 3=0)
                - shift_op: Operación shift (0=NONE, 1=SHL, 2=SHR, 3=ROL, 4=ROR)
                - shift_target: Qué registro hacer shift (0=NONE, 1=A, 2=B, 3=P)
                - write_p: Escribir resultado ALU a P
                - write_temp: Escribir resultado ALU a TEMP
                - write_b: Escribir a B (para decrementos)
        
        Returns:
            Estado actual de registros y flags
        """
        # 1. Seleccionar fuentes de la ALU
        src_a = self._select_alu_source(controls.get('alu_src_a', 0), 'a')
        src_b = self._select_alu_source(controls.get('alu_src_b', 0), 'b')
        
        # 2. Ejecutar operación ALU
        alu_result, carry = self._alu_operation(
            controls.get('alu_op', 0),
            src_a,
            src_b
        )
        
        # 3. Actualizar flags de ALU
        self.flag_z = 1 if alu_result == 0 else 0
        self.flag_c = carry
        
        # 4. Escribir resultados según señales de control
        if controls.get('write_p', 0):
            self.reg_p = alu_result
            
        if controls.get('write_temp', 0):
            self.reg_temp = alu_result
            
        if controls.get('write_b', 0):
            self.reg_b = alu_result
            self.flag_lsb = alu_result & 1
        
        # 5. Ejecutar operaciones de shift
        self._execute_shift(
            controls.get('shift_op', 0),
            controls.get('shift_target', 0)
        )
        
        # 6. Incrementar contador de ciclos
        self.cycle_count += 1
        
        return self.get_state()
    
    def _select_alu_source(self, selector: int, source_type: str) -> int:
        """Selecciona la fuente para la ALU"""
        if source_type == 'a':
            if selector == 0:
                return self.reg_a
            elif selector == 1:
                return self.reg_p
            elif selector == 2:
                return self.reg_temp
        else:  # source_type == 'b'
            if selector == 0:
                return self.reg_b
            elif selector == 1:
                return self.reg_p
            elif selector == 2:
                return self.reg_temp
            elif selector == 3:
                return 0
        return 0
    
    def _alu_operation(self, op: int, a: int, b: int) -> Tuple[int, int]:
        """
        Ejecuta operación de la ALU sin usar multiplicación
        
        Returns:
            (resultado, carry_flag)
        """
        carry = 0
        
        if op == 0:  # ADD
            result = a + b
            carry = 1 if result > self.mask else 0
            result = result & self.mask
            
        elif op == 1:  # SUB
            result = a - b
            carry = 1 if a < b else 0  # Borrow
            result = result & self.mask
            
        elif op == 2:  # AND
            result = a & b
            
        elif op == 3:  # OR
            result = a | b
            
        elif op == 4:  # XOR
            result = a ^ b
            
        elif op == 5:  # PASS_A
            result = a
            
        elif op == 6:  # PASS_B
            result = b
            
        else:
            result = 0
            
        return result & self.mask, carry
    
    def _execute_shift(self, shift_op: int, target: int):
        """Ejecuta operaciones de shift/rotate"""
        if shift_op == 0 or target == 0:  # NONE
            return
            
        # Seleccionar registro objetivo
        if target == 1:
            reg_val = self.reg_a
        elif target == 2:
            reg_val = self.reg_b
        elif target == 3:
            reg_val = self.reg_p
        else:
            return
        
        # Ejecutar shift
        if shift_op == 1:  # SHL (Shift Left)
            new_val = (reg_val << 1) & self.mask
            self.flag_c = (reg_val >> (self.bit_width - 1)) & 1
            
        elif shift_op == 2:  # SHR (Shift Right)
            self.flag_c = reg_val & 1
            new_val = reg_val >> 1
            
        elif shift_op == 3:  # ROL (Rotate Left)
            new_val = ((reg_val << 1) | (reg_val >> (self.bit_width - 1))) & self.mask
            
        elif shift_op == 4:  # ROR (Rotate Right)
            new_val = ((reg_val >> 1) | (reg_val << (self.bit_width - 1))) & self.mask
            
        else:
            return
        
        # Escribir resultado
        if target == 1:
            self.reg_a = new_val
        elif target == 2:
            self.reg_b = new_val
            self.flag_lsb = new_val & 1
        elif target == 3:
            self.reg_p = new_val
    
    def get_state(self) -> Dict[str, int]:
        """Retorna el estado completo del datapath"""
        return {
            'reg_a': self.reg_a,
            'reg_b': self.reg_b,
            'reg_p': self.reg_p,
            'reg_temp': self.reg_temp,
            'flag_z': self.flag_z,
            'flag_lsb': self.flag_lsb,
            'flag_c': self.flag_c,
            'cycle_count': self.cycle_count
        }
    
    def get_result(self) -> int:
        """Retorna el resultado actual en el registro P"""
        return self.reg_p
    
    def verify_multiplication(self, a: int, b: int) -> bool:
        """Verifica si el resultado actual es correcto"""
        expected = (a * b) & ((1 << (self.bit_width * 2)) - 1)
        # Para multiplicación de 8 bits, resultado en 16 bits
        # Por ahora comparamos solo con bit_width
        expected_truncated = expected & self.mask
        return self.reg_p == expected_truncated
