"""
test_datapath.py - Pruebas unitarias para el simulador de datapath
Verifica que las operaciones bit-accurate funcionen correctamente
"""
import pytest
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.datapath import Datapath


class TestDatapathBasics:
    """Pruebas básicas de inicialización y reset"""
    
    def test_initialization(self):
        """Verifica que el datapath se inicialice correctamente"""
        dp = Datapath(bit_width=8)
        
        assert dp.bit_width == 8
        assert dp.mask == 0xFF
        assert dp.reg_a == 0
        assert dp.reg_b == 0
        assert dp.reg_p == 0
        assert dp.cycle_count == 0
    
    def test_reset(self):
        """Verifica que reset configure los valores correctamente"""
        dp = Datapath(bit_width=8)
        dp.reset(5, 7)
        
        assert dp.reg_a == 5
        assert dp.reg_b == 7
        assert dp.reg_p == 0
        assert dp.flag_lsb == 1  # 7 es impar
        assert dp.cycle_count == 0
    
    def test_bit_width_masking(self):
        """Verifica que los valores se enmascaren al ancho de bits"""
        dp = Datapath(bit_width=4)
        dp.reset(20, 18)  # Valores que exceden 4 bits
        
        assert dp.reg_a == 4   # 20 & 0xF = 4
        assert dp.reg_b == 2   # 18 & 0xF = 2


class TestALUOperations:
    """Pruebas de las operaciones de la ALU"""
    
    def test_alu_add_no_carry(self):
        """Prueba suma sin carry"""
        dp = Datapath(bit_width=8)
        dp.reset(10, 20)
        
        controls = {
            'alu_op': 0,  # ADD
            'alu_src_a': 0,  # reg_a
            'alu_src_b': 0,  # reg_b
            'write_p': 1
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 30
        assert state['flag_c'] == 0
        assert state['flag_z'] == 0
    
    def test_alu_add_with_carry(self):
        """Prueba suma con carry (overflow)"""
        dp = Datapath(bit_width=8)
        dp.reset(200, 100)
        
        controls = {
            'alu_op': 0,  # ADD
            'alu_src_a': 0,
            'alu_src_b': 0,
            'write_p': 1
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 44  # (200 + 100) & 0xFF = 44
        assert state['flag_c'] == 1  # Hubo overflow
    
    def test_alu_sub_no_borrow(self):
        """Prueba resta sin borrow"""
        dp = Datapath(bit_width=8)
        dp.reset(50, 20)
        
        controls = {
            'alu_op': 1,  # SUB
            'alu_src_a': 0,
            'alu_src_b': 0,
            'write_p': 1
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 30
        assert state['flag_c'] == 0
    
    def test_alu_sub_with_borrow(self):
        """Prueba resta con borrow (underflow)"""
        dp = Datapath(bit_width=8)
        dp.reset(10, 20)
        
        controls = {
            'alu_op': 1,  # SUB
            'alu_src_a': 0,
            'alu_src_b': 0,
            'write_p': 1
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 246  # (10 - 20) & 0xFF = 246
        assert state['flag_c'] == 1  # Hubo borrow
    
    def test_alu_and(self):
        """Prueba operación AND"""
        dp = Datapath(bit_width=8)
        dp.reset(0b11110000, 0b10101010)
        
        controls = {
            'alu_op': 2,  # AND
            'alu_src_a': 0,
            'alu_src_b': 0,
            'write_p': 1
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 0b10100000
    
    def test_alu_or(self):
        """Prueba operación OR"""
        dp = Datapath(bit_width=8)
        dp.reset(0b11110000, 0b00001111)
        
        controls = {
            'alu_op': 3,  # OR
            'alu_src_a': 0,
            'alu_src_b': 0,
            'write_p': 1
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 0b11111111
    
    def test_alu_xor(self):
        """Prueba operación XOR"""
        dp = Datapath(bit_width=8)
        dp.reset(0b11110000, 0b10101010)
        
        controls = {
            'alu_op': 4,  # XOR
            'alu_src_a': 0,
            'alu_src_b': 0,
            'write_p': 1
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 0b01011010
    
    def test_zero_flag(self):
        """Prueba que el flag Z se active correctamente"""
        dp = Datapath(bit_width=8)
        dp.reset(5, 5)
        
        controls = {
            'alu_op': 1,  # SUB (5 - 5 = 0)
            'alu_src_a': 0,
            'alu_src_b': 0,
            'write_p': 1
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 0
        assert state['flag_z'] == 1


class TestALUSourceSelection:
    """Pruebas de selección de fuentes para la ALU"""
    
    def test_alu_source_selection(self):
        """Verifica que las fuentes se seleccionen correctamente"""
        dp = Datapath(bit_width=8)
        dp.reset(10, 20)
        
        # Primero escribir algo a reg_p
        controls = {
            'alu_op': 0,  # ADD
            'alu_src_a': 0,  # reg_a = 10
            'alu_src_b': 0,  # reg_b = 20
            'write_p': 1
        }
        dp.execute_operation(controls)
        # Ahora reg_p = 30
        
        # Usar reg_p como fuente A
        controls = {
            'alu_op': 0,  # ADD
            'alu_src_a': 1,  # reg_p = 30
            'alu_src_b': 0,  # reg_b = 20
            'write_temp': 1
        }
        state = dp.execute_operation(controls)
        
        assert state['reg_temp'] == 50  # 30 + 20
    
    def test_write_to_temp(self):
        """Verifica escritura a registro temporal"""
        dp = Datapath(bit_width=8)
        dp.reset(15, 25)
        
        controls = {
            'alu_op': 0,  # ADD
            'alu_src_a': 0,
            'alu_src_b': 0,
            'write_temp': 1
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_temp'] == 40
        assert state['reg_p'] == 0  # No se escribió a P


class TestShiftOperations:
    """Pruebas de operaciones de shift y rotate"""
    
    def test_shift_left(self):
        """Prueba shift left"""
        dp = Datapath(bit_width=8)
        dp.reset(0, 0)
        dp.reg_p = 0b00001111
        
        controls = {
            'shift_op': 1,  # SHL
            'shift_target': 3  # Shift P
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 0b00011110
        assert state['flag_c'] == 0
    
    def test_shift_left_with_carry(self):
        """Prueba shift left que genera carry"""
        dp = Datapath(bit_width=8)
        dp.reset(0, 0)
        dp.reg_p = 0b10001111
        
        controls = {
            'shift_op': 1,  # SHL
            'shift_target': 3
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 0b00011110
        assert state['flag_c'] == 1  # El bit más significativo salió
    
    def test_shift_right(self):
        """Prueba shift right"""
        dp = Datapath(bit_width=8)
        dp.reset(0, 0)
        dp.reg_p = 0b11110000
        
        controls = {
            'shift_op': 2,  # SHR
            'shift_target': 3
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 0b01111000
        assert state['flag_c'] == 0
    
    def test_shift_right_with_carry(self):
        """Prueba shift right que genera carry"""
        dp = Datapath(bit_width=8)
        dp.reset(0, 0)
        dp.reg_p = 0b11110001
        
        controls = {
            'shift_op': 2,  # SHR
            'shift_target': 3
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 0b01111000
        assert state['flag_c'] == 1
    
    def test_shift_on_reg_b_updates_lsb(self):
        """Verifica que shift en B actualice el flag LSB"""
        dp = Datapath(bit_width=8)
        dp.reset(0, 0b00000110)  # LSB = 0
        
        assert dp.flag_lsb == 0
        
        controls = {
            'shift_op': 2,  # SHR
            'shift_target': 2  # Shift B
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_b'] == 0b00000011
        assert state['flag_lsb'] == 1  # Ahora es impar
    
    def test_rotate_left(self):
        """Prueba rotate left"""
        dp = Datapath(bit_width=8)
        dp.reset(0, 0)
        dp.reg_p = 0b10000001
        
        controls = {
            'shift_op': 3,  # ROL
            'shift_target': 3
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 0b00000011  # Bit alto rotó al bajo
    
    def test_rotate_right(self):
        """Prueba rotate right"""
        dp = Datapath(bit_width=8)
        dp.reset(0, 0)
        dp.reg_p = 0b10000001
        
        controls = {
            'shift_op': 4,  # ROR
            'shift_target': 3
        }
        
        state = dp.execute_operation(controls)
        
        assert state['reg_p'] == 0b11000000  # Bit bajo rotó al alto


class TestMultiplicationAlgorithm:
    """Pruebas de algoritmos de multiplicación implementados manualmente"""
    
    def test_simple_shift_and_add(self):
        """
        Prueba un algoritmo simple de shift-and-add
        Multiplica 5 * 3 = 15
        """
        dp = Datapath(bit_width=8)
        dp.reset(5, 3)  # A=5, B=3
        
        # Algoritmo: mientras B > 0:
        #   si B es impar: P = P + A
        #   A = A << 1
        #   B = B >> 1
        
        # Ciclo 1: B=3 (impar)
        # P = P + A = 0 + 5 = 5
        controls = {
            'alu_op': 0, 'alu_src_a': 1, 'alu_src_b': 0,
            'write_p': 1, 'shift_op': 0
        }
        if dp.flag_lsb == 1:
            # Como reg_p empieza en 0, usamos reg_a
            controls['alu_src_a'] = 0
            controls['alu_src_b'] = 3  # Constante 0
            controls['alu_op'] = 5  # PASS_A
            dp.execute_operation(controls)
        
        # A = A << 1 = 10
        controls = {
            'shift_op': 1, 'shift_target': 1, 'alu_op': 5, 'write_p': 0
        }
        dp.execute_operation(controls)
        
        # B = B >> 1 = 1
        controls = {
            'shift_op': 2, 'shift_target': 2
        }
        dp.execute_operation(controls)
        
        # Verificar estado
        assert dp.reg_a == 10
        assert dp.reg_b == 1
        assert dp.reg_p == 5
    
    def test_multiplication_verification(self):
        """Prueba la función de verificación de multiplicación"""
        dp = Datapath(bit_width=8)
        
        # Caso correcto
        dp.reset(7, 8)
        dp.reg_p = 56
        assert dp.verify_multiplication(7, 8) == True
        
        # Caso incorrecto
        dp.reg_p = 55
        assert dp.verify_multiplication(7, 8) == False
        
        # Caso con overflow (resultado truncado)
        dp.reset(200, 200)
        # 200 * 200 = 40000, truncado a 8 bits = 40000 & 0xFF = 64
        dp.reg_p = 64
        assert dp.verify_multiplication(200, 200) == True


class TestCycleCounter:
    """Pruebas del contador de ciclos"""
    
    def test_cycle_count_increments(self):
        """Verifica que el contador de ciclos se incremente"""
        dp = Datapath(bit_width=8)
        dp.reset(1, 1)
        
        assert dp.cycle_count == 0
        
        controls = {'alu_op': 0, 'write_p': 1}
        dp.execute_operation(controls)
        
        assert dp.cycle_count == 1
        
        dp.execute_operation(controls)
        assert dp.cycle_count == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
