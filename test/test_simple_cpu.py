import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.simple_cpu import SimpleCPU, OpCode

class TestSimpleCPU:
    @pytest.fixture
    def cpu(self):
        return SimpleCPU(num_registers=16, bit_width=8)

    def test_initialization(self, cpu):
        """Test proper initialization of registers and state."""
        assert cpu.num_registers == 16
        assert cpu.bit_width == 8
        assert cpu.pc == 0
        assert cpu.halted == False
        # Constants verification
        assert cpu.registers[0] == 0
        assert cpu.registers[1] == 1
        assert cpu.registers[2] == 2
        assert cpu.registers[3] == 4  # 1 << 2
        assert cpu.registers[7] == 64 # 1 << 6

    def test_reset(self, cpu):
        """Test reset functionality."""
        cpu.pc = 10
        cpu.halted = True
        cpu.cycle_count = 50
        cpu.registers[10] = 255
        
        cpu.reset()
        
        assert cpu.pc == 0
        assert cpu.halted == False
        assert cpu.cycle_count == 0
        assert cpu.registers[10] == 0  # Should be cleared
        # Constants should still be there
        assert cpu.registers[1] == 1

    def test_opcode_add(self, cpu):
        """Test ADD instruction."""
        # Setup: R8 = 5, R9 = 3
        cpu._write_register(8, 5)
        cpu._write_register(9, 3)
        
        # ADD R10 = R8 + R9. Next PC = 1
        # [ADD, 8, 9, 10, 1]
        instruction = [OpCode.ADD, 8, 9, 10, 1]
        cpu.execute(instruction)
        
        assert cpu.registers[10] == 8
        assert cpu.pc == 1

    def test_opcode_sub(self, cpu):
        """Test SUB instruction."""
        cpu._write_register(8, 10)
        cpu._write_register(9, 3)
        
        # SUB R10 = R8 - R9
        instruction = [OpCode.SUB, 8, 9, 10, 1]
        cpu.execute(instruction)
        
        assert cpu.registers[10] == 7

    def test_opcode_shifts(self, cpu):
        """Test SHL and SHR instructions."""
        cpu._write_register(8, 1)  # Value = 1
        
        # SHL R10 = R8 << 2 (Shift by explicitly shifting 2 using R2 constant)
        # R2 is constant 2
        instruction = [OpCode.SHL, 8, 2, 10, 1]
        cpu.execute(instruction)
        assert cpu.registers[10] == 4  # 1 << 2 = 4
        
        # SHR R11 = R10 >> 1 (using R1 constant=1)
        instruction = [OpCode.SHR, 10, 1, 11, 2]
        cpu.execute(instruction)
        assert cpu.registers[11] == 2  # 4 >> 1 = 2

    def test_opcode_mov(self, cpu):
        """Test MOV instruction."""
        cpu._write_register(8, 42)
        
        # MOV R9 = R8
        instruction = [OpCode.MOV, 8, 0, 9, 1]
        cpu.execute(instruction)
        
        assert cpu.registers[9] == 42

    def test_opcode_branch_gt(self, cpu):
        """Test BGT (Branch if Greater Than)."""
        cpu._write_register(8, 10)
        cpu._write_register(9, 5)
        
        # BGT R8 > R9 ? goto 5 : goto 2
        instruction = [OpCode.BGT, 8, 9, 5, 2]
        cpu.execute(instruction)
        assert cpu.pc == 5
        
        # BGT R9 > R8? (False)
        instruction = [OpCode.BGT, 9, 8, 5, 2]
        cpu.execute(instruction)
        assert cpu.pc == 2

    def test_opcode_branch_eq(self, cpu):
        """Test BEQ (Branch if Equal)."""
        cpu._write_register(8, 10)
        cpu._write_register(9, 10)
        
        # BEQ R8 == R9 ? goto 5 : goto 2
        instruction = [OpCode.BEQ, 8, 9, 5, 2]
        cpu.execute(instruction)
        assert cpu.pc == 5
        
        cpu._write_register(9, 11)
        # BEQ (False)
        instruction = [OpCode.BEQ, 8, 9, 5, 2]
        cpu.execute(instruction)
        assert cpu.pc == 2

    def test_halt(self, cpu):
        """Test HALT instruction."""
        # HALT
        instruction = [OpCode.HALT, 0, 0, 0, 0]
        cpu.execute(instruction)
        
        assert cpu.halted == True

    def test_overflow(self, cpu):
        """Test overflow behavior (8-bit wrapping)."""
        cpu._write_register(8, 250)
        cpu._write_register(9, 10)
        
        # 250 + 10 = 260 -> 4 (in 8-bit)
        instruction = [OpCode.ADD, 8, 9, 10, 1]
        cpu.execute(instruction)
        
        assert cpu.registers[10] == 4
