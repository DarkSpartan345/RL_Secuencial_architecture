import sys
from pathlib import Path

# Add current directory to path to find shift_and_add_demo
sys.path.insert(0, str(Path(__file__).parent))

try:
    from simple_cpu import OpCode
    from shift_and_add_demo import SHIFT_AND_ADD_PROGRAM
except ImportError:
    # Handle running from root
    sys.path.insert(0, 'env')
    from simple_cpu import OpCode
    from shift_and_add_demo import SHIFT_AND_ADD_PROGRAM

def generate_mem_file(program, output_path="hdl/program.mem", memory_size=16):
    """
    Generates a .mem file for Verilog $readmemh.
    Format: 5 hex digits (19 bits)
    [Op(3)][RegA(4)][RegB(4)][Dest1(4)][Dest2(4)]
    """
    print(f"Generando archivo de memoria: {output_path}")
    
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
            # Total 19 bits. 
            # In hex, we need 5 digits (20 bits capacity).
            
            val = (opcode << 16) | (reg_a << 12) | (reg_b << 8) | (dest_1 << 4) | dest_2
            
            # Write as 5-digit hex
            hex_str = f"{val:05X}"
            f.write(f"{hex_str}\n")
            
            print(f"Addr {i:02}: {instruction} -> {hex_str}")
            
        # Pad with HALT instructions (OpCode 7, rest 0 -> 70000)
        remaining = memory_size - len(program)
        if remaining > 0:
            halt_val = (7 << 16)
            halt_hex = f"{halt_val:05X}"
            for _ in range(remaining):
                f.write(f"{halt_hex}\n")
            print(f"Padding {remaining} lines with HALT ({halt_hex})")

    print("Done.")

if __name__ == "__main__":
    # Determine output path relative to script location
    # If running from root: hdl/program.mem
    # If running from env: ../hdl/program.mem
    
    # We'll assume running from root as per standard practice, but check
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    
    output_file = root_dir / "hdl" / "program.mem"
    
    generate_mem_file(SHIFT_AND_ADD_PROGRAM, output_path=output_file)
