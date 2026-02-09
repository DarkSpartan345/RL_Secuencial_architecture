// Unidad de Control para SimpleCPU (con pipeline FETCH/EXECUTE)
// Lee instrucciones de ROM y genera señales de control

module simple_control #(
    parameter PROGRAM_SIZE = 16,
    parameter PROGRAM_FILE = "program.mem"
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // Flags del datapath (evaluados después de aplicar selectores)
    input wire zero_flag,
    input wire equal_flag,
    input wire greater_flag,
    
    // Señales de control hacia el datapath
    output reg [2:0] opcode,
    output reg [3:0] reg_a_sel,
    output reg [3:0] reg_b_sel,
    output reg [3:0] dest_reg,
    output reg reg_write,
    output reg load_operands,
    
    // Estado
    output reg [3:0] pc,
    output reg done
);

    // OpCodes
    localparam OP_ADD = 3'd0;
    localparam OP_SUB = 3'd1;
    localparam OP_SHL = 3'd2;
    localparam OP_SHR = 3'd3;
    localparam OP_MOV = 3'd4;
    localparam OP_BGT = 3'd5;
    localparam OP_BEQ = 3'd6;
    localparam OP_HALT = 3'd7;
    
    // Memoria de programa (ROM)
    // Formato: [OpCode(3)][RegA(4)][RegB(4)][Dest1(4)][Dest2(4)] = 19 bits
    reg [18:0] program_rom [0:PROGRAM_SIZE-1];
    
    // Instrucción actual
    wire [18:0] instruction;
    wire [2:0] instr_opcode;
    wire [3:0] instr_reg_a;
    wire [3:0] instr_reg_b;
    wire [3:0] instr_dest1;
    wire [3:0] instr_dest2;
    
    // Decodificar instrucción
    assign instruction = program_rom[pc];
    assign instr_opcode = instruction[18:16];
    assign instr_reg_a = instruction[15:12];
    assign instr_reg_b = instruction[11:8];
    assign instr_dest1 = instruction[7:4];
    assign instr_dest2 = instruction[3:0];
    
    // Registros de pipeline
    reg [2:0] saved_opcode;
    reg [3:0] saved_dest1;
    reg [3:0] saved_dest2;
    
    // Estados de la FSM de control
    localparam S_IDLE   = 3'd0;
    localparam S_LOAD   = 3'd1;
    localparam S_FETCH  = 3'd2;  // Aplicar selectores, leer registros
    localparam S_EXEC   = 3'd3;  // Evaluar flags, escribir, calcular next PC
    localparam S_DONE   = 3'd4;
    
    reg [2:0] state;
    
    // Cargar programa desde archivo
    initial begin
        $readmemh(PROGRAM_FILE, program_rom);
    end
    
    // Lógica de control
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            pc <= 0;
            done <= 0;
            opcode <= 0;
            reg_a_sel <= 0;
            reg_b_sel <= 0;
            dest_reg <= 0;
            reg_write <= 0;
            load_operands <= 0;
            saved_opcode <= 0;
            saved_dest1 <= 0;
            saved_dest2 <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 0;
                    load_operands <= 0;
                    reg_write <= 0;
                    if (start) begin
                        state <= S_LOAD;
                        load_operands <= 1;
                    end
                end
                
                S_LOAD: begin
                    // Cargar operandos en el mismo ciclo
                    load_operands <= 0;
                    pc <= 0;
                    state <= S_FETCH;
                end
                
                S_FETCH: begin
                    // Aplicar selectores de registro y guardar info de instrucción
                    reg_a_sel <= instr_reg_a;
                    reg_b_sel <= instr_reg_b;
                    dest_reg <= instr_dest1;
                    opcode <= instr_opcode;
                    
                    // Guardar para el siguiente ciclo
                    saved_opcode <= instr_opcode;
                    saved_dest1 <= instr_dest1;
                    saved_dest2 <= instr_dest2;
                    
                    reg_write <= 0;  // No escribir aún
                    
                    // Ir a EXECUTE (los flags se evaluarán con los selectores correctos)
                    state <= S_EXEC;
                end
                
                S_EXEC: begin
                    // Ahora los flags reflejan los valores correctos de los registros
                    case (saved_opcode)
                        OP_ADD, OP_SUB, OP_SHL, OP_SHR, OP_MOV: begin
                            // Escribir resultado
                            reg_write <= 1;
                            pc <= saved_dest2;
                            state <= S_FETCH;
                        end
                        
                        OP_BGT: begin
                            reg_write <= 0;
                            if (greater_flag)
                                pc <= saved_dest1;
                            else
                                pc <= saved_dest2;
                            state <= S_FETCH;
                        end
                        
                        OP_BEQ: begin
                            reg_write <= 0;
                            if (equal_flag)
                                pc <= saved_dest1;
                            else
                                pc <= saved_dest2;
                            state <= S_FETCH;
                        end
                        
                        OP_HALT: begin
                            reg_write <= 0;
                            state <= S_DONE;
                        end
                        
                        default: begin
                            reg_write <= 0;
                            state <= S_DONE;
                        end
                    endcase
                end
                
                S_DONE: begin
                    done <= 1;
                    reg_write <= 0;
                    if (start) begin
                        state <= S_LOAD;
                        load_operands <= 1;
                        done <= 0;
                    end
                end
            endcase
        end
    end

endmodule
