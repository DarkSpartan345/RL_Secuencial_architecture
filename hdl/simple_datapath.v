// Datapath genérico para SimpleCPU
// Contiene: Banco de registros, ALU, Shifter

module simple_datapath #(
    parameter BIT_WIDTH = 16,
    parameter NUM_REGS = 16
)(
    input wire clk,
    input wire rst_n,
    
    // Señales de control
    input wire [2:0] opcode,
    input wire [3:0] reg_a_sel,
    input wire [3:0] reg_b_sel,
    input wire [3:0] dest_reg,
    input wire reg_write,
    
    // Entrada de datos externos
    input wire load_operands,
    input wire [BIT_WIDTH-1:0] operand_a_in,
    input wire [BIT_WIDTH-1:0] operand_b_in,
    
    // Salidas
    output wire [BIT_WIDTH-1:0] reg_a_val,
    output wire [BIT_WIDTH-1:0] reg_b_val,
    output wire [BIT_WIDTH-1:0] result_out,
    output wire zero_flag,
    output wire equal_flag,
    output wire greater_flag
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

    // Banco de registros
    reg [BIT_WIDTH-1:0] registers [0:NUM_REGS-1];
    
    // Registros constantes (solo lectura)
    localparam NUM_CONST_REGS = 8;
    
    // Valores de registros seleccionados
    assign reg_a_val = registers[reg_a_sel];
    assign reg_b_val = registers[reg_b_sel];
    
    // Resultado de la ALU
    reg [BIT_WIDTH-1:0] alu_result;
    
    // Flags
    assign zero_flag = (reg_b_val == 0);
    assign equal_flag = (reg_a_val == reg_b_val);
    assign greater_flag = (reg_a_val > reg_b_val);
    
    // Lógica de ALU (combinacional)
    always @(*) begin
        case (opcode)
            OP_ADD: alu_result = reg_a_val + reg_b_val;
            OP_SUB: alu_result = reg_a_val - reg_b_val;
            OP_SHL: alu_result = reg_a_val << (reg_b_val[3:0]);
            OP_SHR: alu_result = reg_a_val >> (reg_b_val[3:0]);
            OP_MOV: alu_result = reg_a_val;
            default: alu_result = 0;
        endcase
    end
    
    assign result_out = alu_result;
    
    // Inicialización y escritura de registros
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset: inicializar registros constantes
            registers[0] <= 0;   // R0 = 0
            registers[1] <= 1;   // R1 = 1
            registers[2] <= 2;   // R2 = 2
            registers[3] <= 4;   // R3 = 4
            registers[4] <= 8;   // R4 = 8
            registers[5] <= 16;  // R5 = 16
            registers[6] <= 32;  // R6 = 32
            registers[7] <= 64;  // R7 = 64
            
            // Reset registros escribibles
            for (i = NUM_CONST_REGS; i < NUM_REGS; i = i + 1) begin
                registers[i] <= 0;
            end
        end else begin
            // Cargar operandos externos
            if (load_operands) begin
                registers[8] <= operand_a_in;   // R8 = A (multiplicando)
                registers[9] <= operand_b_in;   // R9 = B (multiplicador)
                registers[10] <= 0;             // R10 = P (producto)
            end
            // Escribir resultado en registro destino (si es escribible)
            else if (reg_write && dest_reg >= NUM_CONST_REGS) begin
                registers[dest_reg] <= alu_result;
            end
        end
    end

endmodule
