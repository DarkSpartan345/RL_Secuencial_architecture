// Módulo Top para SimpleCPU
// Conecta la unidad de control con el datapath

module simple_cpu_top #(
    parameter BIT_WIDTH = 16,
    parameter PROGRAM_FILE = "program.mem"
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [BIT_WIDTH-1:0] operand_a,
    input wire [BIT_WIDTH-1:0] operand_b,
    output wire [BIT_WIDTH-1:0] result,
    output wire done,
    output wire [3:0] pc
);

    // Señales internas
    wire [2:0] opcode;
    wire [3:0] reg_a_sel;
    wire [3:0] reg_b_sel;
    wire [3:0] dest_reg;
    wire reg_write;
    wire load_operands;
    
    wire [BIT_WIDTH-1:0] reg_a_val;
    wire [BIT_WIDTH-1:0] reg_b_val;
    wire [BIT_WIDTH-1:0] alu_result;
    wire zero_flag;
    wire equal_flag;
    wire greater_flag;
    
    // Instanciar Unidad de Control
    simple_control #(
        .PROGRAM_FILE(PROGRAM_FILE)
    ) control (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .zero_flag(zero_flag),
        .equal_flag(equal_flag),
        .greater_flag(greater_flag),
        .opcode(opcode),
        .reg_a_sel(reg_a_sel),
        .reg_b_sel(reg_b_sel),
        .dest_reg(dest_reg),
        .reg_write(reg_write),
        .load_operands(load_operands),
        .pc(pc),
        .done(done)
    );
    
    // Instanciar Datapath
    simple_datapath #(
        .BIT_WIDTH(BIT_WIDTH)
    ) datapath (
        .clk(clk),
        .rst_n(rst_n),
        .opcode(opcode),
        .reg_a_sel(reg_a_sel),
        .reg_b_sel(reg_b_sel),
        .dest_reg(dest_reg),
        .reg_write(reg_write),
        .load_operands(load_operands),
        .operand_a_in(operand_a),
        .operand_b_in(operand_b),
        .reg_a_val(reg_a_val),
        .reg_b_val(reg_b_val),
        .result_out(alu_result),
        .zero_flag(zero_flag),
        .equal_flag(equal_flag),
        .greater_flag(greater_flag)
    );
    

    assign result = datapath.registers[10];

endmodule
