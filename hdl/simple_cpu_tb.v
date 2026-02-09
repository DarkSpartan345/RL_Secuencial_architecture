`timescale 1ns/1ps

// Testbench para SimpleCPU con memoria de programa
module simple_cpu_tb;
    parameter BIT_WIDTH = 16;
    parameter CLK_PERIOD = 10;
    
    // Señales
    reg clk;
    reg rst_n;
    reg start;
    reg [BIT_WIDTH-1:0] operand_a;
    reg [BIT_WIDTH-1:0] operand_b;
    wire [BIT_WIDTH-1:0] result;
    wire done;
    wire [3:0] pc;
    
    // Instanciar DUT
    simple_cpu_top #(
        .BIT_WIDTH(BIT_WIDTH),
        .PROGRAM_FILE("hdl/program.mem")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .operand_a(operand_a),
        .operand_b(operand_b),
        .result(result),
        .done(done),
        .pc(pc)
    );
    
    // Generador de reloj
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Variables de test
    integer test_count;
    integer error_count;
    integer cycles;
    reg [BIT_WIDTH-1:0] expected;
    
    // Task para ejecutar una multiplicación
    task run_multiply;
        input [BIT_WIDTH-1:0] a;
        input [BIT_WIDTH-1:0] b;
        input [BIT_WIDTH-1:0] exp;
        begin
            test_count = test_count + 1;
            operand_a = a;
            operand_b = b;
            expected = exp;
            cycles = 0;
            
            // Pulso de start
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;
            
            // Esperar done o timeout
            while (!done && cycles < 500) begin
                @(posedge clk);
                cycles = cycles + 1;
                if (cycles < 20)
                    $display("  Cycle %0d: PC=%0d, state=%0d, opcode=%0d, done=%0d", 
                             cycles, pc, dut.control.state, dut.control.opcode, done);
            end
            
            // Verificar resultado
            if (!done) begin
                $display("Test #%0d: %0d * %0d - ERROR: Timeout after %0d cycles", 
                         test_count, a, b, cycles);
                error_count = error_count + 1;
            end else if (result !== exp) begin
                $display("Test #%0d: %0d * %0d = %0d (expected %0d) - FAILED [%0d cycles]", 
                         test_count, a, b, result, exp, cycles);
                error_count = error_count + 1;
            end else begin
                $display("Test #%0d: %0d * %0d = %0d - PASSED [%0d cycles]", 
                         test_count, a, b, result, cycles);
            end
            
            // Reset para siguiente test
            rst_n = 0;
            @(posedge clk);
            @(posedge clk);
            rst_n = 1;
            @(posedge clk);
        end
    endtask
    
    // Test principal
    initial begin
        $dumpfile("simple_cpu_tb.vcd");
        $dumpvars(0, simple_cpu_tb);
        
        $display("========================================");
        $display("Testbench: SimpleCPU con ROM de Programa");
        $display("Bit width: %0d", BIT_WIDTH);
        $display("========================================\n");
        
        // Inicializar
        test_count = 0;
        error_count = 0;
        rst_n = 0;
        start = 0;
        operand_a = 0;
        operand_b = 0;
        
        // Reset
        #(CLK_PERIOD * 3);
        rst_n = 1;
        #(CLK_PERIOD * 2);
        
        // === Casos básicos ===
        $display("--- Casos básicos ---");
        run_multiply(3, 5, 15);
        run_multiply(7, 8, 56);
        run_multiply(12, 10, 120);
        
        // === Casos con cero ===
        $display("\n--- Casos con cero ---");
        run_multiply(0, 5, 0);
        run_multiply(5, 0, 0);
        
        // === Casos con uno ===
        $display("\n--- Casos con uno ---");
        run_multiply(1, 42, 42);
        run_multiply(42, 1, 42);
        
        // === Casos más grandes ===
        $display("\n--- Casos grandes ---");
        run_multiply(15, 15, 225);
        run_multiply(100, 2, 200);
        run_multiply(255, 255, 65025);
        
        // === Resumen ===
        $display("\n========================================");
        $display("Testbench completado");
        $display("Total tests: %0d", test_count);
        $display("Errores: %0d", error_count);
        if (error_count == 0)
            $display("RESULTADO: TODOS LOS TESTS PASARON ✓");
        else
            $display("RESULTADO: %0d TESTS FALLARON ✗", error_count);
        $display("========================================\n");
        
        $finish;
    end

endmodule
