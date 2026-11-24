// tb_linear_layer.sv - short deterministic self-checking test for linear_layer.sv
// Tiny layer: 4 inputs x 3 neurons. All weights, biases and expected sums are hand-made.
`timescale 1ns/1ps

module tb_linear_layer;

  localparam int N_IN  = 4;
  localparam int N_NPE = 3;
  localparam int DEPTH = 8;                   // rows used: 2 bias + 4 inputs = 6
  localparam int ACC_W = 25;
  localparam int AW    = 3;

  logic                        clk      = 0;
  logic                        rst_n    = 0;
  logic                        wr_en    = 0;
  logic [AW-1:0]               wr_addr  = 0;
  logic [N_NPE*8-1:0]          wr_data  = 0;
  logic                        start    = 0;
  logic                        done;
  logic                        in_valid = 0;
  logic                        in_ready;
  logic [7:0]                  in_data  = 0;
  logic [N_NPE-1:0][ACC_W-1:0] acc;

  int errors = 0;
  int cycles;

  linear_layer #(.N_IN(N_IN), .N_NPE(N_NPE), .DEPTH(DEPTH), .ACC_W(ACC_W)) dut (.*);

  always #10 clk = ~clk;   // 50 MHz

  // ---------------------------------------------------------------------------
  // The hand-made layer
  //   bias:      neuron0 = 5,  neuron1 = -100,           neuron2 = 1000
  //   weights:   neuron0 = [ 1,  2,   3,  4]
  //              neuron1 = [-1, -2, 127,  0]
  //              neuron2 = [10,-128,  0, -5]
  // A row is {lane2, lane1, lane0} = {neuron2, neuron1, neuron0}
  // ---------------------------------------------------------------------------
  task automatic load_weights();
    write_row(3'd0, 24'h03FF00);   // bias high bytes: 0x00 (5), 0xFF (-100), 0x03 (1000)
    write_row(3'd1, 24'hE89C05);   // bias low bytes:  0x05,     0x9C,        0xE8
    write_row(3'd2, 24'h0AFF01);   // k=0: w = 1, -1,   10
    write_row(3'd3, 24'h80FE02);   // k=1: w = 2, -2, -128
    write_row(3'd4, 24'h007F03);   // k=2: w = 3, 127,   0
    write_row(3'd5, 24'hFB0004);   // k=3: w = 4,   0,  -5
  endtask

  task automatic write_row(input logic [AW-1:0] a, input logic [N_NPE*8-1:0] d);
    wr_en = 1; wr_addr = a; wr_data = d;
    @(posedge clk); #1;
    wr_en = 0;
  endtask

  // Run one inference. with_gaps = 1 puts an idle cycle before every input.
  // cycles = clock edges from the start pulse until done is high.
  task automatic run_layer(input logic [7:0] xs [N_IN], input bit with_gaps, output int cycles);
    start = 1;
    @(posedge clk); #1;
    start  = 0;
    cycles = 1;
    for (int k = 0; k < N_IN; k++) begin
      if (with_gaps) begin
        in_valid = 0;
        @(posedge clk); #1; cycles++;
      end
      in_valid = 1;
      in_data  = xs[k];
      while (!in_ready) begin @(posedge clk); #1; cycles++; end   // wait while the bias loads
      @(posedge clk); #1; cycles++;                                // the layer takes x_k here
    end
    in_valid = 0;
    while (!done) begin @(posedge clk); #1; cycles++; end
  endtask

  // Compare the three accumulators with hand-computed sums
  task automatic check_acc(input string name, input int e0, input int e1, input int e2);
    int expected [3];
    expected = '{e0, e1, e2};
    for (int j = 0; j < N_NPE; j++) begin
      if (acc[j] !== ACC_W'(expected[j])) begin
        $display("FAIL  %-26s neuron %0d: got %0d, expected %0d", name, j, $signed(acc[j]), expected[j]);
        errors++;
      end
      else $display("ok    %-26s neuron %0d = %0d", name, j, $signed(acc[j]));
    end
  endtask

  initial begin
    repeat (2) @(posedge clk); #1;
    rst_n = 1;
    check_acc("reset", 0, 0, 0);
    load_weights();

    // Test 1: inputs x = [1, 2, 128, 10], one per cycle, no gaps
    //   neuron0 =    5 + 1*1  + 2*2    + 128*3   + 10*4    =   434
    //   neuron1 = -100 + 1*-1 + 2*-2   + 128*127 + 10*0    = 16151
    //   neuron2 = 1000 + 1*10 + 2*-128 + 128*0   + 10*-5   =   704
    run_layer('{8'd1, 8'd2, 8'd128, 8'd10}, 1'b0, cycles);
    check_acc("T1 x=[1,2,128,10]", 434, 16151, 704);
    // 1 (start) + 2 (bias rows) + 4 (inputs) + 2 (drain) = 9 clock edges
    if (cycles !== 9) begin $display("FAIL  T1 cycles: got %0d, expected 9", cycles); errors++; end
    else $display("ok    T1 cycles to done        = %0d", cycles);

    // Test 2: the same inputs with an idle cycle before each one -> the same sums
    run_layer('{8'd1, 8'd2, 8'd128, 8'd10}, 1'b1, cycles);
    check_acc("T2 same, with gaps", 434, 16151, 704);

    // Test 3: all inputs 0 -> only the biases remain (also proves the old sums were cleared)
    run_layer('{8'd0, 8'd0, 8'd0, 8'd0}, 1'b0, cycles);
    check_acc("T3 x=[0,0,0,0]", 5, -100, 1000);

    // Test 4: all inputs 128 (= 1.0)
    //   neuron0 =    5 + 128*(1+2+3+4)       =   1285
    //   neuron1 = -100 + 128*(-1-2+127+0)    =  15772
    //   neuron2 = 1000 + 128*(10-128+0-5)    = -14744
    run_layer('{8'd128, 8'd128, 8'd128, 8'd128}, 1'b0, cycles);
    check_acc("T4 x=[128,128,128,128]", 1285, 15772, -14744);

    if (errors == 0) $display("\nPASS: linear_layer - all checks passed\n");
    else             $display("\nFAIL: linear_layer - %0d check(s) failed\n", errors);
    $finish;
  end

endmodule
