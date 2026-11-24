// tb_npe.sv - short deterministic self-checking test for npe.sv
// Every expected value below is computed by hand (see the comment on each line).
`timescale 1ns/1ps

import sympa_pkg::*;

module tb_npe;

  localparam int ACC_W = 25;


  logic                    clk   = 0;
  logic                    rst_n = 0;
  logic [2:0]              op    = OP_NOP;
  logic [7:0]              x     = 0;
  logic [7:0]              w     = 0;
  logic signed [ACC_W-1:0] sr_in = 0;
  logic signed [ACC_W-1:0] acc;
  logic signed [ACC_W-1:0] sr;

  int errors = 0;

  npe #(.ACC_W(ACC_W)) dut (.*);

  always #10 clk = ~clk;   // 20 ns period = 50 MHz

  // Apply one operation for one clock cycle, then return to NOP
  task automatic step(input logic [2:0] o, input logic [7:0] xi, input logic [7:0] wi);
    op = o; x = xi; w = wi;
    @(posedge clk); #1;
    op = OP_NOP;
  endtask

  // Compare a value with the hand-computed expectation
  task automatic check(input string name, input logic signed [ACC_W-1:0] got, input int expected);
    if (got !== ACC_W'(expected)) begin
      $display("FAIL  %-16s got %0d, expected %0d", name, got, expected);
      errors++;
    end
    else $display("ok    %-16s = %0d", name, got);
  endtask

  initial begin
    repeat (2) @(posedge clk); #1;
    rst_n = 1;
    check("reset acc", acc, 0);
    check("reset sr",  sr,  0);

    // Test 1: bias +10 (0x000A), then four MACs
    step(OP_BIAS_HI, 8'd0, 8'h00);
    step(OP_BIAS_LO, 8'd0, 8'h0A);
    check("T1 bias", acc, 10);
    step(OP_MAC, 8'd2,   8'd3);       //  10 + 2*3        =     16
    check("T1 mac 1", acc, 16);
    step(OP_MAC, 8'd128, 8'hFF);      //  16 + 128*(-1)   =   -112
    check("T1 mac 2", acc, -112);
    step(OP_MAC, 8'd0,   8'd127);     // -112 + 0*127     =   -112
    check("T1 mac 3", acc, -112);
    step(OP_MAC, 8'd128, 8'd127);     // -112 + 128*127   =  16144
    check("T1 mac 4", acc, 16144);
    step(OP_NOP, 8'd99,  8'd99);      // NOP changes nothing
    check("T1 nop", acc, 16144);

    // Test 2: LOAD_SR copies acc; SHIFT takes the neighbour's value; acc is untouched
    step(OP_LOAD_SR, 8'd0, 8'd0);
    check("T2 load_sr", sr, 16144);
    sr_in = -12345;
    step(OP_SHIFT, 8'd0, 8'd0);
    check("T2 shift", sr, -12345);
    check("T2 acc kept", acc, 16144);

    // Test 3: negative bias -300 (0xFED4), then the most negative product
    step(OP_BIAS_HI, 8'd0, 8'hFE);
    step(OP_BIAS_LO, 8'd0, 8'hD4);
    check("T3 bias", acc, -300);
    step(OP_MAC, 8'd128, 8'h80);      // -300 + 128*(-128) = -16684
    check("T3 mac", acc, -16684);

    // Test 4: worst negative case for 784 inputs: bias -32768 (0x8000), 784 x (128 * -128)
    step(OP_BIAS_HI, 8'd0, 8'h80);
    step(OP_BIAS_LO, 8'd0, 8'h00);
    check("T4 bias", acc, -32768);
    repeat (784) step(OP_MAC, 8'd128, 8'h80);
    check("T4 784 MACs", acc, -12877824);   // -32768 - 784*16384

    // Test 5: worst positive case: bias +32767 (0x7FFF), 784 x (128 * 127)
    step(OP_BIAS_HI, 8'd0, 8'h7F);
    step(OP_BIAS_LO, 8'd0, 8'hFF);
    check("T5 bias", acc, 32767);
    repeat (784) step(OP_MAC, 8'd128, 8'd127);
    check("T5 784 MACs", acc, 12777471);    // 32767 + 784*16256

    if (errors == 0) $display("\nPASS: npe - all checks passed\n");
    else             $display("\nFAIL: npe - %0d check(s) failed\n", errors);
    $finish;
  end

endmodule
