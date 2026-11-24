// tb_weight_mem.sv - short deterministic self-checking test for weight_mem.sv
// Small memory: 3 lanes (3 NPEs) x 8 rows. Every expected value is written by hand.
`timescale 1ns/1ps

module tb_weight_mem;

  localparam int N_NPE = 3;
  localparam int DEPTH = 8;
  localparam int AW    = 3;
  localparam int W     = N_NPE*8;             // 24-bit rows

  logic          clk     = 0;
  logic          wr_en   = 0;
  logic [AW-1:0] wr_addr = 0;
  logic [W-1:0]  wr_data = 0;
  logic [AW-1:0] rd_addr = 0;
  logic [W-1:0]  rd_data;

  int errors = 0;

  weight_mem #(.N_NPE(N_NPE), .DEPTH(DEPTH)) dut (.*);

  always #10 clk = ~clk;   // 50 MHz

  // Write one full row in one clock
  task automatic write_row(input logic [AW-1:0] a, input logic [W-1:0] d);
    wr_en = 1; wr_addr = a; wr_data = d;
    @(posedge clk); #1;
    wr_en = 0;
  endtask

  // Compare rd_data with the expected row and print the three lanes
  task automatic check(input string name, input logic [W-1:0] expected);
    if (rd_data !== expected) begin
      $display("FAIL  %-24s got %h, expected %h", name, rd_data, expected);
      errors++;
    end
    else $display("ok    %-24s = %h  (lane0=%h lane1=%h lane2=%h)",
                  name, rd_data, rd_data[7:0], rd_data[15:8], rd_data[23:16]);
  endtask

  initial begin
    @(posedge clk); #1;

    // Write four rows. Lane j is bits [8j+7:8j]: 24'h332211 means lane0=11, lane1=22, lane2=33
    write_row(3'd0, 24'h332211);
    write_row(3'd1, 24'h007F80);      // lane0=80 (-128), lane1=7F (+127), lane2=00
    write_row(3'd5, 24'hAB01FE);      // lane0=FE (-2),   lane1=01,        lane2=AB
    write_row(3'd7, 24'hC0FFEE);      // the last row

    // Test 1: read row 5; the data appears 2 clocks after the address
    rd_addr = 3'd5;
    repeat (2) @(posedge clk); #1;
    check("T1 row 5", 24'hAB01FE);

    // Test 2: the latency is exactly 2 - after 1 clock the previous row is still there
    rd_addr = 3'd1;
    @(posedge clk); #1;
    check("T2 after 1 clock (old)", 24'hAB01FE);
    @(posedge clk); #1;
    check("T2 after 2 clocks (new)", 24'h007F80);

    // Test 3: a new address every clock, as the layer will read; each row comes out 2 clocks later
    rd_addr = 3'd0; @(posedge clk); #1;
    rd_addr = 3'd7; @(posedge clk); #1;  check("T3 row 0", 24'h332211);
    rd_addr = 3'd5; @(posedge clk); #1;  check("T3 row 7", 24'hC0FFEE);
    rd_addr = 3'd1; @(posedge clk); #1;  check("T3 row 5", 24'hAB01FE);
                    @(posedge clk); #1;  check("T3 row 1", 24'h007F80);

    // Test 4: overwrite row 1 and read the new value
    write_row(3'd1, 24'h123456);
    rd_addr = 3'd1;
    repeat (2) @(posedge clk); #1;
    check("T4 row 1 rewritten", 24'h123456);

    // Test 5: the other rows did not change
    rd_addr = 3'd0;
    repeat (2) @(posedge clk); #1;
    check("T5 row 0 unchanged", 24'h332211);

    if (errors == 0) $display("\nPASS: weight_mem - all checks passed\n");
    else             $display("\nFAIL: weight_mem - %0d check(s) failed\n", errors);
    $finish;
  end

endmodule
