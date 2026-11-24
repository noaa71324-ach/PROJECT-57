// tb_linear_layer_full.sv - linear_layer at its real size: 784 inputs x 64 neurons
// Deterministic: weights, biases and inputs follow simple rules, so every expected sum
// is a short formula (written next to each test, with sample values).
`timescale 1ns/1ps

module tb_linear_layer_full;

  localparam int N_IN  = 784;
  localparam int N_NPE = 64;
  localparam int DEPTH = 1024;                // rows used: 2 bias + 784 inputs = 786
  localparam int ACC_W = 25;
  localparam int AW    = 10;

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

  logic [7:0] xs [N_IN];                      // the inputs of the current test
  int errors = 0;
  int cycles;

  linear_layer #(.N_IN(N_IN), .N_NPE(N_NPE), .DEPTH(DEPTH), .ACC_W(ACC_W)) dut (.*);

  always #10 clk = ~clk;   // 50 MHz

  // ---------------------------------------------------------------------------
  // Rules for the weights and biases
  //   bias_j          = 100*j - 3200            (-3200 .. 3100)
  //   weight set A:   w_jk = j - 32             (same for every input k; -32 .. 31)
  //   weight set B:   w_jk = 2 if k == 10*j+5, else 0
  //                   (each neuron "picks" one input, so a wrong row/input pairing shows up)
  // ---------------------------------------------------------------------------
  function automatic int bias(input int j);
    return 100*j - 3200;
  endfunction

  function automatic logic [7:0] weight(input bit set_b, input int j, input int k);
    if (!set_b) return 8'(j - 32);
    else        return (k == 10*j + 5) ? 8'd2 : 8'd0;
  endfunction

  task automatic write_row(input logic [AW-1:0] a, input logic [N_NPE*8-1:0] d);
    wr_en = 1; wr_addr = a; wr_data = d;
    @(posedge clk); #1;
    wr_en = 0;
  endtask

  // Fill the weight memory: row 0/1 = bias high/low bytes, row 2+k = weights of input k
  task automatic load_weights(input bit set_b);
    logic [N_NPE*8-1:0] row_hi, row_lo, row;
    logic [15:0]        b;
    for (int j = 0; j < N_NPE; j++) begin
      b = 16'(bias(j));
      row_hi[8*j +: 8] = b[15:8];
      row_lo[8*j +: 8] = b[7:0];
    end
    write_row(AW'(0), row_hi);
    write_row(AW'(1), row_lo);
    for (int k = 0; k < N_IN; k++) begin
      for (int j = 0; j < N_NPE; j++) row[8*j +: 8] = weight(set_b, j, k);
      write_row(AW'(2 + k), row);
    end
  endtask

  // Run one inference on xs[]; cycles = clock edges from start until done
  task automatic run_layer(output int cycles);
    start = 1;
    @(posedge clk); #1;
    start  = 0;
    cycles = 1;
    for (int k = 0; k < N_IN; k++) begin
      in_valid = 1;
      in_data  = xs[k];
      while (!in_ready) begin @(posedge clk); #1; cycles++; end
      @(posedge clk); #1; cycles++;
    end
    in_valid = 0;
    while (!done) begin @(posedge clk); #1; cycles++; end
  endtask

  // Check all 64 neurons against expected[]; print neurons 0, 1, 32 and 63 as samples
  task automatic check_all(input string name, input int expected [N_NPE]);
    int bad = 0;
    for (int j = 0; j < N_NPE; j++) begin
      if (acc[j] !== ACC_W'(expected[j])) begin
        if (bad < 5) $display("FAIL  %s neuron %0d: got %0d, expected %0d", name, j, $signed(acc[j]), expected[j]);
        bad++;
      end
    end
    $display("%s  %-28s neuron0=%0d neuron1=%0d neuron32=%0d neuron63=%0d  (%0d of 64 wrong)",
             (bad == 0) ? "ok  " : "FAIL", name,
             $signed(acc[0]), $signed(acc[1]), $signed(acc[32]), $signed(acc[63]), bad);
    errors += bad;
  endtask

  int expected [N_NPE];

  initial begin
    repeat (2) @(posedge clk); #1;
    rst_n = 1;

    // ---- Weight set A ----
    load_weights(1'b0);

    // Test 1: all inputs 1  ->  acc_j = bias_j + 784*1*(j-32) = 884*j - 28288
    //   neuron0 = -28288, neuron1 = -27404, neuron32 = 0, neuron63 = 27404
    foreach (xs[k]) xs[k] = 8'd1;
    run_layer(cycles);
    foreach (expected[j]) expected[j] = bias(j) + 784*1*(j - 32);
    check_all("T1 set A, x=1", expected);
    // 1 (start) + 2 (bias rows) + 784 (inputs) + 2 (drain) = 789 clock edges
    if (cycles !== 789) begin $display("FAIL  T1 cycles: got %0d, expected 789", cycles); errors++; end
    else $display("ok    T1 cycles to done = %0d", cycles);

    // Test 2: all inputs 128 (= 1.0)  ->  acc_j = bias_j + 784*128*(j-32)
    //   neuron0 = -3,214,464, neuron32 = 0, neuron63 = 3,114,012 (fits in 25 bits)
    foreach (xs[k]) xs[k] = 8'd128;
    run_layer(cycles);
    foreach (expected[j]) expected[j] = bias(j) + 784*128*(j - 32);
    check_all("T2 set A, x=128", expected);

    // ---- Weight set B ----
    load_weights(1'b1);

    // Test 3: x_k = k mod 128  ->  acc_j = bias_j + 2 * x_(10j+5) = bias_j + 2*((10j+5) mod 128)
    //   neuron0 = -3200 + 2*5 = -3190, neuron1 = -3100 + 2*15 = -3070,
    //   neuron32 = 0 + 2*69 = 138,     neuron63 = 3100 + 2*123 = 3346
    foreach (xs[k]) xs[k] = 8'(k % 128);
    run_layer(cycles);
    foreach (expected[j]) expected[j] = bias(j) + 2*((10*j + 5) % 128);
    check_all("T3 set B, x=k mod 128", expected);

    if (errors == 0) $display("\nPASS: linear_layer full size (784 x 64) - all checks passed\n");
    else             $display("\nFAIL: linear_layer full size - %0d check(s) failed\n", errors);
    $finish;
  end

endmodule
