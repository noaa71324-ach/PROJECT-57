// weight_mem.sv - weight memory of the layer (spec section 6.1)
// One row per input index k. Each row holds one weight byte per NPE:
// lane j = bits [8j+7:8j] goes to NPE j. All NPEs read the same row in the same cycle.
// Read latency is 2 clocks (registered address, then registered data), so Quartus maps it to M9K.

module weight_mem #(
  parameter int N_NPE = 64,                   // number of lanes (one byte per NPE)
  parameter int DEPTH = 1024,                 // number of rows
  parameter int AW    = $clog2(DEPTH)         // address width
)(
  input  logic               clk,
  // write port: one full row per cycle
  input  logic               wr_en,
  input  logic [AW-1:0]      wr_addr,
  input  logic [N_NPE*8-1:0] wr_data,
  // read port: data appears 2 clocks after rd_addr
  input  logic [AW-1:0]      rd_addr,
  output logic [N_NPE*8-1:0] rd_data
);

  logic [N_NPE*8-1:0] mem [DEPTH];            // the RAM (no reset: block RAM cannot be reset)
  logic [AW-1:0]      rd_addr_q;              // registered read address

  always_ff @(posedge clk) begin
    if (wr_en) mem[wr_addr] <= wr_data;
    rd_addr_q <= rd_addr;                     // clock 1: register the address
    rd_data   <= mem[rd_addr_q];              // clock 2: register the data
  end

endmodule
