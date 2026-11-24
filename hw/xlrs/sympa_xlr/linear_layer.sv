// linear_layer.sv - first linear layer: weight_mem + N_NPE x npe + a small controller
// After start: loads the bias (2 rows), then takes N_IN inputs one per cycle (gaps allowed)
// and broadcasts each input to all NPEs with its weight row. When done is high,
// acc[j] holds  bias_j + sum_k x_k * w_jk  for every neuron j.
//
// Weight memory layout (spec section 6.1): row 0 = bias high bytes, row 1 = bias low bytes,
// row 2+k = weights of input k. Lane j of every row belongs to NPE j.

import sympa_pkg::*;

module linear_layer #(
  parameter int N_IN  = 784,                  // inputs per inference
  parameter int N_NPE = 64,                   // neurons (one NPE each)
  parameter int DEPTH = 1024,                 // weight memory rows
  parameter int ACC_W = 25,                   // accumulator width
  parameter int AW    = $clog2(DEPTH)
)(
  input  logic                         clk,
  input  logic                         rst_n,
  // weight memory write port (used to load the weights before start)
  input  logic                         wr_en,
  input  logic [AW-1:0]                wr_addr,
  input  logic [N_NPE*8-1:0]           wr_data,
  // control
  input  logic                         start,     // one-cycle pulse: begin a new inference
  output logic                         done,      // high when acc holds the result
  // input stream: x_0 .. x_(N_IN-1), one byte per accepted cycle
  input  logic                         in_valid,
  output logic                         in_ready,
  input  logic [7:0]                   in_data,
  // results: acc[j] = neuron j
  output logic [N_NPE-1:0][ACC_W-1:0]  acc
);


  typedef enum logic [2:0] {IDLE, BIAS_HI, BIAS_LO, RUN, DRAIN, DONE} state_t;

  state_t                  state;
  logic [$clog2(N_IN)-1:0] k;                 // index of the next input
  logic                    drain_cnt;         // counts the 2 drain cycles

  // ---------------------------------------------------------------------------
  // Stage 0 (this cycle): choose the weight row, the operation and the input
  // ---------------------------------------------------------------------------
  logic [AW-1:0] rd_addr;
  logic [2:0]    op_now;
  logic [7:0]    x_now;
  logic          accept;

  assign in_ready = (state == RUN);
  assign accept   = in_valid && in_ready;
  assign done     = (state == DONE);

  always_comb begin
    rd_addr = '0;
    op_now  = OP_NOP;
    x_now   = '0;
    case (state)
      BIAS_HI: begin rd_addr = AW'(0); op_now = OP_BIAS_HI; end
      BIAS_LO: begin rd_addr = AW'(1); op_now = OP_BIAS_LO; end
      RUN:     if (accept) begin
                 rd_addr = AW'(2 + k);
                 op_now  = OP_MAC;
                 x_now   = in_data;
               end
      default: ;                              // IDLE, DRAIN, DONE: NOP
    endcase
  end

  // ---------------------------------------------------------------------------
  // Controller
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= IDLE;
      k         <= '0;
      drain_cnt <= 1'b0;
    end
    else begin
      case (state)
        IDLE, DONE: if (start) begin state <= BIAS_HI; k <= '0; end
        BIAS_HI:    state <= BIAS_LO;
        BIAS_LO:    state <= RUN;
        RUN:        if (accept) begin
                      if (k == N_IN-1) begin state <= DRAIN; drain_cnt <= 1'b0; end
                      else k <= k + 1'b1;
                    end
        DRAIN:      if (drain_cnt) state <= DONE;  // 2 cycles: the last MAC reaches the NPEs
                    else drain_cnt <= 1'b1;
        default:    state <= IDLE;
      endcase
    end
  end

  // ---------------------------------------------------------------------------
  // Delay op and x by 2 cycles, to meet the weight row (weight_mem latency = 2)
  // ---------------------------------------------------------------------------
  logic [2:0] op_p1, op_p2;
  logic [7:0] x_p1,  x_p2;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      op_p1 <= OP_NOP; op_p2 <= OP_NOP;
      x_p1  <= '0;     x_p2  <= '0;
    end
    else begin
      op_p1 <= op_now; x_p1 <= x_now;
      op_p2 <= op_p1;  x_p2 <= x_p1;
    end
  end

  // ---------------------------------------------------------------------------
  // Weight memory and the NPEs
  // ---------------------------------------------------------------------------
  logic [N_NPE*8-1:0] w_row;                  // one row: one weight byte per NPE

  weight_mem #(.N_NPE(N_NPE), .DEPTH(DEPTH)) u_wmem (
    .clk     (clk),
    .wr_en   (wr_en),
    .wr_addr (wr_addr),
    .wr_data (wr_data),
    .rd_addr (rd_addr),
    .rd_data (w_row)
  );

  genvar j;
  generate
    for (j = 0; j < N_NPE; j++) begin : g_npe
      logic signed [ACC_W-1:0] acc_j;
      logic signed [ACC_W-1:0] sr_j;          // SR chain is the next step; unused for now

      npe #(.ACC_W(ACC_W)) u_npe (
        .clk   (clk),
        .rst_n (rst_n),
        .op    (op_p2),
        .x     (x_p2),
        .w     (w_row[8*j +: 8]),             // lane j
        .sr_in ('0),
        .acc   (acc_j),
        .sr    (sr_j)
      );

      assign acc[j] = acc_j;
    end
  endgenerate

endmodule
