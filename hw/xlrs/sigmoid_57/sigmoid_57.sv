
import xbox_def_pkg::*;

module sigmoid_57 (
  input        clk,
  input        rst_n,

  // Command Status Register Interface
  input        [XBOX_NUM_REGS-1:0][31:0] host_regs,              // regs accelerator write data, reflecting logicisters content as most recently written by SW over APB
  input  logic [XBOX_NUM_REGS-1:0]       host_regs_valid_pulse,  // logic written by host (APB) (one per register)
  output logic [XBOX_NUM_REGS-1:0][31:0] host_regs_data_out,     // regs accelerator write data, this is what SW will read when accessing the register
                                                                 // provided that the register specific host_regs_valid_out is asserted
  output logic [XBOX_NUM_REGS-1:0]       host_regs_valid_out,    // logic accelerator (one per register)
  input  logic [XBOX_NUM_REGS-1:0]       host_regs_read_pulse,   // Indicate register actual read by host to allow clear on read if desired.

  mem_intf_read.client_read   mem_intf_read,
  mem_intf_write.client_write mem_intf_write
);

  // Register indexes must be compliant with the relative used address in the accelerator code, this is NOT automated.
  enum {
    SIGMOID_IN_REG_IDX  = 0,  // HW_SIGMOID_IN_REG  : ACCELERATOR_BASE_ADDR + 0x00
    SIGMOID_OUT_REG_IDX = 1   // HW_SIGMOID_OUT_REG : ACCELERATOR_BASE_ADDR + 0x04
  } reg_idx_e;

  enum logic [1:0] {IDLE, LOAD, CALC, DONE} next_state, state; // state machine

  logic       start;

  // Registered input and output
  logic [7:0] x_reg;
  logic [7:0] y_reg;

  // Combinational datapath, same names as software_sigmoid_model() in the C code
  logic        sign;
  logic [6:0]  abs_x;
  logic [6:1]  s;
  logic [2:0]  interval_idx;
  logic [12:0] abs_x_q1_10;
  logic [12:0] shift_add_result;
  logic [12:0] intercept_q1_10;
  logic [12:0] mx_plus_c_q1_10;
  logic [7:0]  sigma_abs_x;
  logic [7:0]  sigma_x;

  // SW writing the input register starts a new calculation
  assign start = host_regs_valid_pulse[SIGMOID_IN_REG_IDX];

  // ---------------------------------------------------------------------------
  // State machine
  //   IDLE : waiting for the first input after reset
  //   LOAD : sample the input register. Done one cycle after the write pulse,
  //          so host_regs already holds the new value
  //   CALC : sample the datapath result into y_reg
  //   DONE : result is valid for SW, held until the next write
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= IDLE;
    else        state <= next_state;
  end

  always_comb begin
    next_state = state;

    case (state)
      IDLE:    next_state = IDLE;
      LOAD:    next_state = CALC;
      CALC:    next_state = DONE;
      DONE:    next_state = DONE;
      default: next_state = IDLE;
    endcase

    // A new write always restarts the calculation, from any state
    if (start) next_state = LOAD;
  end

  // ---------------------------------------------------------------------------
  // Input and output registers
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      x_reg <= '0;
      y_reg <= '0;
    end
    else begin
      if (state == LOAD) x_reg <= host_regs[SIGMOID_IN_REG_IDX][7:0];
      if (state == CALC) y_reg <= sigma_x;
    end
  end

  // ---------------------------------------------------------------------------
  // Datapath, matches software_sigmoid_model()
  // ---------------------------------------------------------------------------

  // Extract sign (MSB) and absolute value
  assign sign  = x_reg[7];
  assign abs_x = x_reg[6:0];

  // Domain interval decoder, comparator free (based on Eq. 23 of the paper).
  // s[1] and s[2] are completed so all of |x| >= 0.6875 and [2, 2.25) are covered,
  // matching the intervals of Table II and the C model.
  assign s[1] = abs_x[4] | (abs_x[3] & (abs_x[2] | (abs_x[1] & abs_x[0]))); // 0.6875 <= |x|
  assign s[2] = abs_x[5] | (abs_x[3] & abs_x[4]);                           // 1.5    <= |x|
  assign s[3] = (abs_x[2] | abs_x[3]) & ~abs_x[4] & abs_x[5];               // 2.25   <= |x|
  assign s[4] = abs_x[4] & abs_x[5] & ~abs_x[6];                            // 3      <= |x|
  assign s[5] = ~abs_x[4] & ~abs_x[5] & abs_x[6];                           // 4      <= |x|
  assign s[6] = (abs_x[4] | abs_x[5]) & abs_x[6];                           // 5      <= |x|

  // Priority encoder, s[6] has the highest priority
  assign interval_idx[2] = s[6] | s[5] | s[4];
  assign interval_idx[1] = s[6] | (~(s[4] | s[5]) & (s[2] | s[3]));
  assign interval_idx[0] = ~s[6] & (s[5] | (~s[4] & (s[3] | (~s[2] & s[1]))));

  assign abs_x_q1_10 = {abs_x, 6'b0}; // abs_x << 6

  // Shift add network (y = mx + c) calculated in Q1.10 for precision
  always_comb begin
    case (interval_idx)
      3'd0: begin
        intercept_q1_10  = 13'd64 << 3;
        shift_add_result = abs_x_q1_10 >> 2;
      end
      3'd1: begin
        intercept_q1_10  = 13'd69 << 3;
        shift_add_result = (abs_x_q1_10 >> 3) + (abs_x_q1_10 >> 4);
      end
      3'd2: begin
        intercept_q1_10  = 13'd80 << 3;
        shift_add_result = abs_x_q1_10 >> 3;
      end
      3'd3: begin
        intercept_q1_10  = 13'd98 << 3;
        shift_add_result = abs_x_q1_10 >> 4;
      end
      3'd4: begin
        intercept_q1_10  = 13'd110 << 3;
        shift_add_result = abs_x_q1_10 >> 5;
      end
      3'd5: begin
        intercept_q1_10  = 13'd118 << 3;
        shift_add_result = abs_x_q1_10 >> 6;
      end
      default: begin // interval 6, saturation
        intercept_q1_10  = 13'd128 << 3;
        shift_add_result = '0;
      end
    endcase
  end

  assign mx_plus_c_q1_10 = shift_add_result + intercept_q1_10;

  // Return to Q1.7
  assign sigma_abs_x = mx_plus_c_q1_10[10:3];

  // Conditional 2's complement for symmetry
  assign sigma_x = sign ? (8'd128 - sigma_abs_x) : sigma_abs_x;

  // ---------------------------------------------------------------------------
  // Register read back to SW
  // ---------------------------------------------------------------------------
  always_comb begin
    host_regs_data_out  = '0;
    host_regs_valid_out = '0;

    host_regs_data_out[SIGMOID_OUT_REG_IDX]  = {24'b0, y_reg};
    host_regs_valid_out[SIGMOID_OUT_REG_IDX] = (state == DONE);
  end

  // ---------------------------------------------------------------------------
  // Memory interfaces are not used by this accelerator, outputs are tied inactive
  // ---------------------------------------------------------------------------
  assign mem_intf_read.mem_req         = 1'b0;
  assign mem_intf_read.mem_start_addr  = '0;
  assign mem_intf_read.mem_size_bytes  = '0;

  assign mem_intf_write.mem_req        = 1'b0;
  assign mem_intf_write.mem_start_addr = '0;
  assign mem_intf_write.mem_size_bytes = '0;
  assign mem_intf_write.mem_data       = '0;

endmodule
