// npe.sv - one Neural Processing Element (spec section 6.2)
// One neuron of layer 1: loads a bias, then adds x*w once per cycle.
// x is the broadcast input (unsigned 0..128, Q1.7), w is this NPE's weight byte (signed).

import sympa_pkg::*;

module npe #(
  parameter int ACC_W = 25                    // accumulator width (spec section 3)
)(
  input  logic                    clk,
  input  logic                    rst_n,      // active-low reset, same style as sigmoid_57
  input  logic [2:0]              op,         // operation for this cycle (see below)
  input  logic [7:0]              x,          // broadcast input, unsigned 0..128
  input  logic [7:0]              w,          // weight (signed) or a bias byte
  input  logic signed [ACC_W-1:0] sr_in,      // SR value of the next NPE (used by SHIFT)
  output logic signed [ACC_W-1:0] acc,        // running sum of this neuron
  output logic signed [ACC_W-1:0] sr          // shift register (result of the layer)
);


  logic [7:0]         bias_hi;                // bias high byte, held between BIAS_HI and BIAS_LO
  logic signed [16:0] prod;                   // x*w, fits in 17 bits: -16,384 .. 16,256

  // {1'b0, x} makes x a positive 9-bit signed number, so a 9x9 embedded multiplier is used
  assign prod = $signed({1'b0, x}) * $signed(w);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc     <= '0;
      sr      <= '0;
      bias_hi <= '0;
    end
    else begin
      case (op)
        OP_BIAS_HI: bias_hi <= w;
        OP_BIAS_LO: acc     <= ACC_W'($signed({bias_hi, w}));   // 16-bit bias, sign-extended
        OP_MAC:     acc     <= acc + ACC_W'(prod);              // 17-bit product, sign-extended
        OP_LOAD_SR: sr      <= acc;
        OP_SHIFT:   sr      <= sr_in;
        default:    ;                                           // OP_NOP: hold
      endcase
    end
  end

endmodule
