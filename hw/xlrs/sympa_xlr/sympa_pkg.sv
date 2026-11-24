// sympa_pkg.sv - definitions shared by all SYMPA-Lite modules (spec section 4)
// Compile this file first.

package sympa_pkg;

  // npe operations (spec section 6.2)
  localparam logic [2:0] OP_NOP     = 3'd0;   // hold everything
  localparam logic [2:0] OP_BIAS_HI = 3'd1;   // remember the bias high byte (from w)
  localparam logic [2:0] OP_BIAS_LO = 3'd2;   // acc <= sign-extend({high byte, w})
  localparam logic [2:0] OP_MAC     = 3'd3;   // acc <= acc + x*w
  localparam logic [2:0] OP_LOAD_SR = 3'd4;   // sr  <= acc
  localparam logic [2:0] OP_SHIFT   = 3'd5;   // sr  <= sr_in

endpackage
