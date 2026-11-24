# SYMPA-Lite Sigmoid: SystemVerilog Implementation Spec (K5-XBOX, 8-bit)

Nov 22, 2026 · @sigal · Draft v0.5

## 1. Purpose and scope

Build SYMPA-Lite Sigmoid: an 8-bit, inference-only SystemVerilog accelerator for the Fashion-MNIST MLP 784-64-10, running inside the K5-XBOX platform on the DE10-Lite. It follows the SYMPA architecture ([Medus et al., IEEE Access 2019](https://doi.org/10.1109/ACCESS.2019.2920885)) and uses our sigmoid\_57 ([Meher & Aggarwal, IEEE TCAS-I 2026](https://doi.org/10.1109/TCSI.2025.3641204)) as the single activation block.

This spec starts from the SYMPA-Lite example spec (16 Nov 2026). Where the two differ, this spec wins for our project. "Must" is checked at acceptance (section 11); "should" is recommended.

**In scope:** sympa\_core RTL, the K5-XBOX wrapper sympa\_xlr, the Python quantizer and bit-exact golden model (done), a C driver and C reference model, unit and system testbenches, and qsyn\_xlr synthesis.

**Out of scope for v1:** on-chip training, convolution, floating point, run-time topology changes, and the example's UART link, demo ROM, 7-segment display and PLL.

| Item | Medus paper | SYMPA-Lite example | This spec | Why |
| --- | --- | --- | --- | --- |
| Activation | One AFB, several functions | ReLU + requantize | Requantize to Q3.4 + sigmoid\_57 | The project's sigmoid; Medus shows the AFB limits clock speed |
| Number format | 18-bit Q6.12 | u8 activations 0..255, s8 weights, any scale | u8 activations Q1.7 (0..128), s8 weights with a power-of-2 scale | Sigmoid is not scale-invariant, so its input must be exact Q3.4 |
| Weight storage | One RAM per NPE | One wide bank, one byte lane per NPE | Same as example | All NPEs share one read address |
| Host link | None | UART + board top level | K5-XBOX host\_regs + mem\_intf | Course platform |
| Clock | 490 MHz, Virtex-7 | 100 MHz through a PLL | 50 MHz platform clock | qsyn\_xlr default; no PLL inside an accelerator |
| Output layer | Through the AFB | RAW + argmax | RAW + argmax | Matches the PyTorch model (no activation after fc2) |

## 2. Target platform and hard constraints

The accelerator runs inside K5-XBOX on the MAX 10 10M50DAF484C7G of the DE10-Lite, clocked by the platform at 50 MHz, within a budget of 20K logic elements.

| Item | Value |
| --- | --- |
| Board | Terasic DE10-Lite |
| FPGA | Intel MAX 10 10M50DAF484C7G (speed grade C7) |
| Device resources (approximate) | About 49,760 LEs, 182 M9K blocks (9,216 bits each), 144 18×18 multipliers (each splits into two 9×9) |
| Accelerator budget | Up to about 20K LEs of logic. Embedded multipliers and M9K memory are not counted and may be used freely (confirmed by the supervisor) |
| Clock | Platform clock, 50 MHz by default; the -mhz option changes it, about 65 MHz at most |
| Tools | qsyn\_xlr and comp\_fpga (Quartus 24.1 Lite), launch\_k5\_sim for hardware and software together, prog\_fpga and launch\_k5\_app on the board, Python 3.10+ with numpy and PyTorch |

Baseline: sigmoid\_57 alone synthesizes to 90 LEs, no memory and no multipliers, with Fmax 148.5 MHz (slow 1200 mV, 85 °C). Its report shows negative slack only because basic.sdc was missing and Quartus assumed a 1 ns clock.

**Hard constraints (must)**

1. One clock domain: the platform clk. Reset follows the platform style used by sigmoid\_57 (rst\_n, active low).
2. `qsyn_xlr sympa_xlr -all` completes: synthesizable, fits, and meets 50 MHz with no negative setup or hold slack.
3. With the REF\_FMNIST config (section 3) the accelerator uses at most 20K LEs, and at most 50% of the 9×9 multipliers and 60% of the M9K blocks, leaving room for the platform.
4. The weight bank is inferred as M9K. The fitter report shows no RAM built from logic.
5. RTL is synthesizable SystemVerilog-2012: logic, always\_ff, always\_comb, packages, enum, packed struct, generate, parameters. No latches, no initial blocks except RAM init through `$readmemh`, no vendor IP.
6. sympa\_core and everything below it use no SV interface. Only sympa\_xlr uses the platform's mem\_intf\_read and mem\_intf\_write interfaces, because the platform requires them.
7. Every new module is listed in `sympa_xlr.f`, which simulation and qsyn\_xlr both read.

## 3. Parameters and number formats

Arithmetic is exact integer, and the golden model reproduces every output bit. The REF\_FMNIST values come from the Python model (`mlp_bit_accurate.py`, H = 64); all parameters live in a generated `sympa_cfg_pkg.sv`.

| Parameter | REF\_FMNIST | REF\_TINY | Meaning |
| --- | --- | --- | --- |
| L | 2 | 2 | Layers after the input layer |
| N\[0..L\] | {784, 64, 10} | {4, 8, 3} | Units per layer; N\[0\] is the input count |
| N\_NPE | 64 | 8 | Processing elements; at least the largest of N\[1..L\] |
| DW | 8 | 8 | Activation and weight width |
| BIAS\_BYTES | 2 | 2 | Bias stored as 16-bit signed, most significant byte first |
| ACC\_W | 25 | 25 | Accumulator width; must satisfy the rule below |
| MEM\_DEPTH / AW | 1024 / 10 | 64 / 6 | Weight-bank rows / address width |
| RAM\_LAT | 2 | 2 | Weight read latency (registered address and output) |
| BASE\[1..L\] | {0, 786} | {0, 6} | First weight-bank row of each layer (section 6) |
| F\[1..L\] | {6, 6} | from quantizer | Weight fraction bits per layer (weight = int / 2^F) |
| SHIFT\[1..L-1\] | {9} | from quantizer | Requantize shift into the sigmoid, 7 + F − 4 |
| MODE\[1..L\] | {SIGMOID, RAW} | {SIGMOID, RAW} | Layer L must be RAW; all others SIGMOID |
| OUT\_FIFO\_DEPTH | 16 | 4 | At least N\[L\] |

| Quantity | Format | Range |
| --- | --- | --- |
| Input and hidden activation x | Unsigned 8-bit, value = int / 128 (Q1.7) | 0 to 128 |
| Weight w | Signed 8-bit two's complement, value = int / 2^F | −128 to 127 |
| Product x·w | Signed 17-bit, from a 9×9 multiply of {1'b0, x} and w | −16,384 to 16,256 |
| Bias b | Signed 16-bit, in accumulator scale 2^−(7+F) | −32,768 to 32,767 |
| Accumulator, output logit | Signed ACC\_W bits | No wrap allowed |
| Sigmoid input z | 8-bit sign-magnitude, Q3.4 | −127 to 127 |
| Sigmoid output y | Unsigned 8-bit, Q1.7 | 0 to 128 |

**Bit-exact semantics.** For unit j of layer i with inputs x\_0 … x\_{N\[i−1\]−1}:

```latex
\mathrm{acc}_j = b_j + \sum_{k=0}^{N_{i-1}-1} x_k \, w_{jk}
```

- Hidden layer (SIGMOID), s = SHIFT\[i\]: z = clamp((acc + 2^(s−1)) >>> s, −127, 127), then y = sigmoid\_57(z). The arithmetic right shift rounds half up. z enters sigmoid\_57 as sign-magnitude: z for z ≥ 0, else 8'h80 | (−z).
- Output layer (RAW): logit\_j = acc\_j. The argmax is the lowest index holding the maximum logit.

**Accumulator width rule (must be checked at elaboration).** For every layer, ACC\_W ≥ 1 + ceil(log2(128 · 128 · N\[i−1\] + 2^15)). For 784 inputs this gives 25. sympa\_core must `$error` at elaboration if the rule fails, if any N\[i\] > N\_NPE, or if OUT\_FIFO\_DEPTH < N\[L\].

**Quantization recipe** (`quant_vs_ideal.py`, `mlp_bit_accurate.py`)

1. Train the float MLP with a sigmoid hidden layer (PyTorch, Adam, 5 epochs).
2. Map each input pixel p (0..255) to x = (256p + 255) div 510, which equals round(p · 128 / 255). The C driver does this before the image reaches the accelerator.
3. Per layer, F = floor(log2(127 / max|W|)) and w = clip(round(W · 2^F), −128, 127). The scale is a power of 2 so the accumulator scale is exactly 2^−(7+F).
4. Set b = round(B · 2^(7+F)); it must fit in BIAS\_BYTES.
5. For SIGMOID layers, SHIFT = 7 + F − 4, which maps the accumulator exactly to Q3.4. The example picks SHIFT from activation percentiles instead; that works for ReLU, which is scale-invariant, but not for a sigmoid.
6. Report float and integer accuracy on the 10,000-image test set. A gap above 1 percentage point fails acceptance. Reference run: float 85.94%, 8-bit with ideal sigmoid 85.95%, 8-bit with sigmoid\_57 85.93%.

## 4. Module hierarchy and file list

The design is one board-independent core under a thin platform wrapper, with one file per module.

&#91;embedded content: module hierarchy · wrapper, core, 5 blocks, 4 sub-blocks\]

Only sympa\_xlr changes if the platform interface turns out different from section 8; everything under sympa\_core is tested on its own.

```text
rtl/
  sympa_pkg.sv        opcodes, token struct, latency constants (T_NPE, T_AFB)
  sympa_cfg_pkg.sv    GENERATED: section 3 parameters
  sympa_xlr.sv        K5-XBOX wrapper (section 8)
  sympa_xlr.f         file list for simulation and qsyn_xlr
    sympa_core.sv
      sympa_ctrl.sv   FSM, op sequencer, input mux, cycle counter
      npe_array.sv    broadcast register, weight_bank, N_NPE x npe, SR chain
        weight_bank.sv  dual-port RAM, N_NPE*8 wide, written through wr_*
        npe.sv
      afb.sv          requantize + sigmoid_core, or RAW pass-through
        requantize.sv
        sigmoid_core.sv  datapath of sigmoid_57, unchanged
      sync_fifo.sv    output FIFO
      argmax.sv
tb/       unit and system testbenches, register and memory models
tools/    quant_vs_ideal.py, mlp_bit_accurate.py (quantizer, golden model, weights.hex)
data/     weights.hex, golden vectors (generated)
```

## 5. sympa\_core ports

sympa\_core keeps the example's board-independent ports, so it can be written and simulated before the platform files arrive. All outputs are registered, and no combinational path runs from in\_valid to in\_ready or from out\_ready to out\_valid.

| Port | Dir | Width | Meaning |
| --- | --- | --- | --- |
| clk, rst\_n | in | 1 | Platform clock; active-low reset |
| wr\_valid | in | 1 | Write one full weight-bank row. Ignored while busy; an assertion flags it |
| wr\_addr | in | AW | Row address |
| wr\_data | in | N\_NPE·8 | Lane j is wr\_data\[8j+7:8j\] |
| in\_valid, in\_ready | in, out | 1 | Input stream handshake. A byte accepted while idle starts an inference. Gaps are allowed |
| in\_data | in | 8 | Input byte x\_k (Q1.7), in order k = 0 … N\[0\]−1; exactly N\[0\] bytes per inference |
| out\_valid, out\_ready | out, in | 1 | Output stream handshake from the output FIFO |
| out\_data | out | ACC\_W | Signed logit, unit order 0 … N\[L\]−1 |
| out\_last | out | 1 | High with the last logit |
| argmax\_valid | out | 1 | One-cycle pulse once the last logit is in the FIFO |
| argmax\_idx | out | 8 | Index of the largest logit (lowest index on ties) |
| busy | out | 1 | High from the first accepted input byte until the last logit is in the FIFO |
| cycles | out | 32 | Cycles from the first accepted input byte to the last logit written; held until the next start |

## 6. Datapath blocks

Every NPE reads the same weight-bank row in the same cycle, so one row holds one weight index for all units, one byte lane per NPE.

**6.1 weight\_bank.** Simple dual-port RAM, MEM\_DEPTH × (N\_NPE · 8) bits, read latency RAM\_LAT. The wrapper writes it row by row through the wr\_\* port when the C program loads the weights (section 8); testbenches may preload it from `weights.hex` with `$readmemh`. Lane j drives NPE j. As in Medus, the bias comes first in each layer, so it loads before the first MAC. Lanes of units j ≥ N\[i\] hold 0.

| Rows (REF\_FMNIST) | Content |
| --- | --- |
| 0–1 | Layer 1 bias b\_j, high byte then low byte |
| 2–785 | Layer 1 weights: row 2 + k holds w\_jk for input k = 0 … 783 |
| 786–787 | Layer 2 bias, lanes 0–9 used |
| 788–851 | Layer 2 weights: row 788 + k holds w\_jk for hidden unit k = 0 … 63, lanes 0–9 used |
| 852–1023 | Unused, 0 |

The 852 used rows × 64 lanes take 54,528 bytes, about 58 M9K blocks (32% of the device).

**6.2 npe.** One neuron. Opcodes travel with the data through the pipeline as in-order tokens (from the example), so input bubbles are harmless.

| Opcode | Effect |
| --- | --- |
| BIAS\_HI | Hold the bias high byte |
| BIAS\_LO | acc ← sign-extend({high, low}) |
| MAC | acc ← acc + {1'b0, x} · w, on one 9×9 embedded multiplier |
| LOAD\_SR | sr ← acc |
| SHIFT | sr ← sr of NPE j + 1 (NPE N\_NPE − 1 shifts in 0) |
| NOP | Hold |

**6.3 SR chain.** N\_NPE registers of ACC\_W bits, one per NPE, loaded together by LOAD\_SR. SHIFT moves every value one place toward NPE 0, and sr of NPE 0 drives the AFB, so units leave in order 0, 1, 2 … and match the weight-row order of the next layer.

**6.4 afb.** One block for the whole network, as in Medus. In SIGMOID mode it computes requantize (add 2^(SHIFT−1), arithmetic shift right by SHIFT, saturate to ±127, convert to sign-magnitude), then sigmoid\_core, the combinational datapath of sigmoid\_57 unchanged, then a register. The 8-bit result y goes to the input mux as the next layer's x. In RAW mode it passes the logit to the output FIFO. Latency T\_AFB = 2 cycles (to be confirmed at timing closure).

**6.5 argmax.** Watches logits as they enter the FIFO and keeps the index of the largest, replacing it only on a strictly greater value, so ties keep the lowest index. It pulses argmax\_valid after logit N\[L\] − 1.

**6.6 sync\_fifo.** Output FIFO, OUT\_FIFO\_DEPTH ≥ N\[L\] entries of ACC\_W bits, so the whole output layer fits even if the reader stalls.

## 7. Control sequence and cycle count

One inference takes about 875 cycles, 17.5 µs at 50 MHz, when the input arrives at one byte per cycle. sympa\_ctrl issues this sequence, which follows Medus Algorithms 1 and 3 with all 64 hidden units computed in one pass.

1. **Idle.** Wait for the first accepted in\_data byte; raise busy and clear the cycle counter.
2. **Layer 1 bias.** Read rows BASE\[1\] and BASE\[1\] + 1; issue BIAS\_HI then BIAS\_LO to all NPEs.
3. **Layer 1 MAC.** For k = 0 … 783: accept x\_k from in\_data, read row BASE\[1\] + 2 + k, broadcast x\_k to all NPEs with MAC. An input gap inserts a bubble.
4. **Load SR.** After the last MAC leaves the NPE pipeline, issue LOAD\_SR.
5. **Layer 2 bias.** Rows BASE\[2\] and BASE\[2\] + 1, as in step 2.
6. **Layer 2 MAC.** For k = 0 … 63: SHIFT the chain; the value leaving NPE 0 passes through the AFB (SIGMOID) and becomes x\_k; read row BASE\[2\] + 2 + k and broadcast x\_k with MAC. The input mux now selects the AFB instead of in\_data.
7. **Load SR** for layer 2, as in step 4.
8. **Output.** For j = 0 … 9: SHIFT; the value leaving NPE 0 passes through the AFB (RAW) into the FIFO and argmax. Raise out\_last with j = 9, pulse argmax\_valid, drop busy, and hold cycles.

```latex
C \approx (N_0 + 2) + (N_1 + 2) + N_2 + 2\,(T_{NPE} + T_{RAM} + T_{AFB}) \approx 786 + 66 + 10 + 12 \approx 875
```

This is Medus Eq. 12 with 2 bias rows per layer. The wrapper first fetches the image into its input buffer (25 XMEM reads, latency to be measured in simulation), so the core itself never stalls.

## 8. K5-XBOX wrapper (sympa\_xlr)

sympa\_xlr connects sympa\_core to K5-XBOX the way the course example xmemcpy\_ref does: addresses and commands go in host registers, and data moves through mem\_intf\_read and mem\_intf\_write in 32-byte lines. The protocol below is read from xmemcpy\_ref.sv and xmemcpy\_ref.c; signal widths come from xbox\_def\_pkg.sv, mem\_intf\_read.sv and mem\_intf\_write.sv.

| Item | Value | Source |
| --- | --- | --- |
| Host registers | 16 × 32 bit (XBOX\_NUM\_REGS), register i at XBOX\_REGS\_BASE\_ADDR + 4·i | k5\_xbox\_defines.h, xmemcpy\_ref.c |
| Register base | 0x4001FFC0 on DE10-Lite, at the top of XSPACE | Derived from k5\_xbox\_defines.h |
| XMEM | 64 KB at 0x40000000: 2 memories × 1,024 lines × 32 bytes, XMEM\_ADDR\_WIDTH = 16 (xbox\_def\_pkg.sv) | k5\_xbox\_defines.h |
| XMEM address in hardware | Low 16 bits (XMEM\_ADDR\_WIDTH) of the C pointer written to a register | xmemcpy\_ref.sv |
| Start | Software writes the START register; hardware acts on host\_regs\_valid\_pulse for that index | xmemcpy\_ref.sv |
| Done | Hardware sets bit 0 of DONE and its host\_regs\_valid\_out; it clears on host\_regs\_read\_pulse after software reads it | xmemcpy\_ref.sv |
| Clock | 50 MHz by default; the -mhz option of qsyn\_xlr and comp\_fpga changes it, about 65 MHz at most | FPGA\_Clock\_Configuration.md |

| Interface | Accelerator drives | Platform returns | Handshake |
| --- | --- | --- | --- |
| mem\_intf\_read | mem\_req, mem\_start\_addr, mem\_size\_bytes (1 to 32) | mem\_valid, mem\_data (32 bytes) | Hold mem\_req until mem\_valid, drop it in that cycle, register mem\_data |
| mem\_intf\_write | mem\_req, mem\_start\_addr, mem\_size\_bytes, mem\_data (32 bytes) | mem\_ack | Hold mem\_req and the data until mem\_ack |

Data written to memory must come straight from a register: the course timing note traces failures on the board to computed values driven into memory in the same cycle.

**Integration.** The platform builds one accelerator at a time: qsyn\_xlr passes CRNT\_XLR=\<name> and wraps the module of that name. So sympa\_xlr keeps exactly the xmemcpy\_ref port list, lives in $MY\_K5\_XLRS/sympa\_xlr/, and lists every file in sympa\_xlr.f. mem\_size\_bytes is 6 bits (1 to 32), and XMEM buffers should start on 32-byte boundaries, since XMEM is built from 32-byte lines.

**Register map** (index = byte offset / 4, xmemcpy\_ref style)

| Index | Offset | Name | Dir | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 0x00 | IMG\_ADDR | W | XMEM address of the 784 input bytes (Q1.7) |
| 1 | 0x04 | W\_ADDR | W | XMEM address of the current weight chunk |
| 2 | 0x08 | W\_ROW0 | W | First weight-bank row of the chunk |
| 3 | 0x0C | W\_NROWS | W | Rows in the chunk, 64 bytes each |
| 4 | 0x10 | OUT\_ADDR | W | XMEM address for the 10 logits, 4 bytes each |
| 5 | 0x14 | START | W | 1 = load one weight chunk, 2 = run one inference |
| 6 | 0x18 | DONE | R | Bit 0 set when the command finishes; cleared when software reads it |
| 7 | 0x1C | RESULT | R | Predicted class 0–9 |
| 8 | 0x20 | CYCLES | R | Core cycles of the last inference |
| 9–15 | 0x24–0x3C | Reserved |  |  |

**Weights (decided with the supervisor).** The weights come from the C program through XMEM, never through the registers. Python writes the weight image (852 rows × 64 bytes = 54,528 bytes) as a hex file; the C program loads it into XMEM with bm\_start\_soc\_load\_hex\_file, as xmemcpy\_ref loads its data. All 64 KB of XMEM is ours (supervisor), so everything fits at once: 54,528 bytes of weights, 784 input bytes and 40 bytes of logits take 55,352 bytes. The default is one load of all 852 rows followed by START = 1; W\_ROW0 and W\_NROWS stay so the weights can also be loaded in parts. Each row takes two 32-byte reads. The weights then stay in the M9K bank until power-off.

**Inference.** On START = 2 the wrapper reads the 784 input bytes (25 reads) into a 784-byte input buffer in one M9K block, then streams it into in\_data at one byte per cycle, so layer 1 never waits on memory. When argmax\_valid rises it writes the 10 logits to OUT\_ADDR (two writes), latches RESULT and CYCLES, and sets DONE.

## 9. Software driver and performance measurement

The speedup is the C reference model's time on the platform CPU divided by the accelerator's time, for the same image and the same integer weights.

**C driver, once at start-up:** load the weights hex file into XMEM (all 852 rows in one chunk) with bm\_start\_soc\_load\_hex\_file, write W\_ADDR, W\_ROW0 and W\_NROWS, write START = 1, and poll DONE.

**C driver** (one inference)

1. Put the 784 input bytes in XMEM, converted to Q1.7 with x = (256p + 255) / 510.
2. Write IMG\_ADDR and OUT\_ADDR, then write START = 2.
3. Poll DONE, then read RESULT and CYCLES; the 10 logits are at OUT\_ADDR.

**C reference model.** The same integer network in C, using `mlp_weights.h` exported by `mlp_bit_accurate.py`: 50,816 MACs per image, the requantize shift, and the sigmoid\_57 arithmetic, as in `software_sigmoid_model()`. It must return the same class as the accelerator for every test image.

**Measurement.** Use the platform cycle counter as xmemcpy\_ref.c does: ENABLE\_CYCLE\_COUNT, RESET\_CYCLE\_COUNT, then GET\_CYCLE\_COUNT\_START and GET\_CYCLE\_COUNT\_END around the call, reported as K5 effective cycles (a 32-bit counter read by get\_cycle\_count in k5\_utils\_lib.h). The software-only result is divided by 8, because the K5 core runs 8 threads and the reference uses one. The XON and XOFF macros, set by launch\_k5\_app -ccd1 XON, choose the path. Report the C model's cycles, the accelerator call's cycles (image fetch and logit write included), the core's CYCLES, and the speedup; report the one-time weight load separately.

## 10. Verification plan

Every block is checked against vectors from the Python golden model before it joins the next level, so a mismatch points at one block.

**Golden data.** One frozen weight set (H = 64) exported by `mlp_bit_accurate.py`: weights.hex, mlp\_weights.h, and for 100 test images the input bytes, sigmoid inputs z, sigmoid outputs y, the 10 logits and the class.

| Level | Test | Pass condition |
| --- | --- | --- |
| sigmoid\_core | All 255 legal inputs | Equal to the Python table, which already matches the sigmoid\_57 RTL bit for bit |
| requantize | Random accumulators plus edges: half-way ties, ±127 saturation, the most negative accumulator | Equal to the Python formula |
| npe | Directed: bias load, 784 MACs, LOAD\_SR, SHIFT, bubbles between MACs | acc and sr equal to a NumPy reference |
| weight\_bank | Read after $readmemh and after a wr\_\* write, RAM\_LAT = 2 | Data in the right lane at the right cycle |
| argmax, sync\_fifo | Ties, all equal, maximum at index 0 and 9, output stalls | Lowest index on ties; no lost logit |
| sympa\_core, REF\_TINY | 4-8-3 network, random weights | Every logit bit-exact; quick to debug |
| sympa\_core, REF\_FMNIST | The 100 golden images | Every logit bit-exact; class equal; cycles within 5% of section 7 |
| sympa\_xlr | Register model + memory model for mem\_intf | Same classes as the core test through host\_regs |
| Synthesis | `qsyn_xlr sympa_xlr -all` | Section 2 constraints met |

## 11. Acceptance criteria

The design is accepted when all six hold; the first is already met.

1. Integer golden model within 1 percentage point of float on the 10,000 test images (met: 85.93% against 85.94%).
2. sympa\_core gives bit-exact logits and the same class for all 100 golden images.
3. `qsyn_xlr sympa_xlr -all` passes at 50 MHz with at most 20K LEs and the weight bank in M9K.
4. Cycles per image within 5% of the section 7 formula (about 875).
5. In launch\_k5\_sim, the C driver's class equals the C reference model's class for every test image (on the board too, if a board run is added).
6. Compute and end-to-end speedups are measured and reported (section 9).

## 12. Open questions

Sections 8 and 9 now follow xmemcpy\_ref; these questions close the rest. Sections 3 to 7 do not depend on them.

**Answered**

- XMEM and M9K: all FPGA memory is free for the accelerator (supervisor).
- Target: simulation with launch\_k5\_sim; a run on the physical board may be added later.
- Timing: judge by the Fmax that qsyn\_xlr reports, as the course hackathon does. The slack lines assume a 1 ns clock because basic.sdc is missing.
- Platform widths: XMEM\_ADDR\_WIDTH = 16, 32-byte lines, mem\_size\_bytes up to 32, 16 host registers (xbox\_def\_pkg.sv, mem\_intf\_read.sv, mem\_intf\_write.sv, host\_regs\_intrf.sv).
- SYMPA-Lite example: this spec no longer depends on its sections 6 to 12.

**Still open** (both settle in the first simulation run)

- [ ] Measure the mem\_intf\_read latency (cycles from mem\_req to mem\_valid) in the first launch\_k5\_sim run; it sets the image-fetch and weight-load times.
- [ ] Check whether hw/xlrs/xlrs.f must list sympa\_xlr for launch\_k5\_sim; xbox.f and xbox\_xlr\_wrap.sv under $K5\_XBOX\_SRC show it.

**Our next steps**

- [ ] Export the weight image as a hex file in bank layout (852 rows × 64 bytes) and freeze one weight set.
- [ ] Write and unit-test sigmoid\_core, requantize, npe, weight\_bank and argmax against the golden vectors.
- [ ] Start sympa\_xlr from xmemcpy\_ref.sv, whose register and mem\_intf handling is the template.
