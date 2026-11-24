# SYMPA-Lite Sigmoid accelerator (K5-XBOX, DE10-Lite)

Student project (two students, Bar-Ilan, course DDP26). Goal: a SystemVerilog hardware
accelerator for a Fashion-MNIST MLP **784-64-10** (Flatten -> Linear -> Sigmoid -> Linear -> argmax),
running as an accelerator inside the course's **K5-XBOX** platform, measured against the same
network in C on the platform CPU.

**The spec is the source of truth:** `SYMPA-Lite Sigmoid SystemVerilog Implementation Spec (K5-XBOX, 8-bit).md`
(draft v0.5). Read it fully before writing code. If the code must differ from the spec, update the spec
first and tell the students.

## Background already settled (do not reopen without asking)

- Architecture: Medus et al., "A Novel Systolic Parallel Hardware Architecture for the FPGA Acceleration of
  Feedforward Neural Networks", IEEE Access 2019 (SYMPA): 64 NPEs (one per hidden unit), input broadcast one
  byte per cycle, shift-register (SR) chain, ONE activation block, the same NPEs reused for layer 2.
- Activation: the students' own `sigmoid_57` (Meher & Aggarwal, IEEE TCAS-I 2026), already in
  `$MY_K5_XLRS/sigmoid_57/sigmoid_57.sv`. Reuse its combinational datapath **unchanged** as `sigmoid_core`.
  Standalone it is 90 LEs, Fmax 148.5 MHz.
- 8-bit formats (settled earlier in Python): activations unsigned Q1.7 (0..128), weights int8 Q0.6
  (both layers), bias int16 in accumulator scale, accumulator 25 bit, requantize `(acc + 256) >>> 9`,
  saturate to +-127, sign-magnitude Q3.4 into the sigmoid. Output layer is RAW (no activation) + argmax,
  lowest index on ties. Float 85.94% vs 8-bit with our sigmoid 85.93% on 10,000 test images.
- Weights come from the C program through XMEM (supervisor's decision), never through registers.
  All 64 KB of XMEM and all M9K are free for us. Target is simulation (`launch_k5_sim`); a board run may come later.
- Budget: <= 20K LEs of logic; embedded multipliers and M9K are not counted. Clock 50 MHz.

## Platform facts (from the course files)

- Shared platform RTL: `$K5_XBOX_SRC` (= /project/tsmc65/shared/k5_share/k5_xbox/hw/xbox):
  `xbox_def_pkg.sv`, `host_regs_intrf.sv`, `xbox_xlrs/mem/mem_intf_read.sv`, `xbox_xlrs/mem/mem_intf_write.sv`,
  `xbox_xlr_wrap.sv`, `xbox.f`. **Read-only: never modify anything outside `$MY_K5_PROJ`.**
- XMEM: 64 KB at 0x40000000 (2 x 1024 lines x 32 bytes), `XMEM_ADDR_WIDTH = 16`, `mem_size_bytes` 1..32.
- 16 host registers at `XBOX_REGS_BASE_ADDR` = 0x4001FFC0, `HOST_REG(i)` at base + 4*i.
- Template accelerator: `$MY_K5_XLRS/xmemcpy_ref/xmemcpy_ref.sv` + `$MY_K5_PROJ/sw/apps/xmemcpy_ref/xmemcpy_ref.c`.
  Copy its port list exactly, its START (valid pulse) / DONE (clear-on-read) handling, and its mem_intf
  handshake (hold mem_req until mem_valid / mem_ack; register mem_data).
- The platform picks the accelerator by name (`CRNT_XLR=<name>`): module name = folder name = `sympa_xlr`,
  in `$MY_K5_XLRS/sympa_xlr/`, every file listed in `sympa_xlr.f`.
- Data written to memory must come straight from a register (course timing note).
- C timing: `ENABLE_CYCLE_COUNT; RESET_CYCLE_COUNT; GET_CYCLE_COUNT_START(s); ... GET_CYCLE_COUNT_END(e);`
  (k5_utils_lib.h). Software-only cycles are divided by 8 (8 hardware threads). `XON`/`XOFF` macros come from
  `launch_k5_app ... -ccd1 XON`.

## Commands

```
cd $MY_K5_XLRS/sympa_xlr && qsyn_xlr sympa_xlr -all     # synthesis, fit, timing (judge by Fmax)
launch_k5_sim sympa_xlr                                 # HW + C co-simulation
cd $MY_K5_PROJ/hw/gen_fpga && comp_fpga sympa_xlr       # full FPGA build (later)
```

## Python files (in this folder)

- `quant_vs_ideal.py`: float vs 8-bit vs 8-bit + our sigmoid (the accuracy result above). The only Python
  file; do not change it.

## Work order

Current scope: the **first linear layer only** (784 -> 64, bias + MACs, no activation). Not yet: the sigmoid,
layer 2, the K5-XBOX wrapper.

RTL one module at a time, in this order:

1. `npe`: one neuron (spec section 6.2).
2. `weight_mem`: the weight memory, one byte lane per NPE (spec section 6.1).
3. `linear_layer`: `weight_mem` + the NPEs + control. Small parameters first (4 inputs x 3 neurons),
   then 784 x 64.

Each module gets a short deterministic self-checking testbench with a few hand-computed values that prints
PASS or FAIL. No trained weights, no golden files. Show the students the code and the test before moving on
to the next module.

RTL goes in `$MY_K5_XLRS/sympa_xlr/`, testbenches in `$MY_K5_XLRS/sympa_xlr/tb/`.

## Working with the students

They write in Hebrew; answer in Hebrew, keep code, comments and the spec in English. Explain what each step
does and why, since they are learning. Ask before deleting files or changing anything they wrote.
