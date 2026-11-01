#include <stdio.h>
#include <stdint.h>

// Hardware Accelerator Memory-Mapped I/O
// Base address for the custom K5 accelerator (update based on physical mapping)
#define ACCELERATOR_BASE_ADDR 0x30000000 

// Write pointer: sends 8 bit input to the FPGA
#define HW_SIGMOID_IN_REG  (*((volatile uint32_t*)(ACCELERATOR_BASE_ADDR + 0x00)))

// Read pointer: fetches 8 bit Q1.7 output from the FPGA
#define HW_SIGMOID_OUT_REG (*((volatile uint32_t*)(ACCELERATOR_BASE_ADDR + 0x04)))

// Helper func to print 8 bit binary
void print_binary_8b(uint8_t val) {
    for (int i = 7; i >= 0; i--) {
        printf("%d", (val >> i) & 1);
        if (i == 4) printf("_");
    }
}

// Software Reference Model
uint8_t software_sigmoid_model(uint8_t input_x) {
    // Extract sign (MSB) and absolute value
    uint8_t sign = (input_x >> 7) & 0x01;
    uint8_t abs_x = input_x & 0x7F;

    uint8_t interval_idx = 0;
    uint32_t shift_add_result = 0;
    uint32_t intercept_q1_10 = 0;

    // Interval decoder (7 PWL segments based on research)
    if (abs_x >= 80) { interval_idx = 6; }
    else if (abs_x >= 64) { interval_idx = 5; }
    else if (abs_x >= 48) { interval_idx = 4; }
    else if (abs_x >= 36) { interval_idx = 3; }
    else if (abs_x >= 24) { interval_idx = 2; }
    else if (abs_x >= 11) { interval_idx = 1; }
    else { interval_idx = 0; }

    // Shift add network (y = mx + c) calculated in Q1.10 for precision
    switch (interval_idx) {
    case 0:
        intercept_q1_10 = 64 << 3;
        shift_add_result = (abs_x << 6) >> 2;
        break;
    case 1:
        intercept_q1_10 = 69 << 3;
        shift_add_result = ((abs_x << 6) >> 3) + ((abs_x << 6) >> 4);
        break;
    case 2:
        intercept_q1_10 = 80 << 3;
        shift_add_result = (abs_x << 6) >> 3;
        break;
    case 3:
        intercept_q1_10 = 98 << 3;
        shift_add_result = (abs_x << 6) >> 4;
        break;
    case 4:
        intercept_q1_10 = 110 << 3;
        shift_add_result = (abs_x << 6) >> 5;
        break;
    case 5:
        intercept_q1_10 = 118 << 3;
        shift_add_result = (abs_x << 6) >> 6;
        break;
    case 6:
        intercept_q1_10 = 128 << 3;
        shift_add_result = 0;
        break;
    }

    uint32_t mx_plus_c_q1_10 = shift_add_result + intercept_q1_10;
    
    // return to Q1.7
    uint8_t sigma_abs_x = mx_plus_c_q1_10 >> 3;

    // Conditional 2's complement for symmetry
    if (sign == 1) {
        return 128 - sigma_abs_x;
    }
    else {
        return sigma_abs_x;
    }
}

// Main Bare Metal Application
int main() {
    printf("\n--- K5 Bare Metal: HW/SW Co-Simulation ---\n");
    printf("Starting automated verification...\n\n");

    int errors = 0;
    
    // Verify HW against SW for all 256 possible 8 bit inputs
    for (int i = 0; i < 256; i++) {
        uint8_t test_input = (uint8_t)i;

        // Get expected output from C model
        uint8_t expected_sw_result = software_sigmoid_model(test_input);

        // Drive input to the FPGA (Write to memory-mapped reg)
        HW_SIGMOID_IN_REG = test_input;
        
        // Read calculated output from the FPGA
        uint8_t actual_hw_result = HW_SIGMOID_OUT_REG;

        // Compare and log errors
        if (expected_sw_result != actual_hw_result) {
            printf("[ERROR] Mismatch at Input 0x%02X! SW Expected: 0x%02X, HW Actual: 0x%02X\n", 
                   test_input, expected_sw_result, actual_hw_result);
            errors++;
        }
    }

    printf("-----------------------------------------------\n");
    if (errors == 0) {
        printf("SUCCESS: Hardware accelerator perfectly matches the C model\n");
    } else {
        printf("FAILED: Found %d mismatches between HW and SW\n", errors);
    }
    printf("-----------------------------------------------\n");

    return 0;
}