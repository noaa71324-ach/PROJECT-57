#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

// Helper function to print 8 bits in binary format
void print_binary_8b(uint8_t val) {
    for (int i = 7; i >= 0; i--) {
        printf("%d", (val >> i) & 1);
        if (i == 4) printf("_");
    }
}

// Mathematical true sigmoid function
float true_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Hardware accelerator model
// Takes an 8-bit sign-magnitude input and returns an 8-bit Q1.7 output
uint8_t hardware_sigmoid_accelerator(uint8_t input_x) {
    // Step 1: Extract sign bit (MSB) and absolute value (remaining 7 bits)
    uint8_t sign = (input_x >> 7) & 0x01;
    uint8_t abs_x = input_x & 0x7F;

    uint8_t interval_idx = 0;
    uint32_t shift_add_result = 0;
    uint32_t intercept_q1_10 = 0;

    // Step 2: Interval decoder
    // Checks which of the 7 paper-defined segments the absolute input belongs to
    if (abs_x >= 80) { interval_idx = 6; } // For values > 5
    else if (abs_x >= 64) { interval_idx = 5; } // For values between 4 and 5
    else if (abs_x >= 48) { interval_idx = 4; } // For values between 3 and 4
    else if (abs_x >= 36) { interval_idx = 3; } // For values between 2.25 and 3
    else if (abs_x >= 24) { interval_idx = 2; } // For values between 1.5 and 2.25
    else if (abs_x >= 11) { interval_idx = 1; } // For values between 0.6875 and 1.5
    else { interval_idx = 0; } // For values < 0.6875

    // Step 3: Shift-Add network and intercept LUT
    // Linear approximation calculation using bit shifts (y = m*x + c)
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
        shift_add_result = 0; // In this range the function is constant and equals 1
        break;
    }

    // Step 4: Add slope result to intercept and return to Q1.7 format
    uint32_t mx_plus_c_q1_10 = shift_add_result + intercept_q1_10;
    uint8_t sigma_abs_x = mx_plus_c_q1_10 >> 3;

    // Step 5: Symmetry handling via conditional 2's complement
    // If original number was negative, subtract result from 1
    if (sign == 1) {
        return 128 - sigma_abs_x;
    }
    else {
        return sigma_abs_x;
    }
}

int main() {
    // Initialize random seed and generate float input (e.g., between -6 and 6)
    srand((unsigned int)time(NULL));
    float original_float_input = ((float)rand() / (float)(RAND_MAX)) * 12.0f - 6.0f;

    // Convert float input to fixed-point Q3.4 representation
    int integer_representation = (int)roundf(original_float_input * 16.0f);

    uint8_t input_hw = 0;
    if (integer_representation < 0) {
        input_hw = (uint8_t)(-integer_representation);
        input_hw = input_hw | 0x80; // Set sign bit for negative numbers
    }
    else {
        input_hw = (uint8_t)(integer_representation);
    }

    // Run hardware accelerator
    uint8_t hw_output = hardware_sigmoid_accelerator(input_hw);

    // Convert hardware output back to float
    float hw_float_result = hw_output / 128.0f;

    // Check error against true sigmoid considering quantized input 
    float quantized_input_float = (float)integer_representation / 16.0f;
    float true_val = true_sigmoid(quantized_input_float);

    float abs_error = fabsf(hw_float_result - true_val);
    float fsr_error_percent = (abs_error / 1.0f) * 100.0f;

    printf("--- Sigmoid Hardware Accelerator Simulation ---\n");
    printf("Original Float Input    : %f\n", original_float_input);
    printf("Quantized Float Input   : %f\n", quantized_input_float);
    printf("-----------------------------------------------\n");

    printf("Hardware Input (Bin)    : ");
    print_binary_8b(input_hw);
    printf(" (Sign_magnitude)\n");

    printf("Hardware Output (Bin)   : ");
    print_binary_8b(hw_output);
    printf(" (Q1.7)\n");

    printf("-----------------------------------------------\n");
    printf("Hardware Float Result   : %f\n", hw_float_result);
    printf("True Mathematical Result: %f (Based on Quantized Input)\n", true_val);
    printf("Absolute Error          : %f\n", abs_error);
    printf("Absolute Error %% (FSR)  : %.4f%%\n", fsr_error_percent);
    printf("-----------------------------------------------\n");

    // Open file to write test vectors for Verilog simulation
    FILE* fp = fopen("test_vectors.txt", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    float max_error = 0.0f;
    uint8_t worst_case_input = 0;

    printf("Running Exhaustive Hardware Simulation for all 256 inputs\n");

    // Run on all 256 possible 8-bit values
    for (int i = 0; i < 256; i++) {
        uint8_t input_hw = (uint8_t)i;

        // Reconstruct float from hardware sign-magnitude input
        uint8_t sign = (input_hw >> 7) & 0x01;
        uint8_t abs_val = input_hw & 0x7F;
        float quantized_input_float = (float)abs_val / 16.0f; // Q3.4 representation
        if (sign == 1) {
            quantized_input_float = -quantized_input_float;
        }

        // Run hardware accelerator
        uint8_t hw_output = hardware_sigmoid_accelerator(input_hw);

        // Write binary input-output pairs to file for testbench
        for (int b = 7; b >= 0; b--) fprintf(fp, "%d", (input_hw >> b) & 1);
        fprintf(fp, "_");
        for (int b = 7; b >= 0; b--) fprintf(fp, "%d", (hw_output >> b) & 1);
        fprintf(fp, "\n");

        // Calculate error against true sigmoid
        float hw_float_result = hw_output / 128.0f;
        float true_val = true_sigmoid(quantized_input_float);
        float abs_error = fabsf(hw_float_result - true_val);

        // Update maximum error
        if (abs_error > max_error) {
            max_error = abs_error;
            worst_case_input = input_hw;
        }
    }

    fclose(fp);

    printf("Simulation Complete. Test vectors saved to 'test_vectors.txt'\n");
    printf("-----------------------------------------------\n");
    printf("Maximum Absolute Error  : %f\n", max_error);
    printf("Max Error %% (FSR)        : %.4f%%\n", (max_error / 1.0f) * 100.0f);
    printf("Worst Case Input - Hex: 0x%02X | Dec: %d | Bin: ", worst_case_input, worst_case_input);
    print_binary_8b(worst_case_input);
    printf("\n");
    printf("-----------------------------------------------\n");

    // Check if max error meets the target spec
    // 0.0078125 * 1.2 = 0.009375
    if (max_error <= 0.009375f) {
        printf("SUCCESS: Hardware logic meets the 7-bit architecture targets (< 1.2 LSB error).\n");
    }
    else {
        printf("WARNING: Maximum error exceeds hardware tolerances.\n");
    }

    return 0;
}