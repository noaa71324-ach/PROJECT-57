#include <stdio.h>
#include <stdint.h>
#include <math.h>

/*
 * Original true sigmoid calculation
 */
float true_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/*
 * Model 1: Full Range Model (as specified in the paper: [-5, 5])
 * Wide segment distribution to maintain accuracy at the edges.
 */
uint8_t hw_sigmoid_full_range(uint8_t input_x) {
    // Step 1: Extract sign bit (MSB) and absolute value (Sign & Magnitude)
    uint8_t sign = (input_x >> 7) & 0x01;
    uint8_t abs_x = input_x & 0x7F;

    uint8_t interval_idx = 0;
    uint32_t shift_add_result = 0;
    uint32_t intercept_q1_10 = 0;

    // Step 2: Interval Decoder - split into 7 segments
    if (abs_x >= 80) { interval_idx = 6; } // [4.0, 5.0+)
    else if (abs_x >= 64) { interval_idx = 5; } // [3.0, 4.0)
    else if (abs_x >= 48) { interval_idx = 4; } // [2.25, 3.0)
    else if (abs_x >= 36) { interval_idx = 3; } // [1.5, 2.25)
    else if (abs_x >= 24) { interval_idx = 2; } // [0.6875, 1.5)
    else if (abs_x >= 11) { interval_idx = 1; } // [0.0, 0.6875)
    else { interval_idx = 0; }

    // Step 3: Shift-Add network and intercept LUT
    // Internal values are multiplied by 8 (left shift by 3) to preserve fractional precision
    switch (interval_idx) {
    case 0: // Slope 16
        intercept_q1_10 = 64 << 3;
        shift_add_result = (abs_x << 4);
        break;
    case 1: // Slope 12
        intercept_q1_10 = 69 << 3;
        shift_add_result = (abs_x << 3) + (abs_x << 2);
        break;
    case 2: // Slope 8
        intercept_q1_10 = 80 << 3;
        shift_add_result = (abs_x << 3);
        break;
    case 3: // Slope 4
        intercept_q1_10 = 98 << 3;
        shift_add_result = (abs_x << 2);
        break;
    case 4: // Slope 2
        intercept_q1_10 = 110 << 3;
        shift_add_result = (abs_x << 1);
        break;
    case 5: // Slope 1
        intercept_q1_10 = 118 << 3;
        shift_add_result = abs_x;
        break;
    case 6: // Flat region where function equals 1, so slope is 0
        intercept_q1_10 = 128 << 3;
        shift_add_result = 0;
        break;
    }

    // Step 4: Add linear equation and divide by 8 (>> 3) to return to output format
    uint32_t mx_plus_c = shift_add_result + intercept_q1_10;
    uint8_t sigma_abs_x = (uint8_t)(mx_plus_c >> 3);

    // Step 5: Negative value handling (Conditional 2's Complement)
    // Subtract from 128 using logic operations instead of a dedicated subtractor
    if (sign == 1) {
        sigma_abs_x = (~sigma_abs_x) + 1;
        sigma_abs_x = sigma_abs_x & 0x7F;
    }

    // Step 6: Clamp result to system legal maximum (128 represents 1.0)
    if (sigma_abs_x > 128 && sign == 0) {
        sigma_abs_x = 128;
    }

    return sigma_abs_x;
}

/*
 * Model 2: Narrow Range Model ([-2, 2])
 * Tests hardware reduction impact by forcing early saturation.
 */
uint8_t hw_sigmoid_narrow_range(uint8_t input_x) {
    uint8_t sign = (input_x >> 7) & 0x01;
    uint8_t abs_x = input_x & 0x7F;

    uint8_t interval_idx = 0;
    uint32_t shift_add_result = 0;
    uint32_t intercept_q1_10 = 0;

    // Narrow interval decoder: forces early saturation (reaches 1) starting from 2.0
    if (abs_x >= 32) { interval_idx = 3; } // [2.0, infinity) - Early Saturation
    else if (abs_x >= 24) { interval_idx = 2; } // [1.5, 2.0)
    else if (abs_x >= 11) { interval_idx = 1; } // [0.6875, 1.5)
    else { interval_idx = 0; } // [0.0, 0.6875)

    // Intercept LUT and shift-add network
    switch (interval_idx) {
    case 0: // Slope 15
        intercept_q1_10 = 64 << 3;
        shift_add_result = (abs_x << 3) + (abs_x << 2) + (abs_x << 1) + abs_x;
        break;
    case 1: // Slope 12
        intercept_q1_10 = 69 << 3;
        shift_add_result = (abs_x << 3) + (abs_x << 2);
        break;
    case 2: // Slope 8
        intercept_q1_10 = 81 << 3;
        shift_add_result = (abs_x << 3);
        break;
    case 3: // Flat region where function equals 1, so slope is 0
        intercept_q1_10 = 128 << 3;
        shift_add_result = 0;
        break;
    }

    uint32_t mx_plus_c = shift_add_result + intercept_q1_10;
    uint8_t sigma_abs_x = (uint8_t)(mx_plus_c >> 3);

    if (sign == 1) {
        sigma_abs_x = (~sigma_abs_x) + 1;
        sigma_abs_x = sigma_abs_x & 0x7F;
    }

    if (sigma_abs_x > 128 && sign == 0) {
        sigma_abs_x = 128;
    }

    return sigma_abs_x;
}

/*
 * Hardware-Software Translation Functions
 * Convert floats to 8-bit format before entering the circuit,
 * and translate binary fixed-point output back to float for comparison.
 */
float simulate_full_range_float(float x) {
    // Step A: Extract sign and absolute value in software before hardware conversion
    uint8_t sign = (x < 0.0f) ? 1 : 0;
    float abs_x = (x < 0.0f) ? -x : x;

    // Step B: Multiply by 16 for 4-bit fractional representation (Q3.4) and round
    int integer_part = (int)roundf(abs_x * 16.0f);

    // Clamp max value to 127 to prevent overflow and sign bit override
    if (integer_part > 127) integer_part = 127;

    // Step C: Combine MSB sign bit with absolute value
    uint8_t input_hw = (sign << 7) | (integer_part & 0x7F);

    // Run simulated hardware calculation
    uint8_t hw_output = hw_sigmoid_full_range(input_hw);

    // Step D: Decode result - divide by 128 to convert int back to float (7 fractional bits)
    return hw_output / 128.0f;
}

float simulate_narrow_range_float(float x) {
    uint8_t sign = (x < 0.0f) ? 1 : 0;
    float abs_x = (x < 0.0f) ? -x : x;

    int integer_part = (int)roundf(abs_x * 16.0f);
    if (integer_part > 127) integer_part = 127;

    uint8_t input_hw = (sign << 7) | (integer_part & 0x7F);
    uint8_t hw_output = hw_sigmoid_narrow_range(input_hw);

    return hw_output / 128.0f;
}

/*
 * Main Testbench
 * Performance comparison between full and narrow range models against theoretical math.
 */
int main() {
    float test_points[] = { 0.0f, 0.8f, 1.8f, 2.5f, -3.0f, 4.5f };
    int num_points = 6;

    printf(" x      | HW Full Range [-5,5] | HW Narrow Range [-2,2] | True Math Sigmoid | Error (Full) | Error (Narrow)\n");
    printf("----------------------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < num_points; i++) {
        float x = test_points[i];

        // Run simulation for both hardware models and theoretical reference
        float hw_full = simulate_full_range_float(x);
        float hw_narrow = simulate_narrow_range_float(x);
        float true_val = true_sigmoid(x);

        // Calculate conversion and approximation error against true math value
        float err_full = fabs(hw_full - true_val);
        float err_narrow = fabs(hw_narrow - true_val);

        printf("%6.2f | %20.4f | %22.4f | %17.4f | %12.4f | %14.4f\n",
            x, hw_full, hw_narrow, true_val, err_full, err_narrow);
    }

    return 0;
}