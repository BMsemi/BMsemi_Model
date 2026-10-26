`timescale 1ns/1ps

// Deterministic PL-side INT4 RRAM programming model.
//
// Units:
//   conductance: 0.1 uS
//   voltage:     mV
//
// This is the synthesizable LUT counterpart of:
//   scripts/model_rram_int4_programming_accuracy.py
//
// The source heatmap is diagnostic preparation-pulse data, not a qualified
// characterization grid. PL intentionally omits Monte Carlo noise; it performs
// deterministic verify-and-pulse updates from the same SET/RESET mean dG LUTs.
module rram_int4_programming_accuracy_lut #(
    parameter integer MAX_PULSES = 24
) (
    input  logic        clk,
    input  logic        rstn,
    input  logic        start,
    input  logic [3:0]  target_code,
    input  logic [9:0]  initial_g_tenth_us,

    output logic        busy,
    output logic        done,
    output logic        correct,
    output logic [3:0]  decoded_code,
    output logic [9:0]  target_g_tenth_us,
    output logic [9:0]  final_g_tenth_us,
    output logic [5:0]  pulse_count,
    output logic        last_mode_reset,
    output logic signed [15:0] last_delta_tenth_us,
    output logic [15:0] last_voltage_mv
);
    localparam signed [15:0] INVALID = -16'sd32768;
    localparam logic [5:0] MAX_PULSE_COUNT = MAX_PULSES;

    logic [9:0] current_g;
    logic [9:0] target_g;
    logic [3:0] current_decoded;
    logic signed [15:0] selected_delta;
    logic [15:0] selected_voltage_mv;
    logic do_reset;
    logic at_target;

    assign target_g = 10'd50 + ({6'd0, target_code} * 10'd60);
    assign target_g_tenth_us = target_g;
    assign final_g_tenth_us = current_g;
    assign current_decoded = decode_int4(current_g);
    assign decoded_code = current_decoded;
    assign at_target = (current_decoded == target_code);
    assign do_reset = (current_g > target_g);

    always_comb begin
        if (do_reset) begin
            selected_delta = choose_reset_delta(current_g, current_g - target_g);
            selected_voltage_mv = choose_reset_voltage(current_g, current_g - target_g);
        end else begin
            selected_delta = choose_set_delta(current_g, target_g - current_g);
            selected_voltage_mv = choose_set_voltage(current_g, target_g - current_g);
        end
    end

    always_ff @(posedge clk) begin
        if (!rstn) begin
            busy <= 1'b0;
            done <= 1'b0;
            correct <= 1'b0;
            current_g <= '0;
            pulse_count <= '0;
            last_mode_reset <= 1'b0;
            last_delta_tenth_us <= '0;
            last_voltage_mv <= '0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                busy <= 1'b1;
                current_g <= clamp_g(initial_g_tenth_us);
                pulse_count <= '0;
                correct <= 1'b0;
                last_mode_reset <= 1'b0;
                last_delta_tenth_us <= '0;
                last_voltage_mv <= '0;
            end else if (busy) begin
                if (at_target || pulse_count >= MAX_PULSE_COUNT || selected_delta <= 0) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    correct <= at_target;
                end else begin
                    if (do_reset)
                        current_g <= clamp_g($signed({6'd0, current_g}) - selected_delta);
                    else
                        current_g <= clamp_g($signed({6'd0, current_g}) + selected_delta);
                    pulse_count <= pulse_count + 1'b1;
                    last_mode_reset <= do_reset;
                    last_delta_tenth_us <= selected_delta;
                    last_voltage_mv <= selected_voltage_mv;
                end
            end
        end
    end

    function automatic [9:0] clamp_g(input signed [15:0] value);
        begin
            if (value < 16'sd50)
                clamp_g = 10'd50;
            else if (value > 16'sd950)
                clamp_g = 10'd950;
            else
                clamp_g = value[9:0];
        end
    endfunction

    function automatic [3:0] decode_int4(input [9:0] g_tenth);
        integer code;
        integer best_code;
        integer best_dist;
        integer target;
        integer dist;
        begin
            best_code = 0;
            best_dist = 1024;
            for (code = 0; code < 16; code = code + 1) begin
                target = 50 + (60 * code);
                dist = (g_tenth > target) ? (g_tenth - target) : (target - g_tenth);
                if (dist < best_dist) begin
                    best_dist = dist;
                    best_code = code;
                end
            end
            decode_int4 = best_code[3:0];
        end
    endfunction

    function automatic integer set_bin(input [9:0] g_tenth);
        begin
            if (g_tenth < 10'd200) set_bin = 0;
            else if (g_tenth < 10'd400) set_bin = 1;
            else if (g_tenth < 10'd600) set_bin = 2;
            else if (g_tenth < 10'd800) set_bin = 3;
            else set_bin = 4;
        end
    endfunction

    function automatic integer reset_bin(input [9:0] g_tenth);
        begin
            if (g_tenth < 10'd400) reset_bin = 0;
            else if (g_tenth < 10'd600) reset_bin = 1;
            else if (g_tenth < 10'd800) reset_bin = 2;
            else reset_bin = 3;
        end
    endfunction

    function automatic signed [15:0] set_lut(input integer row, input integer bin);
        begin
            set_lut = INVALID;
            case (row)
                0: case (bin) 3: set_lut = 16'sd101; default: ; endcase
                1: case (bin) 1: set_lut = 16'sd184; 2: set_lut = 16'sd86; 3: set_lut = 16'sd36; default: ; endcase
                2: case (bin) 0: set_lut = 16'sd213; 1: set_lut = 16'sd151; 2: set_lut = 16'sd8; 3: set_lut = -16'sd18; default: ; endcase
                3: case (bin) 0: set_lut = 16'sd2; 1: set_lut = 16'sd33; 2: set_lut = 16'sd2; 3: set_lut = -16'sd20; default: ; endcase
                4: case (bin) 0: set_lut = 16'sd75; 1: set_lut = 16'sd45; 2: set_lut = -16'sd32; 3: set_lut = -16'sd7; default: ; endcase
                5: case (bin) 0: set_lut = 16'sd107; 1: set_lut = 16'sd26; 2: set_lut = 16'sd19; 3: set_lut = -16'sd26; default: ; endcase
                6: case (bin) 0: set_lut = -16'sd42; 1: set_lut = -16'sd42; 2: set_lut = 16'sd28; 3: set_lut = 16'sd20; default: ; endcase
                7: case (bin) 0: set_lut = 16'sd30; 1: set_lut = 16'sd12; 2: set_lut = 16'sd31; 3: set_lut = 16'sd42; default: ; endcase
                8: case (bin) 0: set_lut = 16'sd53; 1: set_lut = 16'sd41; 2: set_lut = 16'sd49; 3: set_lut = 16'sd42; default: ; endcase
                9: case (bin) 0: set_lut = 16'sd65; 1: set_lut = 16'sd40; 2: set_lut = 16'sd69; 3: set_lut = 16'sd48; 4: set_lut = -16'sd85; default: ; endcase
                default: ;
            endcase
        end
    endfunction

    function automatic [15:0] set_voltage(input integer row);
        begin
            case (row)
                0: set_voltage = 16'd1350;
                1: set_voltage = 16'd1250;
                2: set_voltage = 16'd1150;
                3: set_voltage = 16'd1050;
                4: set_voltage = 16'd950;
                5: set_voltage = 16'd850;
                6: set_voltage = 16'd750;
                7: set_voltage = 16'd650;
                8: set_voltage = 16'd550;
                default: set_voltage = 16'd450;
            endcase
        end
    endfunction

    function automatic signed [15:0] reset_lut(input integer row, input integer bin);
        begin
            reset_lut = INVALID;
            case (row)
                0: case (bin) 1: reset_lut = 16'sd299; default: ; endcase
                1: case (bin) 0: reset_lut = 16'sd102; 1: reset_lut = 16'sd162; default: ; endcase
                2: case (bin) 0: reset_lut = 16'sd29; 1: reset_lut = -16'sd123; default: ; endcase
                3: case (bin) 0: reset_lut = -16'sd5; 1: reset_lut = 16'sd57; default: ; endcase
                4: case (bin) 0: reset_lut = 16'sd23; 1: reset_lut = 16'sd67; default: ; endcase
                5: case (bin) 0: reset_lut = -16'sd14; 1: reset_lut = 16'sd77; 2: reset_lut = 16'sd208; default: ; endcase
                6: case (bin) 0: reset_lut = -16'sd24; 1: reset_lut = -16'sd26; 2: reset_lut = 16'sd140; 3: reset_lut = 16'sd139; default: ; endcase
                7: case (bin) 0: reset_lut = -16'sd8; 1: reset_lut = -16'sd46; 2: reset_lut = 16'sd1; 3: reset_lut = 16'sd124; default: ; endcase
                8: case (bin) 0: reset_lut = -16'sd17; 1: reset_lut = 16'sd5; 2: reset_lut = 16'sd12; 3: reset_lut = 16'sd27; default: ; endcase
                9: case (bin) 0: reset_lut = 16'sd49; 1: reset_lut = -16'sd38; 2: reset_lut = 16'sd0; default: ; endcase
                10: case (bin) 0: reset_lut = 16'sd22; 1: reset_lut = -16'sd34; 2: reset_lut = 16'sd15; 3: reset_lut = 16'sd49; default: ; endcase
                11: case (bin) 0: reset_lut = -16'sd2; 1: reset_lut = 16'sd10; 2: reset_lut = -16'sd5; 3: reset_lut = 16'sd74; default: ; endcase
                12: case (bin) 0: reset_lut = -16'sd30; 1: reset_lut = -16'sd28; 2: reset_lut = 16'sd8; 3: reset_lut = -16'sd23; default: ; endcase
                13: case (bin) 0: reset_lut = -16'sd3; 1: reset_lut = 16'sd24; 2: reset_lut = 16'sd6; 3: reset_lut = 16'sd69; default: ; endcase
                14: case (bin) 0: reset_lut = -16'sd11; 1: reset_lut = 16'sd25; 2: reset_lut = 16'sd28; 3: reset_lut = 16'sd61; default: ; endcase
                15: case (bin) 0: reset_lut = 16'sd22; 1: reset_lut = 16'sd8; 2: reset_lut = 16'sd7; 3: reset_lut = 16'sd90; default: ; endcase
                16: case (bin) 0: reset_lut = -16'sd15; 1: reset_lut = 16'sd27; 2: reset_lut = 16'sd7; 3: reset_lut = 16'sd61; default: ; endcase
                17: case (bin) 0: reset_lut = 16'sd17; 1: reset_lut = 16'sd1; 2: reset_lut = 16'sd12; 3: reset_lut = 16'sd70; default: ; endcase
                default: ;
            endcase
        end
    endfunction

    function automatic [15:0] reset_voltage(input integer row);
        begin
            reset_voltage = 16'd2750 - (row[15:0] * 16'd100);
        end
    endfunction

    function automatic signed [15:0] choose_set_delta(input [9:0] g, input [9:0] required);
        integer row;
        integer bin;
        signed [15:0] d;
        signed [15:0] best_under;
        signed [15:0] best_over;
        begin
            bin = set_bin(g);
            best_under = 16'sd0;
            best_over = 16'sd32767;
            for (row = 0; row < 10; row = row + 1) begin
                d = set_lut(row, bin);
                if (d > 0) begin
                    if (d <= required && (best_under == 0 || (required - d) < (required - best_under)))
                        best_under = d;
                    if (d < best_over)
                        best_over = d;
                end
            end
            choose_set_delta = (best_under > 0) ? best_under :
                               ((best_over != 16'sd32767) ? best_over : 16'sd0);
        end
    endfunction

    function automatic [15:0] choose_set_voltage(input [9:0] g, input [9:0] required);
        integer row;
        integer bin;
        signed [15:0] d;
        signed [15:0] chosen;
        begin
            bin = set_bin(g);
            chosen = choose_set_delta(g, required);
            choose_set_voltage = 16'd0;
            for (row = 0; row < 10; row = row + 1) begin
                d = set_lut(row, bin);
                if (d == chosen && choose_set_voltage == 0)
                    choose_set_voltage = set_voltage(row);
            end
        end
    endfunction

    function automatic signed [15:0] choose_reset_delta(input [9:0] g, input [9:0] required);
        integer row;
        integer bin;
        signed [15:0] d;
        signed [15:0] best_under;
        signed [15:0] best_over;
        begin
            bin = reset_bin(g);
            best_under = 16'sd0;
            best_over = 16'sd32767;
            for (row = 0; row < 18; row = row + 1) begin
                d = reset_lut(row, bin);
                if (d > 0) begin
                    if (d <= required && (best_under == 0 || (required - d) < (required - best_under)))
                        best_under = d;
                    if (d < best_over)
                        best_over = d;
                end
            end
            choose_reset_delta = (best_under > 0) ? best_under :
                                 ((best_over != 16'sd32767) ? best_over : 16'sd0);
        end
    endfunction

    function automatic [15:0] choose_reset_voltage(input [9:0] g, input [9:0] required);
        integer row;
        integer bin;
        signed [15:0] d;
        signed [15:0] chosen;
        begin
            bin = reset_bin(g);
            chosen = choose_reset_delta(g, required);
            choose_reset_voltage = 16'd0;
            for (row = 0; row < 18; row = row + 1) begin
                d = reset_lut(row, bin);
                if (d == chosen && choose_reset_voltage == 0)
                    choose_reset_voltage = reset_voltage(row);
            end
        end
    endfunction
endmodule
