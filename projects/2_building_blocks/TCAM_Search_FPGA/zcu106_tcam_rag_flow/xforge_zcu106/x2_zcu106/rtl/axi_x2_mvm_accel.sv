`timescale 1ns / 1ps

// -----------------------------------------------------------------------------
// PE lane: ideal PWM_INT8_ROW_SERIAL analog-emulation multiply-accumulate lane.
//
// This mirrors the software CrossbarArray PWM path:
//   - signed activations are driven as unsigned PWM pulses by adding 128
//   - each signed INT8 weight is stored as four 4-bit differential banks:
//       pos0, pos1, neg0, neg1
//   - the four nibble products are row-serial accumulated
//   - the signed-activation bridge correction, 128 * weight, is subtracted
//
// The external result remains the signed INT8 MVM value, but the PE datapath now
// follows the analog storage/drive structure used by pwm_int8_row_serial.
// -----------------------------------------------------------------------------
module x2_pe_mac_lane (
    input  logic                    clk,
    input  logic                    rstn,
    input  logic                    clear,
    input  logic                    enable,
    input  logic signed [7:0]       activation,
    input  logic signed [7:0]       weight,
    output logic signed [31:0]      acc
);
    function automatic logic signed [31:0] pwm_int8_row_serial_product(
        input logic signed [7:0] activation_i,
        input logic signed [7:0] weight_i
    );
        logic [7:0] activation_drive_u8;
        logic [8:0] weight_mag_u9;
        logic [3:0] a0;
        logic [3:0] a1;
        logic [3:0] pos0;
        logic [3:0] pos1;
        logic [3:0] neg0;
        logic [3:0] neg1;
        logic signed [31:0] p00;
        logic signed [31:0] p01;
        logic signed [31:0] p10;
        logic signed [31:0] p11;
        logic signed [31:0] raw_pwm_sum;
        logic signed [31:0] signed_weight;
        begin
            // signed INT8 -> unsigned PWM drive: -128..127 maps to 0..255.
            activation_drive_u8 = activation_i ^ 8'h80;
            a0 = activation_drive_u8[3:0];
            a1 = activation_drive_u8[7:4];

            // signed INT8 weight -> 4-bit differential magnitude banks.
            weight_mag_u9 = weight_i[7] ? ({1'b0, ~weight_i} + 9'd1) : {1'b0, weight_i};
            pos0 = weight_i[7] ? 4'd0 : weight_mag_u9[3:0];
            pos1 = weight_i[7] ? 4'd0 : weight_mag_u9[7:4];
            neg0 = weight_i[7] ? weight_mag_u9[3:0] : 4'd0;
            neg1 = weight_i[7] ? weight_mag_u9[7:4] : 4'd0;

            p00 = $signed({24'd0, a0}) * $signed({24'd0, pos0})
                - $signed({24'd0, a0}) * $signed({24'd0, neg0});
            p01 = $signed({24'd0, a0}) * $signed({24'd0, pos1})
                - $signed({24'd0, a0}) * $signed({24'd0, neg1});
            p10 = $signed({24'd0, a1}) * $signed({24'd0, pos0})
                - $signed({24'd0, a1}) * $signed({24'd0, neg0});
            p11 = $signed({24'd0, a1}) * $signed({24'd0, pos1})
                - $signed({24'd0, a1}) * $signed({24'd0, neg1});

            raw_pwm_sum = p00 + (p01 <<< 4) + (p10 <<< 4) + (p11 <<< 8);
            signed_weight = {{24{weight_i[7]}}, weight_i};

            // CrossbarX2Backend subtracts 128 * sum(W) after the unsigned-PWM
            // crossbar pass. Applying it per row is algebraically identical.
            pwm_int8_row_serial_product = raw_pwm_sum - (signed_weight <<< 7);
        end
    endfunction

    always_ff @(posedge clk) begin
        if (!rstn) begin
            acc <= 32'sd0;
        end else if (clear) begin
            acc <= 32'sd0;
        end else if (enable) begin
            acc <= acc + pwm_int8_row_serial_product(activation, weight);
        end
    end
endmodule


// -----------------------------------------------------------------------------
// AXI-Stream MVM core.
//
// Input DMA stream format, configured by AXI-Lite rows/cols/batch:
//   - ceil(rows * cols / 4) words of weight bytes, signed INT8, row-major
//     weights[row * cols + col], little-endian per word
//   - batch * ceil(rows / 4) words of vector bytes, signed INT8,
//     little-endian per word, vectors packed batch-major
//
// Output DMA stream format:
//   - batch * cols words of signed INT32 accumulators, batch-major
// -----------------------------------------------------------------------------
module x2_mvm_axis_core #(
    parameter integer MAX_ROWS                    = 32,
    parameter integer MAX_COLS                    = 32,
    parameter integer MAX_BATCH                   = 32,
    parameter integer N_PE                        = 8,
    parameter integer DATA_WIDTH                  = 32,
    parameter integer TDC_TIMING_ENABLE           = 1,
    parameter integer TDC_TIMING_MIN_PS           = 0,
    parameter integer TDC_TIMING_MAX_PS           = 576,
    parameter integer TDC_TIMING_REFERENCE_LEVELS = 256
) (
    input  logic                    clk,
    input  logic                    rstn,

    input  logic                    start_pulse,
    input  logic                    abort_pulse,
    input  logic [7:0]              cfg_rows,
    input  logic [7:0]              cfg_cols,
    input  logic [7:0]              cfg_batch,

    output logic                    idle,
    output logic                    busy,
    output logic [7:0]              state_flags,
    output logic                    done_pulse,
    output logic                    error_pulse,
    output logic [7:0]              error_code,
    output logic [31:0]             expected_input_words,
    output logic [31:0]             words_received,
    output logic [31:0]             outputs_sent,
    output logic [31:0]             cycle_count,
    output logic                    tdc_timing_valid,
    output logic [7:0]              tdc_first_ready_column,
    output logic [7:0]              tdc_last_ready_column,
    output logic [31:0]             tdc_arrival_min_ps,
    output logic [31:0]             tdc_arrival_max_ps,
    output logic [31:0]             tdc_arrival_spread_ps,
    output logic [31:0]             tdc_pe_completion_time_ps,
    output logic [31:0]             debug_shape,
    output logic [31:0]             debug_stream0,
    output logic [31:0]             debug_stream1,
    output logic [31:0]             debug_stream2,
    output logic [31:0]             debug_stream3,
    output logic [31:0]             debug_weight0,
    output logic [31:0]             debug_weight1,
    output logic [31:0]             debug_vector0,
    output logic [31:0]             debug_vector1,
    output logic [31:0]             debug_result0,
    output logic [31:0]             debug_result1,
    output logic [31:0]             debug_result2,
    output logic [31:0]             debug_result3,
    output logic [31:0]             debug_axis0,
    output logic [31:0]             debug_axis1,
    output logic [31:0]             debug_axis2,
    output logic [31:0]             debug_axis3,
    output logic [31:0]             debug_axis_meta,

    input  logic [DATA_WIDTH-1:0]   s_axis_tdata,
    input  logic                    s_axis_tvalid,
    output logic                    s_axis_tready,
    input  logic                    s_axis_tlast,

    output logic [DATA_WIDTH-1:0]   m_axis_tdata,
    output logic                    m_axis_tvalid,
    input  logic                    m_axis_tready,
    output logic                    m_axis_tlast
);
    localparam integer MAX_WEIGHT_BYTES = MAX_ROWS * MAX_COLS;
    localparam integer MAX_VECTOR_BYTES = MAX_BATCH * MAX_ROWS;
    localparam integer WORD_CNT_WIDTH   = 16;
    localparam integer TDC_TIMING_SPAN_PS = TDC_TIMING_MAX_PS - TDC_TIMING_MIN_PS;

    localparam logic [7:0] ERR_NONE           = 8'h00;
    localparam logic [7:0] ERR_BAD_CONFIG     = 8'h01;
    localparam logic [7:0] ERR_TLAST_EARLY    = 8'h02;
    localparam logic [7:0] ERR_TLAST_MISSING  = 8'h03;

    typedef enum logic [3:0] {
        ST_IDLE         = 4'd0,
        ST_LOAD         = 4'd1,
        ST_COMPUTE_INIT = 4'd2,
        ST_COMPUTE      = 4'd3,
        ST_STORE        = 4'd4,
        ST_TDC_SCAN     = 4'd5,
        ST_TDC_DIV_PREP = 4'd6,
        ST_TDC_DIV      = 4'd7,
        ST_TDC_FINISH   = 4'd8,
        ST_SEND_PREP    = 4'd9,
        ST_SEND         = 4'd10,
        ST_ERROR        = 4'd11
    } state_t;

    state_t state;

    logic [7:0] rows_reg;
    logic [7:0] cols_reg;
    logic [7:0] batch_reg;
    logic [WORD_CNT_WIDTH-1:0] vector_words_reg;
    logic [WORD_CNT_WIDTH-1:0] weight_words_reg;
    logic [WORD_CNT_WIDTH-1:0] total_words_reg;
    logic [WORD_CNT_WIDTH-1:0] load_word_idx;
    logic [WORD_CNT_WIDTH-1:0] load_vector_word_idx;
    logic [WORD_CNT_WIDTH-1:0] load_batch_idx;
    logic [WORD_CNT_WIDTH-1:0] row_idx;
    logic [WORD_CNT_WIDTH-1:0] col_base;
    logic [WORD_CNT_WIDTH-1:0] out_idx;
    logic [WORD_CNT_WIDTH-1:0] batch_idx;
    logic [31:0]                 stream_debug [0:3];

    logic signed [7:0]  vector_batch_mem [0:MAX_VECTOR_BYTES-1];
    logic signed [7:0]  weight_mem [0:MAX_WEIGHT_BYTES-1];
    logic signed [31:0] result_mem [0:MAX_COLS-1];

    logic                    pe_clear;
    logic                    pe_enable;
    logic signed [7:0]       pe_activation;
    logic signed [7:0]       pe_weight [0:N_PE-1];
    logic signed [31:0]      pe_acc    [0:N_PE-1];
    logic                    tdc_timing_valid_reg;
    logic [7:0]              tdc_first_ready_column_reg;
    logic [7:0]              tdc_last_ready_column_reg;
    logic [31:0]             scan_tdc_magnitude;
    logic [31:0]             tdc_min_magnitude;
    logic [31:0]             tdc_max_magnitude;
    logic [31:0]             tdc_arrival_min_reg;
    logic [31:0]             tdc_arrival_max_reg;
    logic [63:0]             tdc_dividend_early;
    logic [63:0]             tdc_dividend_late;
    logic [63:0]             tdc_divisor_early;
    logic [63:0]             tdc_divisor_late;
    logic [64:0]             tdc_remainder_early;
    logic [64:0]             tdc_remainder_late;
    logic [63:0]             tdc_quotient_early;
    logic [63:0]             tdc_quotient_late;
    logic [6:0]              tdc_div_bit;
    logic                    final_output_word;
    logic [31:0]             send_data;
    logic                    send_valid;
    logic                    send_last;
    logic [31:0]             axis_debug [0:3];
    logic [15:0]             axis_handshake_count;
    logic [3:0]              axis_debug_lastmask;

    function automatic [WORD_CNT_WIDTH-1:0] ceil_div4(input [WORD_CNT_WIDTH-1:0] value);
        ceil_div4 = (value + 16'd3) >> 2;
    endfunction

    function automatic [31:0] abs_accumulator(input logic signed [31:0] value);
        begin
            abs_accumulator = value[31] ? (~value + 1'b1) : value;
        end
    endfunction

    function automatic logic valid_config(input logic [7:0] rows, input logic [7:0] cols, input logic [7:0] batch);
        valid_config = (rows >= 8'd1) && (rows <= MAX_ROWS) &&
                       (cols >= 8'd1) && (cols <= MAX_COLS) &&
                       (batch >= 8'd1) && (batch <= MAX_BATCH);
    endfunction

    function automatic [WORD_CNT_WIDTH-1:0] expected_words_for(
        input logic [7:0] rows,
        input logic [7:0] cols,
        input logic [7:0] batch
    );
        logic [WORD_CNT_WIDTH-1:0] row_count;
        logic [WORD_CNT_WIDTH-1:0] weight_count;
        begin
            row_count = {8'd0, rows};
            weight_count = {8'd0, rows} * {8'd0, cols};
            expected_words_for = ceil_div4(weight_count) + ({8'd0, batch} * ceil_div4(row_count));
        end
    endfunction

    genvar lane;
    generate
        for (lane = 0; lane < N_PE; lane = lane + 1) begin : pe_lanes
            x2_pe_mac_lane u_lane (
                .clk        (clk),
                .rstn       (rstn),
                .clear      (pe_clear),
                .enable     (pe_enable && ((col_base + lane) < cols_reg)),
                .activation (pe_activation),
                .weight     (pe_weight[lane]),
                .acc        (pe_acc[lane])
            );
        end
    endgenerate

    assign idle                 = (state == ST_IDLE);
    assign busy                 = (state != ST_IDLE) && (state != ST_ERROR);
    assign scan_tdc_magnitude   = (TDC_TIMING_ENABLE != 0)
                                 ? abs_accumulator($signed(result_mem[out_idx]))
                                 : 32'd0;
    assign tdc_timing_valid     = (TDC_TIMING_ENABLE != 0) ? tdc_timing_valid_reg : 1'b0;
    assign tdc_first_ready_column = (TDC_TIMING_ENABLE != 0) ? tdc_first_ready_column_reg : 8'd0;
    assign tdc_last_ready_column  = (TDC_TIMING_ENABLE != 0) ? tdc_last_ready_column_reg : 8'd0;
    assign tdc_arrival_min_ps   = tdc_timing_valid ? tdc_arrival_min_reg : 32'd0;
    assign tdc_arrival_max_ps   = tdc_timing_valid ? tdc_arrival_max_reg : 32'd0;
    assign tdc_arrival_spread_ps = tdc_timing_valid
                                 ? tdc_arrival_max_ps - tdc_arrival_min_ps
                                 : 32'd0;
    assign tdc_pe_completion_time_ps = tdc_arrival_max_ps;
    assign state_flags          = {2'b00, (state == ST_ERROR), ((state == ST_SEND) || (state == ST_SEND_PREP)),
                                   (state == ST_STORE), (state == ST_COMPUTE),
                                   (state == ST_LOAD), (state == ST_IDLE)};
    assign s_axis_tready        = (state == ST_LOAD);
    assign m_axis_tvalid        = send_valid;
    assign m_axis_tdata         = send_data;
    assign final_output_word    = (out_idx == (cols_reg - 8'd1)) &&
                                  (batch_idx == (batch_reg - 8'd1));
    assign m_axis_tlast         = send_last;
    assign expected_input_words = {
        16'd0,
        expected_words_for(cfg_rows, cfg_cols, cfg_batch == 8'd0 ? 8'd1 : cfg_batch)
    };
    assign debug_shape          = {8'd0, batch_reg, cols_reg, rows_reg};
    assign debug_stream0        = stream_debug[0];
    assign debug_stream1        = stream_debug[1];
    assign debug_stream2        = stream_debug[2];
    assign debug_stream3        = stream_debug[3];
    assign debug_weight0        = {weight_mem[3], weight_mem[2], weight_mem[1], weight_mem[0]};
    assign debug_weight1        = {weight_mem[7], weight_mem[6], weight_mem[5], weight_mem[4]};
    assign debug_vector0        = {vector_batch_mem[3], vector_batch_mem[2], vector_batch_mem[1], vector_batch_mem[0]};
    assign debug_vector1        = {vector_batch_mem[7], vector_batch_mem[6], vector_batch_mem[5], vector_batch_mem[4]};
    assign debug_result0        = result_mem[0];
    assign debug_result1        = result_mem[1];
    assign debug_result2        = result_mem[2];
    assign debug_result3        = result_mem[3];
    assign debug_axis0          = axis_debug[0];
    assign debug_axis1          = axis_debug[1];
    assign debug_axis2          = axis_debug[2];
    assign debug_axis3          = axis_debug[3];
    assign debug_axis_meta      = {axis_debug_lastmask, state, m_axis_tready,
                                   send_valid, send_last, 5'd0, axis_handshake_count};

    always_comb begin
        pe_clear      = (state == ST_COMPUTE_INIT);
        pe_enable     = (state == ST_COMPUTE);
        pe_activation = vector_batch_mem[(batch_idx * MAX_ROWS) + row_idx];
        for (int i = 0; i < N_PE; i = i + 1) begin
            if ((col_base + i[WORD_CNT_WIDTH-1:0]) < cols_reg) begin
                pe_weight[i] = weight_mem[(row_idx * cols_reg) + col_base + i];
            end else begin
                pe_weight[i] = 8'sd0;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!rstn) begin
            state              <= ST_IDLE;
            rows_reg           <= MAX_ROWS;
            cols_reg           <= MAX_COLS;
            batch_reg          <= 8'd1;
            vector_words_reg   <= '0;
            weight_words_reg   <= '0;
            total_words_reg    <= '0;
            load_word_idx      <= '0;
            load_vector_word_idx <= '0;
            load_batch_idx     <= '0;
            row_idx            <= '0;
            col_base           <= '0;
            out_idx            <= '0;
            batch_idx          <= '0;
            done_pulse         <= 1'b0;
            error_pulse        <= 1'b0;
            error_code         <= ERR_NONE;
            words_received     <= 32'd0;
            outputs_sent       <= 32'd0;
            cycle_count        <= 32'd0;
            tdc_timing_valid_reg   <= 1'b0;
            tdc_first_ready_column_reg <= 8'd0;
            tdc_last_ready_column_reg  <= 8'd0;
            tdc_min_magnitude  <= 32'd0;
            tdc_max_magnitude  <= 32'd0;
            tdc_arrival_min_reg <= 32'd0;
            tdc_arrival_max_reg <= 32'd0;
            tdc_dividend_early  <= 64'd0;
            tdc_dividend_late   <= 64'd0;
            tdc_divisor_early   <= 64'd0;
            tdc_divisor_late    <= 64'd0;
            tdc_remainder_early <= 65'd0;
            tdc_remainder_late  <= 65'd0;
            tdc_quotient_early  <= 64'd0;
            tdc_quotient_late   <= 64'd0;
            tdc_div_bit         <= 7'd0;
            send_data          <= 32'd0;
            send_valid         <= 1'b0;
            send_last          <= 1'b0;
            axis_handshake_count <= 16'd0;
            axis_debug_lastmask  <= 4'd0;
            for (int dbg = 0; dbg < 4; dbg = dbg + 1) begin
                stream_debug[dbg] <= 32'd0;
                axis_debug[dbg]   <= 32'd0;
            end
        end else begin
            done_pulse  <= 1'b0;
            error_pulse <= 1'b0;

            if (busy) begin
                cycle_count <= cycle_count + 32'd1;
            end

            if (abort_pulse) begin
                state       <= ST_IDLE;
                error_code  <= ERR_NONE;
                send_valid  <= 1'b0;
                send_last   <= 1'b0;
                tdc_timing_valid_reg <= 1'b0;
            end else begin
                case (state)
                    ST_IDLE: begin
                        if (start_pulse) begin
                            if (!valid_config(cfg_rows, cfg_cols, cfg_batch == 8'd0 ? 8'd1 : cfg_batch)) begin
                                state       <= ST_ERROR;
                                error_code  <= ERR_BAD_CONFIG;
                                error_pulse <= 1'b1;
                            end else begin
                                rows_reg         <= cfg_rows;
                                cols_reg         <= cfg_cols;
                                batch_reg        <= (cfg_batch == 8'd0 ? 8'd1 : cfg_batch);
                                vector_words_reg <= ceil_div4({8'd0, cfg_rows});
                                weight_words_reg <= ceil_div4({8'd0, cfg_rows} * {8'd0, cfg_cols});
                                total_words_reg  <= expected_words_for(
                                    cfg_rows,
                                    cfg_cols,
                                    cfg_batch == 8'd0 ? 8'd1 : cfg_batch
                                );
                                load_word_idx    <= '0;
                                load_vector_word_idx <= '0;
                                load_batch_idx   <= '0;
                                batch_idx        <= '0;
                                words_received   <= 32'd0;
                                outputs_sent     <= 32'd0;
                                cycle_count      <= 32'd0;
                                tdc_timing_valid_reg <= 1'b0;
                                tdc_first_ready_column_reg <= 8'd0;
                                tdc_last_ready_column_reg  <= 8'd0;
                                tdc_min_magnitude <= 32'd0;
                                tdc_max_magnitude <= 32'd0;
                                tdc_arrival_min_reg <= 32'd0;
                                tdc_arrival_max_reg <= 32'd0;
                                send_data        <= 32'd0;
                                send_valid       <= 1'b0;
                                send_last        <= 1'b0;
                                axis_handshake_count <= 16'd0;
                                axis_debug_lastmask  <= 4'd0;
                                error_code       <= ERR_NONE;
                                for (int dbg = 0; dbg < 4; dbg = dbg + 1) begin
                                    stream_debug[dbg] <= 32'd0;
                                    axis_debug[dbg]   <= 32'd0;
                                end
                                for (int res = 0; res < MAX_COLS; res = res + 1) begin
                                    result_mem[res] <= 32'sd0;
                                end
                                state            <= ST_LOAD;
                            end
                        end
                    end

                    ST_LOAD: begin
                        if (s_axis_tvalid && s_axis_tready) begin
                            if (load_word_idx < 16'd4) begin
                                stream_debug[load_word_idx[1:0]] <= s_axis_tdata;
                            end

                            for (int b = 0; b < 4; b = b + 1) begin
                                logic [WORD_CNT_WIDTH-1:0] byte_index;
                                logic signed [7:0]         byte_value;

                                byte_value = s_axis_tdata[8*b +: 8];

                                if (load_word_idx < weight_words_reg) begin
                                    byte_index = (load_word_idx << 2) + b[WORD_CNT_WIDTH-1:0];
                                    if (byte_index < ({8'd0, rows_reg} * {8'd0, cols_reg})) begin
                                        weight_mem[byte_index] <= byte_value;
                                    end
                                end else begin
                                    byte_index = (load_vector_word_idx << 2) + b[WORD_CNT_WIDTH-1:0];
                                    if ((load_batch_idx < batch_reg) && (byte_index < rows_reg)) begin
                                        vector_batch_mem[(load_batch_idx * MAX_ROWS) + byte_index] <= byte_value;
                                    end
                                end
                            end

                            words_received <= words_received + 32'd1;

                            if (load_word_idx == (total_words_reg - 1'b1)) begin
                                if (!s_axis_tlast) begin
                                    state       <= ST_ERROR;
                                    error_code  <= ERR_TLAST_MISSING;
                                    error_pulse <= 1'b1;
                                end else begin
                                    row_idx  <= '0;
                                    col_base <= '0;
                                    batch_idx <= '0;
                                    state    <= ST_COMPUTE_INIT;
                                end
                            end else begin
                                load_word_idx <= load_word_idx + 1'b1;
                                if (load_word_idx >= weight_words_reg) begin
                                    if (load_vector_word_idx == (vector_words_reg - 1'b1)) begin
                                        load_vector_word_idx <= '0;
                                        load_batch_idx <= load_batch_idx + 1'b1;
                                    end else begin
                                        load_vector_word_idx <= load_vector_word_idx + 1'b1;
                                    end
                                end
                            end
                        end
                    end

                    ST_COMPUTE_INIT: begin
                        row_idx <= '0;
                        if (col_base == 0) begin
                            tdc_timing_valid_reg <= 1'b0;
                            tdc_first_ready_column_reg <= 8'd0;
                            tdc_last_ready_column_reg  <= 8'd0;
                            tdc_min_magnitude <= 32'd0;
                            tdc_max_magnitude <= 32'd0;
                            tdc_arrival_min_reg <= 32'd0;
                            tdc_arrival_max_reg <= 32'd0;
                        end
                        state   <= ST_COMPUTE;
                    end

                    ST_COMPUTE: begin
                        if (row_idx == (rows_reg - 8'd1)) begin
                            state <= ST_STORE;
                        end else begin
                            row_idx <= row_idx + 1'b1;
                        end
                    end

                    ST_STORE: begin
                        for (int p = 0; p < N_PE; p = p + 1) begin
                            if ((col_base + p) < cols_reg) begin
                                result_mem[col_base + p] <= pe_acc[p];
                            end
                        end

                        if ((col_base + N_PE) >= cols_reg) begin
                            out_idx <= '0;
                            state   <= (TDC_TIMING_ENABLE != 0) ? ST_TDC_SCAN : ST_SEND_PREP;
                        end else begin
                            col_base <= col_base + N_PE;
                            row_idx  <= '0;
                            state    <= ST_COMPUTE_INIT;
                        end
                    end

                    ST_TDC_SCAN: begin
                        if (out_idx == 0) begin
                            tdc_min_magnitude <= scan_tdc_magnitude;
                            tdc_max_magnitude <= scan_tdc_magnitude;
                            tdc_first_ready_column_reg <= 8'd0;
                            tdc_last_ready_column_reg  <= 8'd0;
                        end else begin
                            if (scan_tdc_magnitude > tdc_max_magnitude) begin
                                tdc_max_magnitude <= scan_tdc_magnitude;
                                tdc_first_ready_column_reg <= out_idx[7:0];
                            end
                            if (scan_tdc_magnitude < tdc_min_magnitude) begin
                                tdc_min_magnitude <= scan_tdc_magnitude;
                                tdc_last_ready_column_reg <= out_idx[7:0];
                            end else if (scan_tdc_magnitude == tdc_min_magnitude) begin
                                tdc_last_ready_column_reg <= out_idx[7:0];
                            end
                        end

                        if (out_idx == (cols_reg - 8'd1)) begin
                            out_idx <= '0;
                            state   <= ST_TDC_DIV_PREP;
                        end else begin
                            out_idx <= out_idx + 1'b1;
                        end
                    end

                    ST_TDC_DIV_PREP: begin
                        logic [63:0] numerator;
                        logic [63:0] early_divisor;
                        logic [63:0] late_divisor;

                        if ((TDC_TIMING_SPAN_PS <= 0) || (TDC_TIMING_REFERENCE_LEVELS <= 0)) begin
                            tdc_arrival_min_reg <= TDC_TIMING_MIN_PS;
                            tdc_arrival_max_reg <= TDC_TIMING_MIN_PS;
                            tdc_timing_valid_reg    <= 1'b1;
                            state               <= ST_SEND_PREP;
                        end else begin
                            numerator = TDC_TIMING_SPAN_PS;
                            numerator = numerator * TDC_TIMING_REFERENCE_LEVELS;
                            early_divisor = TDC_TIMING_REFERENCE_LEVELS;
                            early_divisor = early_divisor + tdc_max_magnitude;
                            late_divisor = TDC_TIMING_REFERENCE_LEVELS;
                            late_divisor = late_divisor + tdc_min_magnitude;

                            tdc_dividend_early  <= numerator + (early_divisor >> 1);
                            tdc_dividend_late   <= numerator + (late_divisor >> 1);
                            tdc_divisor_early   <= early_divisor;
                            tdc_divisor_late    <= late_divisor;
                            tdc_remainder_early <= 65'd0;
                            tdc_remainder_late  <= 65'd0;
                            tdc_quotient_early  <= 64'd0;
                            tdc_quotient_late   <= 64'd0;
                            tdc_div_bit         <= 7'd63;
                            state               <= ST_TDC_DIV;
                        end
                    end

                    ST_TDC_DIV: begin
                        logic [64:0] shifted_remainder_early;
                        logic [64:0] shifted_remainder_late;

                        shifted_remainder_early = {
                            tdc_remainder_early[63:0],
                            tdc_dividend_early[tdc_div_bit]
                        };
                        shifted_remainder_late = {
                            tdc_remainder_late[63:0],
                            tdc_dividend_late[tdc_div_bit]
                        };

                        if (shifted_remainder_early >= {1'b0, tdc_divisor_early}) begin
                            tdc_remainder_early <= shifted_remainder_early - {1'b0, tdc_divisor_early};
                            tdc_quotient_early[tdc_div_bit] <= 1'b1;
                        end else begin
                            tdc_remainder_early <= shifted_remainder_early;
                            tdc_quotient_early[tdc_div_bit] <= 1'b0;
                        end

                        if (shifted_remainder_late >= {1'b0, tdc_divisor_late}) begin
                            tdc_remainder_late <= shifted_remainder_late - {1'b0, tdc_divisor_late};
                            tdc_quotient_late[tdc_div_bit] <= 1'b1;
                        end else begin
                            tdc_remainder_late <= shifted_remainder_late;
                            tdc_quotient_late[tdc_div_bit] <= 1'b0;
                        end

                        if (tdc_div_bit == 0) begin
                            state <= ST_TDC_FINISH;
                        end else begin
                            tdc_div_bit <= tdc_div_bit - 1'b1;
                        end
                    end

                    ST_TDC_FINISH: begin
                        logic [63:0] early_arrival;
                        logic [63:0] late_arrival;

                        early_arrival = TDC_TIMING_MIN_PS;
                        early_arrival = early_arrival + tdc_quotient_early;
                        late_arrival = TDC_TIMING_MIN_PS;
                        late_arrival = late_arrival + tdc_quotient_late;
                        tdc_arrival_min_reg <= (early_arrival > TDC_TIMING_MAX_PS)
                                             ? TDC_TIMING_MAX_PS : early_arrival[31:0];
                        tdc_arrival_max_reg <= (late_arrival > TDC_TIMING_MAX_PS)
                                             ? TDC_TIMING_MAX_PS : late_arrival[31:0];
                        tdc_timing_valid_reg <= 1'b1;
                        state <= ST_SEND_PREP;
                    end

                    ST_SEND_PREP: begin
                        send_data  <= result_mem[0];
                        send_valid <= 1'b1;
                        send_last  <= (cols_reg == 8'd1) && (batch_idx == (batch_reg - 8'd1));
                        out_idx    <= '0;
                        state      <= ST_SEND;
                    end

                    ST_SEND: begin
                        if (send_valid && m_axis_tready) begin
                            if (axis_handshake_count < 16'd4) begin
                                axis_debug[axis_handshake_count[1:0]] <= send_data;
                                axis_debug_lastmask[axis_handshake_count[1:0]] <= send_last;
                            end
                            axis_handshake_count <= axis_handshake_count + 16'd1;
                            outputs_sent <= outputs_sent + 32'd1;
                            if (out_idx == (cols_reg - 8'd1)) begin
                                if (batch_idx == (batch_reg - 8'd1)) begin
                                    done_pulse <= 1'b1;
                                    send_valid <= 1'b0;
                                    send_last  <= 1'b0;
                                    state      <= ST_IDLE;
                                end else begin
                                    send_valid <= 1'b0;
                                    send_last  <= 1'b0;
                                    batch_idx  <= batch_idx + 1'b1;
                                    col_base   <= '0;
                                    row_idx    <= '0;
                                    state      <= ST_COMPUTE_INIT;
                                end
                            end else begin
                                out_idx   <= out_idx + 1'b1;
                                send_data <= result_mem[out_idx + 1'b1];
                                send_last <= ((out_idx + 1'b1) == (cols_reg - 8'd1)) &&
                                             (batch_idx == (batch_reg - 8'd1));
                            end
                        end
                    end

                    ST_ERROR: begin
                        if (start_pulse) begin
                            state <= ST_IDLE;
                        end
                    end

                    default: begin
                        state <= ST_IDLE;
                    end
                endcase
            end
        end
    end
endmodule


// -----------------------------------------------------------------------------
// Simple AXI4 writer for result words.
//
// The PE core emits one 32-bit output per AXIS beat. The writer packs four
// outputs into one 128-bit HP-port write so the PS interconnect sees aligned,
// full-width DDR transactions instead of narrow writes.
// -----------------------------------------------------------------------------
module axis_result_axi_writer #(
    parameter integer AXI_ADDR_WIDTH = 32,
    parameter integer AXI_DATA_WIDTH = 128
) (
    input  logic                            clk,
    input  logic                            rstn,
    input  logic                            clear,
    input  logic [AXI_ADDR_WIDTH-1:0]       base_addr,

    input  logic [31:0]                     s_axis_tdata,
    input  logic                            s_axis_tvalid,
    output logic                            s_axis_tready,
    input  logic                            s_axis_tlast,

    output logic                            idle,
    output logic                            busy,
    output logic                            error,
    output logic [1:0]                      last_bresp,
    output logic [31:0]                     write_count,
    output logic [31:0]                     last_data,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "PROTOCOL AXI4, ADDR_WIDTH 32, DATA_WIDTH 128, HAS_BURST 1, HAS_LOCK 1, HAS_CACHE 1, HAS_PROT 1, HAS_QOS 1, HAS_WSTRB 1, HAS_BRESP 1, HAS_RRESP 1, SUPPORTS_NARROW_BURST 0, MAX_BURST_LENGTH 1" *)
    output logic [AXI_ADDR_WIDTH-1:0]       m_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWLEN" *)
    output logic [7:0]                      m_axi_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWSIZE" *)
    output logic [2:0]                      m_axi_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWBURST" *)
    output logic [1:0]                      m_axi_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWLOCK" *)
    output logic                            m_axi_awlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWCACHE" *)
    output logic [3:0]                      m_axi_awcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWPROT" *)
    output logic [2:0]                      m_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWQOS" *)
    output logic [3:0]                      m_axi_awqos,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWVALID" *)
    output logic                            m_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWREADY" *)
    input  logic                            m_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WDATA" *)
    output logic [AXI_DATA_WIDTH-1:0]       m_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WSTRB" *)
    output logic [(AXI_DATA_WIDTH/8)-1:0]   m_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WLAST" *)
    output logic                            m_axi_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WVALID" *)
    output logic                            m_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WREADY" *)
    input  logic                            m_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BRESP" *)
    input  logic [1:0]                      m_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BVALID" *)
    input  logic                            m_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BREADY" *)
    output logic                            m_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARADDR" *)
    output logic [AXI_ADDR_WIDTH-1:0]       m_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARLEN" *)
    output logic [7:0]                      m_axi_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARSIZE" *)
    output logic [2:0]                      m_axi_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARBURST" *)
    output logic [1:0]                      m_axi_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARLOCK" *)
    output logic                            m_axi_arlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARCACHE" *)
    output logic [3:0]                      m_axi_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARPROT" *)
    output logic [2:0]                      m_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARQOS" *)
    output logic [3:0]                      m_axi_arqos,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARVALID" *)
    output logic                            m_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARREADY" *)
    input  logic                            m_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RDATA" *)
    input  logic [AXI_DATA_WIDTH-1:0]       m_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RRESP" *)
    input  logic [1:0]                      m_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RLAST" *)
    input  logic                            m_axi_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RVALID" *)
    input  logic                            m_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RREADY" *)
    output logic                            m_axi_rready
);
    typedef enum logic [1:0] {
        WR_FILL = 2'd0,
        WR_ADDR_DATA = 2'd1,
        WR_RESP = 2'd2
    } wr_state_t;

    wr_state_t state;
    logic aw_done;
    logic w_done;
    logic [AXI_DATA_WIDTH-1:0] pack_data;
    logic [2:0] pack_words;
    logic [2:0] beat_words;
    logic [AXI_DATA_WIDTH-1:0] pack_next_data;
    logic [2:0] pack_next_words;
    logic pack_flush;
    logic axis_fire;

    function automatic [AXI_DATA_WIDTH-1:0] set_pack_lane(
        input [AXI_DATA_WIDTH-1:0] data,
        input [1:0] lane,
        input [31:0] word
    );
        begin
            set_pack_lane = data;
            case (lane)
                2'd0: set_pack_lane[31:0]    = word;
                2'd1: set_pack_lane[63:32]   = word;
                2'd2: set_pack_lane[95:64]   = word;
                default: set_pack_lane[127:96] = word;
            endcase
        end
    endfunction

    assign idle          = (state == WR_FILL) && (pack_words == 3'd0) &&
                           !m_axi_awvalid && !m_axi_wvalid && !m_axi_bready;
    assign busy          = !idle;
    assign s_axis_tready = (state == WR_FILL) && (pack_words < 3'd4);
    assign axis_fire     = s_axis_tvalid && s_axis_tready;

    assign m_axi_awlen   = 8'd0;
    assign m_axi_awsize  = 3'd4;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awlock  = 1'b0;
    assign m_axi_awcache = 4'b0011;
    assign m_axi_awprot  = 3'b000;
    assign m_axi_awqos   = 4'b0000;
    assign m_axi_wlast   = 1'b1;

    assign m_axi_araddr  = '0;
    assign m_axi_arlen   = 8'd0;
    assign m_axi_arsize  = 3'd4;
    assign m_axi_arburst = 2'b01;
    assign m_axi_arlock  = 1'b0;
    assign m_axi_arcache = 4'b0011;
    assign m_axi_arprot  = 3'b000;
    assign m_axi_arqos   = 4'b0000;
    assign m_axi_arvalid = 1'b0;
    assign m_axi_rready  = 1'b1;

    always_comb begin
        pack_next_data  = pack_data;
        pack_next_words = pack_words;
        pack_flush      = 1'b0;
        if (axis_fire) begin
            pack_next_data  = set_pack_lane(pack_data, pack_words[1:0], s_axis_tdata);
            pack_next_words = pack_words + 3'd1;
            pack_flush      = (pack_words == 3'd3) || s_axis_tlast;
        end
    end

    always_ff @(posedge clk) begin
        if (!rstn || clear) begin
            state         <= WR_FILL;
            aw_done       <= 1'b0;
            w_done        <= 1'b0;
            error         <= 1'b0;
            last_bresp    <= 2'b00;
            write_count   <= 32'd0;
            last_data     <= 32'd0;
            pack_data     <= '0;
            pack_words    <= 3'd0;
            beat_words    <= 3'd0;
            m_axi_awaddr  <= '0;
            m_axi_awvalid <= 1'b0;
            m_axi_wdata   <= '0;
            m_axi_wstrb   <= '0;
            m_axi_wvalid  <= 1'b0;
            m_axi_bready  <= 1'b0;
        end else begin
            case (state)
                WR_FILL: begin
                    aw_done       <= 1'b0;
                    w_done        <= 1'b0;
                    m_axi_awvalid <= 1'b0;
                    m_axi_wvalid  <= 1'b0;
                    m_axi_bready  <= 1'b0;
                    if (axis_fire) begin
                        last_data <= s_axis_tdata;
                        if (pack_flush) begin
                            pack_data     <= '0;
                            pack_words    <= 3'd0;
                            beat_words    <= pack_next_words;
                            m_axi_awaddr  <= base_addr + (write_count << 2);
                            m_axi_wdata   <= pack_next_data;
                            m_axi_wstrb   <= {(AXI_DATA_WIDTH/8){1'b1}};
                            m_axi_awvalid <= 1'b1;
                            m_axi_wvalid  <= 1'b1;
                            m_axi_bready  <= 1'b1;
                            state         <= WR_ADDR_DATA;
                        end else begin
                            pack_data  <= pack_next_data;
                            pack_words <= pack_next_words;
                        end
                    end
                end

                WR_ADDR_DATA: begin
                    if (m_axi_awvalid && m_axi_awready) begin
                        m_axi_awvalid <= 1'b0;
                        aw_done       <= 1'b1;
                    end
                    if (m_axi_wvalid && m_axi_wready) begin
                        m_axi_wvalid <= 1'b0;
                        w_done       <= 1'b1;
                    end
                    if ((aw_done || (m_axi_awvalid && m_axi_awready)) &&
                        (w_done  || (m_axi_wvalid  && m_axi_wready))) begin
                        state <= WR_RESP;
                    end
                end

                WR_RESP: begin
                    if (m_axi_bvalid) begin
                        last_bresp   <= m_axi_bresp;
                        error        <= error || (m_axi_bresp != 2'b00);
                        write_count  <= write_count + {29'd0, beat_words};
                        beat_words   <= 3'd0;
                        m_axi_bready <= 1'b0;
                        state        <= WR_FILL;
                    end
                end

                default: begin
                    state <= WR_FILL;
                end
            endcase
        end
    end

    logic unused_writer_inputs;
    assign unused_writer_inputs = m_axi_arready ^ ^m_axi_rdata ^
                                  ^m_axi_rresp ^ m_axi_rlast ^ m_axi_rvalid;
endmodule


// -----------------------------------------------------------------------------
// AXI4 input reader.
//
// Each AXI read fetches lane zero of one 128-bit DDR line and advances the
// physical address by 16 bytes. Software scatters one packed PE word per line.
// This avoids both the faulty AXI DMA burst address path and the HP0 width
// converter's duplicated upper lanes.
// -----------------------------------------------------------------------------
module axi_input_word_reader #(
    parameter integer AXI_ADDR_WIDTH = 32,
    parameter integer AXI_DATA_WIDTH = 128
) (
    input  logic                            clk,
    input  logic                            rstn,
    input  logic                            clear,
    input  logic                            start,
    input  logic [AXI_ADDR_WIDTH-1:0]       base_addr,
    input  logic [31:0]                     word_count,

    output logic [31:0]                     m_axis_tdata,
    output logic                            m_axis_tvalid,
    input  logic                            m_axis_tready,
    output logic                            m_axis_tlast,

    output logic                            idle,
    output logic                            busy,
    output logic                            error,
    output logic [1:0]                      last_rresp,
    output logic [31:0]                     read_count,
    output logic [31:0]                     beat_count,

    output logic [AXI_ADDR_WIDTH-1:0]       m_axi_araddr,
    output logic [7:0]                      m_axi_arlen,
    output logic [2:0]                      m_axi_arsize,
    output logic [1:0]                      m_axi_arburst,
    output logic                            m_axi_arlock,
    output logic [3:0]                      m_axi_arcache,
    output logic [2:0]                      m_axi_arprot,
    output logic [3:0]                      m_axi_arqos,
    output logic                            m_axi_arvalid,
    input  logic                            m_axi_arready,
    input  logic [AXI_DATA_WIDTH-1:0]       m_axi_rdata,
    input  logic [1:0]                      m_axi_rresp,
    input  logic                            m_axi_rlast,
    input  logic                            m_axi_rvalid,
    output logic                            m_axi_rready
);
    typedef enum logic [1:0] {
        RD_IDLE = 2'd0,
        RD_ADDR = 2'd1,
        RD_DATA = 2'd2,
        RD_OUT  = 2'd3
    } rd_state_t;

    rd_state_t state;
    logic [AXI_ADDR_WIDTH-1:0] current_addr;
    logic [AXI_DATA_WIDTH-1:0] read_data;
    logic [31:0] words_remaining;

    assign idle          = (state == RD_IDLE);
    assign busy          = !idle;
    assign m_axi_araddr  = current_addr;
    assign m_axi_arlen   = 8'd0;
    // The HP port and SmartConnect data path are 128 bits wide. Fetch one
    // complete 16-byte line and forward lane zero to the packed-word stream.
    // A 4-byte ARSIZE can pass a permissive AXI model but stalls after the
    // first transaction on the physical ZCU106 upsizer.
    assign m_axi_arsize  = 3'd4;
    assign m_axi_arburst = 2'b01;
    assign m_axi_arlock  = 1'b0;
    assign m_axi_arcache = 4'b0011;
    assign m_axi_arprot  = 3'b000;
    assign m_axi_arqos   = 4'b0000;
    assign m_axi_arvalid = (state == RD_ADDR);
    // Keep the return channel draining while the reader is active. The ZynqMP
    // HP0 path can retain read credits across the RD_OUT gap; withdrawing
    // RREADY there can leave a later single-beat transaction permanently
    // outstanding in SmartConnect.
    assign m_axi_rready  = (state != RD_IDLE);
    assign m_axis_tvalid = (state == RD_OUT);
    assign m_axis_tlast  = (state == RD_OUT) && (words_remaining == 32'd1);

    assign m_axis_tdata = read_data[31:0];

    always_ff @(posedge clk) begin
        if (!rstn || clear) begin
            state           <= RD_IDLE;
            current_addr    <= '0;
            read_data       <= '0;
            words_remaining <= 32'd0;
            error           <= 1'b0;
            last_rresp      <= 2'b00;
            read_count      <= 32'd0;
            beat_count      <= 32'd0;
        end else if (start) begin
            current_addr    <= base_addr;
            words_remaining <= word_count;
            error           <= (word_count == 32'd0) || (base_addr[3:0] != 4'd0);
            last_rresp      <= 2'b00;
            read_count      <= 32'd0;
            beat_count      <= 32'd0;
            state           <= ((word_count == 32'd0) || (base_addr[3:0] != 4'd0)) ? RD_IDLE : RD_ADDR;
        end else begin
            case (state)
                RD_IDLE: begin
                    // Wait for the next MMIO start pulse.
                end

                RD_ADDR: begin
                    if (m_axi_arready) begin
                        // AXI permits a zero-latency slave to return the first
                        // beat in the address-handshake cycle.
                        if (m_axi_rvalid) begin
                            read_data  <= m_axi_rdata;
                            last_rresp <= m_axi_rresp;
                            beat_count <= beat_count + 32'd1;
                            if ((m_axi_rresp != 2'b00) || !m_axi_rlast) begin
                                error <= 1'b1;
                                state <= RD_IDLE;
                            end else begin
                                state <= RD_OUT;
                            end
                        end else begin
                            state <= RD_DATA;
                        end
                    end
                end

                RD_DATA: begin
                    if (m_axi_rvalid) begin
                        read_data  <= m_axi_rdata;
                        last_rresp <= m_axi_rresp;
                        beat_count <= beat_count + 32'd1;
                        if ((m_axi_rresp != 2'b00) || !m_axi_rlast) begin
                            error <= 1'b1;
                            state <= RD_IDLE;
                        end else begin
                            state <= RD_OUT;
                        end
                    end
                end

                RD_OUT: begin
                    if (m_axis_tready) begin
                        read_count <= read_count + 32'd1;
                        if (words_remaining == 32'd1) begin
                            words_remaining <= 32'd0;
                            state <= RD_IDLE;
                        end else begin
                            words_remaining <= words_remaining - 32'd1;
                            current_addr <= current_addr + 32'd16;
                            state        <= RD_ADDR;
                        end
                    end
                end

                default: state <= RD_IDLE;
            endcase
        end
    end
endmodule


// -----------------------------------------------------------------------------
// Top-level AXI-Lite + AXI-Stream accelerator.
// -----------------------------------------------------------------------------
module axi_x2_mvm_accel #(
    parameter integer C_S_AXI_ADDR_WIDTH = 8,
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer DATA_WIDTH         = 32,
    parameter integer MAX_ROWS           = 32,
    parameter integer MAX_COLS           = 32,
    parameter integer MAX_BATCH          = 32,
    parameter integer N_PE               = 8,
    parameter integer TDC_TIMING_ENABLE           = 1,
    parameter integer TDC_TIMING_MIN_PS           = 0,
    parameter integer TDC_TIMING_MAX_PS           = 576,
    parameter integer TDC_TIMING_REFERENCE_LEVELS = 256
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:S_AXIS:M_AXIS:M_AXI, ASSOCIATED_RESET aresetn, FREQ_HZ 99990005" *)
    input  logic                              aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  logic                              aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, ADDR_WIDTH 8, DATA_WIDTH 32" *)
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *)
    input  logic [2:0]                        s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  logic                              s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output logic                              s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  logic [C_S_AXI_DATA_WIDTH-1:0]     s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  logic [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  logic                              s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output logic                              s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output logic [1:0]                        s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output logic                              s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  logic                              s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  logic [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *)
    input  logic [2:0]                        s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  logic                              s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output logic                              s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output logic [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output logic [1:0]                        s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output logic                              s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  logic                              s_axi_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *)
    input  logic [DATA_WIDTH-1:0]             s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *)
    input  logic                              s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *)
    output logic                              s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *)
    input  logic                              s_axis_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output logic [DATA_WIDTH-1:0]             m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output logic [(DATA_WIDTH/8)-1:0]         m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output logic                              m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input  logic                              m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output logic                              m_axis_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "PROTOCOL AXI4, ADDR_WIDTH 32, DATA_WIDTH 128, HAS_BURST 1, HAS_LOCK 1, HAS_CACHE 1, HAS_PROT 1, HAS_QOS 1, HAS_WSTRB 1, HAS_BRESP 1, HAS_RRESP 1, SUPPORTS_NARROW_BURST 1, MAX_BURST_LENGTH 1" *)
    output logic [31:0]                       m_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWLEN" *)
    output logic [7:0]                        m_axi_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWSIZE" *)
    output logic [2:0]                        m_axi_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWBURST" *)
    output logic [1:0]                        m_axi_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWLOCK" *)
    output logic                              m_axi_awlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWCACHE" *)
    output logic [3:0]                        m_axi_awcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWPROT" *)
    output logic [2:0]                        m_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWQOS" *)
    output logic [3:0]                        m_axi_awqos,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWVALID" *)
    output logic                              m_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWREADY" *)
    input  logic                              m_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WDATA" *)
    output logic [127:0]                      m_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WSTRB" *)
    output logic [15:0]                       m_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WLAST" *)
    output logic                              m_axi_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WVALID" *)
    output logic                              m_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WREADY" *)
    input  logic                              m_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BRESP" *)
    input  logic [1:0]                        m_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BVALID" *)
    input  logic                              m_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BREADY" *)
    output logic                              m_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARADDR" *)
    output logic [31:0]                       m_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARLEN" *)
    output logic [7:0]                        m_axi_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARSIZE" *)
    output logic [2:0]                        m_axi_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARBURST" *)
    output logic [1:0]                        m_axi_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARLOCK" *)
    output logic                              m_axi_arlock,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARCACHE" *)
    output logic [3:0]                        m_axi_arcache,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARPROT" *)
    output logic [2:0]                        m_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARQOS" *)
    output logic [3:0]                        m_axi_arqos,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARVALID" *)
    output logic                              m_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARREADY" *)
    input  logic                              m_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RDATA" *)
    input  logic [127:0]                      m_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RRESP" *)
    input  logic [1:0]                        m_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RLAST" *)
    input  logic                              m_axi_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RVALID" *)
    input  logic                              m_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RREADY" *)
    output logic                              m_axi_rready,

    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 irq INTERRUPT" *)
    (* X_INTERFACE_PARAMETER = "SENSITIVITY LEVEL_HIGH" *)
    output logic                              irq
);
    localparam [31:0] VERSION     = 32'h0002_0025;
    localparam [31:0] MAGIC       = 32'h5832_4D56; // "X2MV"
    localparam [31:0] DEFAULT_CFG = (32'h0001_0000) | ((MAX_COLS & 32'hFF) << 8) | (MAX_ROWS & 32'hFF);

    localparam [7:0] REG_CTRL     = 8'h00;
    localparam [7:0] REG_STATUS   = 8'h04;
    localparam [7:0] REG_CONFIG   = 8'h08;
    localparam [7:0] REG_EXPECTED = 8'h0C;
    localparam [7:0] REG_RX_WORDS = 8'h10;
    localparam [7:0] REG_TX_WORDS = 8'h14;
    localparam [7:0] REG_CYCLES   = 8'h18;
    localparam [7:0] REG_ERROR    = 8'h1C;
    localparam [7:0] REG_MAGIC    = 8'h20;
    localparam [7:0] REG_VERSION  = 8'h24;
    localparam [7:0] REG_DBG_SHAPE   = 8'h28;
    localparam [7:0] REG_DBG_STREAM0 = 8'h2C;
    localparam [7:0] REG_DBG_STREAM1 = 8'h30;
    localparam [7:0] REG_DBG_STREAM2 = 8'h34;
    localparam [7:0] REG_DBG_STREAM3 = 8'h38;
    localparam [7:0] REG_DBG_WEIGHT0 = 8'h3C;
    localparam [7:0] REG_DBG_WEIGHT1 = 8'h40;
    localparam [7:0] REG_DBG_VECTOR0 = 8'h44;
    localparam [7:0] REG_DBG_VECTOR1 = 8'h48;
    localparam [7:0] REG_DBG_RESULT0 = 8'h4C;
    localparam [7:0] REG_DBG_RESULT1 = 8'h50;
    localparam [7:0] REG_DBG_RESULT2 = 8'h54;
    localparam [7:0] REG_DBG_RESULT3 = 8'h58;
    localparam [7:0] REG_DBG_AXIS0   = 8'h5C;
    localparam [7:0] REG_DBG_AXIS1   = 8'h60;
    localparam [7:0] REG_DBG_AXIS2   = 8'h64;
    localparam [7:0] REG_DBG_AXIS3   = 8'h68;
    localparam [7:0] REG_DBG_AXIS_META = 8'h6C;
    localparam [7:0] REG_WR_ADDR_LO  = 8'h70;
    localparam [7:0] REG_WR_ADDR_HI  = 8'h74;
    localparam [7:0] REG_WR_COUNT    = 8'h78;
    localparam [7:0] REG_WR_STATUS   = 8'h7C;
    localparam [7:0] REG_RD_ADDR_LO  = 8'h80;
    localparam [7:0] REG_RD_ADDR_HI  = 8'h84;
    localparam [7:0] REG_RD_COUNT    = 8'h88;
    localparam [7:0] REG_RD_STATUS   = 8'h8C;
    localparam [7:0] REG_TDC_ORDER       = 8'h90;
    localparam [7:0] REG_TDC_ARRIVAL_MIN = 8'h94;
    localparam [7:0] REG_TDC_ARRIVAL_MAX = 8'h98;
    localparam [7:0] REG_TDC_ARRIVAL_SPREAD = 8'h9C;
    localparam [7:0] REG_TDC_PE_COMPLETION  = 8'hA0;
    localparam [7:0] REG_TDC_CONFIG       = 8'hA4;
    localparam [7:0] REG_TDC_TIMING_MIN   = 8'hA8;
    localparam [7:0] REG_TDC_TIMING_MAX   = 8'hAC;

    logic [31:0] cfg_reg;
    logic [31:0] writer_addr_lo;
    logic [31:0] writer_addr_hi;
    logic [31:0] reader_addr_lo;
    logic [31:0] reader_addr_hi;
    logic        irq_enable;
    logic        done_sticky;
    logic        error_sticky;

    logic        start_pulse;
    logic        abort_pulse;
    logic        clear_pulse;
    logic        core_idle;
    logic        core_busy;
    logic [7:0]  core_state_flags;
    logic        core_done_pulse;
    logic        core_error_pulse;
    logic [7:0]  core_error_code;
    logic [31:0] expected_input_words;
    logic [31:0] words_received;
    logic [31:0] outputs_sent;
    logic [31:0] cycle_count;
    logic        tdc_timing_valid;
    logic [7:0]  tdc_first_ready_column;
    logic [7:0]  tdc_last_ready_column;
    logic [31:0] tdc_arrival_min_ps;
    logic [31:0] tdc_arrival_max_ps;
    logic [31:0] tdc_arrival_spread_ps;
    logic [31:0] tdc_pe_completion_time_ps;
    logic [31:0] debug_shape;
    logic [31:0] debug_stream0;
    logic [31:0] debug_stream1;
    logic [31:0] debug_stream2;
    logic [31:0] debug_stream3;
    logic [31:0] debug_weight0;
    logic [31:0] debug_weight1;
    logic [31:0] debug_vector0;
    logic [31:0] debug_vector1;
    logic [31:0] debug_result0;
    logic [31:0] debug_result1;
    logic [31:0] debug_result2;
    logic [31:0] debug_result3;
    logic [31:0] debug_axis0;
    logic [31:0] debug_axis1;
    logic [31:0] debug_axis2;
    logic [31:0] debug_axis3;
    logic [31:0] debug_axis_meta;
    logic [31:0] core_m_axis_tdata;
    logic        core_m_axis_tvalid;
    logic        core_m_axis_tready;
    logic        core_m_axis_tlast;
    logic        writer_idle;
    logic        writer_busy;
    logic        writer_error;
    logic [1:0]  writer_bresp;
    logic [31:0] writer_count;
    logic [31:0] writer_last_data;
    logic [31:0] reader_axis_tdata;
    logic        reader_axis_tvalid;
    logic        reader_axis_tready;
    logic        reader_axis_tlast;
    logic        reader_idle;
    logic        reader_busy;
    logic        reader_error;
    logic [1:0]  reader_rresp;
    logic [31:0] reader_count;
    logic [31:0] reader_beats;

    function automatic [31:0] apply_wstrb(
        input [31:0] old_value,
        input [31:0] new_value,
        input [3:0]  wstrb
    );
        apply_wstrb = old_value;
        for (int i = 0; i < 4; i = i + 1) begin
            if (wstrb[i]) begin
                apply_wstrb[8*i +: 8] = new_value[8*i +: 8];
            end
        end
    endfunction

    function automatic [31:0] read_mux(input [7:0] addr);
        begin
            case (addr)
                REG_CTRL:     read_mux = {29'd0, irq_enable, 2'd0};
                REG_STATUS:   read_mux = {16'd0, core_state_flags, 5'd0, error_sticky, done_sticky, core_idle};
                REG_CONFIG:   read_mux = cfg_reg;
                REG_EXPECTED: read_mux = expected_input_words;
                REG_RX_WORDS: read_mux = words_received;
                REG_TX_WORDS: read_mux = outputs_sent;
                REG_CYCLES:   read_mux = cycle_count;
                REG_ERROR:    read_mux = {24'd0, core_error_code};
                REG_MAGIC:    read_mux = MAGIC;
                REG_VERSION:  read_mux = VERSION;
                REG_DBG_SHAPE:   read_mux = debug_shape;
                REG_DBG_STREAM0: read_mux = debug_stream0;
                REG_DBG_STREAM1: read_mux = debug_stream1;
                REG_DBG_STREAM2: read_mux = debug_stream2;
                REG_DBG_STREAM3: read_mux = debug_stream3;
                REG_DBG_WEIGHT0: read_mux = debug_weight0;
                REG_DBG_WEIGHT1: read_mux = debug_weight1;
                REG_DBG_VECTOR0: read_mux = debug_vector0;
                REG_DBG_VECTOR1: read_mux = debug_vector1;
                REG_DBG_RESULT0: read_mux = debug_result0;
                REG_DBG_RESULT1: read_mux = debug_result1;
                REG_DBG_RESULT2: read_mux = debug_result2;
                REG_DBG_RESULT3: read_mux = debug_result3;
                REG_DBG_AXIS0:   read_mux = debug_axis0;
                REG_DBG_AXIS1:   read_mux = debug_axis1;
                REG_DBG_AXIS2:   read_mux = debug_axis2;
                REG_DBG_AXIS3:   read_mux = debug_axis3;
                REG_DBG_AXIS_META: read_mux = debug_axis_meta;
                REG_WR_ADDR_LO:  read_mux = writer_addr_lo;
                REG_WR_ADDR_HI:  read_mux = writer_addr_hi;
                REG_WR_COUNT:    read_mux = writer_count;
                REG_WR_STATUS:   read_mux = {writer_last_data[15:0], 3'd0, writer_bresp,
                                             writer_error, writer_busy, writer_idle,
                                             writer_count[7:0]};
                REG_RD_ADDR_LO:  read_mux = reader_addr_lo;
                REG_RD_ADDR_HI:  read_mux = reader_addr_hi;
                REG_RD_COUNT:    read_mux = reader_count;
                REG_RD_STATUS:   read_mux = {reader_beats[15:0], 10'd0, reader_rresp,
                                             reader_error, reader_busy, reader_idle,
                                             reader_count[0]};
                REG_TDC_ORDER:       read_mux = {tdc_timing_valid, 15'd0,
                                                 tdc_last_ready_column, tdc_first_ready_column};
                REG_TDC_ARRIVAL_MIN: read_mux = tdc_arrival_min_ps;
                REG_TDC_ARRIVAL_MAX: read_mux = tdc_arrival_max_ps;
                REG_TDC_ARRIVAL_SPREAD: read_mux = tdc_arrival_spread_ps;
                REG_TDC_PE_COMPLETION:  read_mux = tdc_pe_completion_time_ps;
                REG_TDC_CONFIG:      read_mux = {
                    TDC_TIMING_ENABLE[0],
                    15'd0,
                    TDC_TIMING_REFERENCE_LEVELS[15:0]
                };
                REG_TDC_TIMING_MIN:  read_mux = TDC_TIMING_MIN_PS;
                REG_TDC_TIMING_MAX:  read_mux = TDC_TIMING_MAX_PS;
                default:      read_mux = 32'd0;
            endcase
        end
    endfunction

    x2_mvm_axis_core #(
        .MAX_ROWS   (MAX_ROWS),
        .MAX_COLS   (MAX_COLS),
        .MAX_BATCH  (MAX_BATCH),
        .N_PE       (N_PE),
        .DATA_WIDTH (DATA_WIDTH),
        .TDC_TIMING_ENABLE           (TDC_TIMING_ENABLE),
        .TDC_TIMING_MIN_PS           (TDC_TIMING_MIN_PS),
        .TDC_TIMING_MAX_PS           (TDC_TIMING_MAX_PS),
        .TDC_TIMING_REFERENCE_LEVELS (TDC_TIMING_REFERENCE_LEVELS)
    ) u_core (
        .clk                  (aclk),
        .rstn                 (aresetn),
        .start_pulse          (start_pulse),
        .abort_pulse          (abort_pulse),
        .cfg_rows             (cfg_reg[7:0]),
        .cfg_cols             (cfg_reg[15:8]),
        .cfg_batch            (cfg_reg[23:16]),
        .idle                 (core_idle),
        .busy                 (core_busy),
        .state_flags          (core_state_flags),
        .done_pulse           (core_done_pulse),
        .error_pulse          (core_error_pulse),
        .error_code           (core_error_code),
        .expected_input_words (expected_input_words),
        .words_received       (words_received),
        .outputs_sent         (outputs_sent),
        .cycle_count          (cycle_count),
        .tdc_timing_valid     (tdc_timing_valid),
        .tdc_first_ready_column(tdc_first_ready_column),
        .tdc_last_ready_column(tdc_last_ready_column),
        .tdc_arrival_min_ps   (tdc_arrival_min_ps),
        .tdc_arrival_max_ps   (tdc_arrival_max_ps),
        .tdc_arrival_spread_ps(tdc_arrival_spread_ps),
        .tdc_pe_completion_time_ps(tdc_pe_completion_time_ps),
        .debug_shape          (debug_shape),
        .debug_stream0        (debug_stream0),
        .debug_stream1        (debug_stream1),
        .debug_stream2        (debug_stream2),
        .debug_stream3        (debug_stream3),
        .debug_weight0        (debug_weight0),
        .debug_weight1        (debug_weight1),
        .debug_vector0        (debug_vector0),
        .debug_vector1        (debug_vector1),
        .debug_result0        (debug_result0),
        .debug_result1        (debug_result1),
        .debug_result2        (debug_result2),
        .debug_result3        (debug_result3),
        .debug_axis0          (debug_axis0),
        .debug_axis1          (debug_axis1),
        .debug_axis2          (debug_axis2),
        .debug_axis3          (debug_axis3),
        .debug_axis_meta      (debug_axis_meta),
        .s_axis_tdata         (reader_axis_tdata),
        .s_axis_tvalid        (reader_axis_tvalid),
        .s_axis_tready        (reader_axis_tready),
        .s_axis_tlast         (reader_axis_tlast),
        .m_axis_tdata         (core_m_axis_tdata),
        .m_axis_tvalid        (core_m_axis_tvalid),
        .m_axis_tready        (core_m_axis_tready),
        .m_axis_tlast         (core_m_axis_tlast)
    );

    axi_input_word_reader u_input_reader (
        .clk            (aclk),
        .rstn           (aresetn),
        .clear          (abort_pulse || clear_pulse),
        .start          (start_pulse),
        .base_addr      (reader_addr_lo),
        .word_count     (expected_input_words),
        .m_axis_tdata   (reader_axis_tdata),
        .m_axis_tvalid  (reader_axis_tvalid),
        .m_axis_tready  (reader_axis_tready),
        .m_axis_tlast   (reader_axis_tlast),
        .idle           (reader_idle),
        .busy           (reader_busy),
        .error          (reader_error),
        .last_rresp     (reader_rresp),
        .read_count     (reader_count),
        .beat_count     (reader_beats),
        .m_axi_araddr   (m_axi_araddr),
        .m_axi_arlen    (m_axi_arlen),
        .m_axi_arsize   (m_axi_arsize),
        .m_axi_arburst  (m_axi_arburst),
        .m_axi_arlock   (m_axi_arlock),
        .m_axi_arcache  (m_axi_arcache),
        .m_axi_arprot   (m_axi_arprot),
        .m_axi_arqos    (m_axi_arqos),
        .m_axi_arvalid  (m_axi_arvalid),
        .m_axi_arready  (m_axi_arready),
        .m_axi_rdata    (m_axi_rdata),
        .m_axi_rresp    (m_axi_rresp),
        .m_axi_rlast    (m_axi_rlast),
        .m_axi_rvalid   (m_axi_rvalid),
        .m_axi_rready   (m_axi_rready)
    );

    axis_result_axi_writer u_result_writer (
        .clk            (aclk),
        .rstn           (aresetn),
        .clear          (start_pulse || abort_pulse || clear_pulse),
        .base_addr      (writer_addr_lo),
        .s_axis_tdata   (core_m_axis_tdata),
        .s_axis_tvalid  (core_m_axis_tvalid),
        .s_axis_tready  (core_m_axis_tready),
        .s_axis_tlast   (core_m_axis_tlast),
        .idle           (writer_idle),
        .busy           (writer_busy),
        .error          (writer_error),
        .last_bresp     (writer_bresp),
        .write_count    (writer_count),
        .last_data      (writer_last_data),
        .m_axi_awaddr   (m_axi_awaddr),
        .m_axi_awlen    (m_axi_awlen),
        .m_axi_awsize   (m_axi_awsize),
        .m_axi_awburst  (m_axi_awburst),
        .m_axi_awlock   (m_axi_awlock),
        .m_axi_awcache  (m_axi_awcache),
        .m_axi_awprot   (m_axi_awprot),
        .m_axi_awqos    (m_axi_awqos),
        .m_axi_awvalid  (m_axi_awvalid),
        .m_axi_awready  (m_axi_awready),
        .m_axi_wdata    (m_axi_wdata),
        .m_axi_wstrb    (m_axi_wstrb),
        .m_axi_wlast    (m_axi_wlast),
        .m_axi_wvalid   (m_axi_wvalid),
        .m_axi_wready   (m_axi_wready),
        .m_axi_bresp    (m_axi_bresp),
        .m_axi_bvalid   (m_axi_bvalid),
        .m_axi_bready   (m_axi_bready)
    );

    assign irq = irq_enable && (done_sticky || error_sticky);
    assign m_axis_tdata  = 32'd0;
    assign m_axis_tkeep = {(DATA_WIDTH/8){1'b1}};
    assign m_axis_tvalid = 1'b0;
    assign m_axis_tlast  = 1'b0;
    assign s_axis_tready = 1'b0;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            s_axi_bresp   <= 2'b00;
            s_axi_bvalid  <= 1'b0;
            s_axi_arready <= 1'b0;
            s_axi_rdata   <= 32'd0;
            s_axi_rresp   <= 2'b00;
            s_axi_rvalid  <= 1'b0;

            cfg_reg       <= DEFAULT_CFG;
            writer_addr_lo <= 32'd0;
            writer_addr_hi <= 32'd0;
            reader_addr_lo <= 32'd0;
            reader_addr_hi <= 32'd0;
            irq_enable    <= 1'b0;
            done_sticky   <= 1'b0;
            error_sticky  <= 1'b0;
            start_pulse   <= 1'b0;
            abort_pulse   <= 1'b0;
            clear_pulse   <= 1'b0;
        end else begin
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            s_axi_arready <= 1'b0;
            start_pulse   <= 1'b0;
            abort_pulse   <= 1'b0;
            clear_pulse   <= 1'b0;

            if (core_done_pulse) begin
                done_sticky <= 1'b1;
            end
            if (core_error_pulse) begin
                error_sticky <= 1'b1;
            end
            if (writer_error) begin
                error_sticky <= 1'b1;
            end
            if (reader_error) begin
                error_sticky <= 1'b1;
            end

            if (!s_axi_bvalid && s_axi_awvalid && s_axi_wvalid) begin
                s_axi_awready <= 1'b1;
                s_axi_wready  <= 1'b1;
                s_axi_bvalid  <= 1'b1;
                s_axi_bresp   <= 2'b00;

                case (s_axi_awaddr[7:0])
                    REG_CTRL: begin
                        if (s_axi_wdata[0]) begin
                            start_pulse  <= 1'b1;
                            done_sticky  <= 1'b0;
                            error_sticky <= 1'b0;
                        end
                        if (s_axi_wdata[1]) begin
                            abort_pulse  <= 1'b1;
                            done_sticky  <= 1'b0;
                            error_sticky <= 1'b0;
                        end
                        if (s_axi_wstrb[0]) begin
                            irq_enable <= s_axi_wdata[2];
                        end
                        if (s_axi_wdata[8]) begin
                            clear_pulse  <= 1'b1;
                            done_sticky  <= 1'b0;
                            error_sticky <= 1'b0;
                        end
                    end

                    REG_CONFIG: begin
                        cfg_reg <= apply_wstrb(cfg_reg, s_axi_wdata, s_axi_wstrb);
                    end

                    REG_WR_ADDR_LO: begin
                        writer_addr_lo <= apply_wstrb(writer_addr_lo, s_axi_wdata, s_axi_wstrb);
                    end

                    REG_WR_ADDR_HI: begin
                        writer_addr_hi <= apply_wstrb(writer_addr_hi, s_axi_wdata, s_axi_wstrb);
                    end

                    REG_RD_ADDR_LO: begin
                        reader_addr_lo <= apply_wstrb(reader_addr_lo, s_axi_wdata, s_axi_wstrb);
                    end

                    REG_RD_ADDR_HI: begin
                        reader_addr_hi <= apply_wstrb(reader_addr_hi, s_axi_wdata, s_axi_wstrb);
                    end

                    default: begin
                        // Read-only or reserved registers ignore writes.
                    end
                endcase
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (!s_axi_rvalid && s_axi_arvalid) begin
                s_axi_arready <= 1'b1;
                s_axi_rvalid  <= 1'b1;
                s_axi_rresp   <= 2'b00;
                s_axi_rdata   <= read_mux(s_axi_araddr[7:0]);
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    // Tie off unused AXI protection inputs while keeping the top-level AXI shape.
    logic unused_axi_prot;
    assign unused_axi_prot = ^s_axi_awprot ^ ^s_axi_arprot ^ core_busy ^
                             m_axis_tready ^ ^writer_addr_hi ^ ^reader_addr_hi ^
                             ^s_axis_tdata ^ s_axis_tvalid ^ s_axis_tlast;
endmodule
