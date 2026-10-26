`timescale 1ns / 1ps

module tb_axi_x2_mvm_core;
    localparam int MAX_ROWS  = 32;
    localparam int MAX_COLS  = 32;
    localparam int MAX_BATCH = 32;

    logic clk = 1'b0;
    logic rstn = 1'b0;

    logic start_pulse = 1'b0;
    logic abort_pulse = 1'b0;
    logic [7:0] cfg_rows = 8'd3;
    logic [7:0] cfg_cols = 8'd2;
    logic [7:0] cfg_batch = 8'd1;

    logic idle;
    logic busy;
    logic [7:0] state_flags;
    logic done_pulse;
    logic error_pulse;
    logic [7:0] error_code;
    logic [31:0] expected_input_words;
    logic [31:0] words_received;
    logic [31:0] outputs_sent;
    logic [31:0] cycle_count;
    logic tdc_timing_valid;
    logic [7:0] tdc_first_ready_column;
    logic [7:0] tdc_last_ready_column;
    logic [31:0] tdc_arrival_min_ps;
    logic [31:0] tdc_arrival_max_ps;
    logic [31:0] tdc_arrival_spread_ps;
    logic [31:0] tdc_pe_completion_time_ps;

    logic [31:0] s_axis_tdata = 32'd0;
    logic s_axis_tvalid = 1'b0;
    logic s_axis_tready;
    logic s_axis_tlast = 1'b0;

    logic [31:0] m_axis_tdata;
    logic m_axis_tvalid;
    logic m_axis_tready = 1'b1;
    logic m_axis_tlast;
    logic output_backpressure = 1'b0;

    logic signed [31:0] observed [0:MAX_BATCH*MAX_COLS-1];
    int observed_count = 0;
    int tlast_count = 0;

    logic signed [7:0] vectors [0:MAX_BATCH-1][0:MAX_ROWS-1];
    logic signed [7:0] weights [0:MAX_ROWS-1][0:MAX_COLS-1];

    always #5 clk = ~clk;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            m_axis_tready <= 1'b1;
        end else if (output_backpressure) begin
            m_axis_tready <= !m_axis_tready;
        end else begin
            m_axis_tready <= 1'b1;
        end
    end

    x2_mvm_axis_core #(
        .MAX_ROWS(MAX_ROWS),
        .MAX_COLS(MAX_COLS),
        .MAX_BATCH(MAX_BATCH),
        .N_PE(8),
        .DATA_WIDTH(32)
    ) dut (
        .clk(clk),
        .rstn(rstn),
        .start_pulse(start_pulse),
        .abort_pulse(abort_pulse),
        .cfg_rows(cfg_rows),
        .cfg_cols(cfg_cols),
        .cfg_batch(cfg_batch),
        .idle(idle),
        .busy(busy),
        .state_flags(state_flags),
        .done_pulse(done_pulse),
        .error_pulse(error_pulse),
        .error_code(error_code),
        .expected_input_words(expected_input_words),
        .words_received(words_received),
        .outputs_sent(outputs_sent),
        .cycle_count(cycle_count),
        .tdc_timing_valid(tdc_timing_valid),
        .tdc_first_ready_column(tdc_first_ready_column),
        .tdc_last_ready_column(tdc_last_ready_column),
        .tdc_arrival_min_ps(tdc_arrival_min_ps),
        .tdc_arrival_max_ps(tdc_arrival_max_ps),
        .tdc_arrival_spread_ps(tdc_arrival_spread_ps),
        .tdc_pe_completion_time_ps(tdc_pe_completion_time_ps),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast)
    );

    task automatic send_word(input logic [31:0] data, input logic last);
        begin
            s_axis_tdata  <= data;
            s_axis_tlast  <= last;
            s_axis_tvalid <= 1'b1;
            do begin
                @(posedge clk);
            end while (!s_axis_tready);
            s_axis_tvalid <= 1'b0;
            s_axis_tlast  <= 1'b0;
            s_axis_tdata  <= 32'd0;
            @(posedge clk);
        end
    endtask

    task automatic send_packet_burst(input int rows, input int cols, input int batch);
        logic [31:0] words [0:2047];
        int weight_words;
        int vector_words;
        int total_words;
        int idx;
        begin
            weight_words = ((rows * cols) + 3) / 4;
            vector_words = (rows + 3) / 4;
            total_words = weight_words + (batch * vector_words);

            for (int w = 0; w < weight_words; w++) begin
                words[w] = pack_weight_word(w, rows, cols);
            end
            idx = weight_words;
            for (int b = 0; b < batch; b++) begin
                for (int w = 0; w < vector_words; w++) begin
                    words[idx] = pack_vector_word(b, w, rows);
                    idx++;
                end
            end

            idx = 0;
            s_axis_tvalid <= 1'b1;
            s_axis_tdata  <= words[0];
            s_axis_tlast  <= (total_words == 1);
            while (idx < total_words) begin
                @(posedge clk);
                if (s_axis_tready) begin
                    idx++;
                    if (idx < total_words) begin
                        s_axis_tdata <= words[idx];
                        s_axis_tlast <= (idx == total_words - 1);
                    end else begin
                        s_axis_tvalid <= 1'b0;
                        s_axis_tlast  <= 1'b0;
                        s_axis_tdata  <= 32'd0;
                    end
                end
            end
            @(posedge clk);
        end
    endtask

    function automatic logic [31:0] pack_weight_word(input int word_idx, input int rows, input int cols);
        logic [31:0] word;
        int byte_idx;
        int row;
        int col;
        begin
            word = 32'd0;
            for (int b = 0; b < 4; b++) begin
                byte_idx = word_idx * 4 + b;
                if (byte_idx < rows * cols) begin
                    row = byte_idx / cols;
                    col = byte_idx % cols;
                    word[8*b +: 8] = weights[row][col];
                end
            end
            return word;
        end
    endfunction

    function automatic logic [31:0] pack_vector_word(input int batch_idx, input int word_idx, input int rows);
        logic [31:0] word;
        int byte_idx;
        begin
            word = 32'd0;
            for (int b = 0; b < 4; b++) begin
                byte_idx = word_idx * 4 + b;
                if (byte_idx < rows) begin
                    word[8*b +: 8] = vectors[batch_idx][byte_idx];
                end
            end
            return word;
        end
    endfunction

    function automatic int reference_pwm_int8_row_serial_product(
        input logic signed [7:0] activation_i,
        input logic signed [7:0] weight_i
    );
        logic [7:0] activation_drive_u8;
        int weight_mag;
        int a0;
        int a1;
        int pos0;
        int pos1;
        int neg0;
        int neg1;
        int raw_pwm_sum;
        begin
            activation_drive_u8 = activation_i ^ 8'h80;
            a0 = activation_drive_u8[3:0];
            a1 = activation_drive_u8[7:4];

            weight_mag = weight_i[7] ? -int'(weight_i) : int'(weight_i);
            pos0 = weight_i[7] ? 0 : (weight_mag & 15);
            pos1 = weight_i[7] ? 0 : ((weight_mag >> 4) & 15);
            neg0 = weight_i[7] ? (weight_mag & 15) : 0;
            neg1 = weight_i[7] ? ((weight_mag >> 4) & 15) : 0;

            raw_pwm_sum = (a0 * pos0 - a0 * neg0)
                        + ((a0 * pos1 - a0 * neg1) <<< 4)
                        + ((a1 * pos0 - a1 * neg0) <<< 4)
                        + ((a1 * pos1 - a1 * neg1) <<< 8);

            return raw_pwm_sum - (int'(weight_i) <<< 7);
        end
    endfunction

    function automatic int reference_output(input int batch_idx, input int col, input int rows);
        int sum;
        begin
            sum = 0;
            for (int r = 0; r < rows; r++) begin
                sum += reference_pwm_int8_row_serial_product(vectors[batch_idx][r], weights[r][col]);
            end
            return sum;
        end
    endfunction

    always_ff @(posedge clk) begin
        if (rstn && m_axis_tvalid && m_axis_tready) begin
            if (observed_count < MAX_BATCH * MAX_COLS) begin
                observed[observed_count] <= $signed(m_axis_tdata);
            end
            if (m_axis_tlast) begin
                tlast_count <= tlast_count + 1;
            end
            observed_count <= observed_count + 1;
        end
    end

    task automatic clear_inputs;
        begin
            for (int b = 0; b < MAX_BATCH; b++) begin
                for (int r = 0; r < MAX_ROWS; r++) begin
                    vectors[b][r] = 8'sd0;
                end
            end
            for (int r = 0; r < MAX_ROWS; r++) begin
                for (int c = 0; c < MAX_COLS; c++) begin
                    weights[r][c] = 8'sd0;
                end
            end
        end
    endtask

    task automatic run_case(input string name, input int rows, input int cols, input int batch);
        int weight_words;
        int vector_words;
        int expected_count;
        int obs_idx;
        int expected_value;
        begin
            cfg_rows = rows[7:0];
            cfg_cols = cols[7:0];
            cfg_batch = batch[7:0];
            observed_count = 0;
            tlast_count = 0;

            repeat (2) @(posedge clk);
            start_pulse <= 1'b1;
            @(posedge clk);
            start_pulse <= 1'b0;
            wait (s_axis_tready);

            weight_words = ((rows * cols) + 3) / 4;
            vector_words = (rows + 3) / 4;
            for (int w = 0; w < weight_words; w++) begin
                send_word(pack_weight_word(w, rows, cols), 1'b0);
            end
            for (int b = 0; b < batch; b++) begin
                for (int w = 0; w < vector_words; w++) begin
                    send_word(
                        pack_vector_word(b, w, rows),
                        (b == batch - 1) && (w == vector_words - 1)
                    );
                end
            end

            wait (done_pulse);
            repeat (2) @(posedge clk);

            if (error_pulse || error_code != 8'd0) begin
                $fatal(1, "%s unexpected error_code=%0d", name, error_code);
            end

            expected_count = batch * cols;
            if (observed_count != expected_count) begin
                $fatal(1, "%s expected %0d outputs, got %0d", name, expected_count, observed_count);
            end
            if (tlast_count != 1) begin
                $fatal(1, "%s expected exactly one output TLAST, got %0d", name, tlast_count);
            end
            if (words_received != (weight_words + batch * vector_words)) begin
                $fatal(1, "%s words_received mismatch got=%0d", name, words_received);
            end
            if (outputs_sent != expected_count) begin
                $fatal(1, "%s outputs_sent mismatch got=%0d", name, outputs_sent);
            end

            for (int b = 0; b < batch; b++) begin
                for (int c = 0; c < cols; c++) begin
                    obs_idx = b * cols + c;
                    expected_value = reference_output(b, c, rows);
                    if (observed[obs_idx] != expected_value) begin
                        $fatal(
                            1,
                            "%s output[%0d,%0d] expected %0d got %0d",
                            name,
                            b,
                            c,
                            expected_value,
                            observed[obs_idx]
                        );
                    end
                end
            end

            $display("PASS: %s rows=%0d cols=%0d batch=%0d cycles=%0d", name, rows, cols, batch, cycle_count);
        end
    endtask

    task automatic run_burst_case(input string name, input int rows, input int cols, input int batch);
        int weight_words;
        int vector_words;
        int expected_count;
        int obs_idx;
        int expected_value;
        begin
            cfg_rows = rows[7:0];
            cfg_cols = cols[7:0];
            cfg_batch = batch[7:0];
            observed_count = 0;
            tlast_count = 0;
            output_backpressure = 1'b1;

            repeat (2) @(posedge clk);
            start_pulse <= 1'b1;
            @(posedge clk);
            start_pulse <= 1'b0;
            wait (s_axis_tready);

            weight_words = ((rows * cols) + 3) / 4;
            vector_words = (rows + 3) / 4;
            send_packet_burst(rows, cols, batch);

            wait (done_pulse);
            output_backpressure = 1'b0;
            repeat (4) @(posedge clk);

            if (error_pulse || error_code != 8'd0) begin
                $fatal(1, "%s unexpected error_code=%0d", name, error_code);
            end

            expected_count = batch * cols;
            if (observed_count != expected_count) begin
                $fatal(1, "%s expected %0d outputs, got %0d", name, expected_count, observed_count);
            end
            if (tlast_count != 1) begin
                $fatal(1, "%s expected exactly one output TLAST, got %0d", name, tlast_count);
            end
            if (words_received != (weight_words + batch * vector_words)) begin
                $fatal(1, "%s words_received mismatch got=%0d", name, words_received);
            end
            if (outputs_sent != expected_count) begin
                $fatal(1, "%s outputs_sent mismatch got=%0d", name, outputs_sent);
            end

            for (int b = 0; b < batch; b++) begin
                for (int c = 0; c < cols; c++) begin
                    obs_idx = b * cols + c;
                    expected_value = reference_output(b, c, rows);
                    if (observed[obs_idx] != expected_value) begin
                        $fatal(
                            1,
                            "%s output[%0d,%0d] expected %0d got %0d",
                            name,
                            b,
                            c,
                            expected_value,
                            observed[obs_idx]
                        );
                    end
                end
            end

            $display("PASS: %s rows=%0d cols=%0d batch=%0d cycles=%0d", name, rows, cols, batch, cycle_count);
        end
    endtask

    initial begin
        repeat (5) @(posedge clk);
        rstn <= 1'b1;
        repeat (2) @(posedge clk);

        clear_inputs();
        vectors[0][0] = 8'sd1;
        weights[0][0] = 8'sd7;
        run_case("rows1", 1, 1, 1);

        clear_inputs();
        vectors[0][0] = 8'sd1;
        vectors[0][1] = 8'sd2;
        weights[0][0] = 8'sd4;
        weights[0][1] = 8'sd5;
        weights[1][0] = 8'sd6;
        weights[1][1] = 8'sd7;
        run_case("rows2", 2, 2, 1);

        clear_inputs();
        vectors[0][0] = 8'sd1;
        vectors[0][1] = 8'sd2;
        vectors[0][2] = 8'sd3;
        weights[0][0] = 8'sd4;
        weights[0][1] = 8'sd5;
        weights[1][0] = 8'sd6;
        weights[1][1] = 8'sd7;
        weights[2][0] = 8'sd8;
        weights[2][1] = 8'sd9;
        run_case("rows3", 3, 2, 1);

        clear_inputs();
        vectors[0][0] = 8'sd1;
        vectors[0][1] = 8'sd2;
        vectors[0][2] = 8'sd3;
        vectors[0][3] = 8'sd4;
        weights[0][0] = 8'sd1;
        weights[0][1] = 8'sd0;
        weights[1][0] = 8'sd0;
        weights[1][1] = 8'sd1;
        weights[2][0] = 8'sd10;
        weights[2][1] = 8'sd0;
        weights[3][0] = 8'sd0;
        weights[3][1] = 8'sd10;
        run_case("rows4", 4, 2, 1);
        if (!tdc_timing_valid
            || tdc_first_ready_column != 8'd1
            || tdc_last_ready_column != 8'd0
            || tdc_arrival_min_ps != 32'd495
            || tdc_arrival_max_ps != 32'd514
            || tdc_arrival_spread_ps != 32'd19
            || tdc_pe_completion_time_ps != 32'd514) begin
            $fatal(
                1,
                "TDC timing mismatch valid=%0d first=%0d last=%0d arrival=%0d..%0d spread=%0d completion=%0d",
                tdc_timing_valid,
                tdc_first_ready_column,
                tdc_last_ready_column,
                tdc_arrival_min_ps,
                tdc_arrival_max_ps,
                tdc_arrival_spread_ps,
                tdc_pe_completion_time_ps
            );
        end

        clear_inputs();
        vectors[0][0] = 8'sd1;
        weights[0][0] = -8'sd42;
        weights[0][1] = 8'sd0;
        weights[0][2] = 8'sd31;
        run_case("tdc_signed_zero", 1, 3, 1);
        if (!tdc_timing_valid
            || tdc_first_ready_column != 8'd0
            || tdc_last_ready_column != 8'd1
            || tdc_arrival_min_ps != 32'd495
            || tdc_arrival_max_ps != 32'd576
            || tdc_arrival_spread_ps != 32'd81
            || tdc_pe_completion_time_ps != 32'd576) begin
            $fatal(
                1,
                "Signed TDC timing mismatch valid=%0d first=%0d last=%0d arrival=%0d..%0d spread=%0d completion=%0d",
                tdc_timing_valid,
                tdc_first_ready_column,
                tdc_last_ready_column,
                tdc_arrival_min_ps,
                tdc_arrival_max_ps,
                tdc_arrival_spread_ps,
                tdc_pe_completion_time_ps
            );
        end

        clear_inputs();
        vectors[0][0] = -8'sd128;
        vectors[0][1] = 8'sd127;
        weights[0][0] = -8'sd128;
        weights[0][1] = 8'sd127;
        weights[1][0] = 8'sd127;
        weights[1][1] = -8'sd128;
        run_case("pwm_extreme_nibbles", 2, 2, 1);

        clear_inputs();
        for (int r = 0; r < 32; r++) begin
            vectors[0][r] = r[7:0] - 8'sd16;
            for (int c = 0; c < 4; c++) begin
                weights[r][c] = (r + c + 1) % 7 - 3;
            end
        end
        run_case("rows32", 32, 4, 1);

        clear_inputs();
        for (int b = 0; b < 3; b++) begin
            for (int r = 0; r < 5; r++) begin
                vectors[b][r] = (b + 1) * (r + 1);
            end
        end
        for (int r = 0; r < 5; r++) begin
            for (int c = 0; c < 3; c++) begin
                weights[r][c] = (r * 2) - c;
            end
        end
        run_case("batch3", 5, 3, 3);

        clear_inputs();
        for (int b = 0; b < 32; b++) begin
            for (int r = 0; r < 32; r++) begin
                vectors[b][r] = ((b * 3 + r) % 17) - 8;
            end
        end
        for (int r = 0; r < 32; r++) begin
            for (int c = 0; c < 32; c++) begin
                weights[r][c] = ((r * 5 + c * 3) % 23) - 11;
            end
        end
        run_burst_case("full_burst_backpressure", 32, 32, 32);

        $display("SIM PASS: AXIS batched MVM core");
        $finish;
    end

    initial begin
        #2000000;
        $fatal(1, "Timeout");
    end
endmodule
