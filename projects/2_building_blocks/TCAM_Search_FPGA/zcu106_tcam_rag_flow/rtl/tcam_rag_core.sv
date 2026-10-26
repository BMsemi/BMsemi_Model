`timescale 1ns/1ps

// Differential TCAM organization scaled from the 6x6 reference model:
// 16 logical query bits -> 32 physical q/qbar rows.
// 16 stored vectors -> 32 physical A/B columns.
//
// differential = B - A
// hamming       = (active - differential) / 2 = A
// tdc_code      = differential + active       = 2 * B
//
// An FPGA has no analog crossbar current front end, so A/B currents and the
// current-to-time conversion are represented by exact digital counts/codes.
module tcam_rag_core #(
    parameter integer KEY_BITS   = 16,
    parameter integer ENTRIES    = 16,
    parameter integer TOP_K      = 4,
    parameter integer FIFO_DEPTH = 8
) (
    input  logic clk,
    input  logic rstn,

    input  logic                         cfg_we,
    input  logic [$clog2(ENTRIES)-1:0]   cfg_index,
    input  logic [KEY_BITS-1:0]          cfg_key,
    input  logic [KEY_BITS-1:0]          cfg_care,
    input  logic [15:0]                  cfg_chunk_id,

    input  logic                         query_valid,
    output logic                         query_ready,
    input  logic [15:0]                  query_id,
    input  logic [KEY_BITS-1:0]          query_value,
    input  logic [KEY_BITS-1:0]          query_care,

    output logic                         result_valid,
    input  logic                         result_ready,
    output logic [15:0]                  result_query_id,
    output logic [$clog2(TOP_K)-1:0]     result_rank,
    output logic [15:0]                  result_chunk_id,
    output logic [$clog2(KEY_BITS+1)-1:0] result_hd,
    output logic [$clog2(2*KEY_BITS+1)-1:0] result_score,
    output logic [$clog2(KEY_BITS+1)-1:0] result_a_current,
    output logic [$clog2(KEY_BITS+1)-1:0] result_b_current,
    output logic signed [$clog2(2*KEY_BITS+1)-1:0] result_b_minus_a,
    output logic                         result_last,

    output logic                         busy,
    output logic [$clog2(FIFO_DEPTH+1)-1:0] input_fifo_level,
    output logic [$clog2(FIFO_DEPTH+1)-1:0] compute_fifo_level,
    output logic [$clog2(FIFO_DEPTH*TOP_K+1)-1:0] output_fifo_level,
    output logic [31:0]                  queries_completed
);
    localparam integer PHYS_ROWS = 2 * KEY_BITS;
    localparam integer PHYS_COLS = 2 * ENTRIES;
    localparam integer ENTRY_W = $clog2(ENTRIES);
    localparam integer RANK_W  = $clog2(TOP_K);
    localparam integer COUNT_W = $clog2(KEY_BITS+1);
    localparam integer SCORE_W = $clog2(2*KEY_BITS+1);
    localparam integer PTR_W = $clog2(FIFO_DEPTH);
    localparam integer OF_DEPTH = FIFO_DEPTH * TOP_K;
    localparam integer OF_PTR_W = $clog2(OF_DEPTH);

    // Entry i owns physical columns 2*i (A) and 2*i+1 (B). Each vector
    // contains the 2*KEY_BITS q/qbar row cells.
    logic [PHYS_ROWS-1:0] a_column_mem [0:ENTRIES-1];
    logic [PHYS_ROWS-1:0] b_column_mem [0:ENTRIES-1];
    logic [15:0] chunk_mem [0:ENTRIES-1];

    typedef struct packed {
        logic [15:0] id;
        logic [KEY_BITS-1:0] value;
        logic [KEY_BITS-1:0] care;
    } query_t;

    typedef struct packed {
        logic [15:0] query_id;
        logic [RANK_W-1:0] rank;
        logic [15:0] chunk_id;
        logic [COUNT_W-1:0] hd;
        logic [SCORE_W-1:0] score;
        logic [COUNT_W-1:0] a_current;
        logic [COUNT_W-1:0] b_current;
        logic signed [SCORE_W-1:0] b_minus_a;
        logic last;
    } result_t;

    query_t input_fifo [0:FIFO_DEPTH-1];
    query_t compute_fifo [0:FIFO_DEPTH-1];
    result_t output_fifo [0:OF_DEPTH-1];

    logic [PTR_W-1:0] in_wr_ptr, in_rd_ptr;
    logic [PTR_W-1:0] comp_wr_ptr, comp_rd_ptr;
    logic [OF_PTR_W-1:0] out_wr_ptr, out_rd_ptr;

    logic [PHYS_ROWS-1:0] query_rows;
    logic [COUNT_W-1:0] a_current_all [0:ENTRIES-1];
    logic [COUNT_W-1:0] b_current_all [0:ENTRIES-1];
    logic [COUNT_W-1:0] active_all [0:ENTRIES-1];
    logic signed [SCORE_W-1:0] b_minus_a_all [0:ENTRIES-1];
    logic [COUNT_W-1:0] hd_all [0:ENTRIES-1];
    logic [SCORE_W-1:0] tdc_code_all [0:ENTRIES-1];

    logic [SCORE_W-1:0] best_score [0:TOP_K-1];
    logic [ENTRY_W-1:0] best_idx [0:TOP_K-1];

    typedef enum logic [1:0] {S_IDLE, S_RANK, S_EMIT} state_t;
    state_t state;
    query_t active_query;
    logic [RANK_W-1:0] emit_rank;

    integer qj;
    integer ci, cj;
    integer si, sk;
    logic [SCORE_W-1:0] cand_score, swap_score;
    logic [ENTRY_W-1:0] cand_idx, swap_idx;

    assign query_ready = (input_fifo_level < FIFO_DEPTH);
    assign result_valid = (output_fifo_level != 0);
    assign result_query_id = output_fifo[out_rd_ptr].query_id;
    assign result_rank = output_fifo[out_rd_ptr].rank;
    assign result_chunk_id = output_fifo[out_rd_ptr].chunk_id;
    assign result_hd = output_fifo[out_rd_ptr].hd;
    assign result_score = output_fifo[out_rd_ptr].score;
    assign result_a_current = output_fifo[out_rd_ptr].a_current;
    assign result_b_current = output_fifo[out_rd_ptr].b_current;
    assign result_b_minus_a = output_fifo[out_rd_ptr].b_minus_a;
    assign result_last = output_fifo[out_rd_ptr].last;
    assign busy = (state != S_IDLE) || (input_fifo_level != 0) ||
                  (compute_fifo_level != 0) || (output_fifo_level != 0);

    // Encode every logical query symbol onto its physical q/qbar row pair.
    always_comb begin
        query_rows = '0;
        for (qj = 0; qj < KEY_BITS; qj = qj + 1) begin
            query_rows[2*qj]   = active_query.care[qj] && active_query.value[qj];
            query_rows[2*qj+1] = active_query.care[qj] && !active_query.value[qj];
        end
    end

    // Accumulate every A/B pair in parallel and explicitly derive B-A, HD,
    // and the digital TDC code used for ranking.
    always_comb begin
        for (ci = 0; ci < ENTRIES; ci = ci + 1) begin
            a_current_all[ci] = '0;
            b_current_all[ci] = '0;
            for (cj = 0; cj < PHYS_ROWS; cj = cj + 1) begin
                a_current_all[ci] = a_current_all[ci] +
                    (query_rows[cj] && a_column_mem[ci][cj]);
                b_current_all[ci] = b_current_all[ci] +
                    (query_rows[cj] && b_column_mem[ci][cj]);
            end
            active_all[ci] = a_current_all[ci] + b_current_all[ci];
            b_minus_a_all[ci] = $signed({1'b0, b_current_all[ci]}) -
                                $signed({1'b0, a_current_all[ci]});
            hd_all[ci] = (active_all[ci] - b_minus_a_all[ci]) >> 1;
            tdc_code_all[ci] = b_minus_a_all[ci] + active_all[ci];
        end
    end

    wire input_push = query_valid && query_ready;
    wire input_to_compute = (input_fifo_level != 0) &&
                            (compute_fifo_level < FIFO_DEPTH);
    wire compute_pop = (state == S_IDLE) && (compute_fifo_level != 0) &&
                       (output_fifo_level <= OF_DEPTH-TOP_K);

    always_ff @(posedge clk) begin
        if (!rstn) begin
            in_wr_ptr <= '0;
            in_rd_ptr <= '0;
            comp_wr_ptr <= '0;
            comp_rd_ptr <= '0;
            input_fifo_level <= '0;
            compute_fifo_level <= '0;
        end else begin
            if (input_push) begin
                input_fifo[in_wr_ptr].id <= query_id;
                input_fifo[in_wr_ptr].value <= query_value;
                input_fifo[in_wr_ptr].care <= query_care;
                in_wr_ptr <= in_wr_ptr + 1'b1;
            end
            if (input_to_compute) begin
                compute_fifo[comp_wr_ptr] <= input_fifo[in_rd_ptr];
                in_rd_ptr <= in_rd_ptr + 1'b1;
                comp_wr_ptr <= comp_wr_ptr + 1'b1;
            end
            if (compute_pop)
                comp_rd_ptr <= comp_rd_ptr + 1'b1;

            case ({input_push, input_to_compute})
                2'b10: input_fifo_level <= input_fifo_level + 1'b1;
                2'b01: input_fifo_level <= input_fifo_level - 1'b1;
                default: input_fifo_level <= input_fifo_level;
            endcase
            case ({input_to_compute, compute_pop})
                2'b10: compute_fifo_level <= compute_fifo_level + 1'b1;
                2'b01: compute_fifo_level <= compute_fifo_level - 1'b1;
                default: compute_fifo_level <= compute_fifo_level;
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (!rstn)
            out_rd_ptr <= '0;
        else if (result_valid && result_ready)
            out_rd_ptr <= out_rd_ptr + 1'b1;
    end

    always_ff @(posedge clk) begin
        if (!rstn) begin
            state <= S_IDLE;
            active_query <= '0;
            emit_rank <= '0;
            out_wr_ptr <= '0;
            output_fifo_level <= '0;
            queries_completed <= '0;
            for (si = 0; si < ENTRIES; si = si + 1) begin
                a_column_mem[si] <= '0;
                b_column_mem[si] <= '0;
                chunk_mem[si] <= si[15:0];
            end
            for (si = 0; si < TOP_K; si = si + 1) begin
                best_score[si] <= '0;
                best_idx[si] <= {ENTRY_W{1'b1}};
            end
        end else begin
            if (cfg_we) begin
                chunk_mem[cfg_index] <= cfg_chunk_id;
                for (si = 0; si < KEY_BITS; si = si + 1) begin
                    if (cfg_care[si]) begin
                        a_column_mem[cfg_index][2*si]   <= !cfg_key[si];
                        b_column_mem[cfg_index][2*si]   <=  cfg_key[si];
                        a_column_mem[cfg_index][2*si+1] <=  cfg_key[si];
                        b_column_mem[cfg_index][2*si+1] <= !cfg_key[si];
                    end else begin
                        a_column_mem[cfg_index][2*si]   <= 1'b0;
                        b_column_mem[cfg_index][2*si]   <= 1'b0;
                        a_column_mem[cfg_index][2*si+1] <= 1'b0;
                        b_column_mem[cfg_index][2*si+1] <= 1'b0;
                    end
                end
            end

            case ({(state == S_EMIT), (result_valid && result_ready)})
                2'b10: output_fifo_level <= output_fifo_level + 1'b1;
                2'b01: output_fifo_level <= output_fifo_level - 1'b1;
                default: output_fifo_level <= output_fifo_level;
            endcase

            case (state)
                S_IDLE: begin
                    if ((compute_fifo_level != 0) &&
                        (output_fifo_level <= OF_DEPTH-TOP_K)) begin
                        active_query <= compute_fifo[comp_rd_ptr];
                        state <= S_RANK;
                    end
                end

                S_RANK: begin
                    // Rank the largest TDC code, with deterministic index ties.
                    for (sk = 0; sk < TOP_K; sk = sk + 1) begin
                        best_score[sk] = '0;
                        best_idx[sk] = {ENTRY_W{1'b1}};
                    end
                    for (si = 0; si < ENTRIES; si = si + 1) begin
                        cand_score = tdc_code_all[si];
                        cand_idx = si[ENTRY_W-1:0];
                        for (sk = 0; sk < TOP_K; sk = sk + 1) begin
                            if ((cand_score > best_score[sk]) ||
                                ((cand_score == best_score[sk]) &&
                                 (cand_idx < best_idx[sk]))) begin
                                swap_score = best_score[sk];
                                swap_idx = best_idx[sk];
                                best_score[sk] = cand_score;
                                best_idx[sk] = cand_idx;
                                cand_score = swap_score;
                                cand_idx = swap_idx;
                            end
                        end
                    end
                    emit_rank <= '0;
                    state <= S_EMIT;
                end

                S_EMIT: begin
                    output_fifo[out_wr_ptr].query_id <= active_query.id;
                    output_fifo[out_wr_ptr].rank <= emit_rank;
                    output_fifo[out_wr_ptr].chunk_id <= chunk_mem[best_idx[emit_rank]];
                    output_fifo[out_wr_ptr].hd <= hd_all[best_idx[emit_rank]];
                    output_fifo[out_wr_ptr].score <= tdc_code_all[best_idx[emit_rank]];
                    output_fifo[out_wr_ptr].a_current <= a_current_all[best_idx[emit_rank]];
                    output_fifo[out_wr_ptr].b_current <= b_current_all[best_idx[emit_rank]];
                    output_fifo[out_wr_ptr].b_minus_a <= b_minus_a_all[best_idx[emit_rank]];
                    output_fifo[out_wr_ptr].last <= (emit_rank == TOP_K-1);
                    out_wr_ptr <= out_wr_ptr + 1'b1;
                    if (emit_rank == TOP_K-1) begin
                        queries_completed <= queries_completed + 1'b1;
                        state <= S_IDLE;
                    end else begin
                        emit_rank <= emit_rank + 1'b1;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    initial begin
        if (PHYS_ROWS != 32 || PHYS_COLS != 32)
            $display("INFO: differential TCAM physical geometry is %0dx%0d",
                     PHYS_ROWS, PHYS_COLS);
    end
endmodule
