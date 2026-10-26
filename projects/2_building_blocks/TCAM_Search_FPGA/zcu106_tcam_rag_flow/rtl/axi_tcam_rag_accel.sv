`timescale 1ns/1ps

// AXI4-Lite wrapper for tcam_rag_core.
// Register map (byte offsets):
// 00 ID/version (RO) = 0x5443414d ("TCAM")
// 04 CONTROL: bit0 enqueue query, bit1 pop result, bit2 soft reset
// 08 STATUS: bit0 query_ready, bit1 result_valid, bit2 busy, levels above
// 0C QUERY_ID
// 10 QUERY_VALUE[15:0]
// 14 QUERY_CARE[15:0]
// 18 TABLE_INDEX[3:0]
// 1C TABLE_KEY[15:0]
// 20 TABLE_CARE[15:0]
// 24 TABLE_CHUNK_ID[15:0]
// 28 TABLE_COMMIT: write bit0=1
// 2C RESULT_QUERY_ID (RO)
// 30 RESULT_META (RO): [31] last, [17:16] rank, [15:0] chunk_id
// 34 RESULT_SCORE (RO): [12:8] HD, [5:0] TDC-equivalent score
// 38 QUERIES_COMPLETED (RO)
// 3C RESULT_DIFF (RO): [29:24] TDC, [21:16] signed B-A,
//                      [12:8] B current, [4:0] A current
module axi_tcam_rag_accel #(
    parameter integer C_S_AXI_ADDR_WIDTH = 8,
    parameter integer C_S_AXI_DATA_WIDTH = 32
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI, ASSOCIATED_RESET aresetn" *)
    input  logic aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  logic aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *) input logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *) input logic [2:0] s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *) input logic s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *) output logic s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *) input logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *) input logic [C_S_AXI_DATA_WIDTH/8-1:0] s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *) input logic s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *) output logic s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *) output logic [1:0] s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *) output logic s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *) input logic s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *) input logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *) input logic [2:0] s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *) input logic s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *) output logic s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *) output logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *) output logic [1:0] s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *) output logic s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *) input logic s_axi_rready,

    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 irq INTERRUPT" *)
    (* X_INTERFACE_PARAMETER = "SENSITIVITY LEVEL_HIGH" *)
    output logic irq
);
    logic [C_S_AXI_ADDR_WIDTH-1:0] awaddr_hold;
    logic [31:0] wdata_hold;
    logic [3:0] wstrb_hold;
    logic have_aw, have_w;
    wire write_fire = have_aw && have_w && !s_axi_bvalid;

    logic [15:0] q_id, q_value, q_care;
    logic [3:0] table_index;
    logic [15:0] table_key, table_care, table_chunk;
    logic query_push, result_pop, cfg_we, soft_reset;

    logic core_query_ready, core_result_valid, core_busy;
    logic [15:0] result_query_id, result_chunk_id;
    logic [1:0] result_rank;
    logic [4:0] result_hd;
    logic [5:0] result_score;
    logic [4:0] result_a_current, result_b_current;
    logic signed [5:0] result_b_minus_a;
    logic result_last;
    logic [3:0] in_level, comp_level;
    logic [5:0] out_level;
    logic [31:0] queries_completed;
    wire core_rstn = aresetn && !soft_reset;

    assign s_axi_awready = !have_aw;
    assign s_axi_wready  = !have_w;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_arready = !s_axi_rvalid;
    assign s_axi_rresp   = 2'b00;
    assign irq = core_result_valid;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            have_aw <= 1'b0; have_w <= 1'b0; s_axi_bvalid <= 1'b0;
            awaddr_hold <= '0; wdata_hold <= '0; wstrb_hold <= '0;
            q_id <= '0; q_value <= '0; q_care <= '0;
            table_index <= '0; table_key <= '0; table_care <= '0; table_chunk <= '0;
            query_push <= 1'b0; result_pop <= 1'b0; cfg_we <= 1'b0; soft_reset <= 1'b0;
        end else begin
            query_push <= 1'b0; result_pop <= 1'b0; cfg_we <= 1'b0; soft_reset <= 1'b0;
            if (s_axi_awvalid && s_axi_awready) begin
                awaddr_hold <= s_axi_awaddr; have_aw <= 1'b1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                wdata_hold <= s_axi_wdata; wstrb_hold <= s_axi_wstrb; have_w <= 1'b1;
            end
            if (write_fire) begin
                case (awaddr_hold[7:0])
                    8'h04: begin
                        query_push <= wdata_hold[0] && core_query_ready;
                        result_pop <= wdata_hold[1] && core_result_valid;
                        soft_reset <= wdata_hold[2];
                    end
                    8'h0C: q_id <= wdata_hold[15:0];
                    8'h10: q_value <= wdata_hold[15:0];
                    8'h14: q_care <= wdata_hold[15:0];
                    8'h18: table_index <= wdata_hold[3:0];
                    8'h1C: table_key <= wdata_hold[15:0];
                    8'h20: table_care <= wdata_hold[15:0];
                    8'h24: table_chunk <= wdata_hold[15:0];
                    8'h28: cfg_we <= wdata_hold[0];
                    default: ;
                endcase
                have_aw <= 1'b0; have_w <= 1'b0; s_axi_bvalid <= 1'b1;
            end
            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;
        end
    end

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_rvalid <= 1'b0; s_axi_rdata <= '0;
        end else begin
            if (s_axi_arvalid && s_axi_arready) begin
                case (s_axi_araddr[7:0])
                    8'h00: s_axi_rdata <= 32'h5443414D;
                    8'h04: s_axi_rdata <= 32'h00020000;
                    8'h08: s_axi_rdata <= {8'd0, out_level, comp_level, in_level, 5'd0,
                                             core_busy, core_result_valid, core_query_ready};
                    8'h0C: s_axi_rdata <= {16'd0, q_id};
                    8'h10: s_axi_rdata <= {16'd0, q_value};
                    8'h14: s_axi_rdata <= {16'd0, q_care};
                    8'h18: s_axi_rdata <= {28'd0, table_index};
                    8'h1C: s_axi_rdata <= {16'd0, table_key};
                    8'h20: s_axi_rdata <= {16'd0, table_care};
                    8'h24: s_axi_rdata <= {16'd0, table_chunk};
                    8'h2C: s_axi_rdata <= {16'd0, result_query_id};
                    8'h30: s_axi_rdata <= {result_last, 12'd0, result_rank, 1'b0, result_chunk_id};
                    8'h34: s_axi_rdata <= {19'd0, result_hd, 2'd0, result_score};
                    8'h38: s_axi_rdata <= queries_completed;
                    8'h3C: s_axi_rdata <= {2'd0, result_score, 2'd0,
                                             result_b_minus_a, 3'd0,
                                             result_b_current, 3'd0,
                                             result_a_current};
                    default: s_axi_rdata <= 32'h0;
                endcase
                s_axi_rvalid <= 1'b1;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    tcam_rag_core u_core (
        .clk(aclk), .rstn(core_rstn),
        .cfg_we(cfg_we), .cfg_index(table_index), .cfg_key(table_key),
        .cfg_care(table_care), .cfg_chunk_id(table_chunk),
        .query_valid(query_push), .query_ready(core_query_ready),
        .query_id(q_id), .query_value(q_value), .query_care(q_care),
        .result_valid(core_result_valid), .result_ready(result_pop),
        .result_query_id(result_query_id), .result_rank(result_rank),
        .result_chunk_id(result_chunk_id), .result_hd(result_hd),
        .result_score(result_score), .result_a_current(result_a_current),
        .result_b_current(result_b_current),
        .result_b_minus_a(result_b_minus_a), .result_last(result_last),
        .busy(core_busy), .input_fifo_level(in_level),
        .compute_fifo_level(comp_level), .output_fifo_level(out_level),
        .queries_completed(queries_completed)
    );
endmodule
