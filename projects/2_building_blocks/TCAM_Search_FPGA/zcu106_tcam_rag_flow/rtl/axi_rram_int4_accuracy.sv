`timescale 1ns/1ps

// AXI4-Lite wrapper for the deterministic INT4 RRAM programming-accuracy LUT.
//
// Register map:
//   0x00 CTRL    W: bit0 start, bit1 clear_done
//   0x04 STATUS  R: bit0 idle, bit1 busy, bit2 done, bit3 correct
//   0x08 TARGET  R/W: target INT4 code [3:0]
//   0x0C INIT_G  R/W: initial conductance in 0.1 uS [9:0]
//   0x10 FINAL_G R: final conductance in 0.1 uS [9:0]
//   0x14 RESULT  R: decoded_code [3:0], pulse_count [13:8]
//   0x18 TARGET_G R: target conductance in 0.1 uS [9:0]
//   0x1C LAST_DG R: signed last delta_G in 0.1 uS [15:0]
//   0x20 LAST_V  R: last selected voltage in mV [15:0]
//   0x24 MAGIC   R: 0x52344143 ("R4AC")
//   0x28 VERSION R: 0x00010000
module axi_rram_int4_accuracy #(
    parameter integer C_S_AXI_ADDR_WIDTH = 8,
    parameter integer C_S_AXI_DATA_WIDTH = 32
) (
    input logic aclk,
    input logic aresetn,

    input  logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  logic [2:0] s_axi_awprot,
    input  logic s_axi_awvalid,
    output logic s_axi_awready,
    input  logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_wdata,
    input  logic [C_S_AXI_DATA_WIDTH/8-1:0] s_axi_wstrb,
    input  logic s_axi_wvalid,
    output logic s_axi_wready,
    output logic [1:0] s_axi_bresp,
    output logic s_axi_bvalid,
    input  logic s_axi_bready,
    input  logic [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  logic [2:0] s_axi_arprot,
    input  logic s_axi_arvalid,
    output logic s_axi_arready,
    output logic [C_S_AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output logic [1:0] s_axi_rresp,
    output logic s_axi_rvalid,
    input  logic s_axi_rready
);
    logic [C_S_AXI_ADDR_WIDTH-1:0] awaddr_hold;
    logic [31:0] wdata_hold;
    logic have_aw, have_w;
    wire write_fire = have_aw && have_w && !s_axi_bvalid;

    logic start_pulse;
    logic done_sticky;
    logic [3:0] target_code;
    logic [9:0] initial_g_tenth_us;

    logic busy, done, correct;
    logic [3:0] decoded_code;
    logic [9:0] target_g_tenth_us, final_g_tenth_us;
    logic [5:0] pulse_count;
    logic last_mode_reset;
    logic signed [15:0] last_delta_tenth_us;
    logic [15:0] last_voltage_mv;

    assign s_axi_awready = !have_aw;
    assign s_axi_wready = !have_w;
    assign s_axi_bresp = 2'b00;
    assign s_axi_arready = !s_axi_rvalid;
    assign s_axi_rresp = 2'b00;

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            have_aw <= 1'b0;
            have_w <= 1'b0;
            s_axi_bvalid <= 1'b0;
            awaddr_hold <= '0;
            wdata_hold <= '0;
            target_code <= '0;
            initial_g_tenth_us <= 10'd500;
            start_pulse <= 1'b0;
            done_sticky <= 1'b0;
        end else begin
            start_pulse <= 1'b0;
            if (s_axi_awvalid && s_axi_awready) begin
                awaddr_hold <= s_axi_awaddr;
                have_aw <= 1'b1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                wdata_hold <= s_axi_wdata;
                have_w <= 1'b1;
            end
            if (write_fire) begin
                case (awaddr_hold[7:0])
                    8'h00: begin
                        start_pulse <= wdata_hold[0] && !busy;
                        if (wdata_hold[1])
                            done_sticky <= 1'b0;
                    end
                    8'h08: target_code <= wdata_hold[3:0];
                    8'h0C: initial_g_tenth_us <= wdata_hold[9:0];
                    default: ;
                endcase
                have_aw <= 1'b0;
                have_w <= 1'b0;
                s_axi_bvalid <= 1'b1;
            end
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;
            if (done)
                done_sticky <= 1'b1;
        end
    end

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rdata <= '0;
        end else begin
            if (s_axi_arvalid && s_axi_arready) begin
                case (s_axi_araddr[7:0])
                    8'h00: s_axi_rdata <= 32'h0;
                    8'h04: s_axi_rdata <= {28'd0, correct, done_sticky, busy, !busy};
                    8'h08: s_axi_rdata <= {28'd0, target_code};
                    8'h0C: s_axi_rdata <= {22'd0, initial_g_tenth_us};
                    8'h10: s_axi_rdata <= {22'd0, final_g_tenth_us};
                    8'h14: s_axi_rdata <= {18'd0, pulse_count, 4'd0, decoded_code};
                    8'h18: s_axi_rdata <= {22'd0, target_g_tenth_us};
                    8'h1C: s_axi_rdata <= {{16{last_delta_tenth_us[15]}}, last_delta_tenth_us};
                    8'h20: s_axi_rdata <= {15'd0, last_mode_reset, last_voltage_mv};
                    8'h24: s_axi_rdata <= 32'h52344143;
                    8'h28: s_axi_rdata <= 32'h00010000;
                    default: s_axi_rdata <= 32'h0;
                endcase
                s_axi_rvalid <= 1'b1;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    rram_int4_programming_accuracy_lut u_rram_model (
        .clk(aclk),
        .rstn(aresetn),
        .start(start_pulse),
        .target_code(target_code),
        .initial_g_tenth_us(initial_g_tenth_us),
        .busy(busy),
        .done(done),
        .correct(correct),
        .decoded_code(decoded_code),
        .target_g_tenth_us(target_g_tenth_us),
        .final_g_tenth_us(final_g_tenth_us),
        .pulse_count(pulse_count),
        .last_mode_reset(last_mode_reset),
        .last_delta_tenth_us(last_delta_tenth_us),
        .last_voltage_mv(last_voltage_mv)
    );
endmodule
