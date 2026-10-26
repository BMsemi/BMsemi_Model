`timescale 1ns/1ps

module tb_axi_input_word_reader;
    localparam int WORDS = 300;

    logic clk = 1'b0;
    logic rstn = 1'b0;
    logic clear = 1'b0;
    logic start = 1'b0;
    logic [31:0] base_addr = 32'h1000_0000;
    logic [31:0] word_count = WORDS;
    logic [31:0] axis_data;
    logic axis_valid;
    logic axis_ready = 1'b0;
    logic axis_last;
    logic idle;
    logic busy;
    logic error;
    logic [1:0] last_rresp;
    logic [31:0] read_count;
    logic [31:0] beat_count;
    logic [31:0] araddr;
    logic [7:0] arlen;
    logic [2:0] arsize;
    logic [1:0] arburst;
    logic arlock;
    logic [3:0] arcache;
    logic [2:0] arprot;
    logic [3:0] arqos;
    logic arvalid;
    logic arready = 1'b0;
    logic [127:0] rdata = '0;
    logic [1:0] rresp = 2'b00;
    logic rlast = 1'b1;
    logic rvalid = 1'b0;
    logic rready;

    int cycle = 0;
    int accepted = 0;
    logic response_pending = 1'b0;
    logic [31:0] response_addr = '0;

    always #5 clk = ~clk;

    axi_input_word_reader dut (
        .clk(clk), .rstn(rstn), .clear(clear), .start(start),
        .base_addr(base_addr), .word_count(word_count),
        .m_axis_tdata(axis_data), .m_axis_tvalid(axis_valid),
        .m_axis_tready(axis_ready), .m_axis_tlast(axis_last),
        .idle(idle), .busy(busy), .error(error),
        .last_rresp(last_rresp), .read_count(read_count),
        .beat_count(beat_count), .m_axi_araddr(araddr),
        .m_axi_arlen(arlen), .m_axi_arsize(arsize),
        .m_axi_arburst(arburst), .m_axi_arlock(arlock),
        .m_axi_arcache(arcache), .m_axi_arprot(arprot),
        .m_axi_arqos(arqos), .m_axi_arvalid(arvalid),
        .m_axi_arready(arready), .m_axi_rdata(rdata),
        .m_axi_rresp(rresp), .m_axi_rlast(rlast),
        .m_axi_rvalid(rvalid), .m_axi_rready(rready)
    );

    always_ff @(posedge clk) begin
        cycle <= cycle + 1;
        arready <= (cycle % 5) != 1;
        axis_ready <= (cycle % 7) != 2 && (cycle % 7) != 3;

        if (!rstn) begin
            response_pending <= 1'b0;
            rvalid <= 1'b0;
        end else begin
            if (arvalid && arready) begin
                if (arlen != 0 || arsize != 3'd4 || arburst != 2'b01 || araddr[3:0] != 0)
                    $fatal(1, "Bad AXI read request at 0x%08x", araddr);
                response_pending <= 1'b1;
                response_addr <= araddr;
            end

            if (response_pending && !rvalid) begin
                rdata <= {96'd0, 16'hCAFE, response_addr[15:4]};
                rvalid <= 1'b1;
                response_pending <= 1'b0;
            end else if (rvalid && rready) begin
                rvalid <= 1'b0;
            end
        end

        if (rstn && busy && !rready)
            $fatal(1, "RREADY dropped while the reader was active");

        if (axis_valid && axis_ready) begin
            if (axis_data !== {16'hCAFE, accepted[11:0]})
                $fatal(1, "Word %0d mismatch: got 0x%08x", accepted, axis_data);
            if (axis_last !== (accepted == WORDS - 1))
                $fatal(1, "TLAST mismatch at word %0d", accepted);
            accepted <= accepted + 1;
        end

        if (cycle > 10000)
            $fatal(1, "Reader timeout: accepted=%0d reads=%0d beats=%0d", accepted, read_count, beat_count);
    end

    initial begin
        repeat (5) @(posedge clk);
        rstn <= 1'b1;
        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        wait (idle && accepted == WORDS);
        @(posedge clk);
        if (error || read_count != WORDS || beat_count != WORDS)
            $fatal(1, "Reader counters/error mismatch");
        $display("READER_SIM_PASS: %0d scattered 128-bit reads completed", WORDS);
        $finish;
    end
endmodule
