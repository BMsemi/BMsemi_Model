`timescale 1ns/1ps
// Verify the exact LPC configuration used by Qwen6, including padded INT8
// packets, batch/column tails and output backpressure. No TCAM timing model.
module tb_lpc;
    reg clk=0, rstn=0, start=0;
    always #5 clk=~clk;
    reg [7:0] rows, cols, batch;
    reg [31:0] txdata=0;
    reg txvalid=0, txlast=0;
    wire txready;
    wire [31:0] rxdata;
    wire rxvalid, rxlast, done, error;
    wire [7:0] error_code;
    reg rxready=0;
    integer received=0, last_count=0, stall_cycle=0;
    reg signed [7:0] x[0:31][0:31];
    reg signed [7:0] w[0:31][0:31];

    x2_mvm_axis_core #(.N_PE(4), .TDC_TIMING_ENABLE(0)) dut (
        .clk(clk), .rstn(rstn), .start_pulse(start), .abort_pulse(1'b0),
        .cfg_rows(rows), .cfg_cols(cols), .cfg_batch(batch),
        .done_pulse(done), .error_pulse(error), .error_code(error_code),
        .s_axis_tdata(txdata), .s_axis_tvalid(txvalid), .s_axis_tready(txready), .s_axis_tlast(txlast),
        .m_axis_tdata(rxdata), .m_axis_tvalid(rxvalid), .m_axis_tready(rxready), .m_axis_tlast(rxlast)
    );

    always @(negedge clk) begin
        stall_cycle=stall_cycle+1;
        rxready=rstn && (stall_cycle%3 != 0);
    end

    function automatic integer expected(input integer t, input integer k);
        integer total;
        begin
            total=0;
            for(integer c=0;c<rows;c=c+1) total=total+$signed(x[t][c])*$signed(w[c][k]);
            expected=total;
        end
    endfunction

    always @(posedge clk) begin
        if(rstn && error) $fatal(1,"LPC error %0d",error_code);
        if(rstn && rxvalid && rxready) begin
            if(received>=batch*cols) $fatal(1,"Too many output words");
            if($signed(rxdata) !== expected(received/cols,received%cols))
                $fatal(1,"Output %0d got %0d expected %0d",received,$signed(rxdata),expected(received/cols,received%cols));
            if(rxlast !== (received==batch*cols-1)) $fatal(1,"Unexpected TLAST at %0d",received);
            if(rxlast) last_count=last_count+1;
            received=received+1;
        end
    end

    task automatic send(input reg[31:0] word, input reg last);
        begin
            @(negedge clk); txdata=word; txvalid=1; txlast=last;
            @(posedge clk);
            while(!txready) @(posedge clk);
            @(negedge clk); txvalid=0; txlast=0;
        end
    endtask

    task automatic run_case(input integer r, input integer k, input integer t);
        reg[31:0] word;
        integer idx, weight_words, vector_words;
        begin
            @(negedge clk); rstn=0; start=0; txvalid=0; received=0; last_count=0;
            rows=r; cols=k; batch=t;
            for(integer b=0;b<t;b=b+1)
                for(integer c=0;c<r;c=c+1) x[b][c]=((b*31+c*7)%255)-127;
            for(integer c=0;c<r;c=c+1)
                for(integer o=0;o<k;o=o+1) w[c][o]=((c*11+o*13)%255)-127;
            repeat(8) @(negedge clk);
            rstn=1;
            repeat(3) @(negedge clk);
            start=1; @(negedge clk); start=0;
            weight_words=(r*k+3)/4;
            vector_words=(r+3)/4;
            for(integer n=0;n<weight_words;n=n+1) begin
                word=0;
                for(integer byte_i=0;byte_i<4;byte_i=byte_i+1) begin
                    idx=n*4+byte_i;
                    if(idx<r*k) word[byte_i*8+:8]=w[idx/k][idx%k];
                end
                send(word,0);
            end
            for(integer b=0;b<t;b=b+1) begin
                for(integer n=0;n<vector_words;n=n+1) begin
                    word=0;
                    for(integer byte_i=0;byte_i<4;byte_i=byte_i+1) begin
                        idx=n*4+byte_i;
                        if(idx<r) word[byte_i*8+:8]=x[b][idx];
                    end
                    send(word,(b==t-1)&&(n==vector_words-1));
                end
            end
            while(!done) @(negedge clk);
            if(received != t*k || last_count != 1) $fatal(1,"Incorrect output count");
            $display("PASS LPC N_PE=4 TDC=0 C=%0d K=%0d T=%0d",r,k,t);
        end
    endtask

    initial begin
        run_case(32,32,11);
        run_case(31,29,32);
        run_case(1,1,1);
        run_case(5,3,2);
        $display("QWEN6 LPC RTL PASS");
        $finish;
    end
    initial begin #100000000; $fatal(1,"LPC simulation timeout"); end
endmodule
