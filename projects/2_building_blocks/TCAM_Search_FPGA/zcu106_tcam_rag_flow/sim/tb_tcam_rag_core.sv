`timescale 1ns/1ps
module tb_tcam_rag_core;
    logic clk=0, rstn=0;
    always #5 clk=~clk;

    logic cfg_we; logic [3:0] cfg_index; logic [15:0] cfg_key,cfg_care,cfg_chunk_id;
    logic query_valid,query_ready; logic [15:0] query_id,query_value,query_care;
    logic result_valid,result_ready; logic [15:0] result_query_id,result_chunk_id;
    logic [1:0] result_rank; logic [4:0] result_hd; logic [5:0] result_score;
    logic [4:0] result_a_current,result_b_current;
    logic signed [5:0] result_b_minus_a;
    logic result_last,busy; logic [3:0] in_level,comp_level; logic [5:0] out_level;
    logic [31:0] completed;

    tcam_rag_core dut(
        .clk,.rstn,.cfg_we,.cfg_index,.cfg_key,.cfg_care,.cfg_chunk_id,
        .query_valid,.query_ready,.query_id,.query_value,.query_care,
        .result_valid,.result_ready,.result_query_id,.result_rank,
        .result_chunk_id,.result_hd,.result_score,
        .result_a_current,.result_b_current,.result_b_minus_a,
        .result_last,.busy,
        .input_fifo_level(in_level),.compute_fifo_level(comp_level),
        .output_fifo_level(out_level),.queries_completed(completed));

    task automatic program_entry(input [3:0] idx,input [15:0] key,input [15:0] care,input [15:0] chunk);
      begin @(negedge clk); cfg_index=idx;cfg_key=key;cfg_care=care;cfg_chunk_id=chunk;cfg_we=1;
            @(negedge clk);cfg_we=0; end
    endtask
    task automatic send_query(input [15:0] id,input [15:0] value,input [15:0] care);
      begin @(negedge clk); while(!query_ready) @(negedge clk);
            query_id=id;query_value=value;query_care=care;query_valid=1;
            @(negedge clk);query_valid=0; end
    endtask

    integer n;
    reg [15:0] exp_chunk[0:3]; reg [4:0] exp_hd[0:3]; reg [5:0] exp_score[0:3];
    reg [4:0] exp_a[0:3],exp_b[0:3]; reg signed [5:0] exp_diff[0:3];
    initial begin
      cfg_we=0;query_valid=0;result_ready=0;cfg_index=0;cfg_key=0;cfg_care=0;cfg_chunk_id=0;
      query_id=0;query_value=0;query_care=0;
      repeat(5) @(negedge clk); rstn=1;

      program_entry(0,16'hA5A5,16'hFFFF,16'h1000); // HD 0
      program_entry(1,16'hA5A4,16'hFFFF,16'h1001); // HD 1
      program_entry(2,16'hA5A7,16'hFFFF,16'h1002); // HD 1, tie broken by index
      program_entry(3,16'hA5A0,16'hFFFF,16'h1003); // HD 2
      for(n=4;n<16;n=n+1) program_entry(n[3:0],16'h0000,16'hFFFF,16'h1000+n);

      // Full-care entries must occupy complementary 32-row A/B columns.
      for(n=0;n<16;n=n+1) begin
        if((dut.a_column_mem[n] ^ dut.b_column_mem[n]) !== 32'hFFFFFFFF)
          $fatal(1,"A/B columns are not complementary for entry %0d",n);
      end

      exp_chunk[0]=16'h1000;exp_hd[0]=0;exp_score[0]=32;
      exp_chunk[1]=16'h1001;exp_hd[1]=1;exp_score[1]=30;
      exp_chunk[2]=16'h1002;exp_hd[2]=1;exp_score[2]=30;
      exp_chunk[3]=16'h1003;exp_hd[3]=2;exp_score[3]=28;
      exp_a[0]=0;exp_b[0]=16;exp_diff[0]=16;
      exp_a[1]=1;exp_b[1]=15;exp_diff[1]=14;
      exp_a[2]=1;exp_b[2]=15;exp_diff[2]=14;
      exp_a[3]=2;exp_b[3]=14;exp_diff[3]=12;

      send_query(16'h55AA,16'hA5A5,16'hFFFF);
      // Hold output to prove output FIFO retains ranked records.
      repeat(12) @(negedge clk);
      if(out_level != 4) $fatal(1,"Expected 4 output records, got %0d",out_level);
      result_ready=1;
      for(n=0;n<4;n=n+1) begin
        while(!result_valid) @(negedge clk);
        if(result_query_id!==16'h55AA || result_rank!==n[1:0] ||
           result_chunk_id!==exp_chunk[n] || result_hd!==exp_hd[n] ||
           result_score!==exp_score[n] || result_a_current!==exp_a[n] ||
           result_b_current!==exp_b[n] || result_b_minus_a!==exp_diff[n] ||
           result_last!==(n==3))
          $fatal(1,"Bad rank %0d: q=%h rank=%0d chunk=%h A=%0d B=%0d B-A=%0d TDC=%0d HD=%0d last=%b",
                 n,result_query_id,result_rank,result_chunk_id,
                 result_a_current,result_b_current,result_b_minus_a,
                 result_score,result_hd,result_last);
        if(result_b_minus_a !== ($signed({1'b0,result_b_current}) -
                                 $signed({1'b0,result_a_current})))
          $fatal(1,"Differential equation failed at rank %0d",n);
        if(result_score !== (result_b_minus_a + result_a_current + result_b_current))
          $fatal(1,"TDC equation failed at rank %0d",n);
        if(result_hd !== ((result_a_current + result_b_current - result_b_minus_a) >> 1))
          $fatal(1,"HD decode equation failed at rank %0d",n);
        $display("RESULT rank=%0d chunk=%h A=%0d B=%0d B-A=%0d TDC=%0d HD=%0d",
                 result_rank,result_chunk_id,result_a_current,result_b_current,
                 result_b_minus_a,result_score,result_hd);
        @(negedge clk);
      end
      result_ready=0;
      repeat(3) @(negedge clk);
      if(completed!==1 || out_level!==0) $fatal(1,"Bad completion/output state");

      // Recreate the attached 6x6 example in the low three logical features.
      // String keys "100", "000", "011" map bit 0 to the first symbol.
      program_entry(0,16'h0001,16'h0007,16'h2000);
      program_entry(1,16'h0000,16'h0007,16'h2001);
      program_entry(2,16'h0006,16'h0007,16'h2002);
      for(n=3;n<16;n=n+1) program_entry(n[3:0],16'h0000,16'h0000,16'h2000+n);

      send_query(16'h6006,16'h0001,16'h0003); // ternary query "10X"
      while(out_level != 4) @(negedge clk);
      result_ready=1;
      for(n=0;n<4;n=n+1) begin
        while(!result_valid) @(negedge clk);
        if(n==0 && (result_chunk_id!==16'h2000 || result_a_current!==0 ||
                   result_b_current!==2 || result_b_minus_a!==2 ||
                   result_score!==4 || result_hd!==0))
          $fatal(1,"6x6-scaled rank 0 failed");
        if(n==1 && (result_chunk_id!==16'h2001 || result_a_current!==1 ||
                   result_b_current!==1 || result_b_minus_a!==0 ||
                   result_score!==2 || result_hd!==1))
          $fatal(1,"6x6-scaled rank 1 failed");
        if(n==2 && (result_chunk_id!==16'h2002 || result_a_current!==2 ||
                   result_b_current!==0 || result_b_minus_a!==-2 ||
                   result_score!==0 || result_hd!==2))
          $fatal(1,"6x6-scaled rank 2 failed");
        $display("DEMO6_SCALED rank=%0d chunk=%h A=%0d B=%0d B-A=%0d TDC=%0d HD=%0d",
                 result_rank,result_chunk_id,result_a_current,result_b_current,
                 result_b_minus_a,result_score,result_hd);
        @(negedge clk);
      end
      result_ready=0;
      repeat(3) @(negedge clk);
      if(completed!==2 || out_level!==0) $fatal(1,"Bad second completion/output state");
      $display("TCAM_RAG_CORE_TEST_PASS");
      $finish;
    end
    initial begin #10000; $fatal(1,"TIMEOUT"); end
endmodule
