set build_dir [file normalize "C:/fpga_build/yolo26n_axi_x2_mvm_synth"]
set reports_dir [file join $build_dir reports]
set dcp_file [file join $build_dir axi_x2_mvm_accel_synth.dcp]

if {![file exists $dcp_file]} {
    puts "ERROR: Missing synthesized checkpoint: $dcp_file"
    exit 2
}

open_checkpoint $dcp_file
create_clock -period 10.000 -name aclk [get_ports aclk]
report_timing_summary -file [file join $reports_dir timing_summary_synth_100mhz.rpt]
puts "TIMING_100MHZ_REPORT: [file normalize [file join $reports_dir timing_summary_synth_100mhz.rpt]]"
exit
