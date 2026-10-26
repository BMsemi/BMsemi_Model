set source_dir [file dirname [file normalize [info script]]]
set build_dir D:/fpga_build/qwen6_nlpc
if {[info exists ::env(QWEN_HLS_BUILD)]} {set build_dir $::env(QWEN_HLS_BUILD)}
file mkdir [file dirname $build_dir]
cd [file dirname $build_dir]
open_project -reset [file tail $build_dir]
set_top qwen_nlpc
add_files [file join $source_dir qwen_nlpc.cpp]
add_files -tb [file join $source_dir test_qwen_nlpc.cpp]
open_solution -reset solution1 -flow_target vivado
set_part xczu7ev-ffvc1156-2-e
create_clock -period 10 -name default
config_interface -m_axi_addr64=true
csim_design
csynth_design
if {[info exists ::env(QWEN_HLS_COSIM)] && $::env(QWEN_HLS_COSIM) eq "1"} {cosim_design -rtl verilog}
export_design -format ip_catalog -vendor qwen.local -library hls -version 1.0
exit
