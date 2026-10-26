set project_dir D:/fpga_build/qwen6_nlpc
if {[info exists ::env(QWEN_HLS_BUILD)]} {set project_dir $::env(QWEN_HLS_BUILD)}
cd [file dirname $project_dir]
open_project [file tail $project_dir]
open_solution solution1
cosim_design -rtl verilog
exit
