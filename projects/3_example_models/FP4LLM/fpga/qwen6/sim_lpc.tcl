set here [file dirname [file normalize [info script]]]
set rtl [file normalize [file join $here ../../../../TCAM_Search_FPGA/zcu106_tcam_rag_flow/xforge_zcu106/x2_zcu106/rtl/axi_x2_mvm_accel.sv]]
set testbench [file join $here tb_lpc.sv]
set top tb_lpc
set build D:/fpga_build/q6_lpcsim
if {[info exists ::env(QWEN_LPC_LEGACY)] && $::env(QWEN_LPC_LEGACY) eq "1"} {
    set testbench [file normalize [file join [file dirname $rtl] ../sim/tb_axi_x2_mvm_core.sv]]
    set top tb_axi_x2_mvm_core
    set build D:/fpga_build/q6_lpcsim_legacy
}
create_project -force qwen6_lpc_sim $build -part xczu7ev-ffvc1156-2-e
set_property target_simulator XSim [current_project]
add_files -norecurse $rtl
add_files -fileset sim_1 -norecurse $testbench
set_property top $top [get_filesets sim_1]
update_compile_order -fileset sim_1
launch_simulation -simset sim_1 -mode behavioral
run all
close_sim
exit
