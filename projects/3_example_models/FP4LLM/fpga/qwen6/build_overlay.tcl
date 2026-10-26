# Standalone Qwen overlay. Does not replace the TCAM/YOLO deploy directory.
set here [file dirname [file normalize [info script]]]
set root [file normalize [file join $here ../..]]
set rtl [file normalize [file join $root ../../TCAM_Search_FPGA/zcu106_tcam_rag_flow/xforge_zcu106/x2_zcu106/rtl]]
if {[info exists ::env(QWEN_X2_RTL)]} {set rtl $::env(QWEN_X2_RTL)}
set hls_dir D:/fpga_build/qwen6_nlpc/solution1/impl/ip
if {[info exists ::env(QWEN_HLS_IP)]} {set hls_dir $::env(QWEN_HLS_IP)}
set build_dir D:/fpga_build/q6
if {[info exists ::env(QWEN_BD_BUILD)]} {set build_dir $::env(QWEN_BD_BUILD)}
set deploy [file join $here overlay]
set design q6
file mkdir $build_dir
file mkdir $deploy
if {![file exists [file join $hls_dir component.xml]]} {error "Build the NLPC HLS IP first"}
set wrapper_handle [open [file join $rtl axi_x2_mvm_accel_bd.v] r]
set wrapper_text [read $wrapper_handle]
close $wrapper_handle
if {![regexp {parameter[^,\n]*TDC_TIMING_ENABLE} $wrapper_text]} {
    error "LPC wrapper must expose and forward TDC_TIMING_ENABLE; update from the working tcam_rag_zcu106 RTL"
}
# Snapshot sources so a later workspace edit cannot change an active build.
file mkdir [file join $build_dir src]
foreach source {axi_x2_mvm_accel.sv axi_x2_mvm_accel_bd.v} {
    file copy -force [file join $rtl $source] [file join $build_dir src $source]
}
create_project -force $design $build_dir -part xczu7ev-ffvc1156-2-e
set_property board_part xilinx.com:zcu106:part0:2.6 [current_project]
set_property target_language Verilog [current_project]
set_property ip_repo_paths [list $hls_dir] [current_project]
update_ip_catalog
add_files -norecurse [list [file join $build_dir src axi_x2_mvm_accel.sv] [file join $build_dir src axi_x2_mvm_accel_bd.v]]
update_compile_order -fileset sources_1
create_bd_design $design
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* psu
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells psu]
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__S_AXI_GP0 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__SAXIGP2__DATA_WIDTH {64} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100}] [get_bd_cells psu]
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst_pl
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {6}] [get_bd_cells ctrl]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* mem
set_property -dict [list CONFIG.NUM_SI {6} CONFIG.NUM_MI {1}] [get_bd_cells mem]
set clk [get_bd_pins psu/pl_clk0]
set rst [get_bd_pins rst_pl/peripheral_aresetn]
foreach pin {psu/maxihpm0_fpd_aclk psu/saxihp0_fpd_aclk rst_pl/slowest_sync_clk ctrl/aclk mem/aclk} {
    connect_bd_net $clk [get_bd_pins $pin]
}
connect_bd_net [get_bd_pins psu/pl_resetn0] [get_bd_pins rst_pl/ext_reset_in]
connect_bd_net $rst [get_bd_pins ctrl/aresetn] [get_bd_pins mem/aresetn]
connect_bd_intf_net [get_bd_intf_pins psu/M_AXI_HPM0_FPD] [get_bd_intf_pins ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins mem/M00_AXI] [get_bd_intf_pins psu/S_AXI_HP0_FPD]
set names {axi_x2_mvm axi_x2_mvm_1 axi_x2_mvm_2 axi_x2_mvm_3 qwen_nlpc_4 qwen_nlpc_5}
for {set i 0} {$i < 6} {incr i} {
    set name [lindex $names $i]
    if {$i < 4} {
        create_bd_cell -type module -reference axi_x2_mvm_accel_bd $name
        set_property -dict [list CONFIG.N_PE {4} CONFIG.TDC_TIMING_ENABLE {0}] [get_bd_cells $name]
        set clk_pin aclk
        set rst_pin aresetn
        set control_pin S_AXI
        set memory_pin M_AXI
        set segment S_AXI/reg0
    } else {
        create_bd_cell -type ip -vlnv qwen.local:hls:qwen_nlpc:1.0 $name
        set clk_pin ap_clk
        set rst_pin ap_rst_n
        set control_pin s_axi_control
        set memory_pin m_axi_mem
        set segment s_axi_control/Reg
    }
    connect_bd_net $clk [get_bd_pins $name/$clk_pin]
    connect_bd_net $rst [get_bd_pins $name/$rst_pin]
    connect_bd_intf_net [get_bd_intf_pins ctrl/[format M%02d_AXI $i]] [get_bd_intf_pins $name/$control_pin]
    connect_bd_intf_net [get_bd_intf_pins $name/$memory_pin] [get_bd_intf_pins mem/[format S%02d_AXI $i]]
    assign_bd_address -offset [format 0x%08X [expr {0xA0000000+$i*0x10000}]] -range 64K \
        -target_address_space [get_bd_addr_spaces psu/Data] [get_bd_addr_segs $name/$segment] -force
}
assign_bd_address
validate_bd_design
save_bd_design
if {[info exists ::env(QWEN_VALIDATE_ONLY)] && $::env(QWEN_VALIDATE_ONLY) eq "1"} {exit}
set bd [get_files -all */q6.bd]
set_property synth_checkpoint_mode None $bd
generate_target all $bd
set wrapper [make_wrapper -files $bd -top]
add_files -norecurse $wrapper
set_property top q6_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
set_param general.maxThreads 2
launch_runs synth_1 -jobs 2
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Synthesis failed"}
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation failed"}
open_run impl_1
report_utilization -file [file join $build_dir utilization.rpt]
report_timing_summary -file [file join $build_dir timing.rpt]
# Require both setup and hold closure before exporting deployment artifacts.
foreach delay_type {max min} {
    set worst [get_timing_paths -delay_type $delay_type -max_paths 1]
    if {[llength $worst] == 0 || [get_property SLACK $worst] < 0} {
        error "Timing closure failed ($delay_type); inspect timing.rpt"
    }
}
file copy -force [file join $build_dir q6.runs impl_1 q6_wrapper.bit] [file join $deploy qwen6.bit]
file copy -force [file join $build_dir q6.gen sources_1 bd q6 hw_handoff q6.hwh] [file join $deploy qwen6.hwh]
write_hw_platform -force -fixed -include_bit -file [file join $deploy qwen6.xsa]
puts "QWEN6_DEPLOY: $deploy"
exit
