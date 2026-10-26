set script_dir [file dirname [file normalize [info script]]]
set root_dir [file dirname $script_dir]

set build_dir [file normalize "C:/fpga_build/yolo26n_axi_input_reader_sim"]
set src_dir [file join $build_dir src]
file mkdir $build_dir
file mkdir $src_dir

set rtl_src [file join $root_dir rtl axi_x2_mvm_accel.sv]
set tb_src [file join $root_dir sim tb_axi_input_word_reader.sv]
set rtl_file [file join $src_dir axi_x2_mvm_accel.sv]
set tb_file [file join $src_dir tb_axi_input_word_reader.sv]

file copy -force $rtl_src $rtl_file
file copy -force $tb_src $tb_file

create_project -force axi_input_word_reader_sim $build_dir -part xczu7ev-ffvc1156-2-e
set_property target_language Verilog [current_project]
set_property target_simulator XSim [current_project]

add_files -norecurse [list $rtl_file]
set_property file_type SystemVerilog [get_files $rtl_file]

add_files -fileset sim_1 -norecurse [list $tb_file]
set_property file_type SystemVerilog [get_files $tb_file]
set_property top tb_axi_input_word_reader [get_filesets sim_1]
update_compile_order -fileset sim_1

launch_simulation -simset sim_1 -mode behavioral
run all
close_sim
puts "SIM_OK: tb_axi_input_word_reader"
exit
