# Program the connected ZCU106 with the packaged shared-PE YOLO/TCAM overlay.

set script_dir [file dirname [file normalize [info script]]]
set repo_dir   [file normalize [file join $script_dir ".."]]
set bit_file   [file normalize [file join $repo_dir "deploy" "tcam_yolo_shared_pe" "overlay" "zcu106_axi_x2_mvm.bit"]]

if {![file exists $bit_file]} {
    error "Bitstream not found: $bit_file"
}

open_hw_manager
connect_hw_server
open_hw_target

set devices [get_hw_devices -quiet]
set matching_devices {}
foreach device $devices {
    set device_name [get_property NAME $device]
    set device_part [get_property PART $device]
    puts "INFO: Detected JTAG device $device_name (part $device_part)"
    if {[string match -nocase "xczu7*" $device_name] ||
        [string match -nocase "xczu7*" $device_part]} {
        lappend matching_devices $device
    }
}

if {[llength $matching_devices] == 0} {
    error "No xczu7 device found. Check that the ZCU106 is powered and JTAG is connected."
}

set dev [lindex $matching_devices 0]
current_hw_device $dev
refresh_hw_device $dev
set_property PROGRAM.FILE $bit_file $dev
program_hw_devices $dev

puts "INFO: Programmed $dev with $bit_file"
