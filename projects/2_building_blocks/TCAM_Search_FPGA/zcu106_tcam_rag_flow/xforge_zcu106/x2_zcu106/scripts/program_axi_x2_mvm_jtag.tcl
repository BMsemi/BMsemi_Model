set script_dir [file dirname [file normalize [info script]]]
set root_dir [file dirname $script_dir]
set bit_file [file normalize [file join $root_dir deploy zcu106_axi_x2_mvm.bit]]

if {![file exists $bit_file]} {
    puts "ERROR: Missing bitstream: $bit_file"
    exit 2
}

open_hw_manager
connect_hw_server -allow_non_jtag

set targets [get_hw_targets -quiet *]
if {[llength $targets] == 0} {
    puts "ERROR: No JTAG hardware targets found. Check USB/JTAG cable and board power."
    exit 3
}

open_hw_target [lindex $targets 0]

set devices [get_hw_devices -quiet]
puts "JTAG_DEVICES: $devices"

set device ""
foreach d $devices {
    if {[string match -nocase "*xczu7ev*" $d] || [string match -nocase "*xcz?7ev*" $d]} {
        set device $d
        break
    }
}

if {$device eq ""} {
    foreach d $devices {
        set part_name [get_property PART $d]
        if {[string match -nocase "*xczu7ev*" $part_name]} {
            set device $d
            break
        }
    }
}

if {$device eq "" && [llength $devices] > 0} {
    set device [lindex $devices 0]
    puts "WARNING: Could not identify xczu7ev by name; using first JTAG device: $device"
}

if {$device eq ""} {
    puts "ERROR: No programmable JTAG devices found."
    exit 4
}

current_hw_device $device
refresh_hw_device $device
set_property PROGRAM.FILE $bit_file $device

puts "PROGRAMMING_DEVICE: $device"
puts "PROGRAM_FILE: $bit_file"
program_hw_devices $device
refresh_hw_device $device

set programmed_file [get_property -quiet PROGRAM.FILE $device]
puts "JTAG_PROGRAM_FILE: $programmed_file"

foreach property_name {
    REGISTER.IR.BIT5_DONE
    REGISTER.CONFIG_STATUS.BIT13_DONE_INTERNAL_SIGNAL
} {
    set property_value [get_property -quiet $property_name $device]
    if {$property_value ne ""} {
        puts "JTAG_STATUS: $property_name $property_value"
    }
}

puts "JTAG_PROGRAM_OK: $device"
exit
