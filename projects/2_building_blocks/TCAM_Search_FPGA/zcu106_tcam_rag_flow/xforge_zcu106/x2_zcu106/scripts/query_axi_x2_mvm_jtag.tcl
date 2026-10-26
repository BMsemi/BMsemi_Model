open_hw_manager
connect_hw_server -allow_non_jtag

set targets [get_hw_targets -quiet *]
puts "JTAG_TARGET_COUNT: [llength $targets]"

foreach target $targets {
    puts "JTAG_TARGET: $target"
    open_hw_target $target

    foreach device [get_hw_devices -quiet] {
        set part_name [get_property -quiet PART $device]
        puts "JTAG_DEVICE: $device PART: $part_name"
    }

    close_hw_target
}

disconnect_hw_server
close_hw_manager
exit
