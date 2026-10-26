if {$argc != 1} {
    puts "Usage: vivado -mode batch -source inspect_axi_reader_arsize.tcl -tclargs <routed.dcp>"
    exit 2
}

set dcp [file normalize [lindex $argv 0]]
open_checkpoint $dcp

puts "ARSIZE_NETS_BEGIN"
foreach net [get_nets -hierarchical -quiet -filter {NAME =~ "*arsize*"}] {
    puts "NET $net"
    foreach pin [get_pins -quiet -of_objects $net] {
        puts "  PIN $pin REF=[get_property -quiet REF_PIN_NAME $pin] DIR=[get_property -quiet DIRECTION $pin]"
    }
}
puts "ARSIZE_NETS_END"
exit
