connect
puts "INFO: Connected to XSDB targets"
targets
catch {rst -system} result
puts "INFO: rst -system result: $result"
after 5000
disconnect
exit
