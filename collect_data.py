import os

''' ************* Notes *************

## TCPDUMP
$ tcpdump not src net 10.0.0.0/8 and not src net 172.16.0.0/12 and not src net 192.168.0.0/16 and not src net fd00::/8 and not src net fe80::/10 and not src net MYNET -i en0 -w \%y-\%m-\%d-\%H-\%M-\%S -G 5 -C 50
#G -> Seconds until tcpdump creates next file 
#C -> Or 50 mgb worth of files
#Ignore IPv4 private address and IPv6 (Need to block our pub net ipv6 address in command)

## SILK FLOW
['sIP' 'dIP' 'sPort' 'dPort' 'pro' 'packets' 'bytes' 'flags' 'sTime' 'duration' 'eTime' 'sen']

************* Notes *************''' 

def generate_flows():

	os.chdir("/Users/edubarros/Desktop/Thesis/Probe/")

	#See wich file is the newest
	last_file = max(os.listdir(), key=os.path.getctime)

	#Convert the pcap in .silk 
	os.system("rwp2yaf2silk --in=" + last_file + " --out=" + last_file + ".silk")

	#Convert .silk in human readable
	os.system("rwcut " + last_file + ".silk --out=" + last_file + ".flow")

	#Compute new flow file path
	flow_path = "/Users/edubarros/Desktop/Thesis/Probe/" + last_file + ".flow"

	return flow_path

	tcpdump -i en0 -w \%y-\%m-\%d-\%H-\%M-\%S -G 5 -C 50