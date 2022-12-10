import os
import numpy as np
import pandas as pd
import ipaddress
from ipwhois.net import Net
from ipwhois.asn import IPASN
import socket
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

''' ************* Notes *************

## SPAMHAUS FEED
https://www.spamhaus.org/statistics/botnet-isp/

## FASTER WHOIS 
https://team-cymru.com/community-services/ip-asn-mapping/

## TIME SERIES
https://towardsdatascience.com/mastering-time-series-analysis-in-python-8219047a0351

## DATASET
https://www.unb.ca/cic/datasets/ddos-2019.html
http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/
************* Notes *************''' 

''' ************* Lookup Table *************

	1- AS numbers
	2- Subnets
	3- Port frequency

************* Lookup Table ************* '''

def generate_array(flow_path):
	#Uploads file to memory and normalizes it
	header = []
	lines = []
	text_file = open(flow_path, "r")
	for i, line in enumerate(text_file.readlines()):
		line = line.replace(" ","")
		line = line.split("|")
		line = line[:-1]
		if i == 0:
			header = line
		else:
			lines.append(line)
	text_file.close()

	#Converts lists in Dataframe
	df = pd.DataFrame(lines, columns=header) #print(df.to_string()) #print(df.columns.values)

	return df

def data_normalization(df, labels):

	## IP ##
	sIP = df.loc[ : , 'sIP' ]
	dIP = df.loc[ : , 'dIP' ]
	dIP = dIP.values.tolist()

	subnets = []
	AS_numbers = []
	int_ipv6_srcs = []
	int_ipv6_dsts = []
	int_ipv4_srcs = []
	int_ipv4_dsts = []

	for i, IP in enumerate(sIP):

		#IPv6
		if ":" in IP:
			int_ipv6_src = int(ipaddress.ip_address(IP)) # This int conversion allready gives diferent weights based on the octet position
			int_ipv6_dst = int(ipaddress.ip_address(dIP[i]))
			int_ipv4_src = 0
			int_ipv4_dst = 0
			#ipv6_src = str(ipaddress.ip_address(int_ipv6_src))
			#ipv6_dst = str(ipaddress.ip_address(int_ipv6_dst))
		#IPv4 
		else:
			int_ipv4_src = int(ipaddress.ip_address(IP))
			int_ipv4_dst = int(ipaddress.ip_address(dIP[i]))
			int_ipv6_src = 0
			int_ipv6_dst = 0

		int_ipv6_srcs.append(int_ipv6_src)
		int_ipv6_dsts.append(int_ipv6_dst)
		int_ipv4_srcs.append(int_ipv4_src)
		int_ipv4_dsts.append(int_ipv4_dst)

		## Calculate src IP Subnet and AS 
		try:
			if ipaddress.ip_address(IP).is_private == 0:
				net = Net(IP)
				ipwhois_obj = IPASN(net)
				asn_info = ipwhois_obj.lookup()
				subnet = asn_info["asn_cidr"]
				AS_number = asn_info["asn"]
			else:   
				subnet = ""
				AS_number = ""
		except:
			subnet = ""
			AS_number = ""
			print("Sleeping for 3 hours because whois rate limit exceeded")
			time.sleep(11111)

		subnets.append(subnet)
		AS_numbers.append(AS_number)

	#Add Subnet value and AS value to new array
	df2 = df.assign(subnet = subnets, AS_number = AS_numbers)

	## Calculate Subnet and AS frequency ## 
	subnets = df2.loc[ : , 'subnet' ]
	subnets = subnets.values.tolist() #print(subnets)
	updates_lookup_table_and_returns_frequency_array("subnet", subnets) #print(subnets)
	
	AS_numbers = df2.loc[ : , 'AS_number' ]
	AS_numbers = AS_numbers.values.tolist() #print(AS_numbers)
	updates_lookup_table_and_returns_frequency_array("as", AS_numbers) #print(AS_numbers)

	## Ports ##
	sPort = df.loc[ : , 'sPort' ]
	sPort = sPort.values.tolist()
	dPort = df.loc[ : , 'dPort' ]
	dPort = dPort.values.tolist()

	sPort_frequency = df.loc[ : , 'sPort' ]
	sPort_frequency = sPort_frequency.values.tolist()
	dPort_frequency = df.loc[ : , 'dPort' ]
	dPort_frequency = dPort_frequency.values.tolist()

	## Port frequency ##
	updates_lookup_table_and_returns_frequency_array("sport", sPort_frequency) #print(sPort_frequency)
	updates_lookup_table_and_returns_frequency_array("dport", dPort_frequency) #print(dPort_frequency)

	## Protocol frequency ##
	protocol = df.loc[ : , 'pro' ]
	protocol = protocol.values.tolist() #print(protocol)
	updates_lookup_table_and_returns_frequency_array("protocol", protocol) #print(protocol)
 
	## Packet Count ## 
	packet_count = df.loc[ : , 'packets' ]
	packet_count = packet_count.values.tolist()

	## Bytes Count ##
	byte_count = df.loc[ : , 'bytes' ]
	byte_count = byte_count.values.tolist()

	## Flow Duration Time #
	duration = df.loc[ : , 'duration' ]
	duration = duration.values.tolist()

	## Cumulative of TCP Flags ##
	flags = df.loc[ : , 'flags' ]

	syn_flag, ack_flag, push_flag, finish_flag, reset_flag, urgent_flag, ECN_echo_flag, congestion_window_flag = ([] for i in range(8))

	for x in flags:
		syn = ack = push = finish = reset = urgent = ECN = congestion = 0

		if "S" in x:
			syn = 1
		if "A" in x:
			ack = 1
		if "P" in x:
			push = 1
		if "F" in x:
			finish = 1
		if "R" in x:
			reset = 1 
		if "U" in x:
			urgent = 1
		if "E" in x:
			ECN = 1
		if "C" in x:
			congestion = 1

		syn_flag.append(syn)
		ack_flag.append(ack)
		push_flag.append(push)
		finish_flag.append(finish)
		reset_flag.append(reset)
		urgent_flag.append(urgent)
		ECN_echo_flag.append(ECN)
		congestion_window_flag.append(congestion)

	#Compiling last data frame and dictionary
	final_dict = {
		'int_ipv6_src': int_ipv6_srcs, #(ip convertido em int)
		'int_ipv6_dst': int_ipv6_dsts, #(ip convertido em int)
		'int_ipv4_src': int_ipv4_srcs, #(ip convertido em int)
		'int_ipv4_dst': int_ipv4_dsts, #(ip convertido em int)
		'subnet': subnets, #(frequência de uma subnet)
		'AS_number': AS_numbers, #(frequência de um AS)
		'dPort': dPort,  #(port em int)
		'sPort': sPort,  #(port em int)
		'sPort_frequency': sPort_frequency, #(frequência de um sPort)
		'dPort_frequency': dPort_frequency, #(frequência de um dPort)
		'protocol': protocol, #(frequência de um protocol)
		'packet_count': packet_count, #(numero de pacotes de um flow em int)
		'byte_count': byte_count, #(numero de bytes de um flow em int)
		'duration': duration, #(tempo de duração de um flow em int)
		'syn_flag': syn_flag, #(tem a flag a 1 - int)
		'ack_flag': ack_flag, #(tem a flag a 1 - int)
		'push_flag': push_flag, #(tem a flag a 1 - int)
		'finish_flag': finish_flag, #(tem a flag a 1 - int)
		'reset_flag': reset_flag, #(tem a flag a 1 - int)
		'urgent_flag': urgent_flag, #(tem a flag a 1 - int)
		'ECN_echo_flag': ECN_echo_flag, #(tem a flag a 1 - int)
		'congestion_window_flag': congestion_window_flag #(tem a flag a 1 - int)
	} 
	
	for k, v in final_dict.items():
		for x, i in enumerate(v):
			if i != "" and len(str(i)) > 6:
				final_dict[k][x] =  f'{float(i):.6f}' #print(type(i)) #print(type(final_dict[k][x]))
			else:
				final_dict[k][x] = float(i)

	#df.to_csv("/Users/edubarros/Desktop/rawflow.csv") print(final_dict)
	final_df = pd.DataFrame(final_dict) #print(final_df.to_string())

	final_df['is_malicious'] = labels.tolist()

	final_df = final_df.drop_duplicates()
	final_df = final_df.astype(float)
		
	final_df.corr().to_csv("/Users/edubarros/Desktop/correlation.csv")
	labels = final_df['is_malicious']
	final_df = final_df.drop(columns=['is_malicious'])

	## Data Standardization ## -> Z SCORE (StandardScaler)
	standard_np = final_df.loc[:, list(final_dict.keys())].values
	standard_np = StandardScaler().fit_transform(standard_np)
	
	#np.savetxt("/Users/edubarros/Desktop/datanormalized_standardized.csv", standard_np, delimiter=",")

	return standard_np, labels

def updates_lookup_table_and_returns_frequency_array(data, new_array):
	path = '/Users/edubarros/Desktop/tables/' + str(data) + '.txt'
	table_memory = {}
	total_len = 0

	#Se a lookup table não existir, ele cria uma de forma automática
	if not os.path.isfile(path):
		for i in new_array:
			if str(i) not in table_memory.keys():
				table_memory[i] = 1
			else:
				table_memory[i] += 1
		#print(table_memory)

		#Write updated values in file 
		table = pd.DataFrame(table_memory.items(), columns=[str(data), 'freq'])
		table.to_csv(path)	

		#Calculate total number
		for i in table_memory.values():
			total_len += i
		#print(total_len)

		#Replace values with their frequency
		for n, i in enumerate(new_array):
			new_array[n] = (table_memory[i] / total_len)

	else:
		#Load values from Lookup Table 
		table = pd.read_csv(path)
		type_of_data = table.iloc[:, 1]
		data_freq = table.iloc[:, 2]
		for i, x in enumerate(type_of_data):
			table_memory[str(x)] = data_freq[i]

		#Update values from Lookup Table
		for i in new_array:
			if str(i) in table_memory.keys():
				table_memory[i] += 1
			else:
				table_memory[i] = 1
		#print(table_memory)

		#Calculate total number
		for i in table_memory.values():
			total_len += i
		#print(total_len)

		#Write updated values in file 
		table = pd.DataFrame(table_memory.items(), columns=[str(data), 'freq'])
		table.to_csv(path)	

		#Replace values with their frequency
		for n, i in enumerate(new_array):
			new_array[n] = (table_memory[i] / total_len)
		#print(new_array)

def PCA_extraction(array):
	#PCA - Reduce x features to 3 features
	print(array.shape)
	pca = PCA(n_components=3)
	principalComponents = pca.fit_transform(array)
	principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'pc2', 'pc3'])
	print(principalDf)