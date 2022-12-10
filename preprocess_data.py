import os
import numpy as np
import pandas as pd

def uploads_training_data():
    
    path = "/Users/edubarros/Desktop/total"
    frames = []
    packets = []
    byts = []
    flags = []
    labels = []

    for i in os.listdir(path): # Lista todos os flows
        if ".csv" == os.path.splitext(i)[1]: # Apenas aceita os .csv
            df1 = pd.read_csv(path + "/" + i, dtype=str)
            frames.append(df1)

    concatenated = pd.concat(frames) #print(concatenated.columns.values)    

    #Choose the headers that i need
    concatenated = concatenated[[" Source IP", " Destination IP", " Source Port", " Destination Port", " Protocol", " Flow Duration", " Total Fwd Packets", " Total Backward Packets", "Total Length of Fwd Packets", " Total Length of Bwd Packets", "FIN Flag Count", " SYN Flag Count", " RST Flag Count", " PSH Flag Count", " ACK Flag Count", " URG Flag Count", " CWE Flag Count", " ECE Flag Count", " Label"]]
    
    #Preprocess "Packets" data: Total Fwd Packets + Total Backward Packets
    fwd_packets = concatenated.loc[ : , ' Total Fwd Packets' ]
    fwd_packets = fwd_packets.values.tolist()
    bkw_packets = concatenated.loc[ : , ' Total Backward Packets' ]
    bkw_packets = bkw_packets.values.tolist()

    for x in range(0, len(fwd_packets)):
        packets.append(float(fwd_packets[x]) + float(bkw_packets[x]))

    #Preprocess "Bytes" data: Total Length of Fwd Packets + Total Length of Bwd Packets
    fwd_bytes = concatenated.loc[ : , 'Total Length of Fwd Packets' ]
    fwd_bytes = fwd_bytes.values.tolist()
    bkw_bytes = concatenated.loc[ : , ' Total Length of Bwd Packets' ]
    bkw_bytes = bkw_bytes.values.tolist()

    for x in range(0, len(fwd_bytes)):
        byts.append(float(fwd_bytes[x]) + float(bkw_bytes[x]))

    #Preprocess "Label" data:
    labels_lst = concatenated.loc[ : , ' Label' ]
    labels_lst = labels_lst.values.tolist()

    for x in range(0, len(labels_lst)):
        if "BENIGN" in labels_lst[x]:
            labels.append(0)
        else:
            labels.append(1)

    #Preprocess "Flags" data:
    FIN = concatenated.loc[ : , 'FIN Flag Count' ]
    FIN = FIN.values.tolist()
    SYN = concatenated.loc[ : , ' SYN Flag Count' ]
    SYN = SYN.values.tolist()
    RST = concatenated.loc[ : , ' RST Flag Count' ]
    RST = RST.values.tolist()
    PSH = concatenated.loc[ : , ' PSH Flag Count' ]
    PSH = PSH.values.tolist()
    ACK = concatenated.loc[ : , ' ACK Flag Count' ]
    ACK = ACK.values.tolist()
    URG = concatenated.loc[ : , ' URG Flag Count' ]
    URG = URG.values.tolist()
    CWE = concatenated.loc[ : , ' CWE Flag Count' ]
    CWE = CWE.values.tolist()
    ECE = concatenated.loc[ : , ' ECE Flag Count' ]
    ECE = ECE.values.tolist()

    for x in range(0, len(FIN)):
        tmp_flag = ""
        if "1" in FIN[x]:
            tmp_flag += "F"
        if "1" in SYN[x]:
            tmp_flag += "S"
        if "1" in RST[x]:
            tmp_flag += "R"
        if "1" in PSH[x]:
            tmp_flag += "P"
        if "1" in ACK[x]:
            tmp_flag += "A"
        if "1" in URG[x]:
            tmp_flag += "U"
        if "1" in CWE[x]:
            tmp_flag += "C"
        if "1" in ECE[x]:
            tmp_flag += "E"

        flags.append(tmp_flag)
    
    #Creates 3 new fields
    concatenated = concatenated.assign(bytes = byts, packets = packets, flags = flags, is_malicious = labels)
    #Drops outdated columns
    concatenated = concatenated.drop(columns=[' Total Fwd Packets', ' Total Backward Packets', "Total Length of Fwd Packets", " Total Length of Bwd Packets", "FIN Flag Count", " SYN Flag Count", " RST Flag Count", " PSH Flag Count", " ACK Flag Count", " URG Flag Count", " CWE Flag Count", " ECE Flag Count", " Label"])
    #Rename the columns
    concatenated = concatenated.rename(columns={" Source IP": "sIP", " Destination IP": "dIP", " Source Port": "sPort", " Destination Port": "dPort", " Protocol": "pro", " Flow Duration": "duration"})
    
    #concatenated.to_csv("/Users/edubarros/Desktop/tmp5.csv")

    return concatenated
