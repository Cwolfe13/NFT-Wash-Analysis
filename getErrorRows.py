import pandas as pd
import requests
import pickle
import os
cwd = os.getcwd()

readIn = ['seller_address']

collection = input("Enter name of collection: ")

# Read in seller addrs, drop empty rows
csvLocation = cwd + "/data/" + collection + ".csv"
if (not os.path.exists(csvLocation)):
    print("ERROR: " + csvLocation + " DOES NOT EXIST")
    quit()
csv = pd.read_csv(csvLocation, usecols = readIn)
csv.dropna(subset=['seller_address'], inplace=True)

# Load in pickle file containing dict of seller transactions
pickleLocation = cwd + "/" + collection + ".pkl"
if (not os.path.exists(pickleLocation)):
    print("ERROR: " + pickleLocation + " DOES NOT EXIST")
    quit()
inFile = open(pickleLocation, "rb")
sellerTxns = pickle.load(inFile)
inFile.close()

# Load in rows which request failed
errorLocation = cwd + "/errors/" + collection + "_ErrorRows.txt"
if (not os.path.exists(errorLocation)):
    print("ERROR: " + errorLocation + " DOES NOT EXIST")
    quit()
file1 = open(errorLocation, 'r')
Lines = file1.readlines()
file1.close()
 
for line in Lines:

    # Dict to hold individual seller txns, gets added to dict in pickle file
    wallet_dict = {}

    if line.strip() != "":
        sellerAddr = csv.iloc[int(line.strip())]['seller_address']
        #print(sellerAddr)

        if sellerAddr in sellerTxns:
            print("ADDRESS " + sellerAddr + " ON ROW " + line.strip() + " ALREADY IN DICT!")
        else:
            payload = {'module':'account', 
                'action':'tokentx', 
                'address':sellerAddr,
                'startblock':'0',
                'endblock':'99999999',
                'page':'0',
                'offset':'0',
                'sort':'asc',
                'apikey':'H2V5PTEJQ35UGTIXBZHW5SR37ID9HG2RKR'
                }
            r = requests.get('https://api.etherscan.io/api', params=payload)

            for transaction in r.json()['result']:
                if transaction['from'] == sellerAddr:
                    wallet_dict[transaction['to']] = 'sent'
                else:
                    wallet_dict[transaction['from']] = 'recieved'
            
            sellerTxns[sellerAddr] = wallet_dict
            print(sellerAddr + " wallet transactions added to dictionary")

outFile = collection + ".pkl"
with open(outFile, 'wb+') as f:
    pickle.dump(sellerTxns, f)
print("File saved: " + outFile)