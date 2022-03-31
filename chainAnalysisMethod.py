import pandas as pd
import pickle
import os
cwd = os.getcwd()

collections = [
    "0n1_force",
    "axie_infinity",
    "azuki",
    "bored_ape",
    "clone_x",
    "coolmonkes",
    "creature_world",
    "creepz_reptile",
    "creepz",
    "cryptoadz",
    "cryptobatz",
    "cryptokitties",
    "cryptopunks",
    "cryptoskulls",
    "cyberkongz_vx",
    "DeadFellaz",
    "decentraland_wearables",
    "doge_pound",
    "doodles",
    "dr_ETHvil",
    "emblem_vaul",
    "FLUF_world_thingies",
    "fomo_mofos",
    "full_send",
    "hape_prime",
    "hashmasks",
    "lil_heroes",
    "lostpoets",
    "meebits",
    "mekaverse",
    "metroverse",
    "mutant_ape",
    "my_curio_cards",
    "phantabear",
    "pudgypenguins",
    "punkcomics",
    "rarible",
    "rtfkt",
    "sorare",
    "superrare",
    "wolf_game",
    "world of women",
    "wvrps",
    "x_rabbits"
]

# Change range depending on which collections you want to test
for i in range(8,21):

    collection = collections[i]

    # Load in pickle file containing dict of seller transactions
    pickleLocation = cwd + "/" + collection + ".pkl"
    if (not os.path.exists(pickleLocation)):
        print("ERROR: " + pickleLocation + " DOES NOT EXIST")
        quit()
    inFile = open(pickleLocation, "rb")
    sellerTxns = pickle.load(inFile)
    inFile.close()

    # Read in buyers & sellers associated with each transaction
    readIn = ['winner_account_address', 'seller_address']
    buyersLocation = cwd + "/data/" + collection + ".csv"
    if (not os.path.exists(buyersLocation)):
        print("ERROR: " + buyersLocation + " DOES NOT EXIST")
        quit()
    nftSales = pd.read_csv(buyersLocation, usecols = readIn)
    nftSales.dropna(subset=['seller_address'], inplace=True)

    # List to store buyers that received ETH from sellers
    # Stored as buyer, seller
    buyersSellers = []

    for index, row in nftSales.iterrows():
        for addr in sellerTxns[row['seller_address']]:

            # If the address from txn list is the one who bought the NFT and received ETH...
            if(addr == row['winner_account_address'] and
            sellerTxns[row['seller_address']][addr] == "sent"):
                    buyersSellers.append(tuple((addr, row['seller_address'])))


# Write list of suspicious buyers/sellers to file
    outFileLoc = cwd + "/chain_analysis_results/" + collection + "_results.txt" 
    outFile = open(outFileLoc, "w+")
    for pair in buyersSellers:
        line = ' '.join(str(tmp) for tmp in pair)
        outFile.write(line + '\n')
    if len(buyersSellers) > 0:
        print("/chain_analysis_results/" + collection + "_results.txt successfully written")
    else:
        print(collection + " sellers did not send ETH to buyers")
    outFile.close()