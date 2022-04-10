##Data import code here.
import os
import pandas as pd
import requests
import pickle
cwd = os.getcwd()

#List of csv collections
collectionCSVs = [
    "0n1_force.csv",
    "axie_infinity.csv",
    "azuki.csv",
    "bored_ape.csv",
    "clone_x.csv",
    "coolmonkes.csv",
    "creature_world.csv",
    "creepz_reptile.csv",
    "creepz.csv",
    "cryptoadz.csv",
    "cryptobatz.csv",
    "cryptokitties.csv",
    "cryptopunks.csv",
    "cryptoskulls.csv",
    "cyberkongz_vx.csv",
    "DeadFellaz.csv",
    "decentraland_wearables.csv",
    "doge_pound.csv",
    "doodles.csv",
    "dr_ETHvil.csv",
    "emblem_vault.csv",
    "FLUF_world_thingies.csv",
    "fomo_mofos.csv",
    "full_send.csv",
    "hape_prime.csv",
    "hashmasks.csv",
    "lil_heroes.csv",
    "lostpoets.csv",
    "meebits.csv",
    "mekaverse.csv",
    "metroverse.csv",
    "mutant_ape.csv",
    "my_curio_cards.csv",
    "phantabear.csv",
    "pudgypenguins.csv",
    "punkcomics.csv",
    "rarible.csv",
    "rtfkt.csv",
    "sorare.csv",
    "superrare.csv",
    "wolf_game.csv",
    "world of women.csv",
    "wvrps.csv",
    "x_rabbits.csv"
]

readIn = ['seller_address']

#Change range depending on whether you are getting data
#for first or second half of list
#for i in range(1, 21):

missingIndices = [11]

for i in missingIndices:
    #Load in the csv
    bored_ape_location = cwd + "/data/" + collectionCSVs[i]
    bored_ape = pd.read_csv(bored_ape_location, usecols = readIn)

    #Infers some column data types to float and int to save some space
    bored_ape_converted = bored_ape.infer_objects()

    #Delete rows without wallet address
    bored_ape_converted.dropna(subset=['seller_address'], inplace=True)

    #Usernames that are listed as NaN are unnamed on OpenSea but still have a wallet attached.

    #Create a dict to store all the transactions
    all_wallet_dict = {}
    count = 1

    for seller in bored_ape_converted['seller_address']:
        
        wallet_dict = {}
        address = seller
        print("Row: " + str(count - 1) + " out of " + str(len(bored_ape_converted.index)))
        
        #Build a simple request on etherscan to scale up for each seller wallet in opensea data
        payload = {'module':'account', 
                   'action':'tokentx', 
                   'address':address,
                   'startblock':'0',
                   'endblock':'99999999',
                   'page':'0',
                   'offset':'0',
                   'sort':'asc',
                   'apikey':'H2V5PTEJQ35UGTIXBZHW5SR37ID9HG2RKR'
                  }

        try:
            r = requests.get('https://api.etherscan.io/api', params=payload)
            
            for transaction in r.json()['result']:
                if transaction['from'] == str(address):
                    wallet_dict[transaction['to']] = 'sent'
                else:
                    wallet_dict[transaction['from']] = 'recieved'
            all_wallet_dict[address] = wallet_dict
            
        except:
            print("===== ERROR ON ROW " + str(count -1) + " =====")
            errorFile, sep, tail = collectionCSVs[i].partition('.')
            errorFile = 'errors/' + errorFile + '_ErrorRows.txt'
            with open(errorFile, 'a+') as f:
                f.write('\n' + str(count - 1))

        finally:
            count = count + 1

    #Write collection dictionary to pickle file
    outFile, sep, tail = collectionCSVs[i].partition('.')
    outFile = outFile + '.pkl'
    with open(outFile, 'wb+') as f:
        pickle.dump(all_wallet_dict, f)
    print("File saved: " + outFile)