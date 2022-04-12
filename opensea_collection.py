#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import powerlaw
import math
from math import log10
import pickle
from prettytable import PrettyTable
from collections import Counter

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


class collection():
    """The collection class is meant to act as a holder, the idea is that for each of the CSVs above, we can
    load them in by name, all of the necessary calculations can be done on initialization, and later
    on when we want some useful visualizations of the data at the end we can call the proper methods.
    
    Params
    ------
    name - The name of the collection from the above list of csvs.
    
    Methods
    -------
    """
    def __init__(self, name):
        self.name = name
        self.cwd = os.getcwd()
        directory = self.cwd + '/data/' + self.name
        touseCols = ['payment_token_id', 
                     'total_price', 
                     'payment_token_decimals', 
                     'payment_token_usd_price']
        
        use_dtypes = {'payment_token_id':'float', 'total_price':'float', 
                       'payment_token_decimals':'float', 'winner_account_address':'string',
                       'payment_token_usd_price':'float', 'seller_address':'string'}
        self.panda = pd.read_csv(directory, usecols=touseCols, dtype = use_dtypes, low_memory=False)
        
        #Now we do the work on it
        self.panda = self.clean_panda(self.panda)
        self.panda['adj_price'] = self.make_adjprice(self.panda)
        self.roundness = self.roundness_check(self.panda['adj_price'])
        self.panda['eth_first_sig'] = self.make_first_sig(self.panda['adj_price'])
        single, tenths, hundreths, thousandths = self.make_eth_clusters(self.panda['adj_price'])
        self.panda['eth_single'] = single
        self.panda['eth_tenths'] = tenths
        self.panda['eth_hundreths'] = hundreths
        self.panda['eth_thousandths'] = thousandths
        self.panda['usd_price'] = self.make_usdprice(self.panda['adj_price'], self.panda['payment_token_usd_price'])
        self._make_usd_first_sig(self.panda['usd_price'])
        self._make_fusd()
        
    def clean_panda(self, dataframe):
        """
        Takes in a Panda dataframe read from an opensea csv, drops bad rows, 
        bundle data, non ETH transactions then deletes original panda from memory.

        Params
        ------
        panda - The panda to take in

        Returns
        -------
        cleaned_dataframe - The cleaned dataframe
        """
        def main(dataframe):
            dataframe = drop_bad_rows(dataframe)
            #dataframe = drop_bundle(dataframe)
            dataframe = drop_nETH(dataframe)
            dataframe.reset_index(inplace=True, drop=True)
            return dataframe
        
        def drop_bad_rows(dataframe):
            ret = dataframe.dropna(subset=['total_price'])
            return ret
        
        def drop_bundle(dataframe):
            bad_data_gather=['world of women.csv', 'wvrps.csv', 'x_rabbits.csv', 
                             'creature_world.csv']
            if self.name not in bad_data_gather:
                return dataframe.iloc[:, 1:150]
            else:
                return dataframe
        
        def drop_nETH(dataframe):
            # What happens if there are no non ETH indices? 
            # Likely needs error handling
            bad_indices = dataframe[(dataframe.payment_token_id != 1) & (dataframe.payment_token_id != 2)].index
            ret = dataframe.drop(bad_indices)
            return ret
        
        return main(dataframe)
            
    def make_adjprice(self, dataframe):
        """
        Makes the adj_price column to append to the dataframe in init
        
        params
        ------
        dataframe - The dataframe that the method will use to calculate
        adj_price
        
        returns
        -------
        adj_price - The float representation of the ETH price
        """
        #Make an adj_price column to represent ETH price
        adj_price = dataframe.apply(lambda row: float(row.total_price) / (10**row.payment_token_decimals), axis = 1)
        return adj_price
    
    def roundness_check(self, adj_prices):
        """"
        Roundness check returns the counts of all last sig figs for a collection.
        It uses the last_sig_fig member function do this.
        
        params
        ------
        adj_prices - The adj_price column of the dataframe you are attempt to get
        the values for
        
        returns
        -------
        returndict - Dict with keys of unique last sig fig and values as counts
        """
        def last_sig_fig(number):
            """
            Returns an integer indicating how many places after the decimal the last significant digit is. 1 returned is considered to be 
            the tenths place, while -1 would indicate the ones place
            
            Params
            ------
            number - the number to find the last sigfig of
            
            Returns
            -------
            The integer representation of the last sigfig's place
            """
            
            if type(number) != float and type(number) != int:
                raise ValueError(f'{number} is neither a float or int')
            
            #Convert to string to use indexing
            strnum = str(number)
            
            def calc_ones(strnum):
                last_sig_index = -9999
                #Strip decimal if we have one
                if '.' in strnum:
                    strnum = strnum[0:strnum.rfind('.')]
                #Get length -1 for range
                length = len(strnum)
                for i in range(0, length):
                    if strnum[i] != '0':
                        last_sig_index = i
                if last_sig_index != -9999:
                    return -(len(strnum) - last_sig_index)
                else:
                    #Edge case where we're processing 0.0
                    return -1
                
            #0 and 1 can both be confused in boolean expressions, uses a decided 'null' value instead.
            last_sig_index = -9999
            if '.' in strnum:
                #rfinds gets the index of last .
                dec_loc = strnum.rfind('.')
                for i in range(dec_loc+1, (len(strnum))):
                    #Zeros after the . will never be the last sigfig
                    if strnum[i]!='0':
                        last_sig_index = i
                #We found sigfig past decimal
                if last_sig_index != -9999:
                    #Return the last occurence, but first calculate the place:
                    return last_sig_index-dec_loc
            #If there isnt a decimal, or we didn't find a sigfig past
            return calc_ones(strnum)
        
        #Trying to improve performance
        counts = []
        for adj_price in adj_prices:
            #Find the last significant digit
            place = last_sig_fig(adj_price)
            #counts = np.append(counts, place)
            counts.append(place)
        #Get all uniques, and their values, add to a dict
        returndict = Counter(counts)
        return returndict
    
    def make_first_sig(self, adj_price):
        """
        Makes the first_sig_fig series for a dataframe
        
        params
        ------
        adj_price - The adj_price series of the dataframe
        
        returns
        -------
        series - The first sig fig series for the dataframe
        """
        
        def first_sig_fig(number):
            """
            Returns the first significant digit of a provided number as string
            
            Parameters
            ----------
            number: The number whose first significant digit will be returned
            
            Raises
            ------
            TypeError: If the provided variable is not a number a TypeError will be raised
            """
            #Check that what is provided is actually a number
            if type(number) != int and type(number) != float and isinstance(number, np.ndarray) == False:
                raise TypeError(f"{number} is not a number, it is of type {type(number)}")
            
            #Turn number into string so that it's iterable
            snumber = str(number)
            
            #Sentinel value to determine if we've hit the decimals yet.
            decimal_encountered = False
            for i in range(0,len(snumber)):
                if snumber[i].isdigit():
                    temp = snumber[i]
                    if snumber[i] == '0':
                        pass
                    else:
                        return snumber[i]
                else:
                    pass
        
        #Build the series to return with the function
        series = []
        for i in adj_price:
            #Call the member function and append to a series list.
            series.append(first_sig_fig(i))
        return series
    
    #TODO: DELETE
    def make_second_sig(self, adj_price):
        def second_sig_fig(number, commFSD):
            """
            Returns the first significant digit of a provided number as string
            
            Parameters
            ----------
            number: The number whose first significant digit will be returned
            
            Raises
            ------
            TypeError: If the provided variable is not a number a TypeError will be raised
            """
            #Check that what is provided is actually a number
            if type(number) != int and type(number) != float and isinstance(number, np.ndarray) == False:
                raise TypeError(f"{number} is not a number, it is of type {type(number)}")
            
            #Turn number into string so that it's iterable
            snumber = str(number)
            
            #Sentinel value to determine if we've hit the first sig dig yet.
            for i in range(0,len(snumber)):
                # If digit is most common FSD from collection, find SSD
                if snumber[i] == str(commFSD):
                    for j in range(i + 1, len(snumber)):
                        
                        if snumber[j].isdigit():
                            if snumber[j] == '0':
                                pass
                            else:
                                return snumber[j]
                        else:
                            pass
                # Not the FSD we're looking for
                elif snumber[i] != '0' and snumber[i] != '.':
                    break
                
            # Only one sig fig, return 0, skip over it when doing analysis
            return str(0)
        
        #Build the series to return with the function
        series = []

        vals = self.panda['eth_first_sig'].value_counts().sort_index()
        maxVal = max(vals)
        index = 0
        commonFSD = 0
        for val in vals:
            if val == maxVal:
                commonFSD = index + 1
                break
            index = index + 1
    
        for i in adj_price:
            series.append(second_sig_fig(i, commonFSD))
        return series

    def make_eth_clusters(self, adj_price):
        """
        Takes in the adj price representing the total ETH/WETH traded, which is
        stored in the panda from opensea csv, returns two columns to be added by using the float round function.
        note: Uses data generated from make_adj_price
        
        Params
        ------
        adj_price - The adj price series in a panda representing the ETH/WETH amounts
        
        Returns
        -------
        As a tuple ('single_digit, tenths, hundreths')
        single_digit - A series containing the ETH/WETH as a integer
        tenths - A series containing the ETH/WETH as a float and one decimal place
        hundreths - A series containing the ETH/WETH as float and two decimal places
        thousandths - A series containing the ETH/WETH as a float and three decimal places
        """
        
        #Single round
        single_digit = []
        for i in adj_price:
            single_digit.append(round(i))
        
        #Tenths
        tenths = []
        for i in adj_price:
            tenths.append(round(i, 1))
        
        #Hundreths
        hundreths = []
        for i in adj_price:
            hundreths.append(round(i, 2))
            
        #thousandths
        thousandths = []
        for i in adj_price:
            thousandths.append(round(i, 3))
        
        return single_digit, tenths, hundreths, thousandths
    
    def make_usdprice(self, adj_price, payment_token_eth_price):
        """
        Generates the USD price for the panda using pythons built in zip class
        
        Params
        ------
        
        Returns
        -------
        """
        series = []
        #Zip the values into a tuple
        for i, j in zip(adj_price, payment_token_eth_price):
            #Multiply the values in the tuple and append to a list.
            series.append(i*j)
        return series

    def plot_eth_fsd(self):
        """
        Plots the benford standard in ETH/WETH
        I think its important that this function is also recreated using
        USD. While benford distribution might not be followed with coin amount,
        benford distribution may be closer in USD.
        
        Params
        ------
        
        Returns
        -------
        """
        
        #Get the number of times each unique first sig fig occurs in the data
        values = self.panda['eth_first_sig'].value_counts().sort_index()
        #Make a list to hold the percentages
        percentages = []
        
        for i in values:
            #Get the percentage rather than the count
            percentages.append((i/sum(values))*100)
        
        #List for the xticks
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        #Make it look nice
        plt.style.use('seaborn')
        #Make the bar chart
        plt.bar(x,percentages, width=0.75)
        #Set the ticks
        plt.xticks(x)
        
        #This handles overlaying the benford standard dots
        benford_standard = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
        plt.scatter(x, benford_standard, c='black')
        #Set the labels
        plt.xlabel('First Significant Digit')
        plt.ylabel('Percentage')
        
        #Name the chart for the appendix
        plt.title(self.name)
        plt.show()

    def plot_eth_ssd(self):
        """
        Plots the benford standard in ETH/WETH
        I think its important that this function is also recreated using
        USD. While benford distribution might not be followed with coin amount,
        benford distribution may be closer in USD.
        
        Params
        ------
        
        Returns
        -------
        """
        

        values = self.panda['eth_second_sig'].value_counts().sort_index()
        percentages = []

        # If ssd value of 0 has been included in values drop it (not applicable to Benford Analysis)
        if values.size == 10:
            #zero = True
            values = values.drop(values.index[0])
        
        for i in values:
                #Get the percentage rather than the count
                percentages.append((i/sum(values))*100)
            
        #X values for chart
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        #Make it look nice
        plt.style.use('seaborn')
        #Plot the bar chart
        plt.bar(x,percentages, width=0.75)
        #Set the xticks
        plt.xticks(x)
        
        #This handles overlaying the benford standard dots
        benford_standard = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
        #Plot the scatter plot
        plt.scatter(x, benford_standard, c='black')
        plt.xlabel('Second Significant Digit')
        plt.ylabel('Percentage')
        
        #Probably want to change this for the report
        plt.title(self.name)
        plt.show()

    def plot_usd_fsd(self):
        """
        Plots the beford standard in USD.
        
        Params
        ------
        
        Returns
        -------
        plt - The pyplot to be saved somewhere
        """
        values = self.panda['usd_first_sig'].value_counts().sort_index()
        percentages = []
        for i in values:
            #Get the percentage rather than the count
            percentages.append((i/sum(values))*100)
        
        #Plot the actual percentages and set appropriate ticks
        plt.style.use('seaborn')
        x = [1, 2, 3 , 4, 5, 6, 7, 8, 9]
        plt.bar(x,percentages, width=0.75)
        plt.xticks(x)
        
        #This handles overlaying the benford standard dots
        benford_standard = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
        plt.scatter(x, benford_standard, c='black')
        plt.xlabel('First Significant Digit')
        plt.ylabel('Percentage')
        
        #Probably want to change this for the report
        plt.title(self.name[:-4])
        #plt.savefig('imgs/'+self.name+'_usd_fsd.png')
        plt.show()
        
    def _make_usd_first_sig(self, usd_price):
        def first_sig_fig(number):
            """Returns the first significant digit of a provided number as string
            
            Parameters
            ----------
            number: The number whose first significant digit will be returned
            
            Raises
            ------
            TypeError: If the provided variable is not a number a TypeError will be raised
            """
            #Check that what is provided is actually a number
            if type(number) != int and type(number) != float and isinstance(number, np.ndarray) == False:
                raise TypeError(f"{number} is not a number, it is of type {type(number)}")
            #Turn number into string so that it's iterable
            snumber = str(number)
            #Sentinel value to determine if we've hit the decimals yet.
            decimal_encountered = False
            for i in range(0,len(snumber)):
                if snumber[i].isdigit():
                    temp = snumber[i]
                    if snumber[i] == '0':
                        pass
                    else:
                        return snumber[i]
                else:
                    pass
        series = []
        for i in usd_price:
            series.append(first_sig_fig(i))
        self.panda['usd_first_sig'] = series
    
    def _make_fusd(self):
        """
        Makes a floating point usd column for the panda.
        
        This is the way I should have structured each function
        that was supposed to update the panda, instead of setting
        the value in the initialization of the object, haven't
        changed over all the methods due to time constraints.
        """
        #Would usually return a generator object, by calling list python forces
        #the list comprehension to be saved to memory
        self.panda['fusd_price'] = list(float(usd) for usd in self.panda['usd_price'])
        
    def plot_usd_tail(self):
        """ 
        This method consists of plotting the logarithm of an estimator
        of the probability that a particular number of the distribution occurs 
        versus the logarithm of that particular number. 
        Usually, this estimator is the proportion of times that the number occurs in the data set
        We want to generate the top 10 percent of the data set (aka the tail end of the distribution) 
        
        Params
        ------
        
        Returns
        -------
        """
        length = len(self.panda['adj_price'])
        #Get the top ten percent of trades
        ten_percent = math.floor(length/10)
        top_ten_counts = self.panda['adj_price'].nlargest(ten_percent)
        top_ten_counts = top_ten_counts.value_counts().sort_index()        
        #Calling the powerlaw function attempts to fit the data in top ten counts
        results = powerlaw.Fit(top_ten_counts)
        
        #Optional print statements to determine tail exponent
        #print(f'alpha: {results.power_law.alpha}')
        #print(f'xmin: {results.power_law.xmin}')
        
        #Calling this function evaluates how close the data's distribution was to a powerlaw distribution, 
        #Compared to a lognormal distribution, I was getting very low coefficients for powerlaw distributions
        R, p = results.distribution_compare('power_law', 'lognormal')
        print(f'R: {R} , p: {p}')
        
        #This will plot the value counts on a log log plot
        #Powerlaw distributions usually conform into a straight line,
        #but the scatter plots I was testing with did not, possibly not enough data
        #collection to collection to get a good distribution.
        fig = plt.figure(figsize=(5, 6))
        ax1 = fig.add_subplot()
        plt.loglog()
        plt.scatter(top_ten_counts.index, top_ten_counts, c='r')
        plt.show()
        
        #ax2 = fig.add_subplot()
        #fit = powerlaw.Fit(top_ten_counts)
        #x,y = powerlaw.pdf(top_ten_counts, linear_bins=False)
        #print(f'{len(top_ten_counts)} counts, and x: {x} and y:{y}')
        #ind = y>0
        #y = y[ind]
        #x = x[:-1]
        #x = x[ind]
        #ax1.scatter(x, y, color='r', s=.5)
        ##linear_model=np.polyfit(x,y,1)
        ##linear_model_fn=np.poly1d(linear_model)
        #ax1.plot(x, linear_model_fn)
        #powerlaw.plot_pdf(top_ten_counts[top_ten_counts>0], ax=ax1, color='b', linewidth=2)
        #powerlaw.plot_pdf(sorted(top_ten_counts, reverse=True), ax=ax1, color='b', linewidth=2)
        #plt.show()
        
        # Heres where I was a little confused, calculating the probability 
        # density also can be done by using the probability density function
        # (PDF). This function is calculated by making a histogram of value counts (frequencies),
        # and then the value at any point is calculated by taking the area 
        # underneath a curve that fits the heights of the histogram
        # this is the probability density (PD) which was plotted against trade size
        # in Dr. Li's paper on crypto wash trading.
        # create log bins: (by specifying the multiplier)
        
        """
        What this function was supposed to do can be seen at the following link
        http://www.mkivela.com/binning_tutorial.html
        
        bins = [np.min(self.panda['fusd_price'])]
        cur_value = bins[0]
        multiplier = 2.5
        while cur_value < np.max(values):
            cur_value = cur_value * multiplier
            bins.append(cur_value)
        
        bins = np.array(bins)
        pdf, bins, _ = ax2.hist(values, bins=len(values), density=True)"""
        
        """_ = ax2.scatter(x=top_ten, y=pdv, norm=True)
        ax2.set_title('PDF, log-log, power-law')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('x')
        ax2.set_ylabel('PDF')
        plt.show()"""
    
    def plot_cluster(self, prange):
        """"
        Plots and displays a histogram from 0 to prange,
        also highlights every 5th bin to accentuate clustering.
        
        Params
        ------
        prange - the max range to display as an integer or float
        
        Raises
        ------
        TypeError - if the range provided is not an int or float.
        """
        if not isinstance(prange,int) and not isinstance(prange, float):
            raise TypeError('Range provided must be an int or a float')
        if (prange > max(self.panda.eth_hundreths)):
            raise ValueError(f'The range provided {prange} was greater' +
                             f'than the max transaction size {max(self.panda.eth_hundreths)}')
        #Make the figure
        fig = plt.figure(figsize=(8,6))
        plt.style.use('seaborn')
        ax1 = fig.add_subplot()
        
        category = 'eth_'
        if prange < 10:
            category = category + 'hundreths'
            #true_range = prange + 0.01
        elif prange < 100:
            category = category + 'tenths'
            #true_range = prange + 0.1
        else:
            category = category + 'singles'
            #true_range = prange + 1
        
        #Make the histogram for the figure
        n, bins, patches = ax1.hist(self.panda[category], align='mid', bins=100, range=(0,prange), color='gray')
        
        counter = 0
        print(f'Patches: {patches}')
        for i in range(0, len(patches)):
            if counter % 5 == 0:
                patches[i].set_fc('k')
            counter = counter + 1
        ax1.set_xlabel('Trade Size ETH/USD')
        ax1.set_ylabel('Trade Frequency%')
        plt.xticks(np.arange(0, prange+.01, prange/10))
        plt.show()
        
    def t_test(self, selection):
        """
        Creates a histogram to bin values within the observation window, 
        builds the cluster point frequency and highest frequency of a neighbor 
        within the bounds into a tuple. The tuple is appended to a list and the
        proccess is repeated untill we have the final list of all observations.
        The student t-test is then performed at 1, 5, and 10% levels of 
        significance.
        
        
        Params
        ------
        selection - A string to specify what test you'd like 
        ('all', '100', '500', '1000', '5000')
        
        Raises
        ------
        ValueError - If selection type is not 
        
        Returns
        -------
        """
        valid_selections = ['all', 100, 500, 1000, 5000]
        if selection not in valid_selections:
            raise ValueError(f'Selection {selection} is not a valid option'
                             + f'(\'all\', 100, 500, 1000, 5000)')
        
        max_eth_traded = max(self.panda['adj_price'])
        #Has an upper bound of 100 to limit iterations, but should provide
        #a lengthy sample regardless
        if (max_eth_traded) > 100:
            max_eth_traded = 100
        
        
        #Each iteration should be 100+50 and then repeat 0.01 is the unit 
        def make_t_100():
            its_thousandths = int(max_eth_traded*100)
            lowerbound = 0.005
            upperbound = 0.015
            all_observations=[]
            for i in range(0, its_thousandths):
                # Construct histogram for the observation window
                counts, bins, rects = plt.hist(self.panda['eth_thousandths'],
                                               bins=10, 
                                               range=(lowerbound, upperbound),
                                               align='mid')
                # sum taken beforehand because a count is reset to find the 2nd max
                all_counts = sum(counts)
                
                #Not a valid window, get to the next iteration
                if all_counts == 0:
                    lowerbound = lowerbound + .010
                    upperbound = upperbound + .010
                    continue
                
                
                #Calculate the percentage of a cluster
                cluster_freq = round((counts[5]/all_counts*100), 2)
                # Cluster value accounted for, reset to get 2nd max
                counts[5] = 0.0
                
                #Calculate the 2nd highest index
                index_highest_number = counts.argmax() #argmax is numpys index() function
                #Make the offender piece of the tuple for testing and visualization
                offender = lowerbound + index_highest_number
                offender = round(offender, 3)
                
                #Get the 2nd max
                highest_neighbor = round((max(counts)/all_counts*100), 2)
                #Build a tuple and add it to all views
                this_observation = (cluster_freq, highest_neighbor, offender, all_counts)
                all_observations.append(this_observation)
                #Increment the lower and upper
                lowerbound = lowerbound + .010
                upperbound = upperbound + .010
                #Repeat
            plt.cla()
            self.t_100_observations = all_observations
                
        #Each iteration should be 500+100  0.05 is the unit.
        #Holding off on making this, just not that many thousandth transactions
        def make_t_500():
            its_thousandths = int((max_eth_traded*10)/2)
            #print(f'iterations of histograms: {its_tenths}\n')
            lowerbound = 0.040
            upperbound = 0.060
            all_observations=[]
            for i in range(0, its_thousandths):
                
                # Construct histogram for the observation window
                counts, bins, rects = plt.hist(self.panda['eth_thousandths'],
                                               bins=20,
                                               range=(lowerbound, upperbound),
                                               align='mid')
                
                # sum taken beforehand because a count is reset to find the 2nd max
                all_counts = sum(counts)
                # What happens if no transactions in a window?
                # Not a valid window, get to the next iteration
                if all_counts == 0:
                    lowerbound = lowerbound + .050
                    upperbound = upperbound + .050
                    continue
                # Calculate the percentage of the cluster
                cluster_freq = round((counts[10]/all_counts*100), 2)
                # This value accounted for, reset to be able to grab max
                counts[10] = 0.0
                
                #Calculate the 2nd highest index
                index_highest_number = counts.argmax() #argmax is numpys index() function
                #Make the offender piece of the tuple for testing
                offender = lowerbound + index_highest_number
                offender = round(offender, 2)
                
                #Get the 2nd max
                highest_neighbor = round((max(counts)/all_counts*100), 2)
                
                #Build a tuple and add it to all views
                this_observation = (cluster_freq,
                                    highest_neighbor,
                                    offender,
                                    all_counts)
                all_observations.append(this_observation)
                
                #Increment the lower and upper
                lowerbound = lowerbound + .050
                upperbound = upperbound + .050
                #Repeat
            plt.cla()
            self.t_500_observations = all_observations
        
        #Each iteration should be at 1000+500 0.1 is the unit
        def make_t_1000():
            its_tenths = int(max_eth_traded*10)
            lowerbound = 0.05
            upperbound = 0.15
            all_observations=[]
            for i in range(0, its_tenths):
                
                # Construct histogram for the observation window
                counts, bins, rects = plt.hist(self.panda['eth_hundreths'],
                                               bins=10,
                                               range=(lowerbound, upperbound),
                                               align='mid')
                
                # sum taken beforehand because a count is reset to find the 2nd max
                all_counts = sum(counts)
                # What happens if no transactions in a window?
                # Not a valid window, get to the next iteration
                if all_counts == 0:
                    lowerbound = lowerbound + .10
                    upperbound = upperbound + .10
                    continue
                # Calculate the percentage of the cluster
                cluster_freq = round((counts[5]/all_counts*100), 2)
                # This value accounted for, reset to be able to grab max
                counts[5] = 0.0
                
                #Calculate the 2nd highest index
                index_highest_number = counts.argmax() #argmax is numpys index() function
                #Make the offender piece of the tuple for testing
                offender = lowerbound + index_highest_number
                offender = round(offender, 2)
                
                #Get the 2nd max
                highest_neighbor = round((max(counts)/all_counts*100), 2)
                
                #Build a tuple and add it to all views
                this_observation = (cluster_freq,
                                    highest_neighbor,
                                    offender,
                                    all_counts)
                all_observations.append(this_observation)
                
                #Increment the lower and upper
                lowerbound = lowerbound + .10
                upperbound = upperbound + .10
                #Repeat
            plt.cla()
            self.t_1000_observations = all_observations
        
        #Each iteration should be at 5000+1000 0.5 is the unit.
        def make_t_5000():
            
            its_5tenths = int((max_eth_traded*10)/5)
            
            #The first value we want to obeserve is 0.5
            lowerbound = 0.40
            upperbound = 0.60
            
            all_observations=[]
            for i in range(0, its_5tenths):
                
                # Construct histogram for the observation window
                counts, bins, rects = plt.hist(self.panda['eth_hundreths'], 
                                               bins=20,
                                               range=(lowerbound, upperbound),
                                               align='mid')
                
                # sum taken beforehand because a count is reset to find the 2nd max
                all_counts = sum(counts)
                # What happens if no transactions in a window?
                # Not a valid window, get to the next iteration
                if all_counts == 0:
                    lowerbound = lowerbound + .50
                    upperbound = upperbound + .50
                    continue
                # Calculate the percentage of the cluster
                cluster_freq = round((counts[10]/all_counts*100), 2)
                # This value accounted for, reset to be able to grab max
                counts[10] = 0.0
                
                #Calculate the 2nd highest index
                index_highest_number = counts.argmax() #argmax is numpys index() function
                #Make the offender piece of the tuple for testing
                offender = lowerbound + index_highest_number
                offender = round(offender, 2)
                
                #Get the 2nd max
                highest_neighbor = round((max(counts)/all_counts*100), 2)
                
                #Build a tuple and add it to all views
                this_observation = (cluster_freq,
                                    highest_neighbor,
                                    offender,
                                    all_counts)
                all_observations.append(this_observation)
                
                #Increment the lower and upper
                lowerbound = lowerbound + .50
                upperbound = upperbound + .50
                #Repeat
            plt.cla()
            self.t_5000_observations = all_observations
        
        #Get data ready
        def student_t(units):
            cluster_all = []
            neighbor_all = []
            if (units == 100):
                for observation in self.t_100_observations:
                    cluster_freq = observation[0]
                    neighbor_freq = observation[1]
                    cluster_all.append(cluster_freq)
                    neighbor_all.append(neighbor_freq)
                samples = pd.DataFrame({'cluster_freq':cluster_all}, dtype='float64')
                samples['neighbor_freq'] = neighbor_all
            elif (units == 500):
                for observation in self.t_500_observations:
                    cluster_freq = observation[0]
                    neighbor_freq = observation[1]
                    cluster_all.append(cluster_freq)
                    neighbor_all.append(neighbor_freq)
                samples = pd.DataFrame({'cluster_freq':cluster_all}, dtype='float64')
                samples['neighbor_freq'] = neighbor_all
            elif (units == 1000):
                for observation in self.t_1000_observations:
                    cluster_freq = observation[0]
                    neighbor_freq = observation[1]
                    cluster_all.append(cluster_freq)
                    neighbor_all.append(neighbor_freq)
                samples = pd.DataFrame({'cluster_freq':cluster_all}, dtype='float64')
                samples['neighbor_freq'] = neighbor_all
            elif (units == 5000):
                for observation in self.t_5000_observations:
                    cluster_freq = observation[0]
                    neighbor_freq = observation[1]
                    cluster_all.append(cluster_freq)
                    neighbor_all.append(neighbor_freq)
                samples = pd.DataFrame({'cluster_freq':cluster_all}, dtype='float64')
                samples['neighbor_freq'] = neighbor_all
            
            #Necessary variables of cluster
            cl_mean = samples['cluster_freq'].mean()
            cl_std = samples['cluster_freq'].std()
            cl_variance = cl_std**2
            cl_obs = samples['cluster_freq'].count()
            
            #Necessary variables of neighbor
            ne_mean = samples['neighbor_freq'].mean()
            ne_std = samples['neighbor_freq'].std()
            ne_variance = ne_std**2
            ne_obs = samples['neighbor_freq'].count()
            
            #Equation
            #print(self.t_1000_observations)
            #print(f'cl_mean: {cl_mean}\ncl_variance: {cl_variance}\ncl_obs: {cl_obs}\n'
            #    + f'ne_mean: {ne_mean}\nnne_variance: {ne_variance}\nne_obs: {ne_obs}')
            tval = abs(cl_mean - ne_mean)/math.sqrt((cl_variance/cl_obs) + (ne_variance/ne_obs))
            
            #Null hypothesis is there is no statistical difference between samples
            null_hypothesis = True
            #if t val lower don't reject
            #if t val higher than there is some statistical difference
            
            #(n1 - 1) + (n2 - 1) = (n1+n2)-2
            degsfreedom = (cl_obs+2)-2
            #pval is also known as the critical value
            #print(f'tval: {tval}\ndf:{degsfreedom}')
            pval = scipy.stats.t.sf(abs(tval), df=degsfreedom)*2
            if(units==100):
                self.pval100 = pval
            elif(units==500):
                self.pval500 = pval
            elif(units==1000):
                self.pval1000 = pval
            elif(units==5000):
                self.pval5000 = pval
            
            significance = 0.0
            #now compare the value at 1%, 5%, 10%
            if (pval <= 0.01): #Some statistical difference at 99/100 times
                significance = 0.01
            elif(pval <= 0.05): #Some statistical difference 95/100 times
                significance = 0.05
            elif(pval <= 0.10): #Some statistical difference 90/100 times
                significance = 0.10
            
            if (significance>0.0):
                null_hypothesis = False
            if (units == 100):
                self.t100null_hypothesis = null_hypothesis
                self.t100significance = significance
            elif (units == 500):
                self.t500null_hypothesis = null_hypothesis
                self.t500significance = significance
            elif (units == 1000):
                self.t1000null_hypothesis = null_hypothesis
                self.t1000significance = significance
            elif (units == 5000):
                self.t5000null_hypothesis = null_hypothesis
                self.t5000significance = significance
    
        
        #main work here
        """
        In the bored ape collection there was a total of 9 transactions
        that fell into the 100 unit range, probably serves us better to
        use the 1000 and 5000 unit range.
        """
        if (selection == 'all'):
            make_t_100()
            make_t_500()
            make_t_1000()
            make_t_5000()
            student_t(100)
            student_t(500)
            student_t(1000)
            student_t(5000)
        elif (selection == 100):
            make_t_100()
            student_t(100)
        elif (selection == 500):
            make_t_500()
            student_t(500)
        elif (selection == 1000):
            make_t_1000()
            student_t(1000)
        elif (selection == 5000):
            make_t_5000()
            student_t(5000)

    def print_t_results(self, units):
        if (units==100):
            print(f't-test hypothesis at {units} is: {self.t100null_hypothesis}'
            + f'      at significance: {self.t100significance}'
            + f'      pval was: {self.pval100}')
        elif (units==500):
            print(f't-test hypothesis at {units} is: {self.t500null_hypothesis}'
            + f'      at significance: {self.t500significance}'
            + f'      pval was: {self.pval500}')
        elif (units==1000):
            print(f't-test hypothesis at {units} is: {self.t1000null_hypothesis}'
            + f'      at significance: {self.t1000significance}'
            + f'      pval was: {self.pval1000}')
        elif (units==5000):
            print(f't-test hypothesis at {units} is: {self.t5000null_hypothesis}'
            + f'      at significance: {self.t5000significance}'
            + f'      pval was: {self.pval5000}')
            
    def percentage_true(self, units):
        """
        Returns the percentage of clusters observed that held
        true to the fact that their trade count is expected to
        be the highest in the observation window.
        
        params
        ------
        units - the base unit for which observation windows will
        be built around as an int
        
        Returns
        -------
        float - percentage of clusters true
        """
        
        valid_units = ['all', 100, 500, 1000, 5000]        
        if units not in valid_units:
            raise ValueError(f'Provided units is not one of {valid_units}')
        
        if units == 100:
            self.t_test(100)
            holds = 0
            total_obs = len(self.t_100_observations)
            for this_tuple in self.t_100_observations:
                clusterp = this_tuple[0]
                neighborp = this_tuple[1]
                if clusterp > neighborp:
                    holds = holds+1
            plt.cla()
            return holds/total_obs
        elif units == 500:
            self.t_test(500)
            holds = 0
            total_obs = len(self.t_500_observations)
            for this_tuple in self.t_500_observations:
                clusterp = this_tuple[0]
                neighborp = this_tuple[1]
                if clusterp > neighborp:
                    holds = holds+1
            plt.cla()
            return holds/total_obs
        elif units == 1000:
            self.t_test(1000)
            holds = 0
            total_obs = len(self.t_1000_observations)
            for this_tuple in self.t_1000_observations:
                clusterp = this_tuple[0]
                neighborp = this_tuple[1]
                if clusterp > neighborp:
                    holds = holds+1
            plt.cla()
            return holds/total_obs
        elif units == 5000:
            self.t_test(5000)
            holds = 0
            total_obs = len(self.t_5000_observations)
            for this_tuple in self.t_5000_observations:
                clusterp = this_tuple[0]
                neighborp = this_tuple[1]
                if clusterp > neighborp:
                    holds = holds+1
            plt.cla()
            return holds/total_obs
        elif units == 'all':
            self.t_test('all')
        
        plt.cla()
            
    def BenfordChiTest(self):
        """
        Returns Chi Square value
        A value over 20.057 has a p value of under .01, meaning
        data does not follow Benford's Law.
        """
        first_sigs = []
        for i in self.panda['usd_first_sig'].value_counts().sort_index():
            first_sigs.append(i)
        expected = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
        observed = []
        for i in first_sigs:
            observed.append((i/sum(first_sigs))*100)

        chiSquare = 0
        for (obs, exp) in zip(observed, expected):
            chiSquare = chiSquare + (((obs - exp)**2)/exp)
        return chiSquare

    def buyer_seller_txns(self, plot):
        """
        Loads in seller and buyer addresses from collection CSV,
        pickle file containing transaction history of seller.
        Finds how many sellers sent ETH to buyers, indicating possible
        collusion toward wash trading, or that the seller created a
        second (buyer) wallet funded by the original (seller) wallet.
        Displays results in pie chart. 
                
        Raises
        ------
        FileNotFoundError - If pickle file or csv is not present in 
        specified locations.
        
        Returns
        -------
        Percentage of trades in which the seller sent ETH to the buyer
        """
        # Load in pickle file containing dict of seller transactions
        cwd = os.getcwd()
        collection = self.name[:-4]
        pickleLocation = cwd + "/" + collection + ".pkl"
        if (not os.path.exists(pickleLocation)):
            raise FileNotFoundError("ERROR: " + pickleLocation + " DOES NOT EXIST")
        inFile = open(pickleLocation, "rb")
        sellerTxns = pickle.load(inFile)
        inFile.close()

        # Read in buyers & sellers associated with each transaction
        readIn = ['winner_account_address', 'seller_address']
        buyersLocation = cwd + "/data/" + collection + ".csv"
        if (not os.path.exists(buyersLocation)):
            raise FileNotFoundError("ERROR: " + buyersLocation + " DOES NOT EXIST")
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
        # Might not be necessary?
        '''outFileLoc = cwd + "/chain_analysis_results/" + collection + "_results.txt" 
        outFile = open(outFileLoc, "w+")
        for pair in buyersSellers:
            line = ' '.join(str(tmp) for tmp in pair)
            outFile.write(line + '\n')
        if len(buyersSellers) > 0:
            print("/chain_analysis_results/" + collection + "_results.txt successfully written")
        else:
            print(collection + " sellers did not send ETH to buyers")
        outFile.close()'''

        # Make list of sellers with dupes removed, make array with number of nonoffenders & offenders
        if plot:
            data = np.array([len(nftSales.index) - len(buyersSellers), len(buyersSellers)])
            myColors = ['skyblue', 'red']
            pieLabels = ["Did not send ETH to buyer (" + str(data[0]) + ")", "Sent ETH to buyer (" + str(data[1]) + ")"]
            myExplode = [0.2,0]

        # Display pie chart of result
        
            plt.pie(data, labels=pieLabels, explode=myExplode, colors=myColors)
            plt.title("Seller Transaction history for " + collection +  " (" + str(len(nftSales.index)) + " total transactions)", bbox={'facecolor':'0.8', 'pad':5})
            plt.show()

        percent = (len(buyersSellers) / len(nftSales['seller_address'])) * 100
        return percent

def makeRoundnessVals():
    '''
    Finds the average roundness for all collections.

    Returns
    -------
    avgRoundnessVals - A dictionary containing the collection name as a 
    key and the average roundness value for its respective collection
    '''
    avgRoundnessVals = {}
    for i in collectionCSVs:
        my_obj = collection(i)
        roundnessDict = my_obj.roundness_check(my_obj.panda['adj_price'])
        roundnessSum = 0
        for key in roundnessDict.keys():
            roundnessSum += float(key) * roundnessDict[key]
        avg = roundnessSum / sum(roundnessDict.values())
        avgRoundnessVals[my_obj.name[:-4]] =  avg
        print(my_obj.name[:-4] + " avg roundness calculated: " + str(avg))
        
    return avgRoundnessVals

def plotRoundness(avgVals):
    '''
    Plots average roundness for each collection. Will need to adjust
    layout to fit all collection names
    '''
    def roundnessOutliers(avgVals):
        valsSum = sum(avgVals.values())
        avg = valsSum / len(avgVals)
        print("Collective average roundess: " + str(avg))
        numerator = 0
        for key in avgVals.keys():
            numerator += (avgVals[key] - avg)**2
        stdDev = math.sqrt(numerator / len(avgVals))
        print("Standard devation: " + str(stdDev))
        for key in avgVals.keys():
            z = (avgVals[key] - avg) / stdDev
            if abs(z) > 3:
                print(key + " is a statistical outlier with a z-score of " + str(z))

    plt.cla()
    plt.clf()
    plt.close('all')
    #plt.figure(figsize=(20, 3))
    fig = plt.figure(figsize=(20, 3))
    plt.bar(list(range(1, len(collectionCSVs)+1)), avgVals.values(), align='edge', width = .3)
    plt.xticks(list(range(1, len(collectionCSVs)+1)))
    plt.xticks(rotation=80)
    fig.suptitle('Average Roundness by Collection', fontsize=20)
    plt.xlabel('Collection Key', fontsize=16)
    plt.ylabel('Average Roundness', fontsize=16)
    plt.show()

    tbl = PrettyTable(['Graph Key', 'Collection'])
    count = 1
    for i in collectionCSVs:
        tbl.add_row([count, i[:-4]])
        count += 1
    print(tbl)

    roundnessOutliers(avgVals)

def plotClusterPercentages():
    '''
    Calls percentage_true() for every collection, adds them all to a dict
    with the key being the collection name. Then plots collection percentages
    and detects outliers.
    '''
    def clusterOutliers(percents):
        pctSum = sum(percents.values())
        avg = pctSum / len(percents.values())
        print("Collective average cluster percentage: " + str(avg))
        numerator = 0
        for key in percents.keys():
            numerator += (percents[key] - avg)**2
        stdDev = math.sqrt(numerator / len(percents))
        print("Standard deviation: " + str(stdDev))
        for key in percents.keys():
            z = (percents[key] - avg) / stdDev
            if abs(z) > 3:
                print(key + " is a statistical outlier with a z-score of " + str(z))

    clusterPercents = {}
    for i in collectionCSVs:
            my_obj = collection(i)
            holdPercent = my_obj.percentage_true(5000)
            print(my_obj.name[:-4] + ": " + str(holdPercent))
            clusterPercents[my_obj.name[:-4]] = holdPercent

    plt.clf()
    plt.cla()
    plt.close('all')
    #plt.figure(figsize=(20, 3))
    fig = plt.figure(figsize=(20, 3))
    plt.bar(list(range(1, len(collectionCSVs)+1)), clusterPercents.values(), align='edge', width = .3)
    plt.xticks(list(range(1, len(collectionCSVs)+1)))
    plt.xticks(rotation=45)
    fig.suptitle('Cluster Percentages by Collection', fontsize=20)
    plt.xlabel('Collection Key', fontsize=18)
    plt.ylabel('Cluster Percentage', fontsize=16)
    plt.show()

    tbl = PrettyTable(['Graph Key', 'Collection'])
    count = 1
    for i in collectionCSVs:
        tbl.add_row([count, i[:-4]])
        count += 1
    print(tbl)

    clusterOutliers(clusterPercents)

def plotAllTxns():
    txnPercents = {}
    for i in collectionCSVs:
        my_obj = collection(i)
        txnPctg = my_obj.buyer_seller_txns(False)
        print(my_obj.name[:-4] + ": " + str(txnPctg))
        txnPercents[my_obj.name[:-4]] = txnPctg

    plt.figure(figsize=(20, 3))
    fig = plt.figure()
    plt.bar(list(range(1, len(collectionCSVs)+1)), txnPercents.values(), align='edge', width = .3)
    plt.xticks(list(range(1, len(collectionCSVs)+1)))
    plt.xticks(rotation=45)
    fig.suptitle('Percentage of Transactions in Which Seller sent ETH to Buyer', fontsize=20)
    plt.xlabel('Collection Key', fontsize=18)
    plt.ylabel('Transction Percentage', fontsize=16)
    plt.show()

    tbl = PrettyTable(['Graph Key', 'Collection'])
    count = 1
    for i in collectionCSVs:
        tbl.add_row([count, i[:-4]])
        count += 1
    print(tbl)

def plotAllBenfordChis():
    chiSquares = {}
    for i in collectionCSVs:
            my_obj = collection(i)
            chi = my_obj.BenfordChiTest()
            if chi > 20.057:
                print(i + " does not follow Benford's Law (chi square=" + str(chi) + ")")
            else:
                print(my_obj.name[:-4] + ": " + str(chi))
            chiSquares[my_obj.name[:-4]] = chi

    plt.clf()
    plt.cla()
    plt.close('all')
    #plt.figure(figsize=(20, 3))
    fig = plt.figure(figsize=(20, 3))
    plt.bar(list(range(1, len(collectionCSVs)+1)), chiSquares.values(), align='edge', width = .3)
    plt.xticks(list(range(1, len(collectionCSVs)+1)))
    plt.xticks(rotation=45)
    fig.suptitle('Benford\'s Law Chi Square Values by Collection', fontsize=20)
    plt.xlabel('Collection Key', fontsize=18)
    plt.ylabel('Chi Square', fontsize=16)
    plt.axhline(y=20.057,linewidth=1, color='red')
    plt.show()

    tbl = PrettyTable(['Graph Key', 'Collection'])
    count = 1
    for i in collectionCSVs:
        tbl.add_row([count, i[:-4]])
        count += 1
    print(tbl)

if __name__ == '__main__':
    """
    Heres where I've been testing all the functions that I'm creating
    just create an object from a csv that you have in your data folder,
    init methods will be run on instantiation which handle getting the
    panda prepared, and then you can call any of the functions above.
    """
    '''test = collection(collectionCSVs[24])
    benford_standard = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
    sorted = (test.panda['usd_first_sig'].value_counts().sort_index())
    expected = []
    total_obs = sum(test.panda['usd_first_sig'].value_counts())
    #Generate expected distribution
    for i in range(0, 9):
        expected.append(round(total_obs*benford_standard[i]/100, 3))
    print(expected)
    print(sorted)'''

    '''seed(69)
    dict = {}
    count = 0
    for i in collectionCSVs:
        randNum = randint(1,1000)
        dict[i] = randNum
        count += 1
    plt.figure(figsize=(20, 3))
    plt.bar(list(range(1, len(collectionCSVs)+1)), dict.values(), align='edge', width = .3)
    plt.xticks(list(range(1, len(collectionCSVs)+1)))
    plt.xticks(rotation=45)
    plt.show()

    tbl = PrettyTable(['Graph Key', 'Collection'])
    count = 1
    for i in collectionCSVs:
        tbl.add_row([count, i])
        count += 1
    print(tbl)'''
    roundness = makeRoundnessVals()
    plotRoundness(roundness)
    # test.t_test()
    # test.print_t_results(100)
    # test.print_t_results(1000)
    # test.print_t_results(5000)
    # nan results likely due to no transactions falling within a region
    # multiplying 0 in numpy probably returns nan
    #print(test.observations)
    #print(test.panda['eth_hundreths'][test.panda.eth_hundreths==9.15])