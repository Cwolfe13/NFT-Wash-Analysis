# NFT Wash Analysis
 The purpose of this project is to analyze NFT trades across different markets to determine if wash trading is occuring, and then peform further analysis of those wash trades.
 ## The Data
 The data for this project was collected by a number of students enrolled at the University of Florida using public APIs different markets provided.
 The data used was the following market's [OpeaSea.io](https://www.OpenSea.io) top 43 collections :
    
    -0n1_force

    -Axie_Infinity

    -Azuki

    -Bored Ape 

    -Clone X

    -Cool Monkes

    -Creepz Reptile

    -Creepz

    -Cryptotoadz

    -Cryptobatz

    -Crpytokitties

    -Cryptopunks

    -Cryptoskulls

    -Cyberkongz VX

    -DeadFellaz

    -Decentraland Wearables

    -Doge Pound

    -Doodles

    -dr ETHvil

    -Emblem Vault

    -FLUF World Thingies

    -FOMO Mofos

    -Full Send

    -Hape Prime

    -Hashmasks

    -Lil Heroes

    -Lost Poets

    -Meebits

    -Mekaverse

    -Metroverse

    -Mutant Ape

    -My Curio Cards

    -Phantabear

    -Pudgy Penguins

    -Punk Comics

    -Rarible

    -RTFKT

    -SoRare

    -SuperRare

    -Wolf Game

    -World of Women

    -WVRPS

    -X Rabbits

## Where is the data?
Storage of large files on GitHub is generally not recommended, for the purposes of this project all data should be stored in a folder `data` in the source directory of the project. You should create this folder and store the data there yourself. The data will be part of the zip if transmitted.
# Python version
The repo was initialized with Python 3.9.9, any version equal or higher should be sufficient.
# Jupyterlabs
Jupyter seems like the most obvious choice to use when working with this type of data, it has good visualization tools, and all of the results will be reproducible in a step by step manner for others to inspect.
## Jupyter-lab vs Jupyter-notebook
Jupyter-lab is intended to completely replace Jupyter-notebook at some point, and has a built in debugger, so it seems like the superior choice.
# The virtual environment
Virtual environments are useful for python projects because they indicate to the user the minimum amount of packages needed to run the project. A pip freeze outside of the virtual environment may list installed packages that are not used in the project.
## Creating a virtual environments
To create a virtual environment use the following command from the source directory of the project:  
`python3 -m venv venv`  
or for Windows, invoke the venv command as follows:  
`c:\>c:\Python35\python -m venv c:\path\to\myenv` , make sure to name the path as appropriate and have `myenv` changed to `venv`.

This will create a virtual environment inside of a folder `venv` located in the source directory.
## Activating the virtual environment
Navigate into the venv folder and use `source bin/activate`, or on windows `.Scripts\bin\activate`
## Installing required packages
All of the packages used for this project can be loaded in using the following:  
`pip3 install -r requirements.txt`
## Adding new packages while developing
When installing a new package you'd like to use, make sure to update the requirements.txt using the following command `pip3 freeze > requirements.txt` and then commit the change. If you see the requirements.txt has been updated you should install the requirements again to have the latest packages used.
## Deactivating the virtual environment
Use `deactivate` inside of your shell.