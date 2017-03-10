# Bs_worldAverages

# Content:
- Code - contains Python code
- Mathematica - (obsolete) Mathematica code
- Report - report on the project
- Results - all figures and one txt file created

# To run:
- $ cd Code
- $ python2 GsDGs_and_PhisDGs_worldAverages.py
- .txt file and 3 kinds of figures in .eps, .png, .jpg and .pdf formats will appear in ../Results


# GitHub tips:
1. To download the repo for the first time just do the following (no need to be singed in on the GitHub):
- $ git clone https://github.com/dgerstel/Bs_worldAverages.git

2. To update your local repo:
- $ git pull

3. To upload local changes to GitHub repo:
- (can list all changes with $ git status)
- $ git commit -m "_brief comment of changes made_" _filename_
- (e.g. $ git commit -m "updated phis/DGs plot labels" GsDGs_and_PhisDGs_worldAverages.py)
- $ git push

4. If the above steps don't work -- make sure you've taken care of the ssh-keys, as described below.


# ssh keys:
- For quicker pull/push from from/to the repository you can add your ssh-key to the github website (Settings/Deploy keys).
- To generate and display ssh-key on your machine do:
   - $ cd ~/.ssh && ssh-keygen
   - $ cat id_rsa.pub 
- then copy&paste it on the github


# Common problems:
- If you experience the following:
  $ git push -u origin master

  error: The requested URL returned error: 403 Forbidden while accessing https://github.com/dgerstel/Bs_worldAverages.git/info/refs

fatal: HTTP request failed

Then, _perhaps_ it will help:
in your .git/config change the line:

"url = https://github.com/dgerstel/Bs_worldAverages.git" to this one:

"url = ssh://git@github.com/dgerstel/Bs_worldAverages.git"


