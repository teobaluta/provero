#!/bin/bash

[ ! -d nets/ ] && mkdir nets
# download from eth-sri
wget -r -np -R "index.html*" -nH --cut-dirs 1 https://files.sri.inf.ethz.ch/eran/nets/
