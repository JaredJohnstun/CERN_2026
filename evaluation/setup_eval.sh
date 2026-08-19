#!/bin/bash

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh

##### old setup with centos7 not working on lxplus
OS=$(uname -r)
if [[ $OS == *el7* ]]; then
  lsetup "views LCG_105 x86_64-centos7-gcc12-opt"
elif [[ $OS == *el9* ]]; then
  lsetup "views LCG_106 x86_64-el9-gcc13-opt"
else
  echo "Warning ! It appears your OS (from kernel) ${OS} is not el7 or el9 (running a container ?)"
  echo "Will set up for CentOS7"
  lsetup "views LCG_105 x86_64-centos7-gcc12-opt"
fi
