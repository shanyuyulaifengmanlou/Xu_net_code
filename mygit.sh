#!/bin/sh
git add *;
curTime=$(date "+%G-%m-%d %H:%M:%S");
user=$(whoami);
git commit -m $user;
git pull;
git push origin master;
echo $user push the code to the github successfully at $curTime.;
