#!/bin/bash                                                                                     
# chmod +x run_teleop.sh when the first time you're running
# SO-101 Teleoperation with cameras on index 0 and 1                                            
# Detected ports:                                                                               
#   /dev/cu.usbmodem5AB01575561                                                                 
#   /dev/cu.usbmodem5AB01575631                                                                 
# Adjust LEADER_PORT / FOLLOWER_PORT below if they're swapped.                                  
                                                                                                 
LEADER_PORT="/dev/cu.usbmodem5AB01575631"                                                       
FOLLOWER_PORT="/dev/cu.usbmodem5AB01575561"                                                     

lerobot_venv/bin/python teleoperate.py \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5AB01575561 \
  --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 5}, top: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 5}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5AB01575631 \
  --teleop.id=my_leader \
  --display_data=true