#!/usr/bin/env bash


# create gif from all the frames
convert -delay 4 -loop 0 frames/frame_*.ppm frames.gif

# copy gif over ssh
scp frames.gif jhladik@192.168.1.48:frames/

# delete local gif
rm -f frames.gif

