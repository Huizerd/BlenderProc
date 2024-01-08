#!/usr/bin/env bash

# go over all the materials
materials=(aerial_rocks brick carpet fabric grass lava lego water)
# materials=(aerial_rocks brick)

# regular motions
motion_files=(x-slow x-medium x-fast y-slow y-medium y-fast z-slow z-medium z-fast nx-slow nx-medium nx-fast ny-slow ny-medium ny-fast r-slow r-medium r-fast nr-slow nr-medium nr-fast)
for material in ${materials[@]}; do
    for motionf in ${motion_files[@]}; do
        echo "Processing $material with $motionf"
        blenderproc run main.py $material output --motion regular --motion_file motions/$motionf.yaml
    done
done

# random bezier motion
for material in ${materials[@]}; do
    echo "Processing $material with random bezier motion"
    blenderproc run main.py $material output --motion random_bezier --motion_file motions/random-bezier.yaml
done
