CUDA_VISIBLE_DEVICES=0 python track_video.py -v /nfs/130.245.4.102/add_disk2/starc/HardIndividualVideos/Tattoo37.mp4 &
CUDA_VISIBLE_DEVICES=1 python track_video.py -v /nfs/130.245.4.102/add_disk2/starc/HardIndividualVideos/Tattoo39.mp4 &
CUDA_VISIBLE_DEVICES=3 python track_video.py -v /nfs/130.245.4.102/add_disk2/starc/HardIndividualVideos/Tattoo42.mp4 &
wait
CUDA_VISIBLE_DEVICES=0 python track_video.py -v /nfs/130.245.4.102/add_disk2/starc/HardIndividualVideos/Tattoo45.mp4 &
