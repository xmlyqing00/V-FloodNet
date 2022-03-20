rsync --archive --stats --progress 3090:/home/gvc/Sources/WaterNetV2/output\
  /Ship03/Sources/WaterNetV2/ \

rsync --archive --stats --progress 3090:/home/gvc/Sources/WaterNetV2/records\
  /Ship03/Sources/WaterNetV2/ \

rsync --archive --stats --progress 3090:/home/gvc/Datasets/water2\
  /Ship01/Dataset/water_v3/ \
