#### 目标：实时检测，过滤，识别进入摄像头内的人，一个人进入摄像头，满足过滤条件后，追踪该人，并给这个人标上ID，
    在摄像头内监控的场景内，每个人都做到实时追踪，识别，类似的场景为：一个摄像头内的所有人头上都有一个id来标识这个人，
    是新进入摄像头还是一直在摄像头监控的场景内
#### Pipline

1. 摄像头实时采集图片，比如1s两帧， 采集到的图片首先进入检测模型，检测出该帧图片中含有的人，并返回框体box
2. 按照检测返回的box，crop出多张图片，并resize到（96x96） 进入landmark 关键点模型，输出mks,vis,pose
3. 根据关键点给出的vis, pose 过滤图片，符合要求的图片进入识别模型。
4. 过滤后的图片resize到(64x64)进入识别模型，给出识别特征feature
5. feature 搜索特征库，如果该搜到的相似度top1大于给定识别阈值，则说明该人已经在摄像头内被采集到，如果小于阈值，则说明该人是第一次进入摄像头，
    把该feature存入到特征库中，并给该人一个id号。id号按照特征库中存储的特征数量来编号。
    