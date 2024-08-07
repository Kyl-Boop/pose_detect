from ultralytics import YOLO
import cv2
import math
import matplotlib.pyplot as plt


def getAnglebyline(line1, line2):
    dx1 = line1[0][0] - line1[1][0]
    dy1 = line1[0][1] - line1[1][1]
    dx2 = line2[0][0] - line2[1][0]
    dy2 = line2[0][1] - line2[1][1]

    # 求斜率
    m1 = dy1 / dx1
    m2 = dy2 / dx2
    insideAngle = math.atan(abs((m2 - m1) / (1 + (m1 * m2))))

    angle = insideAngle / math.pi * 180
    if angle > -370 and angle < 370:
        angle = int(angle)

    return angle


def aspectRatio(boxes):
    boxes = boxes.cpu().numpy().astype('uint32')
    x = [boxes[0], boxes[2]]
    y = [boxes[1], boxes[3]]
    width = x[1] - x[0]
    height = y[1] - y[0]
    radio = width / height
    return radio


def getAnglebypoint(point_a, point_b, point_c):
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]  # 点a、b、c的x坐标
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]  # 点a、b、c的y坐标

    if len(point_a) == len(point_b) == len(point_c) == 3:
        # print("坐标点为3维坐标形式")
        a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]  # 点a、b、c的z坐标
    else:
        a_z, b_z, c_z = 0, 0, 0  # 坐标点为2维坐标形式，z 坐标默认值设为0
        # print("坐标点为2维坐标形式，z 坐标默认值设为0")

    # 向量 m=(x1,y1,z1), n=(x2,y2,z2)
    x1, y1, z1 = (a_x - b_x), (a_y - b_y), (a_z - b_z)
    x2, y2, z2 = (c_x - b_x), (c_y - b_y), (c_z - b_z)

    # 两个向量的夹角，即角点b的夹角余弦值
    cos_b = (x1 * x2 + y1 * y2 + z1 * z2) / (
            math.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) * (math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)) + 0.01)  # 角点b的夹角余弦值
    B = math.degrees(math.acos(cos_b))  # 角点b的夹角值
    return B


# getAnglebypoint((3 ** 0.5, 1), (0, 0), (3 ** 0.5, 0))  # 结果为 30°
# getAnglebypoint((1, 1), (0, 0), (1, 0))  # 结果为 45°
# getAnglebypoint((-1, 1), (0, 0), (1, 0))  # 结果为 135°


def is_fallen(keypoints, boxes):
    keypoints = keypoints.cpu().numpy().astype('uint32')
    Left_Shoulder = keypoints[5][:2]
    # if Left_Shoulder[0] + Left_Shoulder[1] == 0: self.ATHERPOSE += 1
    Right_Shoulder = keypoints[6][:2]
    # if Right_Shoulder[0] + Right_Shoulder[1] == 0: self.ATHERPOSE += 1
    Left_Hip = keypoints[11][:2]
    # if Left_Hip[0] + Left_Hip[1] == 0: self.ATHERPOSE += 1
    Right_Hip = keypoints[12][:2]
    # if Right_Hip[0] + Right_Hip[1] == 0: self.ATHERPOSE += 1
    Left_Knee = keypoints[13][:2]
    # if Left_Knee[0] + Left_Knee[1] == 0: self.ATHERPOSE += 1
    Right_Knee = keypoints[15][:2]
    # if Right_Knee[0] + Right_Knee[1] == 0: self.ATHERPOSE += 1
    Left_Ankle = keypoints[15][:2]
    # if Left_Ankle[0] + Left_Ankle[1] == 0: self.ATHERPOSE += 1
    Right_Ankle = keypoints[16][:2]
    # if Right_Ankle[0] + Right_Ankle[1] == 0: self.ATHERPOSE += 1

    Shoulders_c = [(Left_Shoulder[0] + Right_Shoulder[0]) // 2,
                   (Left_Shoulder[1] + Right_Shoulder[1]) // 2]

    hips_c = [(Left_Hip[0] + Right_Hip[0]) // 2,
              (Left_Hip[1] + Right_Hip[1]) // 2]

    Knee_c = [(Left_Knee[0] + Right_Knee[0]) // 2,
              (Left_Knee[1] + Right_Knee[1]) // 2]

    Ankle_c = [(Left_Ankle[0] + Right_Ankle[0]) // 2,
               (Left_Ankle[1] + Right_Ankle[1]) // 2]

    '''计算身体中心线与水平线夹角'''
    human_angle = getAnglebyline([Shoulders_c, hips_c], [[0, 0], [10, 0]])
    '''计算检测区域宽高比'''
    aspect_ratio = aspectRatio(boxes)
    '''计算肩部中心点与胯部中心点的垂直距离差'''
    human_shoulderhip = abs(Shoulders_c[1] - hips_c[1])

    '''计算肩部胯部膝盖夹角'''
    Hip_Knee_Shoulders_angle = getAnglebypoint(Shoulders_c, hips_c, Knee_c)
    Hip_Knee_Right_angle = getAnglebypoint(Right_Shoulder.tolist(), Right_Hip.tolist(), Right_Knee.tolist())

    '''计算胯部膝盖小腿夹角'''
    Ankle_Knee_Hip_angle = getAnglebypoint(hips_c, Knee_c, Ankle_c)
    Ankle_Knee_Right_angle = getAnglebypoint(Right_Hip.tolist(), Right_Knee.tolist(), Right_Ankle.tolist())

    '''计算胯部膝盖是否处于相似的垂直位置'''
    vertical_threshold = Left_Knee[1] - Left_Shoulder[1]

    '''计算胯部膝盖是否处于相似的水平位置'''
    horizontal_threshold = Left_Shoulder[0] - Left_Knee[0]

    status_score = {'Stand': 0.0,
                    'Fall': 0.0,
                    'Sit': 0.0,
                    'other': 0.0}
    _weight = ''

    '''判断Shoulder、Hip、Knee是否被检测到'''
    if Knee_c[0] == 0 and Knee_c[1] == 0 and hips_c[0] == 0 and hips_c[1] == 0:
        status_score['Sit'] += 0.69
        status_score['Fall'] += -0.8 * 2
        status_score['Stand'] += -0.8 * 2
        _weight = f'[1]Sit:+0.2, Fall:-1.6 ,Stand: -1.6'

    elif Shoulders_c[1] == 0 and Shoulders_c[0] == 0 and hips_c[0] == 0 and hips_c[1] == 0:
        status_score['Sit'] += -0.8 * 2
        status_score['Fall'] += -0.8 * 2
        status_score['Stand'] += 0.69

    '''身体中心线与水平线夹角+-25'''
    if human_angle in range(-25, 25):
        status_score['Fall'] += 0.8
        status_score['Sit'] += 0.1
        _weight = f'{_weight}, [2]Fall:+0.8, Sit:+0.1'
    else:
        status_score['Fall'] += 0.2 * ((90 - human_angle) / 90)
        _weight = f'{_weight}, [3]Fall:+{0.8 * ((90 - human_angle) / 90)}'

    '''宽高比小与0.6则为站立'''
    if (aspect_ratio < 0.6 and human_angle in range(65, 115)):
        status_score['Stand'] += 0.8
        _weight = f'{_weight}, [4]Stand:+0.8'

    elif (aspect_ratio > 1 / 0.6):  # 5/3
        status_score['Fall'] += 0.8
        _weight = f'{_weight}, [5]Fall:+0.8'
    if horizontal_threshold < 30:
        status_score['Fall'] += 0.6
        status_score['Sit'] += -0.15
    # if 25 < Hip_Knee_Shoulders_angle < 145 and 75 < human_angle < 125:
    #     status_score['Sit'] += 0.8
    #     status_score['Stand'] += -0.035
    #     if vertical_threshold > 30:
    #         status_score['Sit'] += +0.15
    #     _weight = f'{_weight}, [6]Stand:-0.035, Sit:+0.15'
    # elif Hip_Knee_Shoulders_angle > 120 and 75 < human_angle < 125:
    #     status_score['Stand'] += 0.2
    # elif Hip_Knee_Shoulders_angle > 120 and -25 < human_angle < 25:
    #     status_score['Fall'] += 0.2
    # else:
    #     status_score['Fall'] += 0.05
    #     status_score['Stand'] += 0.05
    #     _weight = f'{_weight}, [7]Stand:+0.05, Fall:+0.05'

    score_max, status_max = max(zip(status_score.values(), status_score.keys()))

    return status_max, score_max


def draw_boxes(boxes, image, label):
    boxes = boxes.cpu().numpy().astype('uint32')
    x = [boxes[0], boxes[2]]
    y = [boxes[1], boxes[3]]
    X = x[0]
    Y = y[0]
    color = (0, 255, 0)
    cv2.rectangle(image, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), color, 2)
    cv2.putText(image, label, (int(X + 5), int(Y + 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


if __name__ == '__main__':

    model = YOLO('yolov8x-pose.pt')
    source = 'images'  # 换成自己的图片路径
    results = model(source)
    y_train = []
    for result in results:
        keypoints = result.keypoints.xy
        boxes = result.boxes.xyxy
        image_path = result.path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(keypoints.shape)
        # print(image_path.split("images")[0])
        # with open(f"{image_path}", "r") as f:
        #     for line in f.readlines():
        #         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
        #         y_train.append(line[0])
        # print(f"第{i}张图片：")
        # y2 = []
        for keypoin, box in zip(keypoints, boxes):
            # print(keypoin,box)
            # keypoint = keypoin.cpu().numpy().astype('uint32')
            status_max, score_max = is_fallen(keypoin, box)
            if status_max == 'Fall':
                draw_boxes(box, image, f'{status_max}')
            # y2.append(status_max)
        # y1.append(y2)
        plt.imshow(image)
        plt.show()

# print(y1)
# class BinaryClassifier(nn.Module):
#     def __init__(self, input_features):
#         super(BinaryClassifier, self).__init__()
#         # 定义网络结构
#         self.fc1 = nn.Linear(input_features, 64)  # 第一个全连接层
#         self.fc2 = nn.Linear(64, 32)  # 第二个全连接层
#         self.fc3 = nn.Linear(32, 2)  # 输出层，2个神经元代表两个类别
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))  # 使用ReLU激活函数
#         x = F.relu(self.fc2(x))  # 使用ReLU激活函数
#         x = self.fc3(x)  # 输出层，不需要激活函数，因为后面会使用softmax
#         return x
#
#
# # 实例化网络模型
# input_features = 17  # 输入特征数量
# model = BinaryClassifier(input_features)
#
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类问题
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
#
# # 假设有一些训练数据
# # X_train是训练特征，y_train是训练标签（0或1）
# X_train = torch.randn(100, input_features)  # 100个样本，每个样本17个特征
# y_train = torch.randint(0, 2, (100,))  # 100个样本的标签，0或1
#
# # 训练网络
# num_epochs = 10  # 训练轮数
# for epoch in range(num_epochs):
#     # 前向传播
#     outputs = model(X_train)
#     loss = criterion(outputs, y_train)
#
#     # 反向传播和优化
#     optimizer.zero_grad()  # 清空过往梯度
#     loss.backward()  # 反向传播，计算当前梯度
#     optimizer.step()  # 根据梯度更新网络参数
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
