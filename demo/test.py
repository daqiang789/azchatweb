import cv2
import mediapipe as mp
import numpy as np
import math
import onnxruntime
import copy
from demo_utils import multiclass_nms, demo_postprocess, demo_preprocess

i = 0

class BasketBall(object):
    def __init__(self, input_type):
        super().__init__()
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
        self.hoop = onnxruntime.InferenceSession("hoop.onnx")

        self.input_type = input_type # 超参数，若为“video”模式，一般视频的帧率是30fps，则会默认每帧之间的时间差为0.035s
        self.how_easy = 1.5 # 超参数，投球难度设置，数字越大约简单
        self.max_speed = 20 # 最大力投球时，投球瞬间球的速度，单位m/s
        self.ready_threshold = 10 # 超参数，判断人体是否准备好，连续ready_threshold帧稳定则准备好
        self.re_init_threshold = 30 # 超参数，判断人体是否转变为未准备状态，连续re_init_threshold帧未准备则转为未准备态

        self.is_shoot = False # 是否已投篮
        self.shot_pos = (-1, -1) # 发射时篮球的位置
        self.speed_x = 0 # 发射时篮球的横向速度，单位m/s
        self.speed_y = 0 # 发射时篮球的纵向速度，单位m/s
        self.shot_time = 0 # 发射时的时间
        self.durr_time = 0 # 发射到当前时刻经过的时间

        self.is_init = False # 是否开始准备投球：手腕高过肩膀、手掌向上抬起与手臂成一定夹角时，认为准备投球
        self.not_prepare_time = 0 # 连续没有准备投球的帧数

        self.stable_time = 0 # 手掌与手臂的夹角相对稳定的帧数
        self.stable_theta = 0 # 手掌与手臂相对稳定的夹角
        self.stable_finger = (-1, -1) # 手掌与手臂相对稳定时手掌的位置
        self.stable_wrist = (-1, -1) # 手掌与手臂相对稳定时手腕的位置
        self.stable_shoulder = (-1, -1) # 手掌与手臂相对稳定时肩膀的位置

        self.ready_time = 0 # 人体准备投球时，手掌与手臂的夹角稳定的帧数
        self.ready_theta = 0 # 人体准备投球时，手掌与手臂的夹角
        self.ready_finger = (-1, -1) # 人体准备投球时，手掌的位置
        self.ready_wrist = (-1, -1) # 人体准备投球时，手腕的位置
        self.ready_shoulder = (-1, -1) # 人体准备投球时，肩膀的位置
        self.ready_dis = 0 # 人体准备投球时，手腕与手臂的距离

        self.pix2dis = 0 # 每个像素代表多少米

        picture = cv2.imread("./basketball.png", cv2.IMREAD_UNCHANGED)
        self.ball_mask = np.zeros((picture.shape[1], picture.shape[0], 3))
        self.ball_mask[:, :, 0] = picture[:, :, 3]
        self.ball_mask[:, :, 1] = picture[:, :, 3]
        self.ball_mask[:, :, 2] = picture[:, :, 3]
        self.ball_mask = self.ball_mask / 255
        self.bg_mask = (np.ones(self.ball_mask.shape, dtype='uint8')) - self.ball_mask

        self.ball_pic = np.zeros((picture.shape[1], picture.shape[0], 3))
        self.ball_pic[:, :, 0] = picture[:, :, 0]
        self.ball_pic[:, :, 1] = picture[:, :, 1]
        self.ball_pic[:, :, 2] = picture[:, :, 2]
        self.ball_pic = self.ball_pic * self.ball_mask

    # 将mediapipe返回的归一化坐标值转换为图像坐标
    def get_position(self, mediapipe_landmark, img_w, img_h):
        mediapipe_landmark.x = mediapipe_landmark.x * img_w
        mediapipe_landmark.y = mediapipe_landmark.y * img_h
        return (int(mediapipe_landmark.x), int(mediapipe_landmark.y))

    # 计算两点间的距离
    def get_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]))

    # 计算edge_point1---corner_point0---edge_point2的夹角的cos值
    def get_theta(self, corner_point0, edge_point1, edge_point2):
        edge1 = (edge_point1[0] - corner_point0[0], edge_point1[1] - corner_point0[1])
        length1 = self.get_distance(corner_point0, edge_point1)

        edge2 = (edge_point2[0] - corner_point0[0], edge_point2[1] - corner_point0[1])
        length2 = self.get_distance(corner_point0, edge_point2)

        return (edge1[0] * edge2[0] + edge1[1] * edge2[1]) / (length1 * length2)

    # 判断point0在point1和point2组成的连线的上方还是下方
    def get_up_down(self, point1, point2, point0):
        result = (point0[1] - point1[1]) * (point2[0] - point1[0]) - (point2[1] - point1[1]) * (point0[0] - point1[0])
        if result > 0:
            return 1
        else:
            return -1

    # 计算法向量
    def get_n(self, point1, point2):
        line1 = (point2[0] - point1[0], point2[1] - point1[1]) / self.get_distance(point1, point2)
        if line1[0] != 0:
            point3 = (point1[0], point1[1] + 1)
            line2 = (0, 1)
        else:
            point3 = (point1[0] + 1, point1[1])
            line2 = (1, 0)
        line3 = self.get_theta(point1, point2, point3) * line1
        n = line2 - line3
        n = n / np.sqrt(n[0] * n[0] + n[1] * n[1])
        if n[0] <= 0:
            return (-1 * n[0], -1 * n[1])
        else:
            return (n[0], n[1])

    # 绘制手上的篮球，篮球一般是25cm直径
    def plot_basketball_inhand(self, frame, shoulder, elbow, wrist, finger):
        self.radius = int(0.25 / self.pix2dis / 2)
        self.ball_pic = cv2.resize(self.ball_pic, (self.radius*2, self.radius*2))
        self.ball_mask = cv2.resize(self.ball_mask, (self.radius*2, self.radius*2))
        self.bg_mask = cv2.resize(self.bg_mask, (self.radius*2, self.radius*2))

        palm_direction = self.get_n(finger, wrist)
        self.shot_pos = (int(finger[0] + palm_direction[0] * self.radius), int(finger[1] + palm_direction[1] * self.radius))
        
        self.plot_basketball(frame, self.shot_pos)

    # 绘制其他地方的篮球
    def plot_basketball(self, frame, pos):
        x1 = pos[0] - self.radius
        x2 = pos[0] + self.radius
        y1 = pos[1] - self.radius
        y2 = pos[1] + self.radius
        if x1 >= 0 and x2 <= frame.shape[1] and y1 >= 0 and y2 <= frame.shape[0]:
            plot_frame = frame[y1: y2, x1: x2, :]
            plot_frame = plot_frame * self.bg_mask
            plot_frame = plot_frame + self.ball_pic
            frame[y1: y2, x1: x2, :] = plot_frame

    # get hoop postion
    def get_basketball_hoop(self, frame):
        img, ratio = demo_preprocess(frame, (320, 320))

        ort_inputs = {self.hoop.get_inputs()[0].name: img[None, :, :, :]}
        output = self.hoop.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], (320, 320), False)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            max_rect = (-1, -1, -1, -1)
            max_area = 0
            for i in range(len(final_boxes)):
                box = final_boxes[i]
                cls_id = int(final_cls_inds[i])
                score = final_scores[i]
                if score < 0.4:
                    continue
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])

                area = (y1 - y0) * (x1 - x0)
                if area > max_area:
                    max_area = area
                    max_rect = (x0, y0, x1 - x0, y1 - y0)

            if max_area > 0:
                return True, max_rect
        return False, max_rect

    # plot ball in the moment that shot
    def plot_ball_in_one_pic(self, frame):
        durr = 0
        while True:
            durr += 0.035

            x_pix = int(self.speed_x * durr / self.pix2dis)
            y_pix = int((self.speed_y * durr - 5 * durr * durr) / self.pix2dis)
            x = self.shot_pos[0] + x_pix
            y = self.shot_pos[1] - y_pix
            pos = (x, y)

            if y >= frame.shape[0] or x >= frame.shape[1]:
                break

            self.plot_basketball(frame, pos)
        return

    def process(self, frame):
        img_w = frame.shape[1]
        img_h = frame.shape[0]

        # 篮球框大小一般为50cm，根据篮球框检测的大小估算pix2dis
        # if no hoop, return False
        has_hoop, hoop_rect = self.get_basketball_hoop(frame)
        if not has_hoop:
            return False
        self.pix2dis = 0.5 / hoop_rect[2]
        cv2.rectangle(frame, (hoop_rect[0], hoop_rect[1]), (hoop_rect[0] + hoop_rect[2], hoop_rect[1] + hoop_rect[3]), (255, 255, 255), 1)
        cv2.putText(frame, "hoop", (hoop_rect[0] + 5, hoop_rect[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=1)

        if self.is_shoot:
            curr_time = cv2.getTickCount()
            if self.input_type == "video":
                #self.durr_time += 0.035
                self.durr_time = (curr_time - self.shot_time) / cv2.getTickFrequency()
            else:
                self.durr_time = (curr_time - self.shot_time) / cv2.getTickFrequency()

            # 使用物体斜抛运动公式计算当前时刻篮球的位置
            x_pix = int(self.speed_x * self.durr_time / self.pix2dis)
            y_pix = int((self.speed_y * self.durr_time - 5 * self.durr_time * self.durr_time) / self.pix2dis)
            x = self.shot_pos[0] + x_pix
            y = self.shot_pos[1] - y_pix
            pos = (x, y)

            # 若篮球离开画面最下边或最右边，则认为本次投篮结束
            if y >= img_h or x >= img_w:
                self.is_shoot = False
                self.shot_pos = (-1, -1)
                self.speed_x = 0
                self.speed_y = 0
                self.shot_time = 0
                self.durr_time = 0

            self.plot_basketball(frame, pos)
            return True

        print("not_prepare_time: ", self.not_prepare_time)
        self.not_prepare_time = self.not_prepare_time + 1

        # 若连续re_init_threshold帧没有准备投球，则认为人体从已准备状态转变为未准备
        if self.not_prepare_time >= self.re_init_threshold:
            self.is_init = False
            self.not_prepare_time = 0
            self.stable_time = 0
            self.stable_theta = 0
            self.stable_finger = (-1, -1)
            self.stable_wrist = (-1, -1)
            self.stable_shoulder = (-1, -1)
            self.ready_time = 0
            self.ready_theta = 0
            self.ready_finger = (-1, -1)
            self.ready_wrist = (-1, -1)
            self.ready_shoulder = (-1, -1)
            self.ready_dis = 0

        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks is None:
            return False

        # 使用右肩膀、右手肘、右手腕、右小指、右食指进行判断
        if results.pose_landmarks.landmark[12].visibility >= 0.5 and \
          results.pose_landmarks.landmark[14].visibility >= 0.5 and \
          results.pose_landmarks.landmark[16].visibility >= 0.5 and \
          results.pose_landmarks.landmark[18].visibility >= 0.5 and \
          results.pose_landmarks.landmark[20].visibility >= 0.5:

            R_shoulder = self.get_position(results.pose_landmarks.landmark[12], img_w, img_h)
            R_elbow = self.get_position(results.pose_landmarks.landmark[14], img_w, img_h)
            R_wrist = self.get_position(results.pose_landmarks.landmark[16], img_w, img_h)
            R_pinky = self.get_position(results.pose_landmarks.landmark[18], img_w, img_h)
            R_index = self.get_position(results.pose_landmarks.landmark[20], img_w, img_h)
            R_finger = (int((R_pinky[0] + R_index[0])/2), int((R_pinky[1] + R_index[1])/2))

            if R_shoulder[0] < 0 or R_shoulder[1] < 0 or R_shoulder[0] > img_w or R_shoulder[1] > img_h or \
              R_elbow[0] < 0 or R_elbow[1] < 0 or R_elbow[0] > img_w or R_elbow[1] > img_h or \
              R_wrist[0] < 0 or R_wrist[1] < 0 or R_wrist[0] > img_w or R_wrist[1] > img_h or \
              R_pinky[0] < 0 or R_pinky[1] < 0 or R_pinky[0] > img_w or R_pinky[1] > img_h or \
              R_index[0] < 0 or R_index[1] < 0 or R_index[0] > img_w or R_index[1] > img_h:
                return False

            cv2.circle(frame, R_shoulder, 2, (255, 0, 0), -1)
            cv2.circle(frame, R_elbow, 2, (255, 0, 0), -1)
            cv2.circle(frame, R_wrist, 2, (255, 0, 0), -1)
            cv2.circle(frame, R_finger, 2, (255, 0, 0), -1)

            person_box_w_dis = abs(R_shoulder[0] - hoop_rect[0] + hoop_rect[2]/2) * self.pix2dis
            print("person_box_w_dis: ", person_box_w_dis, "m")

            # 投球时，手腕必须高于肩膀
            if R_wrist[1] < R_shoulder[1]:
                self.plot_basketball_inhand(frame, R_shoulder, R_elbow, R_wrist, R_finger)
                current_theta = math.degrees(np.arccos(self.get_theta(R_wrist, R_elbow, R_finger)))
                current_theta = self.get_up_down(R_wrist, R_elbow, R_finger) * (180 - current_theta)
                print("current_theta:    ", current_theta)

                if not self.is_init:
                    # 投球时，手掌必须上翻
                    if current_theta >= 0:
                        self.is_init = True
                        self.stable_time = 0
                        self.stable_theta = current_theta
                        self.stable_finger = R_finger
                        self.stable_wrist = R_wrist
                        self.stable_shoulder = R_shoulder
                    else:
                        return False
                else:
                    print("stable_theta:     ", self.stable_theta)
                    print("stable_change:    ", self.stable_theta - current_theta)

                    # 判断手掌与手臂的夹角是否稳定，以获取准备投球时真正的出球角度
                    self.not_prepare_time = 0
                    if abs(self.stable_theta - current_theta) <= 10:
                        self.stable_time += 1
                    else:
                        self.stable_time = 0
                        self.stable_theta = current_theta
                        self.stable_finger = R_finger
                        self.stable_wrist = R_wrist
                        self.stable_shoulder = R_shoulder

                    print("stable_time:      ", self.stable_time)

                    # 若手掌与手臂的夹角持续稳定，则认为准备投球，并设置准备投球状态下的相关参数
                    if self.stable_time >= self.ready_threshold:
                        if self.stable_theta >= 0:
                           self.ready_time = self.stable_time
                           self.ready_theta = self.stable_theta
                           self.ready_finger = self.stable_finger
                           self.ready_wrist = self.stable_wrist
                           self.ready_shoulder = self.stable_shoulder
                           self.ready_dis = self.get_distance(self.ready_wrist, self.ready_shoulder)
                        else:
                            self.is_init = False
                            self.stable_time = 0
                            self.stable_theta = 0
                            self.stable_finger = (-1, -1)
                            self.stable_wrist = (-1, -1)
                            self.stable_shoulder = (-1, -1)
                            self.ready_time = 0
                            self.ready_theta = 0
                            self.ready_finger = (-1, -1)
                            self.ready_wrist = (-1, -1)
                            self.ready_shoulder = (-1, -1)
                            self.ready_dis = 0
                            return False

                    print("ready_theta:      ", self.ready_theta)
                    print("ready_change:     ", self.ready_theta - current_theta)
                    print("ready_time:       ", self.ready_time)

                    # 若手掌与手臂的夹角持续稳定一段时间后发生突变，则认为已投球
                    if self.ready_time >= self.ready_threshold and self.ready_theta - current_theta >= 30:
                        self.is_shoot = True
                        self.shot_time = cv2.getTickCount()

                        # 根据大臂和小臂的变化计算投球速度
                        current_dis = self.get_distance(R_wrist, R_shoulder)
                        speed = (current_dis - self.ready_dis) / current_dis * self.max_speed

                        # 以投球前最后一次稳定的手掌角度作为投球方向，计算横向和纵向的分速度
                        shoot_cos = self.get_theta(self.ready_wrist, (self.ready_wrist[0] + 100, self.ready_wrist[1]), self.ready_finger)
                        self.speed_y = -1 * shoot_cos * speed
                        self.speed_x = np.sqrt(speed * speed - self.speed_y * self.speed_y)

                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print("shoot_cos:        ", math.degrees(np.arccos(shoot_cos)))
                        print("speed:            ", speed)
                        cv2.circle(frame, self.ready_finger, 2, (0, 0, 255), -1)
                        cv2.circle(frame, self.ready_wrist, 2, (0, 0, 255), -1)
                        cv2.circle(frame, self.ready_shoulder, 2, (0, 0, 255), -1)
                        cv2.line(frame, self.ready_finger, self.ready_wrist, (0, 255, 0), 1)

                        # 估算本次投球的落点，根据设置的难易程度，调整投球速度，使球更容易进入篮球框
                        infer_t = abs((hoop_rect[0] + hoop_rect[2]/2) - self.shot_pos[0]) * self.pix2dis/ self.speed_x
                        infer_y = self.shot_pos[1] - int((self.speed_y * infer_t - 5 * infer_t * infer_t) / self.pix2dis)
                        if abs(infer_y - (hoop_rect[1] + hoop_rect[3]/2)) <= hoop_rect[3] * self.how_easy:
                            espect_y_dis = (self.shot_pos[1] - (hoop_rect[1] + hoop_rect[3]/2)) * self.pix2dis
                            self.speed_y = (espect_y_dis + 5 * infer_t * infer_t) / infer_t

                        frame_shot = copy.deepcopy(frame)
                        self.plot_ball_in_one_pic(frame_shot)
                        cv2.imshow('test', frame_shot)

                        self.is_init = False
                        self.stable_time = 0
                        self.stable_theta = 0
                        self.stable_finger = (-1, -1)
                        self.stable_wrist = (-1, -1)
                        self.stable_shoulder = (-1, -1)
                        self.ready_time = 0
                        self.ready_theta = 0
                        self.ready_finger = (-1, -1)
                        self.ready_wrist = (-1, -1)
                        self.ready_shoulder = (-1, -1)
                        self.ready_dis = 0
                        return True
            return False
        else:
            return False

if __name__ == '__main__':
    input_type = "camera"
    basketball = BasketBall(input_type)
    #cam = cv2.VideoCapture("3.mp4")
    cam = cv2.VideoCapture(0)

    #TODO: for test only
    hoop_pic = cv2.imread("hoop.jpg")
    while True:
        i += 1
        success, frame = cam.read()
        if not success:
            break
        if i <= 130:
            continue
        print("==================================")
        print("frame: ", i)

        # TODO: for test only
        img_w = frame.shape[1]
        img_h = frame.shape[0]
        frame[100:200, img_w-105: img_w-5, :] = hoop_pic

        start = cv2.getTickCount()
        ok = basketball.process(frame)
        if ok:
            if input_type == "video":
                cv2.waitKey(30)

        end = cv2.getTickCount()
        during = (end - start) / cv2.getTickFrequency()
        print("time: ", during*1000, "ms")

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == "q":
            break
    cam.release()
    cv2.destroyAllWindows()
