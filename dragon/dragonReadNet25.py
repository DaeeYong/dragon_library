import cv2
import numpy as np
'''
이미지와 관련된 함수.
Openpose의 body25모델을 기준으로 함.

일반적인 함수 사용 순서

get_network -> preprocessing_image_blob -> get_position_from_netoutput
-> show_marked_image

사용 예제코드는 doc/ 참고바람.
'''

BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist", 5: "LShoulder",
                        6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip", 10: "RKnee", 11: "RAnkle",
                        12: "LHip", 13: "LKnee", 14: "LAnkle", 15: "REye", 16: "LEye", 17: "REar",
                        18: "LEar", 19: "LBigToe", 20: "LSmallToe", 21: "LHeel", 22: "RBigToe",
                        23: "RSmallToe", 24: "RHeel", 25: "Background"}

POSE_PAIRS_BODY_25 = [[0, 1],[0, 15], [15, 17], [0, 16], [16, 18],[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
                       [1, 8], [8, 9], [9, 10], [10, 11], [11, 22], [22, 23], [11, 24],
                       [8, 12], [12, 13], [13, 14], [14, 19], [19, 20], [14, 21]]

def get_network(prototxt_path:str, caffemodel_path:str):
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    return net

def preprocess_image_blob(image:cv2.UMat, input_width=368, input_height=368):
    inblob = cv2.dnn.blobFromImage(image, 1.0 / 255, (input_width, input_height), mean=(0, 0, 0), swapRB=False, crop=False)
    return inblob

def get_position_from_netoutput(output, image):
    raw_point_list = []
    rounded_point_list = []
    confidence_list = []
    image_height, image_width, _ = image.shape
    for i in range(0, 25):
        prob_map = output[0, i, :, :]
        _, confidence, _, point = cv2.minMaxLoc(prob_map)

        raw_point_list.append((point[0] * image_width / output.shape[3], point[1] * image_height / output.shape[2]))
        rounded_point_list.append((int(point[0] * image_width / output.shape[3]), int(point[1] * image_height / output.shape[2])))
        confidence_list.append(confidence)

    return rounded_point_list, confidence_list, raw_point_list

def get_position_from_netoutput_test(output, image):
    height, width = image.shape[:2]
    
    raw_point_list = []
    rounded_point_list = []
    confidence_list = []
    for i in range(output.shape[1]):
        confidence_map = output[0, i, :, :]
        min_val, confidence, min_loc, point = cv2.minMaxLoc(confidence_map)

        x = int((point[0] / output.shape[3]) * width)
        y = int((point[1] / output.shape[2]) * height)
        # 관절 위치 정보 출력
        print(f"Joint {i}: ({x}, {y}) Confidence: {confidence}")
        rounded_point_list.append((x, y))
    return rounded_point_list

def show_marked_image(image, joint_pos_list, confidence_list):
    for pair in POSE_PAIRS_BODY_25:
        cv2.line(image, joint_pos_list[pair[0]], joint_pos_list[pair[1]], (0, 255, 0), 2)

    for idx, point in enumerate(joint_pos_list):
        cv2.circle(image, point, 4, (0, 0, 255), thickness=2, lineType=cv2.FILLED)
        cv2.putText(image, "{}".format(idx), (point[0]+10, point[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)
        print(f'idx : {idx}[{BODY_PARTS_BODY_25[idx]}] x : {point[0]}, y : {point[1]}, confidence : {round(confidence_list[idx], 5)}')


    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

