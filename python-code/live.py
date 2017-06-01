import detector
from Utils.dump_tools import loadh5
import cv2

# Settings
config_file = "../models/configs/picc-finetune-nopair.config"
model_dir = "../models/picc-best/"
b_save_png = True
mean_std_file = '../models/picc-best/mean_std.h5'
num_keypoint = 50
# img_file = '/home/bryan/LIFT/data/testimg/img1.jpg'
# video_file = '../data/test.mp4'
# video_output = '../results/test_out.avi'
img_dir = '../data/frames'
img_out_dir = '../results/frames'

floatX = 'float32'

def main():
    #cap = cv2.VideoCapture(0) # from webcam
    # cap = cv2.VideoCapture(video_file) # from video
    mean_std_dict = loadh5(mean_std_file)

    # open frame once to get the image info for initialization
    # _, image_color = cap.read()
    image_color = cv2.imread(img_dir + '/' + '0.png')
    image_gray = cv2.cvtColor(image_color,
                    cv2.COLOR_BGR2GRAY).astype(floatX)
    detect = detector.Detector(config_file, model_dir, num_keypoint, image_gray, b_save_png, mean_std_dict)

    # for counting purpose
    index = 424

    while(True):
        # _, image_color = cap.read()
        fname = img_dir + '/' + str(index) + '.png'
        fname_out = img_out_dir + '/' + str(index) + '.png'
        image_color = cv2.imread(fname)
        image_gray = cv2.cvtColor(image_color,
                        cv2.COLOR_BGR2GRAY).astype(floatX)

        # detect keypoints
        frame = detect.detect_keypoint(image_gray, image_color)
        cv2.imshow('frame',frame)

        # Compute orientation
        detect.compute_orientation(image_gray)

        # Compute descriptor
        detect.compute_descriptor(image_gray, index)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # save frames
        cv2.imwrite(fname_out, frame)
        cv2.waitKey(100)
        index += 1

        if index >= 546:
            break

    cap.release()
    cv2.destroyAllWindows()

main()
