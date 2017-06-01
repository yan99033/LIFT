import h5py
import cv2
import numpy as np
from matplotlib import pyplot as plt

min_match_count = 10
match_count = 50

def readh5(h5node):
    ''' Recursive function to read h5 nodes as dictionary '''

    dict_from_file = {}
    for _key in h5node.keys():
        if isinstance(h5node[_key], h5py._hl.group.Group):
            dict_from_file[_key] = readh5(h5node[_key])
        else:
            dict_from_file[_key] = h5node[_key].value

    return dict_from_file

def loadh5(dump_file_full_name):
    ''' Loads a h5 file as dictionary '''

    with h5py.File(dump_file_full_name, 'r') as h5file:
        dict_from_file = readh5(h5file)

    return dict_from_file

frame1 = 0
frame2 = 1
def main():
    frame1 = 0
    frame2 = 1

    for i in range(84):

        # read files
        fname1 = str(frame1) + '.h5'
        fname2 = str(frame2) + '.h5'

        # read original images
        img1 = cv2.imread('../data/frames/' + str(frame2) + '.png')
        img2 = cv2.imread('../data/frames/' + str(frame1) + '.png')

        # load dictionaries
        dict1 = loadh5(fname1)
        dict2 = loadh5(fname2)

        # load keypoints and descriptors
        kp1 = dict1['keypoints']
        desc1 = dict1['descriptors']
        kp2 = dict2['keypoints']
        desc2 = dict2['descriptors']

        # convert to opencv keypoints
        cv_kp1 = []
        for kp in kp1:
            X, Y = kp[0], kp[1]
            size = kp[2]
            # create keypoint object
            keypoint = cv2.KeyPoint(x=X, y=Y, _size=size)
            # keypoint.pt = X, Y
            # keypoint.size = size
            cv_kp1.append(keypoint)

        cv_kp2 = []
        for kp in kp2:
            X, Y = kp[0], kp[1]
            size = kp[2]
            # create keypoint object
            keypoint = cv2.KeyPoint(x=X, y=Y, _size=size)
            cv_kp2.append(keypoint)

        # brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_L2) #, crossCheck=True)
        matches = bf.match(desc1, desc2)

        good = []
        for m in matches:
            if m.distance < 100:
                good.append(m)

        if len(good) > min_match_count:
            src_pts = np.float32([cv_kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([cv_kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

        # matches = sorted(matches, key=lambda x: x.distance)
        else:
            print "Not enough matches are found - %d/%d" % (len(good), min_match_count)
            matchesMask = None


        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)


        frame = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, good, None, **draw_params)

        # plt.imshow(img3), plt.show()

        # #only draw top 20 matches
        # matches = matches[:5]
        # match_idx = []
        # for match in matches:
        #     trainIdx = match.trainIdx
        #     queryIdx = match.queryIdx
        #     match_idx.append([trainIdx, queryIdx])
        #
        # for match in match_idx:
        #     pt1 = int(kp1[match[0], 0]), int(kp1[match[0], 1])
        #     pt2 = int(kp2[match[1], 0]), int(kp2[match[1], 1])
        #     print('pt1:', pt1)
        #     print('pt2:', pt2)
        #     cv2.arrowedLine(img2, pt1, pt2, (0,255,0), thickness=3, line_type=4)

        output = str(frame1) + '.png'
        cv2.imwrite(output, frame)
        # plt.imshow(img2), plt.show()
        frame1 += 1
        frame2 += 1


main()