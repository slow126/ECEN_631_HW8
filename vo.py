import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



PATH1 = "VO Practice Sequence/VO Practice Sequence"
files1 = os.listdir(PATH1)
files1.sort()
txt1 = "VO_RT"

PATH2 = "BYU Hallway Sequence/BYU Hallway Sequence"
files2 = os.listdir(PATH2)
files2.sort()
txt2 = "BYU_RT"

PATH3 = "myVid3"
files3 = os.listdir(PATH3)
files3.sort()
txt3 = "myVid_RT"

CAL_PATH = "Iphone_calibration_Images"
cal_files = os.listdir(CAL_PATH)

minHessian = 400
surf = cv2.xfeatures2d.SURF_create(minHessian)
sensitive_surf = cv2.xfeatures2d.SURF_create(100)
flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

mtx = [[7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02],
      [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02],
      [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]]


def find_corners(in_img):

    gray_img = cv2.cvtColor(in_img, cv2.COLOR_RGB2GRAY)
    # gray_gpu = cv2.UMat(gray)
    pattern = (8, 6)

    objp = np.zeros((pattern[1] * pattern[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    ret, corners = cv2.findChessboardCorners(gray_img, pattern)

    annotated_gray = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    if ret is True:
        objpoints.append(objp)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        sub_corners = cv2.cornerSubPix(gray_img, np.float32(corners), (5, 5), (-1, -1), criteria)
        imgpoints.append(sub_corners)
        annotated_gray = cv2.drawChessboardCorners(annotated_gray, pattern, sub_corners, ret)

    if ret is False:
        sub_corners = None
        imgpoints = None


    # cv2.imshow("img", annotated_gray)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return annotated_gray, sub_corners, objpoints, imgpoints


def calibrateIphone(path, files):
    objpt_vec = []
    imgpt_vec = []
    for file in files:
        img = cv2.imread(os.path.join(path, file))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        annotated_gray, sub_corners, objpoints, imgpoints = find_corners(img)
        objpt_vec += [objpoints[0]]
        imgpt_vec += [imgpoints[0]]

    obj_vect = np.array(objpt_vec)
    img_vect = np.array(imgpt_vec)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_vect, img_vect, gray.shape[::-1], None, None)
    return mtx, dist, rvecs, tvecs


def find_matches(template_key, template_descript, scene_key, scene_descript, my_thresh=0.5):
    matches = flann.knnMatch(template_descript, scene_descript, 2)

    good_matches = []
    for m, n in matches:
        if (m.distance < my_thresh * n.distance):
            good_matches.append(m)

    obj_pts = []
    scene_pts = []
    for i in range(len(good_matches)):
        temp_obj = template_key[good_matches[i].queryIdx]
        obj_pts.append(temp_obj.pt)
        temp_scene_pts = scene_key[good_matches[i].trainIdx]
        scene_pts.append(temp_scene_pts.pt)
    return np.array(obj_pts), np.array(scene_pts)


def my_resize(img, scale):
    small_img = np.zeros((int(img.shape[0] / scale), int(img.shape[1] / scale), img.shape[2]))
    for i in range(img.shape[2]):
        temp = img[:, :, i]
        small_img[:, :, i] = cv2.resize(temp, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
    return small_img.astype("uint8")


def track_features(prev_img, curr_img):
    prev_key, prev_descript = surf.detectAndCompute(prev_img, None)
    curr_key, curr_descript = surf.detectAndCompute(curr_img, None)

    if len(curr_key) < 10 or len(prev_key) < 10:
        prev_key, prev_descript = sensitive_surf.detectAndCompute(prev_img, None)
        curr_key, curr_descript = sensitive_surf.detectAndCompute(curr_img, None)
        prev_key, curr_key = find_matches(prev_key, prev_descript, curr_key, curr_descript, my_thresh=0.75)
    else:
        prev_key, curr_key = find_matches(prev_key, prev_descript, curr_key, curr_descript)



    E, mask = cv2.findEssentialMat(curr_key, prev_key, mtx[0][0], (mtx[0][2], mtx[1][2]), cv2.RANSAC, 0.999, 1.0)
    # E, mask = cv2.findEssentialMat(curr_key, prev_key, 7.070912000000e+02, (6.018873000000e+02, 1.831104000000e+02), cv2.RANSAC, 0.999, 1.0)

    retval, r, t, mask = cv2.recoverPose(E, curr_key, prev_key, cameraMatrix=np.array(mtx))
    return r, t, prev_key, curr_key


def run(scale, path, files, txtFile):
    my_T = []
    C = np.identity(4)
    f = open(txtFile + ".txt", "w")

    img1 = cv2.imread(os.path.join(path, files[0]))
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    # img1 = my_resize(img1, 6)
    img2 = cv2.imread(os.path.join(path, files[0]))
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # img2 = my_resize(img2, 6)
    R, T, prev_key, _ = track_features(img1, img2)

    prev_img = img2


    for j in range(2, len(files) - 1):
        print(j)
        if j == 471:
            pause = 1
        curr_img = cv2.imread(os.path.join(path, files[j]))
        # curr_img = my_resize(curr_img, 6)
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)

        R, T, prev_key, curr_key = track_features(prev_img, curr_img)

        T = T * scale
        temp = np.concatenate((R, T), axis=1)
        flat = temp.flatten()

        for i in range(len(flat)):
            f.write("%s " % str(flat[i]))
        f.write("\n")

        prev_img = curr_img
        temp = np.concatenate((temp, np.reshape([0, 0, 0, 1], (1, 4))), axis=0)
        C = np.matmul(C, temp)
        my_T.append(C[:, 3])
    my_T = np.array(my_T)
    np.save(txtFile + ".npy", my_T)
    plt.plot(my_T[:,0], my_T[:,2])
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def plot(txtFile):
    my_T = np.load(txtFile + '.npy')
    plt.plot(my_T[:,0], my_T[:,2], label="VO")
    plt.grid(True)
    plt.axis('equal')


    # with open("VO Practice Sequence/VO Practice Sequence R and T.txt") as f:
    #     for line in f:
    #         print(map(int, line.split()))

    # with open("VO Practice Sequence/VO Practice Sequence R and T.txt") as f:
    #     data = [map(int, line.split()) for line in f]

    truth = np.loadtxt("VO Practice Sequence/VO Practice Sequence R and T.txt")

    plt.plot(truth[:,3], truth[:,11], label="truth")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot3D(my_T[:,0], my_T[:,1], my_T[:,2])
    # plt.show()





# plot(txt1)
# run(1.0, PATH1, files1, txt1)

# mtx = [[6.7741251774486568e+02, 0.0000000000000000e+00, 3.2312557438767283e+02],
# [0.0000000000000000e+00, 6.8073800850564749e+02, 2.2477413395670021e+02],
# [0.0000000000000000e+00, 0.0000000000000000e+00, 1.0000000000000000e+00]]
#
# run(0.8, PATH2, files2, txt2)
# mtx, dist, rvec, tvec = calibrateIphone(CAL_PATH, cal_files)
mtx = np.load("iphone_mtx.npy")
np.save("iphone_mtx.npy", mtx)
run(1.0, PATH3, files3, txt3)



