import cv2
import numpy as np
import os


PATH = "VO Practice Sequence/VO Practice Sequence"
files = os.listdir(PATH)
files.sort()

minHessian = 400
surf = cv2.xfeatures2d.SURF_create(minHessian)
flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

mtx = [[7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02],
      [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02],
      [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]]


def my_Homography(template, scene):
    template_key, template_descript = surf.detectAndCompute(template, None)
    scene_key, scene_descript = surf.detectAndCompute(scene, None)

    # template_key, template_key = orb.detectAndCompute(template, None)
    # scene_key, scene_descript = orb.detectAndCompute(scene, None)

    matches = flann.knnMatch(template_descript, scene_descript, 2)

    my_thresh = 0.5
    good_matches = []
    for m, n in matches:
        if (m.distance < my_thresh * n.distance):
            good_matches.append(m)

    obj_pts = []
    scene_pts = []
    good_obj_descript = []
    good_scene_descript = []
    for i in range(len(good_matches)):
        temp_obj = template_key[good_matches[i].queryIdx]
        obj_pts.append(temp_obj.pt)
        temp_scene_pts = scene_key[good_matches[i].trainIdx]
        scene_pts.append(temp_scene_pts.pt)
        # temp_descript = template_descript[good_matches[i].queryIdx]
        # good_obj_descript.append(temp_descript)
        # temp_descript = scene_descript[good_matches[i].queryIdx]
        # good_scene_descript.append(temp_descript)


    H = cv2.findHomography(np.array(obj_pts), np.array(scene_pts), cv2.RANSAC)
    # return H, np.array(obj_pts), np.array(good_obj_descript), np.array(scene_pts), np.array(good_scene_descript), good_matches
    return H, np.array(obj_pts), np.array(scene_pts), good_matches



def track_features(current_img, next_img):
    H, current_key,  next_key, good_matches = my_Homography(current_img, next_img)
    F, mask = cv2.findFundamentalMat(current_key, next_key, cv2.FM_LMEDS)
    # E = cv2.findEssentialMat(points1=current_key, points2=next_key, focal=1.0, pp=(6.018873000000e+02, 1.831104000000e+02), method=cv2.RANSAC, threshold=2, prob=0.5)
    E = cv2.findEssentialMat(points1=current_key, points2=next_key, cameraMatrix=np.array(mtx), method=cv2.RANSAC, threshold=2, prob=0.5)
    # r1, r2, t = cv2.decomposeEssentialMat(E[0])
    retval, r, t, mask = cv2.recoverPose(E, current_key, next_key, np.array(mtx))
    return r, t


def run():
    scale = 1.0
    for j in range(len(files) - 1):
        img = cv2.imread(os.path.join(PATH, files[j]))
        next_img = cv2.imread(os.path.join(PATH, files[j+1]))
        R, T = track_features(img, next_img)
        temp = np.concatenate((R,T), axis=1)
        temp = np.concatenate((temp, np.reshape([0, 0, 0, 1], (1, 4))), axis=0)
        T_k = np.concatenate(([R, T], [0,1]), axis=1)


run()