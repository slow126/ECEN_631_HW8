import cv2          # opencv itself
import numpy as np  # matrix manipulations
from matplotlib import pyplot as plt
import os


def featureDetection():
    thresh = dict(threshold=25, nonmaxSuppression=True);
    fast = cv2.FastFeatureDetector_create(**thresh)
    return fast


def getAbsoluteScale(f0, f1):
    x_pre, y_pre, z_pre = f0
    x, y, z = f1
    scale = np.sqrt((x - x_pre) ** 2 + (y - y_pre) ** 2 + (z - z_pre) ** 2)
    return x, y, z, scale


def featureTracking(img_1, img_2, p1):
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
    st = st.reshape(st.shape[0])
    ##find good one
    p1 = p1[st == 1]
    p2 = p2[st == 1]

    return p1, p2


# rotation matrix for camera 0 relative to global?
# see calib.txt in dataset/sequences/00
def getK():
    return np.array([[7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02],
      [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02],
      [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])


def run(images):
    # initialization
    #     ground_truth =getTruePose()

    img_1 = images[0]
    img_2 = images[0]

    if len(img_1) == 3:
        gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    else:
        gray_1 = img_1
        gray_2 = img_2

    # find the detector
    detector = featureDetection()
    kp1 = detector.detect(img_1)
    p1 = np.array([kp.pt for kp in kp1], dtype='float32')
    p1, p2 = featureTracking(gray_1, gray_2, p1)

    # Camera parameters
    fc = 718.8560
    pp = (607.1928, 185.2157)
    K = getK()

    E, mask = cv2.findEssentialMat(p2, p1, fc, pp, cv2.RANSAC, 0.999, 1.0);
    _, R, t, mask = cv2.recoverPose(E, p2, p1, focal=fc, pp=pp);

    # initialize some parameters
    MAX_FRAME = 701
    MIN_NUM_FEAT = 1500

    preFeature = p2
    preImage = gray_2

    R_f = R
    t_f = t

    maxError = 0
    ret_pos = []

    for numFrame in range(2, MAX_FRAME):

        if numFrame % 20 == 0:
            print(numFrame)

        if (len(preFeature) < MIN_NUM_FEAT):
            feature = detector.detect(preImage)
            preFeature = np.array([ele.pt for ele in feature], dtype='float32')

        curImage_c = images[numFrame]

        if len(curImage_c) == 3:
            curImage = cv2.cvtColor(curImage_c, cv2.COLOR_BGR2GRAY)
        else:
            curImage = curImage_c

        kp1 = detector.detect(curImage);
        preFeature, curFeature = featureTracking(preImage, curImage, preFeature)
        E, mask = cv2.findEssentialMat(curFeature, preFeature, fc, pp, cv2.RANSAC, 0.999, 1.0);
        _, R, t, mask = cv2.recoverPose(E, curFeature, preFeature, focal=fc, pp=pp);

        absolute_scale = 1.0

        if numFrame % 20 == 0:
            print('scale', absolute_scale)



        if absolute_scale > 0.1:
            t_f = t_f + absolute_scale * R_f.dot(t)
            R_f = R.dot(R_f)
        else:
            print("crap ... bad scale:", absolute_scale)

        preImage = curImage
        preFeature = curFeature

        ret_pos.append((t_f[0], t_f[2]))

    return ret_pos


# PATH = "VO Practice Sequence/VO Practice Sequence"
# files = os.listdir(PATH)
# files.sort()
# PATH = "BYU Hallway Sequence/BYU Hallway Sequence"
# files = os.listdir(PATH)
# files.sort()
PATH = "myVid2"
files = os.listdir(PATH)
files.sort()


imgs = []
for j in range(len(files) - 1):
    imgs.append(cv2.imread(os.path.join(PATH, files[j])))

pts = run(imgs)

cx = [x[0] for x in pts]
cy = [x[1] for x in pts]
plt.plot(cx, cy, label='Odometry')


plt.title('Results for pts[0 - {}]'.format(len(cx)))
plt.grid(True)
plt.axis('equal')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.show()
