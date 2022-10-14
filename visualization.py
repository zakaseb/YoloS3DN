import numpy as np
import cv2
import os

from evaluator import kitti

def compute_shortest_distance(l, w, h, x, y, z, ry):
    # compute rotational matrix around yaw axis
    c = np.cos(ry)
    s = np.sin(ry)
    R = np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))

    #corners_3d[0, :] = corners_3d[0, :] + x
    #corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z

    return corners_3d[2, :].min()


def convert_cls(c):
    if c == 'Car':
        return 1
    elif c == 'Pedestrian':
        return 2
    elif c == 'Cyclist':
        return 3
    else:
        assert 'no~'

def GenResultImage_fromkitti(kittihome, kittiresultdir, resultpath):
    kittihome = "/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Testing"
    kittiresultdir = "/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/Project/Stereo3D/output/test/data/"
    resultpath = "/home/zakaseb/Thesis/YoloStereo3D/Stereo3D/"
    txt_list = os.listdir(kittiresultdir)

    for txt in txt_list:
        label_list = []
        imgnum = txt.split('.')[0]
        left_img = cv2.imread(kittihome + 'image_2/' + imgnum + '.png')
        #calib = read_calib_file(kittihome + 'calib/' + imgnum + '.txt')
        #K = calib['P2']

        with open(kittiresultdir + imgnum + '.txt', mode='r') as f:
            bbox = f.readlines()
            if len(bbox) == 0:
                pass
            else:
                bbox = [i.replace("\n", "") for i in bbox]
                for box in bbox:
                    box = box.split(' ')
                    cls, xmin, ymin, xmax, ymax = box[0], float(box[4]), float(box[5]), float(box[6]), float(box[7])
                    h, w, l = float(box[8]), float(box[9]), float(box[1])
                    x, y, z = float(box[11]), float(box[12]), float(box[13])
                    ry = float(box[14])
                    score = float(box[15]) if box.__len__() == 16 else 1.00

                    #xmin, ymin, xmax, ymax = compute_2d_bbox_from_3d_bbox(l, w, h, x, y, z, ry, K)
                    depth = compute_shortest_distance(l, w, h, x, y, z, ry)
                    label = '%d %d %d %d %d %.2f %.2f %d %.2f %.2f\n'\
                        % (
                            convert_cls(cls), int(xmin), int(ymin), int(xmax), int(ymax), depth, 
                            0, 0, 0, score
                            )
                    label_list.append(label)
                    continue

        if len(label_list) == 0:
            pass
        else:
            for obj in label_list:
                obj = obj.replace("\n", "").split(' ')
                xmin, ymin, xmax, ymax = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                cv2.putText(left_img, obj[5], (xmin+1, ymin+1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.rectangle(left_img, (xmin, ymin), (xmax ,ymax), (0, 255, 255), 3)
                continue
        cv2.imwrite(resultpath + imgnum + '.png', left_img)
        print(imgnum)
        continue