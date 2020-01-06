import os
import cv2
import numpy as np

from face_detection import face_detection
from face_points_detection import face_points_detection
from face_swap import warp_image_2d, warp_image_3d, mask_from_points, apply_mask, correct_colours, transformation_from_points
from generate_synthetic_face import generate_synthetic_face


def select_face(im, r=10):
    faces = face_detection(im)

    if len(faces) == 0:
        print('Detect 0 Face !!!')
        exit(-1)

    if len(faces) == 1:
        bbox = faces[0]
    else:
        bbox = []

        def click_on_face(event, x, y, flags, params):
            if event != cv2.EVENT_LBUTTONDOWN:
                return

            for face in faces:
                if face.left() < x < face.right() and face.top() < y < face.bottom():
                    bbox.append(face)
                    break

        im_copy = im.copy()
        for face in faces:
            # draw the face bounding box
            cv2.rectangle(im_copy, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 1)
        cv2.imshow('Click the Face:', im_copy)
        cv2.setMouseCallback('Click the Face:', click_on_face)
        while len(bbox) == 0:
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        bbox = bbox[0]

    points = np.asarray(face_points_detection(im, bbox))

    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)

    x, y = max(0, left - r), max(0, top - r)
    w, h = min(right + r, im_h) - x, min(bottom + r, im_w) - y

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y + h, x:x + w]

if __name__ == '__main__':
    src_path = "imgs/source.jpg"
    dst_path = "imgs/test5.jpg"
    result_path = "results/output.jpg"

    # Generate synthetic face using TL-GAN
    image = generate_synthetic_face()
    cv2.imwrite(src_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imshow("Source", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Face swapping on destination image using the synthetic face as source
    src_img = cv2.imread(src_path)
    dst_img = cv2.imread(dst_path)

    # Select src face
    src_points, src_shape, src_face = select_face(src_img)

    # Select dst face
    dst_points, dst_shape, dst_face = select_face(dst_img)

    h, w = dst_face.shape[:2]

    warped_dst_img = warp_image_3d(dst_face, dst_points[:48], src_points[:48], src_face.shape[:2])
    src_face = correct_colours(warped_dst_img, src_face, src_points)

    # Warp face
    warped_src_face = warp_image_2d(src_face, transformation_from_points(dst_points, src_points), (h, w, 3))

    ## Mask for blending
    mask = mask_from_points((h, w), dst_points)
    mask_src = np.mean(warped_src_face, axis=2) > 0
    mask = np.asarray(mask * mask_src, dtype=np.uint8)

    ## Correct color
    warped_src_face = apply_mask(warped_src_face, mask)
    dst_face_masked = apply_mask(dst_face, mask)
    warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)

    ## Shrink the mask
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    ##Poisson Blending
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

    x, y, w, h = dst_shape
    dst_img_cp = dst_img.copy()
    dst_img_cp[y:y + h, x:x + w] = output
    output = dst_img_cp

    dir_path = os.path.dirname(result_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    cv2.imwrite(result_path, output)
    cv2.imshow("From", dst_img)
    cv2.imshow("To", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
