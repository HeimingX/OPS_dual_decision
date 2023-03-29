import numpy as np
# import json
# from pycocotools import mask
# from skimage import measure

from detectron2.utils.visualizer import GenericMask
from detectron2.structures.masks import polygons_to_bitmask
from detectron2.structures import BoxMode


# BOX_MIN_W = 40
# BOX_MIN_H = 40
# MASK_BOX_RATIO = 0.8
# BOX_RANGE_RATIO = 0.9

def get_potential_unk(mask, box_min_w, box_min_h, mask_box_ratio, box_range_ratio, category_id=-1):
    BOX_MIN_W = box_min_w
    BOX_MIN_H = box_min_h
    MASK_BOX_RATIO = mask_box_ratio
    BOX_RANGE_RATIO = box_range_ratio

    h, w = mask.shape
    mask_unk = np.zeros_like(mask)
    mask_unk[mask > 54] = 1
    
    polygons = GenericMask(mask_unk, h, w).polygons
    potential_boxes = []
    potential_poly = []
    for idx, _poly in enumerate(polygons):
        _mask = polygons_to_bitmask([_poly], h, w)
        _mask = _mask & mask_unk
        mask_area = _mask.sum()
        if mask_area > 0:
            _bbox = extract_bboxes_from_poly(_poly)
            box_w, box_h = _bbox[2] - _bbox[0], _bbox[-1] - _bbox[1]
            box_area = box_h * box_w
            _mask_box_r = (mask_area / box_area) >= MASK_BOX_RATIO
            _box_size_r = (box_h > BOX_MIN_H) and (box_w > BOX_MIN_W)
            _box_range_r = ((box_h / h) < BOX_RANGE_RATIO) and ((box_w / w) < BOX_RANGE_RATIO)
            if _mask_box_r and _box_size_r and _box_range_r:
                potential_boxes.append(_bbox)
                potential_poly.append(_poly)
    if len(potential_boxes) > 0:
        annos = []
        for box, poly in zip(potential_boxes, potential_poly):
            # final_mask = mask
            # final_bbox = extract_bboxes(final_mask)
            annotation = {
                "iscrowd": 0,
                "bbox": box,
                "category_id": category_id,
                "segmentation": [poly],
                "bbox_mode": BoxMode.XYXY_ABS,
            }
            annos.append(annotation)
        return annos
    else:
        return None


def extract_bboxes_from_poly(poly):
    # poly: [w, h]
    a = np.array(poly).reshape((-1, 2))
    x1 = a[:, 0].min()
    x2 = a[:, 0].max()
    y1 = a[:, 1].min()
    y2 = a[:, 1].max()
    return [x1, y1, x2, y2]

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    # print("np.any(m, axis=0)",np.any(m, axis=0))
    # print("p.where(np.any(m, axis=0))",np.where(np.any(m, axis=0)))
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    boxes = np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32)


# def test():
#     ground_truth_binary_mask = np.array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
#                                         [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
#                                         [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
#                                         [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
#                                         [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
#                                         [  0,   0,   0,   0,   0,   1,   1,   1,   0,   0],
#                                         [  1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
#                                         [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
#                                         [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=np.uint8)
#     fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
#     encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
#     ground_truth_area = mask.area(encoded_ground_truth)
#     ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
#     contours = measure.find_contours(ground_truth_binary_mask, 0.5)

#     annotation = {
#             "segmentation": [],
#             "area": ground_truth_area.tolist(),
#             "iscrowd": 0,
#             "image_id": 123,
#             "bbox": ground_truth_bounding_box.tolist(),
#             "category_id": 1,
#             "id": 1
#         }

#     for contour in contours:
#         contour = np.flip(contour, axis=1)
#         segmentation = contour.ravel().tolist()
#         annotation["segmentation"].append(segmentation)
        
#     print(json.dumps(annotation, indent=4))

# if __name__ == '__main__':
#     test()




