def iou(boxA, boxB):
    # boxes are (x, y, w, h)
    ax1, ay1, aw, ah = boxA; ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = boxB; bx2, by2 = bx1 + bw, by1 + bh
    interX1, interY1 = max(ax1, bx1), max(ay1, by1)
    interX2, interY2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, interX2 - interX1) * max(0, interY2 - interY1)
    areaA = aw * ah
    areaB = bw * bh
    union = areaA + areaB - inter + 1e-6
    return inter / union