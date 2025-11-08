import numpy as np
from scipy.spatial.distance import cdist

def match_and_evaluate(detected_centers, gt_centers, max_dist=50):
    """
    Сопоставляет найденные центры с GT и считает ошибки.
    """
    if len(detected_centers) == 0 or len(gt_centers) == 0:
        return [], [], float('inf')

    det = np.array(detected_centers)
    gt = np.array(gt_centers)
    dists = cdist(det, gt)

    matches = []
    errors = []
    used_det = set()
    used_gt = set()

    for _ in range(min(len(det), len(gt))):
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        if dists[i, j] > max_dist:
            break
        if i in used_det or j in used_gt:
            dists[i, j] = 1e9
            continue

        matches.append((det[i], gt[j]))
        errors.append(dists[i, j])
        used_det.add(i)
        used_gt.add(j)
        dists[i, :] = 1e9
        dists[:, j] = 1e9

    mean_error = np.mean(errors) if errors else float('inf')
    return matches, errors, mean_error