import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()


def _cv2():
    import cv2
    return cv2


def to_image_frame(Hinv, loc):
    """
    Given H^-1 and world coordinates, returns (u, v) in image coordinates.
    """
    locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
    if locHomogenous.ndim > 1:
        loc_tr = np.transpose(locHomogenous)
        loc_tr = np.matmul(Hinv, loc_tr)  # to camera frame
        locXYZ = np.transpose(loc_tr/loc_tr[2])  # to pixels (from millimeters)
        return locXYZ[:, :2].astype(int)
    else:
        locHomogenous = np.dot(Hinv, locHomogenous)  # to camera frame
        locXYZ = locHomogenous / locHomogenous[2]  # to pixels (from millimeters)
        return locXYZ[:2].astype(int)


def line_cv(im, ll, value, width):
    cv2 = _cv2()
    for tt in range(ll.shape[0] - 1):
        cv2.line(im, (ll[tt][1], ll[tt][0]), (ll[tt + 1][1], ll[tt + 1][0]), value, width)


def text_cv(im, text, org, value):
    cv2 = _cv2()
    cv2.putText(im, text, org, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=value)


def draw_heatmap(im, pred_data, cmap):
    cv2 = _cv2()
    nSmp = pred_data.shape[0]
    nPed = pred_data.shape[1]

    K_im = np.zeros((nSmp, im.shape[0], im.shape[1]), np.uint8)
    for kk in range(nSmp//8):
        for ii in range(nPed):
            preds_our_XY_ik = to_image_frame(Hinv, pred_data[kk, ii])
            line_cv(K_im[kk], preds_our_XY_ik, value=1, width=10)
    lines_im = np.sum(K_im, axis=0).astype(np.uint8)
    lines_im = cv2.blur(lines_im, (15, 15))
    my_dpi = 96
    plt.figure(figsize=(lines_im.shape[1]/my_dpi, lines_im.shape[0]/my_dpi), dpi=my_dpi)

    sns.heatmap(lines_im, cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    plt.margins(0, 0)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig('../tmp.png')
    heatmap = cv2.imread('../tmp.png')
    cv2.addWeighted(im, 1, heatmap, 1, 0, im)


def draw_gt_data(im): # SDD dataset
    # FIXME: set dataset file
    SDD_dataset = '../data/trajnet/train/stanford/nexus_9.npz'
    SDD_dataset = '../data/trajnet/train/stanford/hyang_6.npz'

    dataset = np.load(SDD_dataset)
    obsvs = dataset['dataset_x']
    preds = dataset['dataset_y']
    samples = np.concatenate((obsvs, preds), axis=1)

    max_x = np.max(samples[:, :, 0])
    min_x = np.min(samples[:, :, 0])

    for ii in range(len(samples)):
        pi = samples[ii]
        obsv_XY = to_image_frame(Hinv, samples[ii])
        line_cv(im, obsv_XY, (30, 100, 0), 3)


def main():
    global Hinv
    args = parse_args()
    cv2 = _cv2()

    dataset_name = args.dataset
    preds_dir = args.preds_dir
    out_dir_main = args.output_dir

    if not os.path.isdir(preds_dir):
        print(f"[ERR] Predictions directory '{preds_dir}' does not exist. Use --preds-dir to point to your npz outputs.")
        sys.exit(1)

    available_npz = [
        os.path.join(dp, f)
        for dp, _, files in os.walk(preds_dir)
        for f in files
        if f.endswith('.npz') and 'stats' not in f
    ]
    if not available_npz:
        print(f"[ERR] No prediction npz files found under '{preds_dir}'. Ensure your model saved predictions or choose another --preds-dir.")
        sys.exit(1)

    Hinv = np.eye(3)
    im_size = (480, 480, 3)

    homography_file = args.homography if args.homography else os.path.join('data', dataset_name, 'H.txt')
    if os.path.exists(homography_file):
        Hinv = np.linalg.inv(np.loadtxt(homography_file))
    else:
        print('[INF] No homography file')

    video_file = args.video if args.video else os.path.join('data', dataset_name, 'video.avi')
    image_file = args.image if args.image else os.path.join('data', dataset_name, 'reference.jpg')
    if os.path.exists(video_file):
        print('[INF] Using video file' + video_file)
        cap = cv2.VideoCapture(video_file)
        time_offset = -12
        use_ref_im = False
        ref_im = None
    elif os.path.exists(image_file):
        print('[INF] Using image file ' + image_file)
        cap = None
        use_ref_im = True

        ref_im = np.zeros((600, 600, 3), dtype=np.uint8)
        Hinv = np.zeros((3, 3))
        Hinv[0, 1], Hinv[1, 0], Hinv[2, 2] = 0.6, 0.6, 1
        Hinv[0, 2], Hinv[1, 2] = -400, 0

        ref_im = cv2.imread(image_file)
        Hinv = np.zeros((3, 3))
        Hinv[0, 1], Hinv[1, 0], Hinv[2, 2] = 1, 1, 1
    else:  # toy dataset
        print('[INF] No image nor video file')
        cap = None
        use_ref_im = False
        ref_im = None
        Hinv[0, 0], Hinv[1, 1] = 200, 200
        Hinv[0, 2], Hinv[1, 2] = 240, 240

    epc_counter = 0
    for filename in sorted(available_npz):
        f = os.path.basename(filename)
        epc_str = f[:f.rfind('-')]
        if epc_str.isdigit():
            epc = int(epc_str)
        else:
            epc = epc_counter
            epc_counter += 1
        if epc % 1000 != 0:
            continue
        print('[INF] Plotting results from ' + filename)

        out_file = os.path.join(out_dir_main, '%05d.png' % epc)
        if os.path.exists(out_file):
            continue
        data = np.load(filename)
        obsvs = data['obsvs']
        preds_gtt = data['preds_gtt']
        preds_our = data['preds_our']
        preds_lnr = data['preds_lnr']
        time_stamp = data['timestamp']

        nPed = obsvs.shape[0]
        nPast = obsvs.shape[1]
        nNext = preds_lnr.shape[1]
        nSmp = preds_our.shape[0]

        if nPed < 2:
            continue

        if cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, time_stamp + time_offset))
            ret, im = cap.read()
            if not ret:
                break
        elif use_ref_im:
            im = np.copy(ref_im)
        else:
            im = np.ones(im_size, dtype=np.uint8) * 128

        preds_gtt_aug = np.concatenate((obsvs[:, -1].reshape((nPed, 1, 2)), preds_gtt), axis=1)
        preds_lnr_aug = np.concatenate((obsvs[:, -1].reshape((nPed, 1, 2)), preds_lnr), axis=1)
        cur_loc_K = np.vstack([obsvs[:, -1].reshape((1, nPed, 1, 2)) for _ in range(nSmp)])
        preds_our_aug = np.concatenate((cur_loc_K, preds_our), axis=2)

        cmap = sns.dark_palette("purple")
        draw_heatmap(im, preds_our_aug, cmap)
        text_cv(im, 'Epoch= %05d' % epc, (15, 50), (50, 50, 250))

        for ii in range(nPed):
            obsv_XY = to_image_frame(Hinv, obsvs[ii])
            line_cv(im, obsv_XY, (255, 20, 0), 2)

        out_dir = out_file[:out_file.rfind('/')]
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(out_file, im)
        print('[INF] Writing image to ', out_file)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize prediction npz files as images.')
    parser.add_argument('--preds-dir', default='medium/toy/socialWays', help='Directory containing prediction npz files.')
    parser.add_argument('--output-dir', default='medium/figs/socialWays/', help='Directory where rendered images are stored.')
    parser.add_argument('--dataset', default='toy', help='Dataset name used for resolving homography/video/image files.')
    parser.add_argument('--homography', help='Optional path to a homography matrix text file.')
    parser.add_argument('--video', help='Optional path to a reference video file.')
    parser.add_argument('--image', help='Optional path to a reference image file.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
