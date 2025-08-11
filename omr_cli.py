import argparse, glob, os, sys, csv, json, random
import numpy as np
import cv2 as cv

# ===================== TEMPLATE (matches the A4 sheet I gave you) =====================
DESIGN_W, DESIGN_H = 2480, 3508
MARGIN = 140
QUESTIONS_TOTAL = 30
OPTIONS = ['A','B','C','D','E']
PER_COL = QUESTIONS_TOTAL // 2
COLS = 2
COL_X = [MARGIN + 120, DESIGN_W//2 + 60]  # starting x for each column (before number width)
START_Y = MARGIN + 160 + 140  # below header
ROW_H = 90
BUBBLE_R = 22
GAP_OPT = 70
NUM_W = 60

def template_layout():
    """Return [(q, opt, cx, cy, r), ...] in DESIGN coordinates."""
    rows = []
    for c in range(COLS):
        for r in range(PER_COL):
            qnum = c*PER_COL + r + 1
            y = START_Y + r*ROW_H
            x = COL_X[c] + NUM_W
            for i, opt in enumerate(OPTIONS):
                cx = x + i*GAP_OPT
                cy = y
                rows.append((qnum, opt, cx, cy, BUBBLE_R))
    return rows

TEMPLATE_ENTRIES = template_layout()

# ===================== IMAGE UTILS =====================
def read_bgr(path: str) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img

def save_image(path: str, img: np.ndarray):
    if not cv.imwrite(path, img):
        raise ValueError(f"Failed to write: {path}")

def resize_max(img: np.ndarray, max_side: int = 1600):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    s = max_side / max(h, w)
    return cv.resize(img, (int(w*s), int(h*s)), interpolation=cv.INTER_AREA), s

def canny_edges(gray: np.ndarray):
    blur = cv.GaussianBlur(gray, (5,5), 0)
    v = np.median(blur)
    lo = int(max(0, 0.66*v))
    hi = int(min(255, 1.33*v))
    edges = cv.Canny(blur, lo, hi)
    edges = cv.dilate(edges, np.ones((3,3), np.uint8), 1)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((5,5), np.uint8), 1)
    return edges

def line_angle(x1,y1,x2,y2):
    ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
    return ang if ang >= 0 else ang + 180

def average_line(lines):
    pts = np.array(lines, dtype=np.float32)
    return tuple(np.mean(pts, axis=0).tolist())

def intersect(p1, p2):
    x1,y1,x2,y2 = p1
    x3,y3,x4,y4 = p2
    A1, B1, C1 = y2-y1, x1-x2, (y2-y1)*x1 + (x1-x2)*y1
    A2, B2, C2 = y4-y3, x3-x4, (y4-y3)*x3 + (x3-x4)*y3
    det = A1*B2 - A2*B1
    if abs(det) < 1e-6: return None
    x = (B2*C1 - B1*C2)/det
    y = (A1*C2 - A2*C1)/det
    return (x, y)

def order_quad(pts: np.ndarray):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def warp(img: np.ndarray, quad: np.ndarray, border=2, fixed=True):
    """
    If fixed=True, warp the detected page to the exact template canvas
    (2480x3508). This removes scale/aspect drift so annotations align.
    """
    if fixed:
        W, H = DESIGN_W, DESIGN_H
        dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    else:
        tl, tr, br, bl = quad
        wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
        hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
        W = max(int(max(wA, wB)), 100); H = max(int(max(hA, hB)), 100)
        dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)

    M = cv.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv.warpPerspective(img, M, (W, H), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return cv.copyMakeBorder(warped, border, border, border, border, cv.BORDER_CONSTANT, value=(255,255,255))

def hough_quad(edges: np.ndarray):
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                           minLineLength=int(edges.shape[1]*0.4), maxLineGap=20)
    if lines is None: return None
    horiz, vert = [], []
    for l in lines[:,0]:
        x1,y1,x2,y2 = map(int,l)
        ang = line_angle(x1,y1,x2,y2)
        if (ang < 15) or (ang > 165): horiz.append((x1,y1,x2,y2))
        elif 75 <= ang <= 105:       vert.append((x1,y1,x2,y2))
    if not horiz or not vert: return None
    ymid = lambda L: (L[1]+L[3])/2
    xmid = lambda L: (L[0]+L[2])/2
    top    = min(horiz, key=ymid)
    bottom = max(horiz, key=ymid)
    left   = min(vert,  key=xmid)
    right  = max(vert,  key=xmid)
    t = average_line([top]); b = average_line([bottom])
    l = average_line([left]); r = average_line([right])
    tl = intersect(t, l); tr = intersect(t, r)
    br = intersect(b, r); bl = intersect(b, l)
    if None in (tl,tr,br,bl): return None
    quad = np.array([tl,tr,br,bl], dtype=np.float32)
    if np.any(~np.isfinite(quad)): return None
    return order_quad(quad)

def contour_quad(edges: np.ndarray):
    cnts, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    for c in sorted(cnts, key=cv.contourArea, reverse=True)[:10]:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            return order_quad(approx.reshape(4,2).astype(np.float32))
    return None

def threshold_image(gray: np.ndarray, method: str):
    if method == "otsu":
        blur = cv.GaussianBlur(gray, (5,5), 0)
        _, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    else:
        th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY, 31, 10)
    # Clean noise for fill_ratio
    th = cv.medianBlur(th, 3)
    return th

# ===================== SCORING =====================
def scale_template_to(wW, wH):
    """Return scaled positions for current warped size."""
    sx = wW / DESIGN_W
    sy = wH / DESIGN_H
    scaled = []
    for q, opt, cx, cy, r in TEMPLATE_ENTRIES:
        scaled.append((q, opt, int(round(cx*sx)), int(round(cy*sy)), int(round(r*min(sx, sy)))))
    return scaled

def fill_ratio(binary_img, cx, cy, r):
    """binary_img: 0..255; count dark pixels inside circle."""
    h, w = binary_img.shape[:2]
    r = max(5, r)
    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = min(w-1, cx + r), min(h-1, cy + r)
    roi = binary_img[y0:y1+1, x0:x1+1]
    if roi.size == 0: return 0.0
    Y, X = np.ogrid[:roi.shape[0], :roi.shape[1]]
    mask = (X - (cx - x0))**2 + (Y - (cy - y0))**2 <= r*r
    vals = roi[mask]
    # Count "black" pixels (filled marks)
    black = np.count_nonzero(vals < 128)
    return black / float(mask.sum() + 1e-6)

def decide_answer(ratios, min_fill=0.28, min_margin=0.08):
    """ratios: dict opt->ratio. Returns (choice or None/'multi', confidence)."""
    items = sorted(ratios.items(), key=lambda kv: kv[1], reverse=True)
    top_opt, top_val = items[0]
    if len(items) > 1:
        next_val = items[1][1]
    else:
        next_val = 0.0
    if top_val < min_fill:
        return (None, top_val)
    if (top_val - next_val) < min_margin:
        return ('multi', top_val - next_val)
    return (top_opt, top_val)

def score_sheet(warped_bgr, th_binary, answer_key, annotate=True):
    """Returns (per_question dict, score_count, annotated_image)."""
    h, w = th_binary.shape[:2]
    positions = scale_template_to(w, h)
    per_q = {q: {"selected": None, "ratios": {}, "correct": answer_key.get(q)} for q in range(1, QUESTIONS_TOTAL+1)}

    # compute ratios
    for q, opt, cx, cy, r in positions:
        ratio = fill_ratio(th_binary, cx, cy, r)
        per_q[q]["ratios"][opt] = ratio

    # decide per question
    correct_count = 0
    for q in range(1, QUESTIONS_TOTAL+1):
        choice, conf = decide_answer(per_q[q]["ratios"])
        per_q[q]["selected"] = choice
        per_q[q]["confidence"] = float(conf)
        per_q[q]["is_correct"] = (choice == per_q[q]["correct"])
        if per_q[q]["is_correct"]:
            correct_count += 1

    # annotate image
    ann = warped_bgr.copy()
    for q, opt, cx, cy, r in positions:
        sel = per_q[q]["selected"]
        correct = per_q[q]["correct"]
        color = (200, 200, 200)  # default
        if opt == correct:
            color = (0, 200, 0)   # green ring on correct option
        if opt == sel and sel not in (None, 'multi'):
            color = (0, 0, 255)   # red ring on chosen option
        cv.circle(ann, (cx, cy), r+4, color, 2)
    return per_q, correct_count, ann

# ===================== CORE PIPELINE =====================
def process_one(path: str, method: str, debug: bool, max_side: int, seed: int | None):
    if seed is not None:
        random.seed(seed)

    # 1) Build a random answer key (A–E) per question
    answer_key = {q: random.choice(OPTIONS) for q in range(1, QUESTIONS_TOTAL+1)}

    # 2) Read + warp + thresh (same as before)
    bgr0 = read_bgr(path)
    bgr, _ = resize_max(bgr0, max_side)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    edges = canny_edges(gray)

    quad = hough_quad(edges)
    if quad is None:
        quad = contour_quad(edges)

    if quad is not None:
        warped = warp(bgr, quad)
        g = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        th = threshold_image(g, method)
    else:
        warped = bgr
        th = threshold_image(gray, method)

    # 3) Score against the random key
    per_q, correct_count, ann = score_sheet(warped, th, answer_key)

    # 4) Save outputs
    base, _ = os.path.splitext(path)
    out_bin = f"{base}.warped_{method}.png"
    out_ann = f"{base}.annotated_{method}.jpg"
    out_csv = f"{base}.score_{method}.csv"
    out_key = f"{base}.answer_key_{method}.csv"
    out_sum = f"{base}.summary_{method}.json"

    # Ensure 3-channel for saving bin
    bin3 = cv.cvtColor(th, cv.COLOR_GRAY2BGR)
    save_image(out_bin, bin3)
    save_image(out_ann, ann)

    # per-question CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "selected", "correct", "is_correct", "confidence"] + [f"ratio_{o}" for o in OPTIONS])
        for q in range(1, QUESTIONS_TOTAL+1):
            row = per_q[q]
            ratios = [f"{row['ratios'].get(o,0):.3f}" for o in OPTIONS]
            w.writerow([q, row["selected"], row["correct"], int(row["is_correct"]), f"{row['confidence']:.3f}"] + ratios)

    # answer key CSV
    with open(out_key, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "correct"])
        for q in range(1, QUESTIONS_TOTAL+1):
            w.writerow([q, answer_key[q]])

    # JSON summary
    summary = {
        "file": os.path.basename(path),
        "method": method,
        "total_questions": QUESTIONS_TOTAL,
        "score": correct_count,
        "percent": round(100.0 * correct_count / QUESTIONS_TOTAL, 2),
    }
    with open(out_sum, "w") as f:
        json.dump(summary, f, indent=2)

    # Optional debug collage
    if debug:
        dbg_items = [bgr]
        dbg_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        dbg_items.append(dbg_edges)
        if quad is not None:
            dbg_poly = bgr.copy()
            cv.polylines(dbg_poly, [quad.astype(int)], True, (0,255,0), 2)
            dbg_items.append(dbg_poly)
        dbg_items.append(warped)
        dbg_items.append(bin3)
        dbg = hstack_same_height(dbg_items)
        save_image(f"{base}.debug_{method}.jpg", dbg)

    return summary

def hstack_same_height(images):
    h = min(im.shape[0] for im in images)
    row = []
    for im in images:
        if im.ndim == 2:
            im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
        scale = h / im.shape[0]
        row.append(cv.resize(im, (int(im.shape[1]*scale), h), interpolation=cv.INTER_AREA))
    return np.hstack(row)

# ===================== GENERATE ANSWERED SHEET =====================
def gen_answered_sheet(blank_path: str, out_path: str, seed: int | None):
    """Take the blank A4 template PNG and draw random filled bubbles."""
    if seed is not None:
        random.seed(seed)
    bgr = read_bgr(blank_path)
    if bgr.shape[1] != DESIGN_W or bgr.shape[0] != DESIGN_H:
        print("Warning: expected the provided blank to be the A4 template (2480x3508). Proceeding anyway.", file=sys.stderr)
    answers = {q: random.choice(OPTIONS) for q in range(1, QUESTIONS_TOTAL+1)}
    for q, opt, cx, cy, r in TEMPLATE_ENTRIES:
        if answers[q] == opt:
            cv.circle(bgr, (cx, cy), max(6, r-3), (0,0,0), thickness=-1)
    save_image(out_path, bgr)
    # also dump the chosen answers for reference
    with open(os.path.splitext(out_path)[0] + "_answers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer"])
        for q in range(1, QUESTIONS_TOTAL+1):
            w.writerow([q, answers[q]])
    return out_path

# ===================== CLI =====================
def main():
    ap = argparse.ArgumentParser(description="OMR: warp (Hough) + threshold + scoring")
    sub = ap.add_subparsers(dest="cmd")

    p_proc = sub.add_parser("process", help="Process and score local images")
    p_proc.add_argument("inputs", help="Image path or glob (e.g., ./scans/*.jpg)")
    p_proc.add_argument("--method", choices=["adaptive","otsu"], default="adaptive")
    p_proc.add_argument("--debug", action="store_true")
    p_proc.add_argument("--max-side", type=int, default=1600)
    p_proc.add_argument("--seed", type=int, default=1234, help="Seed for random answer key")

    p_gen = sub.add_parser("gen-answersheet", help="Generate a randomly answered sheet from the blank A4 template")
    p_gen.add_argument("blank_png", help="Path to OMR_Sample_A4.png")
    p_gen.add_argument("--out", required=True, help="Output path for the filled sheet")
    p_gen.add_argument("--seed", type=int, default=5678)

    args = ap.parse_args()

    if args.cmd == "gen-answersheet":
        outp = gen_answered_sheet(args.blank_png, args.out, args.seed)
        print(f"Generated answered sheet -> {outp}")
        return

    if args.cmd == "process":
        paths = glob.glob(args.inputs) or ([args.inputs] if os.path.isfile(args.inputs) else [])
        if not paths:
            print(f"No files matched: {args.inputs}", file=sys.stderr)
            sys.exit(1)
        for p in paths:
            try:
                summary = process_one(p, args.method, args.debug, args.max_side, args.seed)
                print(f"✓ {os.path.basename(p)} -> score {summary['score']}/{summary['total_questions']} ({summary['percent']}%)")
            except Exception as e:
                print(f"✗ {os.path.basename(p)} failed: {e}", file=sys.stderr)
        return

    ap.print_help()

if __name__ == "__main__":
    main()
