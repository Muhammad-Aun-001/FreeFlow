import os, sys, cv2, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import functools

DIR = Path("/home/aun/Desktop/3rd-ppaer/new")
VIDEO_PATH = DIR / "camel" / "camel.mp4"
GT_MASK_DIR = DIR / "camel" / "masks"

MAX_FRAMES_TO_USE = 200
WARMUP_FRAMES = 7
CHECKPOINT = "facebook/sam2-hiera-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROI_TARGET = 256
MARGIN_PX = 30
LAMBDA_RIDGE = 1e-3
EPS_NORM = 1e-8

MU_LAG = 1e-2
MU_BLOCK = 1e-2

ATTN_REPLACE_BLOCKS = list(range(42, 48))
ATTN_PAST_K = 4
FLOW_DOWNSAMPLE = 32

TOTAL_T0 = time.time()
if not VIDEO_PATH.exists(): print(f"[FAIL] {VIDEO_PATH}"); sys.exit(1)

TMP = DIR / "_tmp_final_pacap"
TMP.mkdir(exist_ok=True)
for f in TMP.glob("*.jpg"): f.unlink()
WARM_TMP = DIR / "_tmp_final_warmup"
WARM_TMP.mkdir(exist_ok=True)
for f in WARM_TMP.glob("*.jpg"): f.unlink()

def db_eval_iou(gt, pred):
    gt = np.asarray(gt).astype(bool); pred = np.asarray(pred).astype(bool)
    inter = np.logical_and(gt, pred).sum(); union = np.logical_or(gt, pred).sum()
    return 1.0 if union == 0 else inter / union

def _boundary(m):
    k = np.ones((3, 3), np.uint8)
    return m & ~cv2.erode(m.astype(np.uint8), k, 1).astype(bool)

def _dilate(m, r):
    k = np.ones((2*r+1, 2*r+1), np.uint8)
    return cv2.dilate(m.astype(np.uint8), k, 1).astype(bool)

def db_eval_boundary(gt, pred, dilation=1):
    gt = np.asarray(gt).astype(bool); pred = np.asarray(pred).astype(bool)
    gt_b = _boundary(gt); pr_b = _boundary(pred)
    if gt_b.sum() == 0 and pr_b.sum() == 0: return 1.0
    gt_d = _dilate(gt_b, dilation); pr_d = _dilate(pr_b, dilation)
    tp_p = np.logical_and(pr_b, gt_d).sum()
    precision = tp_p / pr_b.sum() if pr_b.sum() > 0 else 0.0
    tp_r = np.logical_and(gt_b, pr_d).sum()
    recall = tp_r / gt_b.sum() if gt_b.sum() > 0 else 0.0
    if precision + recall == 0: return 0.0
    return 2 * precision * recall / (precision + recall)

def load_gt_masks(mask_dir, num_frames, H, W):
    if not Path(mask_dir).exists():
        return None
    files = sorted(Path(mask_dir).glob("*.png"))[:num_frames]
    if len(files) == 0:
        return None
    gts = []
    for f in files:
        m = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if m is None: gts.append(np.zeros((H, W), bool)); continue
        if m.ndim == 3: m = m.max(axis=2)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        gts.append((m > 0).astype(bool))
    while len(gts) < num_frames: gts.append(np.zeros((H, W), bool))
    return gts

def tight_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

def expand_bbox(bbox, margin, H, W):
    x0, y0, x1, y1 = bbox
    return (max(0, x0-margin), max(0, y0-margin),
            min(W-1, x1+margin), min(H-1, y1+margin))

def compute_flow_np(prev_bgr, curr_bgr, fs=FLOW_DOWNSAMPLE):
    ps = cv2.resize(prev_bgr, (fs, fs)); cs = cv2.resize(curr_bgr, (fs, fs))
    pg = cv2.cvtColor(ps, cv2.COLOR_BGR2GRAY); cg = cv2.cvtColor(cs, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(pg, cg, None, 0.5, 3, 15, 3, 5, 1.2, 0)

print("Extracting frames...")
cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)
raw_frames = []
while True:
    ret, f = cap.read()
    if not ret: break
    raw_frames.append(f)
    if len(raw_frames) >= MAX_FRAMES_TO_USE: break
cap.release()
N = len(raw_frames); H, W = raw_frames[0].shape[:2]
print(f"  {N} frames @ {fps:.1f} FPS, {W}x{H}")

print("Loading GT masks...")
gt_masks = load_gt_masks(GT_MASK_DIR, N, H, W)
if gt_masks is None:
    print(f"  [WARN] No GT found at {GT_MASK_DIR}, metrics will be skipped.")
has_gt = gt_masks is not None

print("Loading SAM 2...")
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.set_grad_enabled(False)
predictor = SAM2VideoPredictor.from_pretrained(CHECKPOINT, device=DEVICE)
img_predictor = SAM2ImagePredictor(predictor)
print("  Loaded")

trunk = predictor.image_encoder.trunk
blocks = trunk.blocks
print(f"  trunk.blocks: {len(blocks)}")
print(f"  replacing attn for blocks: {ATTN_REPLACE_BLOCKS} (AFTER frame {WARMUP_FRAMES})")

from sam2.modeling.backbones.hieradet import window_partition, window_unpartition

_state = {"use_pred": False, "pred": {}, "cap": {}, "block_hw": {}, "block_B": {}}

def unpartition_attn_gpu(idx, attn_out):
    blk = blocks[idx]
    ws = blk.window_size
    if ws <= 0: return attn_out
    hw = _state["block_hw"].get(idx)
    if hw is None: return attn_out
    H_cur, W_cur = hw
    pad_h = (ws - H_cur % ws) % ws
    pad_w = (ws - W_cur % ws) % ws
    pad_hw = (H_cur + pad_h, W_cur + pad_w)
    B = _state["block_B"].get(idx, 1)
    C_ = attn_out.shape[-1]
    nw = (pad_hw[0] // ws) * (pad_hw[1] // ws)
    a = attn_out.reshape(B, nw, ws, ws, C_)
    return window_unpartition(a, ws, pad_hw, (H_cur, W_cur))

def repartition_attn_gpu(idx, x_hw):
    blk = blocks[idx]
    ws = blk.window_size
    if ws <= 0: return x_hw
    x_win, _ = window_partition(x_hw, ws)
    return x_win

def make_wrapped_attn(attn_mod, idx):
    orig = attn_mod.forward
    @functools.wraps(orig)
    def wrapped(x, *args, **kwargs):
        B, H_cur, W_cur, _ = x.shape
        _state["block_hw"][idx] = (H_cur, W_cur)
        _state["block_B"][idx] = B
        pred = _state["pred"].get(idx)
        if _state["use_pred"] and pred is not None:
            _state["cap"][idx] = None
            return pred
        out = orig(x, *args, **kwargs)
        if isinstance(out, torch.Tensor):
            _state["cap"][idx] = out.detach()
        return out
    return wrapped

for idx in ATTN_REPLACE_BLOCKS:
    blocks[idx].attn.forward = make_wrapped_attn(blocks[idx].attn, idx)

def crop_statistics_gpu(crop_gpu):
    gray = (0.1140 * crop_gpu[..., 0] + 0.5870 * crop_gpu[..., 1]
            + 0.2989 * crop_gpu[..., 2])
    gx = gray[:, 2:] - gray[:, :-2]
    gy = gray[2:, :] - gray[:-2, :]
    ge = torch.sqrt((gx.pow(2).mean() + gy.pow(2).mean()).clamp_min(0) + 1e-8)
    z = torch.zeros((), device=crop_gpu.device)
    return torch.stack([gray.mean(), gray.std(), ge, z, z])

def warp_attn_unpart_gpu(attn_hw, flow_gpu):
    B, H_, W_, C_ = attn_hw.shape
    flow = flow_gpu.permute(2, 0, 1).unsqueeze(0)
    flow = F.interpolate(flow, size=(H_, W_), mode='bilinear',
                         align_corners=False).squeeze(0)
    flow = flow * (W_ / flow_gpu.shape[0])
    gy, gx = torch.meshgrid(
        torch.arange(H_, device=flow.device, dtype=torch.float32),
        torch.arange(W_, device=flow.device, dtype=torch.float32),
        indexing='ij')
    sx = (gx + flow[0]) / max(1, W_ - 1) * 2 - 1
    sy = (gy + flow[1]) / max(1, H_ - 1) * 2 - 1
    grid = torch.stack([sx, sy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    x = attn_hw.permute(0, 3, 1, 2).contiguous()
    w = F.grid_sample(x, grid, mode='bilinear', padding_mode='border',
                      align_corners=True)
    return w.permute(0, 2, 3, 1).contiguous()

def _phase_amplitude(Y):
    rho = torch.linalg.vector_norm(Y.float()) + EPS_NORM
    u = Y.float() / rho
    return rho, u

def _lag_laplacian(K, device):
    L = torch.zeros(K, K, device=device, dtype=torch.float32)
    for i in range(K - 1):
        L[i, i]     += 1.0
        L[i+1, i+1] += 1.0
        L[i, i+1]   -= 1.0
        L[i+1, i]   -= 1.0
    return L

def fit_pacap_csrr(stats_list, phase_list, logrho_list,
                   mu_lag=0.0, mu_block=0.0,
                   block_pos=None, prev_coeffs_ref=None):
    T = len(stats_list); K = ATTN_PAST_K
    if T < K + 2: return None

    Theta_dim = 6 * K
    AtA = torch.zeros(Theta_dim, Theta_dim, device=DEVICE, dtype=torch.float32)
    Atb = torch.zeros(Theta_dim, device=DEVICE, dtype=torch.float32)

    u_stack = phase_list
    u_flat = [u.reshape(-1) for u in u_stack]

    for t in range(K, T):
        s = stats_list[t]
        s_aug = torch.cat([s, torch.ones(1, device=DEVICE)])
        past_u = [u_stack[t - k] for k in range(1, K + 1)]
        target = u_flat[t].float()
        pf = [u.reshape(-1).float() for u in past_u]
        rows = []
        for k in range(K):
            rows.append(s_aug.unsqueeze(0) * pf[k].unsqueeze(1))
        D = torch.cat(rows, dim=1)
        AtA += D.T @ D
        Atb += D.T @ target

    AtA += LAMBDA_RIDGE * torch.eye(Theta_dim, device=DEVICE, dtype=torch.float32)

    if mu_lag > 0.0:
        L_lag = _lag_laplacian(K, DEVICE)
        L_full = torch.zeros(Theta_dim, Theta_dim, device=DEVICE, dtype=torch.float32)
        for k1 in range(K):
            for k2 in range(K):
                L_full[k1*6:(k1+1)*6, k2*6:(k2+1)*6] = L_lag[k1, k2] * torch.eye(6, device=DEVICE)
        AtA += mu_lag * L_full

    if mu_block > 0.0 and block_pos is not None and block_pos > 0 and prev_coeffs_ref is not None:
        prev = prev_coeffs_ref.get("phase")
        if prev is not None and prev.numel() == Theta_dim:
            AtA += mu_block * torch.eye(Theta_dim, device=DEVICE, dtype=torch.float32)
            Atb += mu_block * prev.reshape(-1).to(Atb.dtype)

    try:
        Theta_b = torch.linalg.solve(AtA, Atb)
    except Exception:
        return None
    if not torch.all(torch.isfinite(Theta_b)):
        return None
    Theta_b = Theta_b.reshape(K, 6)
    b_coeffs = Theta_b

    Theta_c_dim = 6 * (K + 1)
    AtA_c = torch.zeros(Theta_c_dim, Theta_c_dim, device=DEVICE, dtype=torch.float32)
    Atb_c = torch.zeros(Theta_c_dim, device=DEVICE, dtype=torch.float32)
    for t in range(K, T):
        s = stats_list[t]
        s_aug = torch.cat([s, torch.ones(1, device=DEVICE)])
        past_logrho = torch.stack([logrho_list[t - k] for k in range(1, K + 1)])
        target = logrho_list[t].reshape(1)
        row_parts = []
        for k in range(K):
            row_parts.append(s_aug * past_logrho[k])
        row_parts.append(s_aug)
        row = torch.cat(row_parts, dim=0)
        AtA_c += row.unsqueeze(0).T @ row.unsqueeze(0)
        Atb_c += row * target
    AtA_c += LAMBDA_RIDGE * torch.eye(Theta_c_dim, device=DEVICE, dtype=torch.float32)

    if mu_lag > 0.0:
        L_lag = _lag_laplacian(K, DEVICE)
        L_full = torch.zeros(Theta_c_dim, Theta_c_dim, device=DEVICE, dtype=torch.float32)
        for k1 in range(K):
            for k2 in range(K):
                L_full[k1*6:(k1+1)*6, k2*6:(k2+1)*6] = L_lag[k1, k2] * torch.eye(6, device=DEVICE)
        AtA_c += mu_lag * L_full

    if mu_block > 0.0 and block_pos is not None and block_pos > 0 and prev_coeffs_ref is not None:
        prev_c = prev_coeffs_ref.get("amp")
        if prev_c is not None and prev_c.numel() == Theta_c_dim:
            AtA_c += mu_block * torch.eye(Theta_c_dim, device=DEVICE, dtype=torch.float32)
            Atb_c += mu_block * prev_c.reshape(-1).to(Atb_c.dtype)

    try:
        Theta_c = torch.linalg.solve(AtA_c, Atb_c)
    except Exception:
        return None
    if not torch.all(torch.isfinite(Theta_c)):
        return None
    Theta_c = Theta_c.reshape(K + 1, 6)
    c_coeffs = Theta_c[:K, :]
    mu_coeffs = Theta_c[K, :]

    return {
        "b_coeffs": b_coeffs,
        "c_coeffs": c_coeffs,
        "mu_coeffs": mu_coeffs,
        "_flat_phase": b_coeffs.reshape(-1).clone(),
        "_flat_amp": Theta_c.reshape(-1).clone(),
    }

def predict_pacap_block(block_idx, hist_attn, stats_t):
    params = pacap_params.get(block_idx)
    if params is None: return None
    if len(hist_attn) < ATTN_PAST_K: return None
    K = ATTN_PAST_K
    past = hist_attn[-K:]
    past_u = []; past_logrho = []
    for Y in past:
        rho, u = _phase_amplitude(Y)
        past_u.append(u); past_logrho.append(torch.log(rho))
    s_aug = torch.cat([stats_t, torch.ones(1, device=DEVICE)])

    b = params["b_coeffs"] @ s_aug
    u_raw = None
    for k in range(K):
        term = b[k] * past_u[k]
        u_raw = term if u_raw is None else u_raw + term
    u_norm = torch.linalg.vector_norm(u_raw.float()) + EPS_NORM
    u_t = u_raw / u_norm

    c = params["c_coeffs"] @ s_aug
    mu = params["mu_coeffs"] @ s_aug
    logrho_stack = torch.stack(past_logrho)
    log_rho_t = (c * logrho_stack).sum() + mu
    rho_t = torch.exp(log_rho_t)

    Y_pacap = (rho_t * u_t).to(hist_attn[-1].dtype)
    return Y_pacap

click_points = []
WINDOW_UI = "Click object(s) (N=done)"
def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y)); print(f"  Click ({x}, {y})")
cv2.namedWindow(WINDOW_UI, cv2.WINDOW_NORMAL); cv2.resizeWindow(WINDOW_UI, 1200, 700)
cv2.setMouseCallback(WINDOW_UI, mouse_cb)
while True:
    disp = raw_frames[0].copy()
    cv2.putText(disp, "CLICK OBJECT(S) (N=done)", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
    for pt in click_points: cv2.circle(disp, pt, 6, (0,0,255), -1)
    cv2.imshow(WINDOW_UI, disp); k = cv2.waitKey(10) & 0xFF
    if k == ord('n') and len(click_points) > 0: break
    if k == ord('q') or k == 27: cv2.destroyAllWindows(); sys.exit(0)
cv2.destroyAllWindows()
points = np.array(click_points, dtype=np.float32)
labels = np.ones(len(click_points), dtype=np.int32)

print(f"\nWarmup: frames 0..{WARMUP_FRAMES} — PURE SAM 2 video predictor...")

for i in range(WARMUP_FRAMES + 1):
    cv2.imwrite(str(WARM_TMP / f"{i:05d}.jpg"), raw_frames[i])

_state["use_pred"] = False; _state["pred"] = {}; _state["cap"] = {}

attn_hist = {i: [] for i in ATTN_REPLACE_BLOCKS}
attn_hist_unpart = {i: [] for i in ATTN_REPLACE_BLOCKS}

warm_state = predictor.init_state(video_path=str(WARM_TMP))
predictor.add_new_points_or_box(
    inference_state=warm_state, frame_idx=0, obj_id=1,
    points=points, labels=labels, clear_old_points=True,
)

pred_masks = {}
for frame_pos, obj_ids, mask_logits in predictor.propagate_in_video(warm_state):
    m = (mask_logits[0] > 0).cpu().numpy()
    if m.ndim == 3: m = m[0]
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    pred_masks[frame_pos] = (m > 0).astype(bool).reshape(H, W)

    for idx in ATTN_REPLACE_BLOCKS:
        co = _state["cap"].get(idx)
        if co is not None:
            attn_hist[idx].append(co)
            while len(attn_hist[idx]) > ATTN_PAST_K + 2: attn_hist[idx].pop(0)
            try:
                un = unpartition_attn_gpu(idx, co)
                attn_hist_unpart[idx].append(un.detach())
                while len(attn_hist_unpart[idx]) > ATTN_PAST_K + 2:
                    attn_hist_unpart[idx].pop(0)
            except Exception:
                attn_hist_unpart[idx].append(None)
                while len(attn_hist_unpart[idx]) > ATTN_PAST_K + 2:
                    attn_hist_unpart[idx].pop(0)
    _state["cap"] = {}

print(f"  Warmup masks: {len(pred_masks)}")

prev_mask_all = pred_masks[WARMUP_FRAMES].copy().astype(np.uint8)
last_real_mask = pred_masks[WARMUP_FRAMES].copy()
last_real_bbox = expand_bbox(tight_bbox(last_real_mask), MARGIN_PX, H, W)
current_bbox = last_real_bbox

prev_crop_gpu = torch.from_numpy(
    cv2.resize(raw_frames[WARMUP_FRAMES], (ROI_TARGET, ROI_TARGET),
               interpolation=cv2.INTER_LINEAR)
).float().to(DEVICE) / 255.0

pacap_stats_hist = {i: [] for i in ATTN_REPLACE_BLOCKS}
pacap_phase_hist = {i: [] for i in ATTN_REPLACE_BLOCKS}
pacap_logrho_hist = {i: [] for i in ATTN_REPLACE_BLOCKS}

for wf in range(WARMUP_FRAMES + 1):
    wf_crop = cv2.resize(raw_frames[wf], (ROI_TARGET, ROI_TARGET),
                         interpolation=cv2.INTER_LINEAR)
    wf_gpu = torch.from_numpy(wf_crop).float().to(DEVICE) / 255.0
    wf_stats = crop_statistics_gpu(wf_gpu)
    for idx in ATTN_REPLACE_BLOCKS:
        if wf < len(attn_hist[idx]):
            Y = attn_hist[idx][wf]
            rho, u = _phase_amplitude(Y)
            pacap_stats_hist[idx].append(wf_stats.detach().clone())
            pacap_phase_hist[idx].append(u.detach().clone())
            pacap_logrho_hist[idx].append(torch.log(rho).detach().clone())

print(f"\nFitting CSRR on {WARMUP_FRAMES + 1} warmup frames "
      f"(mu_lag={MU_LAG}, mu_block={MU_BLOCK})...")
pacap_params = {}
fitted_ref = {}
for pos, idx in enumerate(ATTN_REPLACE_BLOCKS):
    sh = pacap_stats_hist[idx]; ph = pacap_phase_hist[idx]; lh = pacap_logrho_hist[idx]
    if len(sh) >= ATTN_PAST_K + 2:
        p = fit_pacap_csrr(sh, ph, lh,
                           mu_lag=MU_LAG, mu_block=MU_BLOCK,
                           block_pos=pos, prev_coeffs_ref=fitted_ref)
        if p is not None:
            pacap_params[idx] = p
            fitted_ref = {"phase": p["_flat_phase"], "amp": p["_flat_amp"]}
print(f"  CSRR fitted for {len(pacap_params)}/{len(ATTN_REPLACE_BLOCKS)} blocks")

print(f"\nRunning PACAP+CSRR on frames {WARMUP_FRAMES + 1}..{N-1}")
t_infer_start = time.time()

for i in range(WARMUP_FRAMES + 1, N):
    frame = raw_frames[i]
    if last_real_bbox is not None: current_bbox = last_real_bbox
    x0, y0, x1, y1 = current_bbox
    crop_np = frame[y0:y1+1, x0:x1+1]
    if crop_np.size == 0:
        x0, y0, x1, y1 = 0, 0, W-1, H-1
        crop_np = frame
    crop_resized_np = cv2.resize(crop_np, (ROI_TARGET, ROI_TARGET),
                                 interpolation=cv2.INTER_LINEAR)
    crop_gpu = torch.from_numpy(crop_resized_np).float().to(DEVICE) / 255.0

    flow_gpu = None
    if prev_crop_gpu is not None:
        prev_np = (prev_crop_gpu.cpu().numpy() * 255).astype(np.uint8)
        flow_np = compute_flow_np(prev_np, crop_resized_np)
        flow_gpu = torch.from_numpy(flow_np).to(DEVICE).float()

    stats_t = crop_statistics_gpu(crop_gpu)
    _state["pred"] = {}; _state["cap"] = {}

    for idx in ATTN_REPLACE_BLOCKS:
        hist = attn_hist[idx]
        if len(hist) < 2: _state["pred"][idx] = None; continue
        F_pacap = predict_pacap_block(idx, hist, stats_t)
        F_warp = None
        if flow_gpu is not None and len(attn_hist_unpart[idx]) >= 1:
            prev_un = attn_hist_unpart[idx][-1]
            if prev_un is not None:
                try:
                    warped = warp_attn_unpart_gpu(prev_un, flow_gpu)
                    F_warp = repartition_attn_gpu(idx, warped)
                except Exception:
                    F_warp = None
        if F_warp is not None and F_pacap is not None and F_warp.shape == F_pacap.shape:
            F_final = 0.5 * F_warp + 0.5 * F_pacap
        elif F_pacap is not None:
            F_final = F_pacap
        else:
            F_final = None
        _state["pred"][idx] = F_final
    _state["use_pred"] = True

    ps = prev_mask_all[y0:y1+1, x0:x1+1].astype(np.uint8)
    if ps.size == 0 or ps.shape[0] < 2 or ps.shape[1] < 2:
        ps = np.zeros((ROI_TARGET, ROI_TARGET), dtype=np.uint8)
    pmr = cv2.resize(ps, (ROI_TARGET, ROI_TARGET),
                     interpolation=cv2.INTER_NEAREST).astype(np.float32)

    img_predictor.set_image(crop_resized_np)
    with torch.no_grad():
        masks, scores, _ = img_predictor.predict(
            point_coords=None, point_labels=None,
            mask_input=pmr[None], multimask_output=False)

    for idx in ATTN_REPLACE_BLOCKS:
        co = _state["cap"].get(idx)
        if co is not None:
            attn_hist[idx].append(co)
            while len(attn_hist[idx]) > ATTN_PAST_K + 2: attn_hist[idx].pop(0)
            try:
                un = unpartition_attn_gpu(idx, co)
                attn_hist_unpart[idx].append(un.detach())
                while len(attn_hist_unpart[idx]) > ATTN_PAST_K + 2:
                    attn_hist_unpart[idx].pop(0)
            except Exception:
                attn_hist_unpart[idx].append(None)
                while len(attn_hist_unpart[idx]) > ATTN_PAST_K + 2:
                    attn_hist_unpart[idx].pop(0)

    roi_mask = (masks[0] > 0).astype(np.uint8)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE,
                                np.ones((3,3), np.uint8), iterations=1)
    roi_full = cv2.resize(roi_mask, (x1-x0+1, y1-y0+1),
                          interpolation=cv2.INTER_NEAREST)
    mask_full = np.zeros((H, W), dtype=np.uint8)
    mask_full[y0:y1+1, x0:x1+1] = roi_full
    pred_masks[i] = mask_full.astype(bool).reshape(H, W)
    prev_mask_all = mask_full
    prev_crop_gpu = crop_gpu

    tb = tight_bbox(pred_masks[i])
    if tb is not None:
        last_real_mask = pred_masks[i].copy()
        last_real_bbox = expand_bbox(tb, MARGIN_PX, H, W)

    if i % 10 == 0 or i == N-1:
        print(f"  frame {i:3d}  [PACAP]")

t_infer = time.time() - t_infer_start
print(f"\nFrame {WARMUP_FRAMES+1} -> {N-1} time: {t_infer:.2f} s "
      f"({t_infer/(N-1-WARMUP_FRAMES)*1000:.1f} ms/frame)")

TOTAL_T = time.time() - TOTAL_T0

print("\n" + "=" * 60)
print("FINAL — PACAP + BBOX + CSRR (blocks 42-47), all frames")
print("=" * 60)
print(f"  Frames:                       {N}")
print(f"  Warmup (pure SAM 2):          0..{WARMUP_FRAMES}")
print(f"  PACAP frames:                 {WARMUP_FRAMES+1}..{N-1}")
print(f"  Per-frame time (0 -> last):   {t_infer/(N-1-WARMUP_FRAMES)*1000:.1f} ms/frame")
print(f"  Inference time (all frames):  {t_infer:.2f} s")
print(f"  Total pipeline time:          {TOTAL_T:.2f} s")
print("-" * 60)

if has_gt:
    Js, Fs = [], []
    for i in range(N):
        gt = gt_masks[i]
        pr = pred_masks.get(i, np.zeros((H, W), bool))
        pr = np.asarray(pr).reshape(H, W).astype(bool)
        if gt.sum() == 0 and pr.sum() == 0: continue
        Js.append(db_eval_iou(gt, pr))
        Fs.append(db_eval_boundary(gt, pr, 1))
    J = float(np.mean(Js)) * 100
    F = float(np.mean(Fs)) * 100
    print(f"  J  (IoU):                     {J:.2f}")
    print(f"  F  (Boundary):                {F:.2f}")
    print(f"  J&F:                          {(J+F)/2:.2f}")
else:
    print("  J / F / J&F:                  [no GT available]")
print("=" * 60)

for f in TMP.glob("*.jpg"): f.unlink()
try: TMP.rmdir()
except OSError: pass
for f in WARM_TMP.glob("*.jpg"): f.unlink()
try: WARM_TMP.rmdir()
except OSError: pass
torch.cuda.empty_cache()
print("\nDone.")