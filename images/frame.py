"""Frame a generated meal image (see ../image-prompt.md) for Food choices.

Every dish gets the same framing: the bowl or plate (its rim, not its shadow) spans
80% of the frame's width, centred, with an even margin of linen around it. That holds
for both outputs, so all dishes look the same size in the app.

  python3 frame.py source.png out_prefix              # finds the dish by itself
  python3 frame.py source.png out_prefix x0,y0,x1,y1  # or give the dish's box in source pixels

Writes out_prefix.jpg (4:3, 1000x750, card and Library tile), out_prefix-sq.jpg
(square 256x256, Today row thumbnail), both JPEG quality 80, and out_prefix-check.jpg
(the source with the box it used drawn in red; look at it, then delete it).

When the source is framed tighter than 80%, the missing edge is filled with matching
linen (blurred edge colour plus the cloth's own texture), so the dish never gets bigger.
Needs only numpy and Pillow.
"""
import sys
from collections import deque
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FILL = 0.80      # dish width / frame width
MAXH = 0.90      # dish height may use at most this share of the frame height
EDGE_PCT = 90    # edge strength percentile kept when finding the dish


def _components(m):
    h, w = m.shape
    seen = np.zeros(m.shape, bool)
    out = []
    for y, x in zip(*np.nonzero(m)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        q, pts = deque([(y, x)]), []
        while q:
            cy, cx = q.popleft()
            pts.append((cy, cx))
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        out.append(np.array(pts))
    return out


def find_dish(im):
    """Box of the dish: strong edges after a blur (the cloth's weave and soft shadows
    drop out), the largest edge group, plus nearby groups (rim arcs, food)."""
    W, H = im.size
    s = 384 / W
    g = np.asarray(im.convert('L').resize((384, round(H * s)), Image.LANCZOS)
                   .filter(ImageFilter.GaussianBlur(1.6))).astype(float)
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]; gy[1:-1] = g[2:] - g[:-2]
    mag = np.hypot(gx, gy)
    mag[:3] = mag[-3:] = 0; mag[:, :3] = mag[:, -3:] = 0
    m = mag > np.percentile(mag, EDGE_PCT)
    m = np.asarray(Image.fromarray((m * 255).astype('uint8')).filter(ImageFilter.MaxFilter(5))) > 127
    cs = sorted(_components(m), key=len, reverse=True)
    bb = lambda p: [p[:, 1].min(), p[:, 0].min(), p[:, 1].max(), p[:, 0].max()]
    box, used, gap, big = bb(cs[0]), {0}, 0.06 * m.shape[1], len(cs[0])
    changed = True
    while changed:
        changed = False
        for i, p in enumerate(cs):
            if i in used or len(p) < 0.03 * big:
                continue
            b = bb(p)
            if b[0] < box[2] + gap and b[2] > box[0] - gap and b[1] < box[3] + gap and b[3] > box[1] - gap:
                box = [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
                used.add(i); changed = True
    return [(box[0] + 2) / s, (box[1] + 2) / s, (box[2] - 1) / s, (box[3] - 1) / s]


def _linen_texture(im, box, size):
    """Tile the cloth's fine texture (image minus its blur) from the widest plain strip."""
    W, H = im.size
    x0, y0, x1, y1 = box
    cands = [(0, 0, x0, H), (x1, 0, W, H), (0, 0, W, y0), (0, y1, W, H)]
    c = max(cands, key=lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1]))
    c = tuple(int(v) for v in c)
    if (c[2] - c[0]) < 32 or (c[3] - c[1]) < 32:
        return np.zeros((size[1], size[0], 3))
    patch = im.crop(c)
    hp = np.asarray(patch).astype(float) - np.asarray(patch.filter(ImageFilter.GaussianBlur(6))).astype(float)
    t = np.concatenate([hp, hp[:, ::-1]], 1); t = np.concatenate([t, t[::-1]], 0)  # mirror-tile: no seams
    reps = (size[1] // t.shape[0] + 1, size[0] // t.shape[1] + 1, 1)
    return np.tile(t, reps)[:size[1], :size[0]]


def take(im, box, cx, cy, cw, ch):
    """Crop (cx, cy, cw, ch) from im, filling anything outside the source with linen."""
    W, H = im.size
    x0, y0 = round(cx - cw / 2), round(cy - ch / 2)
    x1, y1 = x0 + round(cw), y0 + round(ch)
    if x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H:
        return im.crop((x0, y0, x1, y1))
    pl, pt, pr, pb = max(0, -x0), max(0, -y0), max(0, x1 - W), max(0, y1 - H)
    a = np.asarray(im).astype(float)
    ext = np.pad(a, ((pt, pb), (pl, pr), (0, 0)), mode='edge')
    big = Image.fromarray(ext.astype('uint8'))
    base = np.asarray(big.filter(ImageFilter.GaussianBlur(max(pl, pt, pr, pb, 24) * 0.6))).astype(float)
    canvas = base + _linen_texture(im, box, big.size)
    f = int(min(96, max(24, 0.6 * max(pl, pt, pr, pb)))) // 2 * 2  # even, so the MinFilter size below stays odd  # wider feather when the fill is wide, so no visible seam
    mask = np.zeros(ext.shape[:2]); mask[pt:pt + H, pl:pl + W] = 255
    mask = np.asarray(Image.fromarray(mask.astype('uint8')).filter(ImageFilter.MinFilter(f + 1))
                      .filter(ImageFilter.GaussianBlur(f / 2))).astype(float) / 255
    mask[pt + f:pt + H - f, pl + f:pl + W - f] = 1
    mask = mask[..., None]
    out = Image.fromarray(np.clip(ext * mask + canvas * (1 - mask), 0, 255).astype('uint8'))
    return out.crop((x0 + pl, y0 + pt, x1 + pl, y1 + pt))


def main(src, out, box=None):
    im = Image.open(src).convert('RGB')
    box = list(map(float, box.split(','))) if box else find_dish(im)
    x0, y0, x1, y1 = box
    bw, bh, cx, cy = x1 - x0, y1 - y0, (x0 + x1) / 2, (y0 + y1) / 2
    print('dish box', [round(v) for v in box], f'({bw:.0f} x {bh:.0f})')
    cw = max(bw / FILL, bh / MAXH * 4 / 3)
    take(im, box, cx, cy, cw, cw * 3 / 4).resize((1000, 750), Image.LANCZOS).save(
        out + '.jpg', quality=80, optimize=True, progressive=True)
    s = max(bw / FILL, bh / MAXH)
    take(im, box, cx, cy, s, s).resize((256, 256), Image.LANCZOS).save(
        out + '-sq.jpg', quality=80, optimize=True, progressive=True)
    chk = im.copy()
    ImageDraw.Draw(chk).rectangle(box, outline=(255, 0, 0), width=max(3, im.size[0] // 300))
    chk.thumbnail((800, 800)); chk.save(out + '-check.jpg', quality=80)


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(*sys.argv[1:4])
