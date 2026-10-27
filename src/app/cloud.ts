// The cloud: a 3D space of cluster names, drawn on a 2D canvas.
//
// Every image is a faint dot at its 3D position; every cluster is a name at
// the center of its images. Around the cursor a lens turns dots back into
// images: the nearest image is drawn at full size, the next ones smaller by
// rank and by distance, and at the lens edge they are dots again and the names
// take over. A fisheye push plus a collision pass spread the lens images out,
// so the nearest one is never buried under its neighbors.
//
// three.js only supplies the camera and OrbitControls (drag to rotate, scroll
// to zoom toward the cursor). Drawing is plain canvas 2D, so text stays crisp.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { SpaceCluster, SpaceItem } from "./space";

const FOV = 40;
const START_DISTANCE = 3.2; // closest opening distance
const CLOUD_RADIUS = 1.15; // layout radius that holds nearly every image (precompute scales 90% inside 1)
const FISHEYE = 2; // radial magnification at the lens center (Sarkar-Brown)
const LENS_IMAGES = 12; // images shown at once
const RANK_DECAY = 0.86; // each further image is this much smaller
const MIN_IMAGE_PX = 14; // smaller than this, an image is drawn as its dot
const GAP = 6; // px kept between lens images
const RELAX_STEPS = 6;
const EASE = 0.3; // per-frame easing of lens image size and position

interface Projected {
  x: number; // screen position before the lens
  y: number;
  depth: number; // view-space distance
  visible: boolean;
}

// One image in (or leaving) the lens. Sizes and positions ease toward their
// targets, so images grow in, shrink out and swap ranks without popping.
interface LensImage {
  index: number;
  rank: number; // 0 = nearest to the cursor; -1 = leaving
  target: number; // target long side in px
  size: number; // current long side in px
  ax: number; // anchor: fisheye position this frame
  ay: number;
  x: number; // current position after collisions
  y: number;
  w: number;
  h: number;
}

interface LabelState {
  alpha: number; // animated visibility after collisions
}

export interface CloudCallbacks {
  onPick: (id: string) => void;
}

// Camera pose survives closing and reopening the cloud.
let savedPose: { position: THREE.Vector3; target: THREE.Vector3 } | null = null;

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

export class CloudRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private cb: CloudCallbacks;
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private raf = 0;
  private dirty = true;
  private disposed = false;

  private items: SpaceItem[] = [];
  private clusters: SpaceCluster[] = [];
  private focusId: string | null = null;
  private images = new Map<string, HTMLImageElement>();
  private projected: Projected[] = [];
  private labelStates: LabelState[] = [];
  private lens = new Map<number, LensImage>();
  private drawn: LensImage[] = []; // this frame's visible lens images, smallest first
  private settling = false; // lens images still moving toward rest

  private width = 0;
  private height = 0;
  private dpr = 1;
  private sans = "system-ui, sans-serif";
  private mono = "ui-monospace, monospace";
  private colors = {
    text: "#e8e7e3",
    text2: "#a6a59f",
    text3: "#6e6d68",
    dot: "#9c9b94",
    accent: "#9ab8ff",
    bg: "#0b0b0a",
    surface: "#191918",
    border: "rgba(255,255,255,0.14)",
  };

  // Pointer and lens state. The drawn lens trails the pointer slightly, and
  // its strength eases in and out instead of popping.
  private pointer = { x: 0, y: 0, inside: false, dragging: false };
  private lensX = 0;
  private lensY = 0;
  private lensStrength = 0;
  private down: { x: number; y: number; t: number } | null = null;

  private viewProj = new THREE.Matrix4();

  constructor(canvas: HTMLCanvasElement, cb: CloudCallbacks) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.cb = cb;
    this.camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 100);
    if (savedPose) this.camera.position.copy(savedPose.position);
    else this.camera.position.set(1, 0.35, 1.2).normalize();
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.5;
    this.controls.zoomSpeed = 0.7;
    this.controls.panSpeed = 0.6;
    this.controls.zoomToCursor = true;
    this.controls.minDistance = 0.35;
    this.controls.maxDistance = 9;
    if (savedPose) this.controls.target.copy(savedPose.target);
    this.controls.addEventListener("start", this.onControlsStart);
    this.controls.addEventListener("end", this.onControlsEnd);

    canvas.addEventListener("pointermove", this.onPointerMove);
    canvas.addEventListener("pointerleave", this.onPointerLeave);
    canvas.addEventListener("pointerdown", this.onPointerDown);
    canvas.addEventListener("pointerup", this.onPointerUp);

    this.readTheme();
    this.resize();
    if (!savedPose) this.frameAll();
    this.raf = requestAnimationFrame(this.frame);
  }

  // Back the camera off until the whole cloud fits the narrower field of view
  // (portrait screens are narrow horizontally).
  private frameAll() {
    const vertical = (FOV * Math.PI) / 180;
    const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * this.camera.aspect);
    const distance = CLOUD_RADIUS / Math.sin(Math.min(vertical, horizontal) / 2);
    this.camera.position.normalize().multiplyScalar(Math.min(this.controls.maxDistance, Math.max(START_DISTANCE, distance)));
  }

  setData(items: SpaceItem[], clusters: SpaceCluster[], focusId: string | null) {
    this.items = items;
    this.clusters = clusters;
    this.focusId = focusId;
    this.projected = items.map(() => ({ x: 0, y: 0, depth: 1, visible: false }));
    this.labelStates = clusters.map((_, i) => this.labelStates[i] ?? { alpha: 0 });
    this.lens.clear();
    for (const it of items) {
      if (this.images.has(it.thumb)) continue;
      const img = new Image();
      img.decoding = "async";
      img.onload = () => (this.dirty = true);
      img.src = it.thumb;
      this.images.set(it.thumb, img);
    }
    this.dirty = true;
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(1, rect.width);
    this.height = Math.max(1, rect.height);
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.width * this.dpr);
    this.canvas.height = Math.round(this.height * this.dpr);
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  // Fonts and colors come from the stylesheet (next/font variables).
  readTheme() {
    const css = getComputedStyle(document.documentElement);
    const read = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
    this.sans = read("--font-sans", this.sans);
    this.mono = read("--font-mono", this.mono);
    const c = this.colors;
    this.colors = {
      text: read("--text", c.text),
      text2: read("--text-2", c.text2),
      text3: read("--text-3", c.text3),
      dot: read("--dot", c.dot),
      accent: read("--accent", c.accent),
      bg: read("--bg", c.bg),
      surface: read("--surface-2", c.surface),
      border: read("--border-strong", c.border),
    };
    this.dirty = true;
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    savedPose = { position: this.camera.position.clone(), target: this.controls.target.clone() };
    this.controls.removeEventListener("start", this.onControlsStart);
    this.controls.removeEventListener("end", this.onControlsEnd);
    this.controls.dispose();
    this.canvas.removeEventListener("pointermove", this.onPointerMove);
    this.canvas.removeEventListener("pointerleave", this.onPointerLeave);
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    this.canvas.removeEventListener("pointerup", this.onPointerUp);
  }

  // ---- input ----------------------------------------------------------------

  private onControlsStart = () => {
    this.pointer.dragging = true;
    this.canvas.style.cursor = "grabbing";
  };

  private onControlsEnd = () => {
    this.pointer.dragging = false;
    this.dirty = true;
  };

  private localPoint(e: PointerEvent) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  private moveLensTo(x: number, y: number) {
    if (!this.pointer.inside) {
      // Enter at the pointer instead of sliding in from the last position.
      this.lensX = x;
      this.lensY = y;
    }
    this.pointer.x = x;
    this.pointer.y = y;
    this.pointer.inside = true;
    this.dirty = true;
  }

  // Touch has no hover: a tap places the lens, a tap on a lens image opens it.
  private onPointerMove = (e: PointerEvent) => {
    if (e.pointerType === "touch") return;
    const p = this.localPoint(e);
    this.moveLensTo(p.x, p.y);
  };

  private onPointerLeave = (e: PointerEvent) => {
    if (e.pointerType === "touch") return;
    this.pointer.inside = false;
    this.dirty = true;
  };

  private onPointerDown = (e: PointerEvent) => {
    this.down = { ...this.localPoint(e), t: performance.now() };
  };

  private onPointerUp = (e: PointerEvent) => {
    const d = this.down;
    this.down = null;
    if (!d) return;
    const p = this.localPoint(e);
    if (Math.hypot(p.x - d.x, p.y - d.y) > 5 || performance.now() - d.t > 450) return; // a drag
    const hit = this.hitTest(p.x, p.y);
    if (hit >= 0) this.cb.onPick(this.items[hit].id);
    else if (e.pointerType === "touch") this.moveLensTo(p.x, p.y);
  };

  // The lens image under the point; the largest is drawn on top, so test the
  // drawn list from the end.
  private hitTest(x: number, y: number): number {
    for (let k = this.drawn.length - 1; k >= 0; k--) {
      const L = this.drawn[k];
      if (Math.abs(x - L.x) <= L.w / 2 && Math.abs(y - L.y) <= L.h / 2) return L.index;
    }
    return -1;
  }

  // ---- frame ----------------------------------------------------------------

  private frame = () => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.frame);
    if (this.controls.update()) this.dirty = true;

    const wantLens = this.pointer.inside && !this.pointer.dragging ? 1 : 0;
    if (wantLens !== this.lensStrength) {
      this.lensStrength += (wantLens - this.lensStrength) * 0.25;
      if (Math.abs(wantLens - this.lensStrength) < 0.003) this.lensStrength = wantLens;
      this.dirty = true;
    }
    const dx = this.pointer.x - this.lensX;
    const dy = this.pointer.y - this.lensY;
    if (Math.abs(dx) + Math.abs(dy) > 0.3) {
      this.lensX += dx * 0.5;
      this.lensY += dy * 0.5;
      this.dirty = true;
    }
    if (this.labelStates.some((s) => s.alpha > 0.001 && s.alpha < 0.999)) this.dirty = true;
    if (this.settling) this.dirty = true;
    for (const L of this.lens.values()) {
      if (Math.abs(L.target - L.size) > 0.5) {
        this.dirty = true;
        break;
      }
    }
    if (!this.dirty) return;
    this.dirty = false;
    this.project();
    this.layoutLens();
    this.draw();
  };

  private projectPoint(pos: [number, number, number]): Projected {
    const m = this.viewProj.elements;
    const [x, y, z] = pos;
    const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
    return {
      x: ((cx / cw) * 0.5 + 0.5) * this.width,
      y: (0.5 - (cy / cw) * 0.5) * this.height,
      depth: cw,
      visible: cw > 0.05,
    };
  }

  private project() {
    this.camera.updateMatrixWorld();
    this.viewProj.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    this.items.forEach((it, i) => (this.projected[i] = this.projectPoint(it.pos)));
  }

  private get lensRadius() {
    return Math.min(240, Math.max(130, 0.2 * Math.min(this.width, this.height)));
  }

  private get maxImage() {
    return Math.min(280, Math.max(150, 0.27 * Math.min(this.width, this.height)));
  }

  // 1 near the camera's focus plane, smaller further back.
  private perspective(depth: number): number {
    const focus = this.camera.position.distanceTo(this.controls.target);
    return Math.min(1.35, Math.max(0.55, focus / depth));
  }

  // Fisheye push: points inside the lens move away from its center, most
  // strongly near the center, not at all at the edge.
  private fisheye(x: number, y: number): [number, number] {
    const R = this.lensRadius;
    const dx = x - this.lensX;
    const dy = y - this.lensY;
    const dist = Math.hypot(dx, dy);
    const u = dist / R;
    if (u >= 1 || dist < 1e-6 || this.lensStrength < 0.01) return [x, y];
    const g = ((1 + FISHEYE) * u) / (1 + FISHEYE * u);
    const pushed = R * (u + (g - u) * this.lensStrength);
    return [this.lensX + (dx / dist) * pushed, this.lensY + (dy / dist) * pushed];
  }

  private layoutLens() {
    const R = this.lensRadius;
    const S = this.maxImage;
    // Nearest images to the cursor, by screen distance.
    const near: { i: number; u: number }[] = [];
    if (this.lensStrength > 0.01) {
      this.items.forEach((_, i) => {
        const p = this.projected[i];
        if (!p.visible) return;
        const u = Math.hypot(p.x - this.lensX, p.y - this.lensY) / R;
        if (u < 1) near.push({ i, u });
      });
      near.sort((a, b) => a.u - b.u);
    }
    for (const L of this.lens.values()) {
      L.rank = -1;
      L.target = 0;
    }
    near.slice(0, LENS_IMAGES).forEach(({ i, u }, rank) => {
      let L = this.lens.get(i);
      if (!L) {
        const [ax, ay] = this.fisheye(this.projected[i].x, this.projected[i].y);
        L = { index: i, rank, target: 0, size: 0, ax, ay, x: ax, y: ay, w: 0, h: 0 };
        this.lens.set(i, L);
      }
      L.rank = rank;
      L.target = S * (1 - u) ** 0.6 * RANK_DECAY ** rank * this.lensStrength;
    });

    const active: LensImage[] = [];
    const before = new Map([...this.lens.values()].map((L) => [L.index, [L.x, L.y]]));
    for (const L of [...this.lens.values()]) {
      L.size += (L.target - L.size) * EASE;
      if (L.target === 0 && L.size < 2) {
        this.lens.delete(L.index);
        continue;
      }
      const it = this.items[L.index];
      const aspect = it.w / it.h || 1;
      L.w = aspect >= 1 ? L.size : L.size * aspect;
      L.h = aspect >= 1 ? L.size / aspect : L.size;
      const p = this.projected[L.index];
      [L.ax, L.ay] = this.fisheye(p.x, p.y);
      L.x += (L.ax - L.x) * EASE;
      L.y += (L.ay - L.y) * EASE;
      if (L.size >= MIN_IMAGE_PX) active.push(L);
    }

    // Collision passes: bigger images move less, the nearest one barely.
    active.sort((a, b) => b.size - a.size);
    for (let step = 0; step < RELAX_STEPS; step++) {
      for (let a = 0; a < active.length; a++) {
        for (let b = a + 1; b < active.length; b++) {
          const A = active[a];
          const B = active[b];
          const ox = (A.w + B.w) / 2 + GAP - Math.abs(A.x - B.x);
          const oy = (A.h + B.h) / 2 + GAP - Math.abs(A.y - B.y);
          if (ox <= 0 || oy <= 0) continue;
          const share = a === 0 ? 0.06 : B.size / (A.size + B.size);
          if (ox < oy) {
            const s = A.x < B.x ? -1 : 1;
            A.x += s * ox * share;
            B.x -= s * ox * (1 - share);
          } else {
            const s = A.y < B.y ? -1 : 1;
            A.y += s * oy * share;
            B.y -= s * oy * (1 - share);
          }
        }
      }
    }
    this.drawn = active.reverse();
    // Keep drawing until collisions and easing have come to rest.
    this.settling = active.some((L) => {
      const [x, y] = before.get(L.index) ?? [L.x, L.y];
      return Math.abs(L.x - x) + Math.abs(L.y - y) > 0.3;
    });
    const overImage = this.pointer.inside && this.hitTest(this.pointer.x, this.pointer.y) >= 0;
    this.canvas.style.cursor = this.pointer.dragging ? "grabbing" : overImage ? "pointer" : "grab";
  }

  private draw() {
    const ctx = this.ctx;
    const { width: W, height: H, dpr } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = this.colors.bg;
    ctx.fillRect(0, 0, W, H);

    const focusDistance = this.camera.position.distanceTo(this.controls.target);
    const fog = (depth: number) => 1 - 0.72 * smoothstep(focusDistance - 0.9, focusDistance + 1.3, depth);
    const shown = new Set(this.drawn.map((L) => L.index));

    // Dots, far to near; inside the lens they follow the fisheye push.
    const order = this.items.map((_, i) => i).filter((i) => this.projected[i].visible && !shown.has(i));
    order.sort((a, b) => this.projected[b].depth - this.projected[a].depth);
    ctx.fillStyle = this.colors.dot;
    for (const i of order) {
      const p = this.projected[i];
      const [x, y] = this.fisheye(p.x, p.y);
      ctx.globalAlpha = 0.6 * fog(p.depth);
      ctx.beginPath();
      ctx.arc(x, y, 1.1 + 1.3 * this.perspective(p.depth), 0, Math.PI * 2);
      ctx.fill();
    }

    // The image the grid is showing results for.
    const focusIndex = this.focusId ? this.items.findIndex((it) => it.id === this.focusId) : -1;
    if (focusIndex >= 0 && this.projected[focusIndex].visible && !shown.has(focusIndex)) {
      const [x, y] = this.fisheye(this.projected[focusIndex].x, this.projected[focusIndex].y);
      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = this.colors.accent;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.stroke();
    }

    this.drawLabels(fog);
    this.drawLens(focusIndex);
    ctx.globalAlpha = 1;
  }

  private drawLabels(fog: (depth: number) => number) {
    const ctx = this.ctx;
    const R = this.lensRadius;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const placed: { x0: number; y0: number; x1: number; y1: number }[] = [];
    // A group with one cluster would repeat its child's place; skip it.
    const children = new Map<number, number>();
    for (const c of this.clusters) if (c.parent !== null) children.set(c.parent, (children.get(c.parent) ?? 0) + 1);
    // Cluster names claim space first, front to back; group names fill in.
    const order = this.clusters
      .map((c, i) => ({ c, i, p: this.projectPoint(c.pos) }))
      .filter((o) => o.p.visible && (o.c.level === 1 || (children.get(o.i) ?? 0) > 1))
      .sort((a, b) => b.c.level - a.c.level || a.p.depth - b.p.depth);
    for (const { c, i, p } of order) {
      const scale = Math.min(1.25, Math.max(0.8, this.perspective(p.depth)));
      const coarse = c.level === 0;
      const size = (coarse ? 10.5 : 14) * scale;
      ctx.font = coarse ? `500 ${size}px ${this.mono}` : `500 ${size}px ${this.sans}`;
      ctx.letterSpacing = coarse ? "0.12em" : "0px";
      const text = coarse ? c.label.toUpperCase() : c.label;
      const tw = ctx.measureText(text).width;
      const box = { x0: p.x - tw / 2 - 6, y0: p.y - size * 0.7 - 3, x1: p.x + tw / 2 + 6, y1: p.y + size * 1.6 };
      const free = !placed.some((b) => b.x0 < box.x1 && b.x1 > box.x0 && b.y0 < box.y1 && b.y1 > box.y0);
      const state = this.labelStates[i];
      state.alpha += ((free ? 1 : 0) - state.alpha) * 0.2;
      if (Math.abs((free ? 1 : 0) - state.alpha) < 0.002) state.alpha = free ? 1 : 0;
      if (free) placed.push(box);
      // Names fade where the lens shows images and return at its edge.
      const near = 1 - smoothstep(0.5, 1.05, Math.hypot(p.x - this.lensX, p.y - this.lensY) / R);
      const alpha = state.alpha * fog(p.depth) * (1 - this.lensStrength * near);
      if (alpha < 0.01) continue;
      ctx.globalAlpha = alpha;
      ctx.fillStyle = coarse ? this.colors.text3 : this.colors.text;
      ctx.fillText(text, p.x, p.y);
      if (!coarse) {
        ctx.font = `400 ${10 * scale}px ${this.mono}`;
        ctx.letterSpacing = "0px";
        ctx.fillStyle = this.colors.text3;
        ctx.fillText(String(c.size), p.x, p.y + size * 0.95);
      }
    }
    ctx.letterSpacing = "0px";
  }

  private drawLens(focusIndex: number) {
    const ctx = this.ctx;
    if (this.drawn.length === 0) return;
    const S = this.maxImage;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    const top = this.drawn[this.drawn.length - 1];
    for (const L of this.drawn) {
      const it = this.items[L.index];
      const img = this.images.get(it.thumb);
      const x = L.x - L.w / 2;
      const y = L.y - L.h / 2;
      ctx.globalAlpha = Math.min(1, (L.size - MIN_IMAGE_PX) / 24);
      if (L === top) {
        ctx.shadowColor = "rgba(0,0,0,0.55)";
        ctx.shadowBlur = 24;
        ctx.shadowOffsetY = 8;
      }
      if (img && img.complete && img.naturalWidth > 0) ctx.drawImage(img, x, y, L.w, L.h);
      else {
        ctx.fillStyle = this.colors.surface;
        ctx.fillRect(x, y, L.w, L.h);
      }
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;
      ctx.lineWidth = 1;
      ctx.strokeStyle = L.index === focusIndex ? this.colors.accent : "rgba(255,255,255,0.1)";
      ctx.strokeRect(x + 0.5, y + 0.5, L.w - 1, L.h - 1);
    }

    // Caption on a chip under the nearest image: its own tags, which describe
    // it more exactly than the name of the cluster it sits in.
    if (top.rank !== 0 || top.size < 0.5 * S) return;
    const tags = this.items[top.index].tags;
    if (tags.length === 0) return;
    ctx.font = `500 12px ${this.sans}`;
    const first = tags[0];
    const rest = tags.length > 1 ? ` · ${tags.slice(1).join(" · ")}` : "";
    const w1 = ctx.measureText(first).width;
    ctx.font = `400 12px ${this.sans}`;
    const w2 = ctx.measureText(rest).width;
    const cw = w1 + w2 + 20;
    const ch = 26;
    let cy = top.y + top.h / 2 + 8;
    if (cy + ch > this.height - 40) cy = top.y - top.h / 2 - 8 - ch;
    const cx = Math.min(this.width - cw - 8, Math.max(8, top.x - cw / 2));
    ctx.globalAlpha = Math.min(1, (top.size / S - 0.5) * 4);
    ctx.fillStyle = this.colors.surface;
    ctx.strokeStyle = this.colors.border;
    ctx.beginPath();
    ctx.roundRect(cx, cy, cw, ch, 6);
    ctx.fill();
    ctx.stroke();
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.font = `500 12px ${this.sans}`;
    ctx.fillStyle = this.colors.text;
    ctx.fillText(first, cx + 10, cy + ch / 2);
    ctx.font = `400 12px ${this.sans}`;
    ctx.fillStyle = this.colors.text2;
    ctx.fillText(rest, cx + 10 + w1, cy + ch / 2);
  }
}
