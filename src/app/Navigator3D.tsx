"use client";

// 3D latent-space navigator — the view layer.
//
// A full-screen WebGL overlay (mirrors the Converge overlay pattern). The cloud
// is one instanced draw call of camera-facing quads textured from a single atlas;
// UMAP positions come from navigator3d.computeLayout() and are READ ONLY by the
// renderer. Every neighbor lookup goes through getNeighbors() → dot() on the full
// SigLIP vectors — never the 3D layout.

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { ConvergeImage } from "./Converge";
import {
  computeLayout,
  getNeighbors,
  LAYOUT_RADIUS,
  type Neighbor,
} from "./navigator3d";

// ---------------------------------------------------------------------------
// Tunables (all relative to LAYOUT_RADIUS so the scene stays self-consistent)
// ---------------------------------------------------------------------------
const FOV = 50;
const NEIGHBOR_K = 12;
const OVERVIEW_DIST = LAYOUT_RADIUS * 2.35;
const FOCUS_DIST = LAYOUT_RADIUS * 0.34;
const BASE_SIZE = LAYOUT_RADIUS * 0.052; // sprite full height in world units
const DIST_GAIN = 0.6; // how much far sprites grow to stay legible
const FOG_DENSITY = 0.0032;
const FOCUS_SPRITE_SIZE = LAYOUT_RADIUS * 0.22; // hi-res focus billboard size
const TWEEN_MS = 750;
const IDLE_MS = 2500;
const AUTO_ROTATE_SPEED = 0.55;
const TILE = 256;

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function pow2(x: number): number {
  return Math.pow(2, Math.ceil(Math.log2(Math.max(1, x))));
}

interface SceneCallbacks {
  onPick: (id: string | null) => void;
}

const VERT = /* glsl */ `
  attribute vec3 iOffset;
  attribute vec4 iAtlasUV;
  attribute float iAspect;
  attribute float iState;
  attribute float iIndex;

  uniform float uBaseSize;
  uniform float uDistGain;
  uniform float uRefDepth;
  uniform float uFocusedIndex;

  varying vec2 vUv;
  varying vec2 vQuadUv;
  varying float vState;
  varying float vDepth;

  void main() {
    vQuadUv = uv;
    // Atlas UVs are top-left origin (texture.flipY = false); flip V per quad.
    vUv = vec2(iAtlasUV.x + uv.x * iAtlasUV.z,
               iAtlasUV.y + (1.0 - uv.y) * iAtlasUV.w);
    vState = iState;

    vec4 mv = modelViewMatrix * vec4(iOffset, 1.0);
    float depth = -mv.z;
    vDepth = depth;

    float gain = 1.0 + uDistGain * clamp(depth / uRefDepth, 0.0, 2.0);
    float emph = 1.0 + 0.22 * iState;
    float h = uBaseSize * gain * emph;
    float w = h * iAspect;

    // Collapse the focused instance — the hi-res sprite stands in for it.
    if (abs(iIndex - uFocusedIndex) < 0.5) { w = 0.0; h = 0.0; }

    mv.xy += vec2(position.x * w, position.y * h);
    gl_Position = projectionMatrix * mv;
  }
`;

const FRAG = /* glsl */ `
  precision highp float;
  uniform sampler2D uAtlas;
  uniform vec3 uFogColor;
  uniform float uFogDensity;
  uniform vec3 uRingColor;

  varying vec2 vUv;
  varying vec2 vQuadUv;
  varying float vState;
  varying float vDepth;

  void main() {
    vec4 tex = texture2D(uAtlas, vUv);
    vec3 color = tex.rgb;
    float alpha = tex.a;

    // Thin highlight ring hugging the quad edge when emphasized.
    if (vState > 0.001) {
      vec2 d = min(vQuadUv, 1.0 - vQuadUv);
      float edge = min(d.x, d.y);
      float ring = (1.0 - smoothstep(0.04, 0.055, edge)) * vState;
      color = mix(color, uRingColor, ring);
      alpha = max(alpha, ring);
    }

    // Exponential-squared depth fog toward the near-black background.
    float f = 1.0 - exp(-uFogDensity * uFogDensity * vDepth * vDepth);
    color = mix(color, uFogColor, clamp(f, 0.0, 1.0));

    gl_FragColor = vec4(color, alpha);
  }
`;

// ---------------------------------------------------------------------------
// NavigatorScene — the imperative three.js engine. React drives it via focus()
// / reset() and listens through callbacks; it owns the render loop and picking.
// ---------------------------------------------------------------------------
class NavigatorScene {
  private canvas: HTMLCanvasElement;
  private items: ConvergeImage[];
  private cb: SceneCallbacks;

  private renderer!: THREE.WebGLRenderer;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private controls!: OrbitControls;

  private positions!: Float32Array; // n*3, RENDER ONLY
  private aspects!: Float32Array; // per-instance image aspect (w/h)
  private stateArr!: Float32Array; // per-instance emphasis (animated)
  private iStateAttr!: THREE.InstancedBufferAttribute;
  private material!: THREE.ShaderMaterial;
  private geometry!: THREE.InstancedBufferGeometry;
  private mesh!: THREE.Mesh;
  private atlasTexture!: THREE.CanvasTexture;

  private hiResSprite: THREE.Sprite | null = null;
  private hiResMaterial: THREE.SpriteMaterial | null = null;
  private hiResTexture: THREE.Texture | null = null;
  private hiResSeq = 0;

  private idToIndex = new Map<string, number>();
  private n = 0;
  private hoverIndex = -1;
  private focusedIndex = -1;

  private overviewPos = new THREE.Vector3();
  private tween = {
    active: false,
    start: 0,
    dur: TWEEN_MS,
    fromPos: new THREE.Vector3(),
    toPos: new THREE.Vector3(),
    fromTarget: new THREE.Vector3(),
    toTarget: new THREE.Vector3(),
  };
  private idleTimer = 0;
  private raf = 0;
  private disposed = false;

  // scratch
  private tmpV = new THREE.Vector3();
  private tmpT = new THREE.Vector3();
  private down: { x: number; y: number; t: number; id: number } | null = null;

  constructor(canvas: HTMLCanvasElement, items: ConvergeImage[], cb: SceneCallbacks) {
    this.canvas = canvas;
    this.items = items;
    this.cb = cb;
    items.forEach((it, i) => this.idToIndex.set(it.id, i));
    this.n = items.length;
    this.onPointerMove = this.onPointerMove.bind(this);
    this.onPointerDown = this.onPointerDown.bind(this);
    this.onPointerUp = this.onPointerUp.bind(this);
    this.onControlsStart = this.onControlsStart.bind(this);
    this.onControlsEnd = this.onControlsEnd.bind(this);
  }

  // Build atlas + mesh, wire the scene, start the loop. Resolves once visible.
  async build(): Promise<void> {
    // Lay out the cloud (UMAP, cached). RENDER ONLY.
    this.positions = computeLayout(this.items);

    const atlas = await this.buildAtlas();
    if (this.disposed) return;

    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h, false);
    this.renderer.setClearColor(0x050505, 1);

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(FOV, w / h, 0.1, OVERVIEW_DIST * 6);
    this.overviewPos
      .set(1, 0.42, 1)
      .normalize()
      .multiplyScalar(OVERVIEW_DIST);
    this.camera.position.copy(this.overviewPos);
    this.camera.lookAt(0, 0, 0);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.55;
    this.controls.zoomSpeed = 0.8;
    this.controls.enablePan = false;
    this.controls.minDistance = LAYOUT_RADIUS * 0.18;
    this.controls.maxDistance = OVERVIEW_DIST * 2.2;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = AUTO_ROTATE_SPEED;
    this.controls.target.set(0, 0, 0);
    this.controls.addEventListener("start", this.onControlsStart);
    this.controls.addEventListener("end", this.onControlsEnd);

    this.buildMesh(atlas);

    this.canvas.addEventListener("pointermove", this.onPointerMove);
    this.canvas.addEventListener("pointerdown", this.onPointerDown);
    this.canvas.addEventListener("pointerup", this.onPointerUp);

    const animate = () => {
      this.raf = requestAnimationFrame(animate);
      const now = performance.now();
      if (this.tween.active) this.stepTween(now);
      else this.controls.update();
      this.updateEmphasis();
      this.renderer.render(this.scene, this.camera);
    };
    this.raf = requestAnimationFrame(animate);
  }

  private async buildAtlas(): Promise<{
    texture: THREE.CanvasTexture;
    uvs: Float32Array;
    aspects: Float32Array;
  }> {
    const n = this.n;
    const cols = Math.max(1, Math.ceil(Math.sqrt(n)));
    const dim = pow2(cols * TILE);
    const canvas = document.createElement("canvas");
    canvas.width = dim;
    canvas.height = dim;
    const ctx = canvas.getContext("2d")!;

    const uvs = new Float32Array(n * 4);
    const aspects = new Float32Array(n);

    await Promise.all(
      this.items.map(
        (it, i) =>
          new Promise<void>((resolve) => {
            const img = new Image();
            const finish = () => {
              const col = i % cols;
              const row = Math.floor(i / cols);
              const tileX = col * TILE;
              const tileY = row * TILE;
              const a = it.w / it.h || 1;
              let drawW = TILE;
              let drawH = TILE;
              if (a >= 1) drawH = TILE / a;
              else drawW = TILE * a;
              const offX = (TILE - drawW) / 2;
              const offY = (TILE - drawH) / 2;
              try {
                ctx.drawImage(img, tileX + offX, tileY + offY, drawW, drawH);
              } catch {
                /* leave tile transparent if the draw fails */
              }
              uvs[i * 4 + 0] = (tileX + offX) / dim;
              uvs[i * 4 + 1] = (tileY + offY) / dim;
              uvs[i * 4 + 2] = drawW / dim;
              uvs[i * 4 + 3] = drawH / dim;
              aspects[i] = a;
              resolve();
            };
            img.onload = finish;
            img.onerror = () => {
              aspects[i] = it.w / it.h || 1;
              resolve();
            };
            img.src = it.thumb;
          })
      )
    );

    const texture = new THREE.CanvasTexture(canvas);
    texture.flipY = false;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.magFilter = THREE.LinearFilter;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.generateMipmaps = true;
    texture.anisotropy = this.maxAnisotropy();
    texture.needsUpdate = true;
    return { texture, uvs, aspects };
  }

  private maxAnisotropy(): number {
    // renderer may not exist yet during atlas build; create a throwaway probe.
    if (this.renderer) return this.renderer.capabilities.getMaxAnisotropy();
    return 8;
  }

  private buildMesh(atlas: {
    texture: THREE.CanvasTexture;
    uvs: Float32Array;
    aspects: Float32Array;
  }) {
    const n = this.n;
    this.atlasTexture = atlas.texture;
    this.atlasTexture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    this.aspects = atlas.aspects;
    this.stateArr = new Float32Array(n);

    const indices = new Float32Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;

    const base = new THREE.PlaneGeometry(1, 1);
    const geo = new THREE.InstancedBufferGeometry();
    geo.index = base.index;
    geo.setAttribute("position", base.getAttribute("position"));
    geo.setAttribute("uv", base.getAttribute("uv"));
    geo.instanceCount = n;
    geo.setAttribute("iOffset", new THREE.InstancedBufferAttribute(this.positions, 3));
    geo.setAttribute("iAtlasUV", new THREE.InstancedBufferAttribute(atlas.uvs, 4));
    geo.setAttribute("iAspect", new THREE.InstancedBufferAttribute(atlas.aspects, 1));
    this.iStateAttr = new THREE.InstancedBufferAttribute(this.stateArr, 1);
    this.iStateAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("iState", this.iStateAttr);
    geo.setAttribute("iIndex", new THREE.InstancedBufferAttribute(indices, 1));
    base.dispose();

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uAtlas: { value: atlas.texture },
        uBaseSize: { value: BASE_SIZE },
        uDistGain: { value: DIST_GAIN },
        uRefDepth: { value: OVERVIEW_DIST },
        uFocusedIndex: { value: -1 },
        uFogColor: { value: new THREE.Vector3(5 / 255, 5 / 255, 5 / 255) },
        uFogDensity: { value: FOG_DENSITY },
        uRingColor: { value: new THREE.Vector3(1, 1, 1) },
      },
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
      depthTest: true,
      depthWrite: true,
    });

    this.geometry = geo;
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);
  }

  // ---- render loop helpers -------------------------------------------------

  private stepTween(now: number) {
    const t = Math.min(1, (now - this.tween.start) / this.tween.dur);
    const e = easeInOutCubic(t);
    this.camera.position.lerpVectors(this.tween.fromPos, this.tween.toPos, e);
    this.tmpT.lerpVectors(this.tween.fromTarget, this.tween.toTarget, e);
    this.controls.target.copy(this.tmpT);
    this.camera.lookAt(this.tmpT);
    if (t >= 1) {
      this.tween.active = false;
      this.controls.enabled = true;
      if (this.focusedIndex < 0) this.scheduleIdle();
    }
  }

  private updateEmphasis() {
    const arr = this.stateArr;
    let dirty = false;
    for (let i = 0; i < arr.length; i++) {
      const target = i === this.hoverIndex ? 1 : 0;
      const cur = arr[i];
      if (Math.abs(target - cur) < 0.002) {
        if (cur !== target) {
          arr[i] = target;
          dirty = true;
        }
        continue;
      }
      arr[i] = cur + (target - cur) * 0.18;
      dirty = true;
    }
    if (dirty) this.iStateAttr.needsUpdate = true;
  }

  // ---- camera framing ------------------------------------------------------

  private frameCameraTo(toPos: THREE.Vector3, toTarget: THREE.Vector3) {
    this.tween.fromPos.copy(this.camera.position);
    this.tween.toPos.copy(toPos);
    this.tween.fromTarget.copy(this.controls.target);
    this.tween.toTarget.copy(toTarget);
    this.tween.start = performance.now();
    this.tween.dur = TWEEN_MS;
    this.tween.active = true;
    this.controls.enabled = false;
    this.controls.autoRotate = false;
    this.clearIdle();
  }

  // Public: frame and enlarge the image with this id (called by React).
  focus(id: string) {
    const index = this.idToIndex.get(id);
    if (index === undefined) return;
    this.focusedIndex = index;
    this.material.uniforms.uFocusedIndex.value = index;

    const p = this.tmpV.set(
      this.positions[index * 3 + 0],
      this.positions[index * 3 + 1],
      this.positions[index * 3 + 2]
    );
    const dir = this.camera.position.clone().sub(this.controls.target);
    if (dir.lengthSq() < 1e-6) dir.copy(this.overviewPos);
    dir.normalize().multiplyScalar(FOCUS_DIST);
    const toPos = p.clone().add(dir);
    this.frameCameraTo(toPos, p.clone());

    this.loadHiRes(index, p.clone());
  }

  // Public: ease back to the full-cloud overview (called by React).
  reset() {
    this.focusedIndex = -1;
    this.material.uniforms.uFocusedIndex.value = -1;
    this.removeHiRes();
    this.frameCameraTo(this.overviewPos.clone(), this.tmpT.set(0, 0, 0).clone());
  }

  private loadHiRes(index: number, pos: THREE.Vector3) {
    const seq = ++this.hiResSeq;
    const item = this.items[index];
    const loader = new THREE.TextureLoader();

    const apply = (tex: THREE.Texture) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
      if (!this.hiResSprite) {
        this.hiResMaterial = new THREE.SpriteMaterial({
          transparent: true,
          depthTest: false,
          fog: false,
        });
        this.hiResSprite = new THREE.Sprite(this.hiResMaterial);
        this.hiResSprite.renderOrder = 10;
      }
      if (this.hiResTexture) this.hiResTexture.dispose();
      this.hiResTexture = tex;
      this.hiResMaterial!.map = tex;
      this.hiResMaterial!.needsUpdate = true;

      const a = item.w / item.h || 1;
      const sw = a >= 1 ? FOCUS_SPRITE_SIZE : FOCUS_SPRITE_SIZE * a;
      const sh = a >= 1 ? FOCUS_SPRITE_SIZE / a : FOCUS_SPRITE_SIZE;
      this.hiResSprite.scale.set(sw, sh, 1);
      this.hiResSprite.position.copy(pos);
      if (!this.hiResSprite.parent) this.scene.add(this.hiResSprite);
      this.hiResSprite.visible = true;
    };

    // Show the always-present thumbnail immediately so the focused image never
    // blanks, then lazily upgrade to the full-resolution original when one
    // exists on disk (datasets may ship thumbnails only).
    loader.load(item.thumb, (thumbTex) => {
      if (this.disposed || seq !== this.hiResSeq) {
        thumbTex.dispose();
        return;
      }
      apply(thumbTex);
      loader.load(
        item.file,
        (fullTex) => {
          if (this.disposed || seq !== this.hiResSeq) {
            fullTex.dispose();
            return;
          }
          apply(fullTex);
        },
        undefined,
        () => {
          /* no original on disk — keep the thumbnail */
        }
      );
    });
  }

  private removeHiRes() {
    this.hiResSeq++;
    if (this.hiResSprite) this.hiResSprite.visible = false;
    if (this.hiResTexture) {
      this.hiResTexture.dispose();
      this.hiResTexture = null;
    }
  }

  // ---- idle auto-rotate ----------------------------------------------------

  private scheduleIdle() {
    this.clearIdle();
    this.idleTimer = window.setTimeout(() => {
      if (!this.disposed && this.focusedIndex < 0 && !this.tween.active) {
        this.controls.autoRotate = true;
      }
    }, IDLE_MS);
  }

  private clearIdle() {
    if (this.idleTimer) {
      clearTimeout(this.idleTimer);
      this.idleTimer = 0;
    }
  }

  private onControlsStart() {
    this.controls.autoRotate = false;
    this.clearIdle();
  }

  private onControlsEnd() {
    if (this.focusedIndex < 0) this.scheduleIdle();
  }

  // ---- picking -------------------------------------------------------------

  private pickAt(clientX: number, clientY: number): number {
    const rect = this.canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const W = rect.width;
    const H = rect.height;
    const projScale = H / (2 * Math.tan((FOV * Math.PI) / 180 / 2));

    // The focused hi-res sprite takes click priority so its image never resets.
    if (this.focusedIndex >= 0) {
      const i = this.focusedIndex;
      this.tmpV.set(
        this.positions[i * 3],
        this.positions[i * 3 + 1],
        this.positions[i * 3 + 2]
      );
      const dist = this.camera.position.distanceTo(this.tmpV);
      this.tmpV.project(this.camera);
      if (this.tmpV.z <= 1) {
        const sx = (this.tmpV.x * 0.5 + 0.5) * W;
        const sy = (-this.tmpV.y * 0.5 + 0.5) * H;
        const halfH = (FOCUS_SPRITE_SIZE / 2 / dist) * projScale;
        const halfW = halfH * Math.max(1, this.aspects[i]);
        if (Math.abs(px - sx) <= halfW && Math.abs(py - sy) <= halfH) return i;
      }
    }

    let best = -1;
    let bestDist = Infinity;
    for (let i = 0; i < this.n; i++) {
      if (i === this.focusedIndex) continue;
      this.tmpV.set(
        this.positions[i * 3],
        this.positions[i * 3 + 1],
        this.positions[i * 3 + 2]
      );
      const dist = this.camera.position.distanceTo(this.tmpV);
      this.tmpV.project(this.camera);
      if (this.tmpV.z > 1) continue;
      const sx = (this.tmpV.x * 0.5 + 0.5) * W;
      const sy = (-this.tmpV.y * 0.5 + 0.5) * H;
      const gain = 1 + DIST_GAIN * Math.min(2, dist / OVERVIEW_DIST);
      const halfH = ((BASE_SIZE * gain) / 2 / dist) * projScale;
      const halfW = halfH * this.aspects[i];
      if (Math.abs(px - sx) <= halfW && Math.abs(py - sy) <= halfH) {
        if (dist < bestDist) {
          bestDist = dist;
          best = i;
        }
      }
    }
    return best;
  }

  private onPointerMove(e: PointerEvent) {
    const idx = this.pickAt(e.clientX, e.clientY);
    if (idx !== this.hoverIndex) {
      this.hoverIndex = idx;
      this.canvas.style.cursor = idx >= 0 ? "pointer" : "default";
    }
  }

  private onPointerDown(e: PointerEvent) {
    this.down = { x: e.clientX, y: e.clientY, t: performance.now(), id: e.pointerId };
  }

  private onPointerUp(e: PointerEvent) {
    const d = this.down;
    this.down = null;
    if (!d || d.id !== e.pointerId) return;
    const moved = Math.hypot(e.clientX - d.x, e.clientY - d.y);
    if (moved > 6 || performance.now() - d.t > 500) return; // a drag, not a click
    const idx = this.pickAt(e.clientX, e.clientY);
    this.cb.onPick(idx >= 0 ? this.items[idx].id : null);
  }

  // ---- lifecycle -----------------------------------------------------------

  onResize() {
    if (!this.renderer) return;
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  dispose() {
    this.disposed = true;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.clearIdle();
    this.canvas.removeEventListener("pointermove", this.onPointerMove);
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    this.canvas.removeEventListener("pointerup", this.onPointerUp);
    if (this.controls) {
      this.controls.removeEventListener("start", this.onControlsStart);
      this.controls.removeEventListener("end", this.onControlsEnd);
      this.controls.dispose();
    }
    this.geometry?.dispose();
    this.material?.dispose();
    this.atlasTexture?.dispose();
    this.hiResTexture?.dispose();
    this.hiResMaterial?.dispose();
    this.renderer?.dispose();
  }
}

// ---------------------------------------------------------------------------
// React shell
// ---------------------------------------------------------------------------
interface Navigator3DProps {
  items: ConvergeImage[];
  onExit: () => void;
}

export default function Navigator3D({ items, onExit }: Navigator3DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<NavigatorScene | null>(null);

  const [loading, setLoading] = useState(true);
  const [focusedId, setFocusedId] = useState<string | null>(null);
  const [neighbors, setNeighbors] = useState<Neighbor[]>([]);

  const byId = useMemo(() => new Map(items.map((it) => [it.id, it])), [items]);
  const historyRef = useRef<string[]>([]);

  // Latest-value refs so the once-mounted scene callback / key handlers stay fresh.
  const focusedIdRef = useRef<string | null>(null);
  focusedIdRef.current = focusedId;
  const neighborsRef = useRef<Neighbor[]>([]);
  neighborsRef.current = neighbors;

  const focusOn = useCallback(
    (id: string, pushHistory: boolean) => {
      const item = byId.get(id);
      if (!item) return;
      if (pushHistory && focusedIdRef.current && focusedIdRef.current !== id) {
        historyRef.current.push(focusedIdRef.current);
      }
      setFocusedId(id);
      setNeighbors(getNeighbors(item.vec, items, NEIGHBOR_K));
      sceneRef.current?.focus(id);
    },
    [byId, items]
  );

  const resetView = useCallback(() => {
    historyRef.current = [];
    setFocusedId(null);
    setNeighbors([]);
    sceneRef.current?.reset();
  }, []);

  const stepForward = useCallback(() => {
    const ns = neighborsRef.current;
    if (ns.length === 0) return;
    const prev = historyRef.current[historyRef.current.length - 1];
    const next = ns.find((nb) => nb.id !== prev) ?? ns[0];
    focusOn(next.id, true);
  }, [focusOn]);

  const stepBack = useCallback(() => {
    const prev = historyRef.current.pop();
    if (prev) focusOn(prev, false);
    else resetView();
  }, [focusOn, resetView]);

  // Build the scene once on mount.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const scene = new NavigatorScene(canvas, items, {
      onPick: (id) => {
        if (id === null) {
          if (focusedIdRef.current) resetView();
        } else if (id !== focusedIdRef.current) {
          focusOn(id, true);
        }
      },
    });
    sceneRef.current = scene;
    let cancelled = false;
    scene.build().then(() => {
      if (!cancelled) setLoading(false);
    });
    const onResize = () => scene.onResize();
    window.addEventListener("resize", onResize);
    return () => {
      cancelled = true;
      window.removeEventListener("resize", onResize);
      scene.dispose();
      sceneRef.current = null;
    };
    // Built once; latest state reached via refs above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keyboard: ←/→ walk neighbors, Esc exits focus then the overlay.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (focusedIdRef.current) resetView();
        else onExit();
        return;
      }
      if (!focusedIdRef.current) return;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        stepForward();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        stepBack();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onExit, resetView, stepForward, stepBack]);

  // Touch swipe through the strip when focused.
  const touchStartX = useRef<number | null>(null);
  const onTouchStart = (e: React.TouchEvent) => {
    if (!focusedIdRef.current || !e.touches.length) return;
    touchStartX.current = e.touches[0].clientX;
  };
  const onTouchEnd = (e: React.TouchEvent) => {
    if (touchStartX.current === null || !e.changedTouches.length) {
      touchStartX.current = null;
      return;
    }
    const dx = e.changedTouches[0].clientX - touchStartX.current;
    touchStartX.current = null;
    if (Math.abs(dx) < 50) return;
    if (dx < 0) stepForward();
    else stepBack();
  };

  return (
    <div className="navigator-overlay" onTouchStart={onTouchStart} onTouchEnd={onTouchEnd}>
      <canvas ref={canvasRef} className="navigator-canvas" />

      <button className="navigator-close" onClick={onExit} title="Close (Esc)">
        ✕
      </button>

      {loading && (
        <div className="navigator-loading">
          <div className="navigator-spinner" />
          <span>Building cloud…</span>
        </div>
      )}

      {!loading && !focusedId && (
        <div className="navigator-hint">
          Drag to orbit · scroll to zoom · click an image to dive in
        </div>
      )}

      {focusedId && neighbors.length > 0 && (
        <div className="navigator-strip">
          {neighbors.map((nb) => {
            const img = byId.get(nb.id);
            if (!img) return null;
            return (
              <button
                key={nb.id}
                className="navigator-strip-item"
                onClick={() => focusOn(nb.id, true)}
                title={`similarity ${nb.score.toFixed(3)}`}
              >
                <img src={img.thumb} alt="" draggable={false} />
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
