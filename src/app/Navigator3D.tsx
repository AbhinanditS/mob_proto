"use client";

// 3D latent-space navigator — the view layer.
//
// A full-screen WebGL overlay (mirrors the Converge overlay pattern). The cloud
// is one instanced draw call of camera-facing quads textured from a single atlas;
// UMAP positions come from navigator3d.computeLayout() and are READ ONLY by the
// renderer. Every neighbor lookup goes through getNeighbors() → dot() on the full
// SigLIP vectors — never the 3D layout.
//
// Two modes: "explore" (orbit + click-to-focus + neighbor stepping) and "probe"
// (an interactive point-cloud instrument). Probe mode reads positions from the
// SAME frozen layout array and never mutates it; its k-NN/density queries run in
// layout space (Euclidean), distinct from getNeighbors()'s full-vector cosine
// search, and its drag-to-pull warp only ever touches a separate display buffer —
// never the canonical layout, never .vec.

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { ConvergeImage } from "./Converge";
import {
  computeDensity,
  computeLayout,
  getNeighbors,
  kNearestByPosition,
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

// Probe mode tunables.
const PROBE_K = 8; // live reconstruction neighbor count
const PROBE_POOL_SIZE = 9; // hover slot + top-PROBE_K query results
const PROBE_DEBOUNCE_MS = 120;
const PROBE_BASE_SCALE = 0.35; // at-rest shrink factor for probe-mode points
const PROBE_SPRITE_SIZE = LAYOUT_RADIUS * 0.16; // pooled hi-res billboard size
const DENSITY_K = 6;
const WARP_RADIUS = LAYOUT_RADIUS * 0.25;
const WARP_RELEASE_MS = 450;

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function pow2(x: number): number {
  return Math.pow(2, Math.ceil(Math.log2(Math.max(1, x))));
}

// Minimal trailing-edge debounce — no debounce utility exists elsewhere in the
// codebase. cancel() lets callers flush a pending timer on teardown.
function debounce<F extends (...args: never[]) => void>(
  fn: F,
  ms: number
): F & { cancel: () => void } {
  let timer: number | null = null;
  const wrapped = ((...args: Parameters<F>) => {
    if (timer !== null) window.clearTimeout(timer);
    timer = window.setTimeout(() => {
      timer = null;
      fn(...args);
    }, ms);
  }) as F & { cancel: () => void };
  wrapped.cancel = () => {
    if (timer !== null) window.clearTimeout(timer);
    timer = null;
  };
  return wrapped;
}

interface SceneCallbacks {
  onPick: (id: string | null) => void;
  onProbeQuery: (result: { id: string; dist: number }[], meanDist: number) => void;
}

interface HiResSlot {
  sprite: THREE.Sprite;
  material: THREE.SpriteMaterial;
  texture: THREE.Texture | null;
  index: number;
  seq: number;
}

const VERT = /* glsl */ `
  attribute vec3 iOffset;
  attribute vec4 iAtlasUV;
  attribute float iAspect;
  attribute float iState;
  attribute float iIndex;
  attribute float iPromote;
  attribute float iDensity;

  uniform float uBaseSize;
  uniform float uDistGain;
  uniform float uRefDepth;
  uniform float uFocusedIndex;
  uniform float uProbeActive;
  uniform float uProbeBaseScale;

  varying vec2 vUv;
  varying vec2 vQuadUv;
  varying float vState;
  varying float vDepth;
  varying float vDensity;

  void main() {
    vQuadUv = uv;
    // Atlas UVs are top-left origin (texture.flipY = false); flip V per quad.
    vUv = vec2(iAtlasUV.x + uv.x * iAtlasUV.z,
               iAtlasUV.y + (1.0 - uv.y) * iAtlasUV.w);
    vState = iState;
    vDensity = iDensity;

    vec4 mv = modelViewMatrix * vec4(iOffset, 1.0);
    float depth = -mv.z;
    vDepth = depth;

    float gain = 1.0 + uDistGain * clamp(depth / uRefDepth, 0.0, 2.0);
    float emph = 1.0 + 0.22 * iState;
    // Probe mode: shrink at rest, scaled further by local density so sparse
    // points read smaller/dimmer than dense ones. A no-op when uProbeActive=0,
    // so explore mode's output stays pixel-identical to before.
    float probeShrink = mix(1.0, uProbeBaseScale, uProbeActive);
    float densityScale = mix(1.0, mix(0.45, 1.0, iDensity), uProbeActive);
    float h = uBaseSize * gain * emph * probeShrink * densityScale;
    float w = h * iAspect;

    // Collapse the focused instance (explore mode) OR a promoted hi-res-pool
    // instance (probe mode) — a dedicated sprite stands in for it either way.
    bool collapseFocus = abs(iIndex - uFocusedIndex) < 0.5;
    bool collapsePromoted = uProbeActive > 0.5 && iPromote > 0.5;
    if (collapseFocus || collapsePromoted) { w = 0.0; h = 0.0; }

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
  uniform float uProbeActive;

  varying vec2 vUv;
  varying vec2 vQuadUv;
  varying float vState;
  varying float vDepth;
  varying float vDensity;

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

    // Probe mode: dim + desaturate points in sparse regions — same iDensity
    // attribute that already drove point size above, no extra cost.
    if (uProbeActive > 0.5) {
      float gray = dot(color, vec3(0.299, 0.587, 0.114));
      float sat = mix(0.35, 1.0, vDensity);
      color = mix(vec3(gray), color, sat);
      alpha *= mix(0.55, 1.0, vDensity);
    }

    // Exponential-squared depth fog toward the near-black background.
    float f = 1.0 - exp(-uFogDensity * uFogDensity * vDepth * vDepth);
    color = mix(color, uFogColor, clamp(f, 0.0, 1.0));

    gl_FragColor = vec4(color, alpha);
  }
`;

// ---------------------------------------------------------------------------
// NavigatorScene — the imperative three.js engine. React drives it via focus()
// / reset() / setProbeActive() and listens through callbacks; it owns the
// render loop and picking.
// ---------------------------------------------------------------------------
class NavigatorScene {
  private canvas: HTMLCanvasElement;
  private items: ConvergeImage[];
  private cb: SceneCallbacks;

  private renderer!: THREE.WebGLRenderer;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private controls!: OrbitControls;

  private positions!: Float32Array; // n*3, RENDER ONLY, frozen — never mutated
  private displayPositions!: Float32Array; // n*3 = positions + warpOffsets, backs iOffset
  private warpOffsets!: Float32Array; // n*3, persistent drag-warp accumulator
  private aspects!: Float32Array; // per-instance image aspect (w/h)
  private stateArr!: Float32Array; // per-instance emphasis (animated)
  private promoteArr!: Float32Array; // per-instance probe hi-res-pool membership
  private densityArr!: Float32Array; // per-instance local density [0,1]
  private iStateAttr!: THREE.InstancedBufferAttribute;
  private iOffsetAttr!: THREE.InstancedBufferAttribute;
  private iPromoteAttr!: THREE.InstancedBufferAttribute;
  private iDensityAttr!: THREE.InstancedBufferAttribute;
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

  // ---- probe mode -----------------------------------------------------------
  private probeActive = false;
  private probePoint = new THREE.Vector3();
  private probeMarker: THREE.Mesh | null = null;
  private raycaster = new THREE.Raycaster();
  private hiResPool: HiResSlot[] = [];
  private lastProbeIndices: number[] = [];
  private lastClientX: number | null = null;
  private lastClientY: number | null = null;
  private orbiting = false;
  private scheduleProbeQuery!: (() => void) & { cancel: () => void };

  // ---- drag-to-pull warp ------------------------------------------------------
  private warpActive = false;
  private warpAnchorIndex = -1;
  private warpAnchorOrigin = new THREE.Vector3(); // fixed grabbed-point layout pos
  private warpCursorWorld = new THREE.Vector3(); // evolving dragged-to world pos
  private warpReleaseTween: { active: boolean; start: number; from: Float32Array } | null = null;

  // scratch
  private tmpV = new THREE.Vector3();
  private tmpT = new THREE.Vector3();
  private tmpProbe = new THREE.Vector3();
  private tmpNdc = new THREE.Vector2();
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
    this.scheduleProbeQuery = debounce(() => this.runProbeQuery(), PROBE_DEBOUNCE_MS);
  }

  // Build atlas + mesh, wire the scene, start the loop. Resolves once visible.
  async build(): Promise<void> {
    // Lay out the cloud (UMAP, cached). RENDER ONLY.
    this.positions = computeLayout(this.items);
    this.displayPositions = this.positions.slice();
    this.warpOffsets = new Float32Array(this.n * 3);

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
    this.buildProbeMarker();

    this.canvas.addEventListener("pointermove", this.onPointerMove);
    this.canvas.addEventListener("pointerdown", this.onPointerDown);
    this.canvas.addEventListener("pointerup", this.onPointerUp);

    const animate = () => {
      this.raf = requestAnimationFrame(animate);
      const now = performance.now();
      if (this.tween.active) this.stepTween(now);
      else this.controls.update();
      this.updateEmphasis();
      if (this.probeActive) {
        if (this.orbiting && !this.warpActive && this.lastClientX !== null && this.lastClientY !== null) {
          this.updateProbePoint(this.lastClientX, this.lastClientY);
        }
        this.syncPoolPositions();
      }
      if (this.warpReleaseTween?.active) this.stepWarpRelease(now);
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
    this.promoteArr = new Float32Array(n);
    this.densityArr = new Float32Array(n).fill(1); // neutral until computeDensity runs

    const indices = new Float32Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;

    const base = new THREE.PlaneGeometry(1, 1);
    const geo = new THREE.InstancedBufferGeometry();
    geo.index = base.index;
    geo.setAttribute("position", base.getAttribute("position"));
    geo.setAttribute("uv", base.getAttribute("uv"));
    geo.instanceCount = n;
    this.iOffsetAttr = new THREE.InstancedBufferAttribute(this.displayPositions, 3);
    this.iOffsetAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("iOffset", this.iOffsetAttr);
    geo.setAttribute("iAtlasUV", new THREE.InstancedBufferAttribute(atlas.uvs, 4));
    geo.setAttribute("iAspect", new THREE.InstancedBufferAttribute(atlas.aspects, 1));
    this.iStateAttr = new THREE.InstancedBufferAttribute(this.stateArr, 1);
    this.iStateAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("iState", this.iStateAttr);
    geo.setAttribute("iIndex", new THREE.InstancedBufferAttribute(indices, 1));
    this.iPromoteAttr = new THREE.InstancedBufferAttribute(this.promoteArr, 1);
    this.iPromoteAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("iPromote", this.iPromoteAttr);
    this.iDensityAttr = new THREE.InstancedBufferAttribute(this.densityArr, 1);
    this.iDensityAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("iDensity", this.iDensityAttr);
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
        uProbeActive: { value: 0 },
        uProbeBaseScale: { value: PROBE_BASE_SCALE },
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

  private buildProbeMarker() {
    const geo = new THREE.SphereGeometry(LAYOUT_RADIUS * 0.012, 12, 12);
    const mat = new THREE.MeshBasicMaterial({
      color: 0x7dd3fc,
      transparent: true,
      opacity: 0.85,
      depthTest: false,
    });
    this.probeMarker = new THREE.Mesh(geo, mat);
    this.probeMarker.renderOrder = 12;
    this.probeMarker.visible = false;
    this.scene.add(this.probeMarker);
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

  // ---- probe mode: hi-res pool (proximity/hover sharpening) ----------------

  private makeHiResSlot(): HiResSlot {
    const material = new THREE.SpriteMaterial({ transparent: true, depthTest: false, fog: false });
    const sprite = new THREE.Sprite(material);
    sprite.renderOrder = 8;
    sprite.visible = false;
    this.scene.add(sprite);
    return { sprite, material, texture: null, index: -1, seq: 0 };
  }

  private loadIntoSlot(slot: HiResSlot) {
    const seq = ++slot.seq;
    const index = slot.index;
    const item = this.items[index];
    const loader = new THREE.TextureLoader();
    loader.load(item.thumb, (tex) => {
      if (this.disposed || slot.seq !== seq) {
        tex.dispose();
        return;
      }
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
      if (slot.texture) slot.texture.dispose();
      slot.texture = tex;
      slot.material.map = tex;
      slot.material.needsUpdate = true;
      const a = item.w / item.h || 1;
      const sw = a >= 1 ? PROBE_SPRITE_SIZE : PROBE_SPRITE_SIZE * a;
      const sh = a >= 1 ? PROBE_SPRITE_SIZE / a : PROBE_SPRITE_SIZE;
      slot.sprite.scale.set(sw, sh, 1);
      slot.sprite.position.set(
        this.displayPositions[index * 3],
        this.displayPositions[index * 3 + 1],
        this.displayPositions[index * 3 + 2]
      );
      slot.sprite.visible = true;
    });
  }

  private disposeHiResSlot(slot: HiResSlot) {
    slot.seq++;
    this.scene.remove(slot.sprite);
    slot.sprite.visible = false;
    if (slot.texture) {
      slot.texture.dispose();
      slot.texture = null;
    }
    slot.material.dispose();
  }

  private promoteIndex(index: number) {
    if (this.hiResPool.some((s) => s.index === index)) return;
    const slot = this.makeHiResSlot();
    slot.index = index;
    this.hiResPool.push(slot);
    this.loadIntoSlot(slot);
    this.promoteArr[index] = 1;
  }

  private demoteIndex(index: number) {
    const i = this.hiResPool.findIndex((s) => s.index === index);
    if (i < 0) return;
    const [slot] = this.hiResPool.splice(i, 1);
    this.disposeHiResSlot(slot);
    this.promoteArr[index] = 0;
  }

  // Event-driven (called from hover changes and debounced probe queries) —
  // never scanned per frame, so promotion adds no per-frame cost.
  private refreshPromotionSet() {
    const desired: number[] = [];
    if (this.hoverIndex >= 0) desired.push(this.hoverIndex);
    for (const idx of this.lastProbeIndices) {
      if (desired.length >= PROBE_POOL_SIZE) break;
      if (!desired.includes(idx)) desired.push(idx);
    }
    const desiredSet = new Set(desired);
    for (const slot of [...this.hiResPool]) {
      if (!desiredSet.has(slot.index)) this.demoteIndex(slot.index);
    }
    for (const idx of desired) this.promoteIndex(idx);
    this.iPromoteAttr.needsUpdate = true;
  }

  private clearPromotions() {
    for (const slot of [...this.hiResPool]) this.disposeHiResSlot(slot);
    this.hiResPool = [];
    this.promoteArr.fill(0);
    this.iPromoteAttr.needsUpdate = true;
  }

  // Promoted (collapsed) sprites must follow displayPositions every frame so a
  // warp drag doesn't visually separate a hi-res sprite from its dot.
  private syncPoolPositions() {
    for (const slot of this.hiResPool) {
      slot.sprite.position.set(
        this.displayPositions[slot.index * 3],
        this.displayPositions[slot.index * 3 + 1],
        this.displayPositions[slot.index * 3 + 2]
      );
    }
  }

  // ---- probe mode: live sampling --------------------------------------------

  private ensureDensity() {
    const arr = computeDensity(this.positions, this.n, DENSITY_K);
    this.densityArr.set(arr);
    this.iDensityAttr.needsUpdate = true;
  }

  // Point along the camera ray through (clientX, clientY) at a fixed distance
  // equal to the current camera-to-target distance. Tracks both cursor
  // movement (ray direction) and camera orbiting (ray origin/direction), and
  // scales naturally with zoom. Marching a fixed distance along the ray (vs.
  // intersecting a sphere centered on the orbit target) guarantees the probed
  // point is always exactly that distance from the CAMERA, regardless of
  // screen position — a target-centered sphere would sit the camera almost
  // exactly on its own surface, so off-center (near-tangent) rays could
  // intersect arbitrarily close to the camera and blow up billboard sizes.
  private computeRayPoint(clientX: number, clientY: number, out: THREE.Vector3): THREE.Vector3 {
    const rect = this.canvas.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = -((clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.tmpNdc.set(ndcX, ndcY), this.camera);
    const dist = Math.max(this.controls.getDistance(), 0.01);
    return out.copy(this.raycaster.ray.direction).multiplyScalar(dist).add(this.raycaster.ray.origin);
  }

  private updateProbePoint(clientX: number, clientY: number) {
    this.computeRayPoint(clientX, clientY, this.probePoint);
    if (this.probeMarker) this.probeMarker.position.copy(this.probePoint);
    this.scheduleProbeQuery();
  }

  // Debounced. Layout-space k-NN only — never .vec/cosine.
  private runProbeQuery() {
    if (!this.probeActive || this.warpActive) return;
    const knn = kNearestByPosition(this.positions, this.n, this.probePoint, PROBE_K);
    this.lastProbeIndices = knn.map((r) => r.index);
    const result = knn.map((r) => ({ id: this.items[r.index].id, dist: r.dist }));
    const meanDist = knn.length ? knn.reduce((s, r) => s + r.dist, 0) / knn.length : 0;
    this.refreshPromotionSet();
    this.cb.onProbeQuery(result, meanDist);
  }

  // Public: toggle probe mode (called by React).
  setProbeActive(active: boolean) {
    this.probeActive = active;
    if (this.material) this.material.uniforms.uProbeActive.value = active ? 1 : 0;
    this.clearIdle();
    if (active) {
      this.ensureDensity();
      this.controls.autoRotate = false;
      if (this.probeMarker) this.probeMarker.visible = true;
    } else {
      this.scheduleProbeQuery.cancel();
      this.lastProbeIndices = [];
      this.clearPromotions();
      if (this.probeMarker) this.probeMarker.visible = false;
      if (this.warpActive) {
        this.warpActive = false;
        this.controls.enabled = true;
      }
      this.warpReleaseTween = null;
      this.warpOffsets.fill(0);
      this.displayPositions.set(this.positions);
      if (this.iOffsetAttr) this.iOffsetAttr.needsUpdate = true;
      if (this.focusedIndex < 0) this.scheduleIdle();
    }
  }

  // ---- probe mode: drag-to-pull warp ----------------------------------------

  // Falloff-weighted directional offset applied to nearby points' DISPLAY
  // positions only — this.positions (the frozen layout) is never written.
  private applyWarp(clientX: number, clientY: number) {
    this.computeRayPoint(clientX, clientY, this.tmpProbe);
    const dx = this.tmpProbe.x - this.warpCursorWorld.x;
    const dy = this.tmpProbe.y - this.warpCursorWorld.y;
    const dz = this.tmpProbe.z - this.warpCursorWorld.z;
    this.warpCursorWorld.copy(this.tmpProbe);
    if (dx === 0 && dy === 0 && dz === 0) return;

    const r2 = WARP_RADIUS * WARP_RADIUS;
    // Cap how far any single point can be pulled from its true position — a
    // bounded "rubber band" reach. Without this, a long fast drag can fling a
    // point arbitrarily close to the camera, where its billboard size (which
    // scales with 1/distance) blows up to fill the screen.
    const maxOffset = WARP_RADIUS * 1.5;
    const maxOffset2 = maxOffset * maxOffset;
    const ax = this.warpAnchorOrigin.x;
    const ay = this.warpAnchorOrigin.y;
    const az = this.warpAnchorOrigin.z;
    for (let i = 0; i < this.n; i++) {
      const px = this.positions[i * 3];
      const py = this.positions[i * 3 + 1];
      const pz = this.positions[i * 3 + 2];
      const ddx = px - ax;
      const ddy = py - ay;
      const ddz = pz - az;
      const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
      const wgt = Math.exp(-d2 / r2);
      if (wgt < 1e-4) continue;
      let ox = this.warpOffsets[i * 3] + dx * wgt;
      let oy = this.warpOffsets[i * 3 + 1] + dy * wgt;
      let oz = this.warpOffsets[i * 3 + 2] + dz * wgt;
      const oLen2 = ox * ox + oy * oy + oz * oz;
      if (oLen2 > maxOffset2) {
        const scale = maxOffset / Math.sqrt(oLen2);
        ox *= scale;
        oy *= scale;
        oz *= scale;
      }
      this.warpOffsets[i * 3] = ox;
      this.warpOffsets[i * 3 + 1] = oy;
      this.warpOffsets[i * 3 + 2] = oz;
      this.displayPositions[i * 3] = px + ox;
      this.displayPositions[i * 3 + 1] = py + oy;
      this.displayPositions[i * 3 + 2] = pz + oz;
    }
    this.iOffsetAttr.needsUpdate = true;
  }

  private startWarpReleaseTween() {
    this.warpReleaseTween = {
      active: true,
      start: performance.now(),
      from: this.warpOffsets.slice(),
    };
  }

  private stepWarpRelease(now: number) {
    const tw = this.warpReleaseTween;
    if (!tw) return;
    const t = Math.min(1, (now - tw.start) / WARP_RELEASE_MS);
    const e = easeInOutCubic(t);
    for (let i = 0; i < this.warpOffsets.length; i++) {
      const v = tw.from[i] * (1 - e);
      this.warpOffsets[i] = v;
      this.displayPositions[i] = this.positions[i] + v;
    }
    this.iOffsetAttr.needsUpdate = true;
    if (t >= 1) {
      this.warpOffsets.fill(0);
      this.displayPositions.set(this.positions);
      this.iOffsetAttr.needsUpdate = true;
      this.warpReleaseTween = null;
    }
  }

  // ---- idle auto-rotate ----------------------------------------------------

  private scheduleIdle() {
    this.clearIdle();
    this.idleTimer = window.setTimeout(() => {
      if (!this.disposed && this.focusedIndex < 0 && !this.probeActive && !this.tween.active) {
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
    this.orbiting = true;
  }

  private onControlsEnd() {
    this.orbiting = false;
    if (this.focusedIndex < 0 && !this.probeActive) this.scheduleIdle();
  }

  // ---- picking -------------------------------------------------------------

  private pickAt(clientX: number, clientY: number): number {
    const rect = this.canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const W = rect.width;
    const H = rect.height;
    const projScale = H / (2 * Math.tan((FOV * Math.PI) / 180 / 2));

    // The focused hi-res sprite (explore mode) takes click priority so its
    // image never resets.
    if (this.focusedIndex >= 0) {
      const i = this.focusedIndex;
      this.tmpV.set(
        this.displayPositions[i * 3],
        this.displayPositions[i * 3 + 1],
        this.displayPositions[i * 3 + 2]
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

    // Probe-mode promoted instances: hit-test their hi-res sprite bounds (the
    // instanced quad is collapsed to zero while promoted).
    if (this.probeActive && this.hiResPool.length > 0) {
      let promotedBest = -1;
      let promotedBestDist = Infinity;
      for (const slot of this.hiResPool) {
        const i = slot.index;
        this.tmpV.set(
          this.displayPositions[i * 3],
          this.displayPositions[i * 3 + 1],
          this.displayPositions[i * 3 + 2]
        );
        const dist = this.camera.position.distanceTo(this.tmpV);
        this.tmpV.project(this.camera);
        if (this.tmpV.z > 1) continue;
        const sx = (this.tmpV.x * 0.5 + 0.5) * W;
        const sy = (-this.tmpV.y * 0.5 + 0.5) * H;
        const halfH = (PROBE_SPRITE_SIZE / 2 / dist) * projScale;
        const halfW = halfH * Math.max(1, this.aspects[i]);
        if (Math.abs(px - sx) <= halfW && Math.abs(py - sy) <= halfH && dist < promotedBestDist) {
          promotedBestDist = dist;
          promotedBest = i;
        }
      }
      if (promotedBest >= 0) return promotedBest;
    }

    let best = -1;
    let bestDist = Infinity;
    for (let i = 0; i < this.n; i++) {
      if (i === this.focusedIndex) continue;
      if (this.probeActive && this.promoteArr[i] > 0.5) continue; // handled above
      this.tmpV.set(
        this.displayPositions[i * 3],
        this.displayPositions[i * 3 + 1],
        this.displayPositions[i * 3 + 2]
      );
      const dist = this.camera.position.distanceTo(this.tmpV);
      this.tmpV.project(this.camera);
      if (this.tmpV.z > 1) continue;
      const sx = (this.tmpV.x * 0.5 + 0.5) * W;
      const sy = (-this.tmpV.y * 0.5 + 0.5) * H;
      const gain = 1 + DIST_GAIN * Math.min(2, dist / OVERVIEW_DIST);
      let halfH = ((BASE_SIZE * gain) / 2 / dist) * projScale;
      if (this.probeActive) halfH *= PROBE_BASE_SCALE; // match the shrunk at-rest size
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
      if (this.probeActive) this.refreshPromotionSet();
    }
    if (this.probeActive) {
      this.lastClientX = e.clientX;
      this.lastClientY = e.clientY;
      if (this.warpActive) {
        this.applyWarp(e.clientX, e.clientY);
      } else {
        this.updateProbePoint(e.clientX, e.clientY);
      }
    }
  }

  private onPointerDown(e: PointerEvent) {
    this.down = { x: e.clientX, y: e.clientY, t: performance.now(), id: e.pointerId };
    if (this.probeActive) {
      const idx = this.pickAt(e.clientX, e.clientY);
      if (idx >= 0) {
        this.warpActive = true;
        this.warpAnchorIndex = idx;
        this.warpAnchorOrigin.set(
          this.positions[idx * 3],
          this.positions[idx * 3 + 1],
          this.positions[idx * 3 + 2]
        );
        this.computeRayPoint(e.clientX, e.clientY, this.warpCursorWorld);
        this.controls.enabled = false;
        if (this.warpReleaseTween) this.warpReleaseTween.active = false;
      }
    }
  }

  private onPointerUp(e: PointerEvent) {
    if (this.warpActive) {
      this.warpActive = false;
      this.warpAnchorIndex = -1;
      this.controls.enabled = true;
      this.startWarpReleaseTween();
      this.down = null;
      return; // a warp release is never also a click
    }
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
    this.scheduleProbeQuery?.cancel();
    this.canvas.removeEventListener("pointermove", this.onPointerMove);
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    this.canvas.removeEventListener("pointerup", this.onPointerUp);
    if (this.controls) {
      this.controls.removeEventListener("start", this.onControlsStart);
      this.controls.removeEventListener("end", this.onControlsEnd);
      this.controls.dispose();
    }
    if (this.probeMarker) {
      this.scene.remove(this.probeMarker);
      this.probeMarker.geometry.dispose();
      (this.probeMarker.material as THREE.Material).dispose();
    }
    for (const slot of this.hiResPool) this.disposeHiResSlot(slot);
    this.hiResPool = [];
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

  // Probe mode state.
  type Mode = "explore" | "probe";
  const [mode, setMode] = useState<Mode>("explore");
  const [probeNeighbors, setProbeNeighbors] = useState<{ id: string; dist: number }[]>([]);
  const [probeMeanDist, setProbeMeanDist] = useState(0);

  const byId = useMemo(() => new Map(items.map((it) => [it.id, it])), [items]);
  const historyRef = useRef<string[]>([]);

  // Latest-value refs so the once-mounted scene callback / key handlers stay fresh.
  const focusedIdRef = useRef<string | null>(null);
  focusedIdRef.current = focusedId;
  const neighborsRef = useRef<Neighbor[]>([]);
  neighborsRef.current = neighbors;
  const modeRef = useRef<Mode>(mode);
  modeRef.current = mode;

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

  // ---- probe mode ------------------------------------------------------------

  const enterProbe = useCallback(() => {
    resetView(); // drop any focused image so we start from a clean cloud
    setMode("probe");
    setProbeNeighbors([]);
    setProbeMeanDist(0);
    sceneRef.current?.setProbeActive(true);
  }, [resetView]);

  const exitProbe = useCallback(() => {
    setMode("explore");
    setProbeNeighbors([]);
    setProbeMeanDist(0);
    sceneRef.current?.setProbeActive(false);
  }, []);

  // Build the scene once on mount.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const scene = new NavigatorScene(canvas, items, {
      onPick: (id) => {
        if (modeRef.current !== "explore") return;
        if (id === null) {
          if (focusedIdRef.current) resetView();
        } else if (id !== focusedIdRef.current) {
          focusOn(id, true);
        }
      },
      onProbeQuery: (result, meanDist) => {
        setProbeNeighbors(result);
        setProbeMeanDist(meanDist);
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

  // Keyboard: ←/→ walk neighbors, Esc exits probe/focus/overlay.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (modeRef.current === "probe") exitProbe();
        else if (focusedIdRef.current) resetView();
        else onExit();
        return;
      }
      if (modeRef.current === "probe") return; // arrow-stepping is an explore-only affordance
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
  }, [onExit, resetView, stepForward, stepBack, exitProbe]);

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

  // Probe preview panel: cross-faded thumbnail stack weighted by inverse
  // distance, with density-driven blur/saturation/noise (reuses the same k-NN
  // query — no second computation).
  const probeWeights = useMemo(() => {
    const w = probeNeighbors.map((nb) => 1 / (1 + nb.dist));
    const maxW = Math.max(1e-6, ...w);
    return w.map((v) => v / maxW);
  }, [probeNeighbors]);
  const probeSparsity = Math.max(0, Math.min(1, probeMeanDist / (LAYOUT_RADIUS * 0.18)));

  return (
    <div className="navigator-overlay" onTouchStart={onTouchStart} onTouchEnd={onTouchEnd}>
      <canvas ref={canvasRef} className="navigator-canvas" />

      <button className="navigator-close" onClick={onExit} title="Close (Esc)">
        ✕
      </button>

      {/* Top toolbar: enter / leave probe mode. */}
      {!loading && (
        <div className="navigator-toolbar">
          {mode === "explore" ? (
            <button
              className="navigator-probe-btn"
              onClick={enterProbe}
              title="Probe the cloud as an interactive point-cloud instrument"
            >
              ◎ Probe
            </button>
          ) : (
            <button
              className="navigator-probe-btn active"
              onClick={exitProbe}
              title="Exit probe mode (Esc)"
            >
              ✕ Exit probe
            </button>
          )}
        </div>
      )}

      {loading && (
        <div className="navigator-loading">
          <div className="navigator-spinner" />
          <span>Building cloud…</span>
        </div>
      )}

      {!loading && mode === "explore" && !focusedId && (
        <div className="navigator-hint">
          Drag to orbit · scroll to zoom · click an image to dive in
        </div>
      )}

      {!loading && mode === "probe" && (
        <div className="navigator-hint">
          Move to probe the cloud · drag a point to pull its neighbors · release to snap back
        </div>
      )}

      {mode === "explore" && focusedId && neighbors.length > 0 && (
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

      {mode === "probe" && probeNeighbors.length > 0 && (
        <div
          className="navigator-probe-panel"
          style={
            {
              filter: `blur(${(probeSparsity * 3).toFixed(2)}px) saturate(${(1 - probeSparsity * 0.6).toFixed(2)})`,
              "--noise-opacity": probeSparsity * 0.5,
            } as React.CSSProperties
          }
        >
          <div className="navigator-probe-stack">
            {probeNeighbors.map((nb, i) => {
              const img = byId.get(nb.id);
              if (!img) return null;
              return (
                <img
                  key={nb.id}
                  className="navigator-probe-layer"
                  src={img.thumb}
                  alt=""
                  draggable={false}
                  style={{ opacity: probeWeights[i] * (1 - i * 0.06), zIndex: probeNeighbors.length - i }}
                />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
