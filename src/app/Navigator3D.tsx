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
  computePath,
  getNeighbors,
  nearestImage,
  topNeighbors,
  slerp,
  LAYOUT_RADIUS,
  type Neighbor,
  type PathStop,
} from "./navigator3d";

// ---------------------------------------------------------------------------
// Tunables (all relative to LAYOUT_RADIUS so the scene stays self-consistent)
// ---------------------------------------------------------------------------
const FOV = 50;
const NEIGHBOR_K = 12;
const SLIDER_K = 5; // live "neighbors of this arc point" shown while scrubbing
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

  // Latent-space slider path: a polyline through the cloud + a cursor marker.
  private pathLine: THREE.Line | null = null;
  private pathMarker: THREE.Mesh | null = null;

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

  // ---- latent-space slider path -------------------------------------------

  private posOf(id: string, into: THREE.Vector3): boolean {
    const i = this.idToIndex.get(id);
    if (i === undefined) return false;
    into.set(
      this.positions[i * 3],
      this.positions[i * 3 + 1],
      this.positions[i * 3 + 2]
    );
    return true;
  }

  // Draw a polyline through the layout positions of the distinct images the arc
  // passes through, and frame the camera once to fit it. Layout is RENDER ONLY —
  // the path order itself comes from the full-vector slerp walk upstream.
  setPath(ids: string[]) {
    this.clearPath();
    this.controls.autoRotate = false;
    this.clearIdle();
    if (ids.length === 0) return;

    const pts: THREE.Vector3[] = [];
    const p = new THREE.Vector3();
    for (const id of ids) {
      if (this.posOf(id, p)) pts.push(p.clone());
    }
    if (pts.length === 0) return;

    if (pts.length >= 2) {
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.5,
        depthTest: false,
      });
      this.pathLine = new THREE.Line(geo, mat);
      this.pathLine.frustumCulled = false;
      this.pathLine.renderOrder = 9;
      this.scene.add(this.pathLine);
    }

    // Cursor marker — a small emissive sphere riding the current arc point.
    const markerGeo = new THREE.SphereGeometry(LAYOUT_RADIUS * 0.018, 16, 16);
    const markerMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.9,
      depthTest: false,
    });
    this.pathMarker = new THREE.Mesh(markerGeo, markerMat);
    this.pathMarker.renderOrder = 11;
    this.scene.add(this.pathMarker);

    // Frame the camera once to fit the path's bounding box.
    const box = new THREE.Box3().setFromPoints(pts);
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, LAYOUT_RADIUS * 0.2);
    const dir = this.camera.position.clone().sub(this.controls.target);
    if (dir.lengthSq() < 1e-6) dir.copy(this.overviewPos);
    dir.normalize().multiplyScalar(radius / Math.tan((FOV * Math.PI) / 180 / 2) + radius);
    this.frameCameraTo(center.clone().add(dir), center.clone());
  }

  // Per-scrub update: park the hi-res sprite + marker on this image instantly
  // (no camera tween — focus() would lurch on every slider tick) and collapse
  // its instanced quad so the hi-res billboard stands in for it.
  setPathCursor(id: string) {
    const index = this.idToIndex.get(id);
    if (index === undefined) return;
    this.focusedIndex = index;
    this.material.uniforms.uFocusedIndex.value = index;
    const p = new THREE.Vector3();
    if (!this.posOf(id, p)) return;
    if (this.pathMarker) this.pathMarker.position.copy(p);
    this.loadHiRes(index, p.clone());
  }

  clearPath() {
    this.focusedIndex = -1;
    if (this.material) this.material.uniforms.uFocusedIndex.value = -1;
    this.removeHiRes();
    if (this.pathLine) {
      this.scene.remove(this.pathLine);
      this.pathLine.geometry.dispose();
      (this.pathLine.material as THREE.Material).dispose();
      this.pathLine = null;
    }
    if (this.pathMarker) {
      this.scene.remove(this.pathMarker);
      this.pathMarker.geometry.dispose();
      (this.pathMarker.material as THREE.Material).dispose();
      this.pathMarker = null;
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
    if (this.pathLine) {
      this.pathLine.geometry.dispose();
      (this.pathLine.material as THREE.Material).dispose();
    }
    if (this.pathMarker) {
      this.pathMarker.geometry.dispose();
      (this.pathMarker.material as THREE.Material).dispose();
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

  // Latent-space slider state.
  type Mode = "explore" | "pickA" | "pickB" | "slider";
  const [mode, setMode] = useState<Mode>("explore");
  const [endpointA, setEndpointA] = useState<string | null>(null);
  const [endpointB, setEndpointB] = useState<string | null>(null);
  const [pathStops, setPathStops] = useState<PathStop[]>([]);
  const [t, setT] = useState(0);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [currentScore, setCurrentScore] = useState(1);
  const [arcNeighbors, setArcNeighbors] = useState<Neighbor[]>([]);

  const byId = useMemo(() => new Map(items.map((it) => [it.id, it])), [items]);
  const historyRef = useRef<string[]>([]);

  // Latest-value refs so the once-mounted scene callback / key handlers stay fresh.
  const focusedIdRef = useRef<string | null>(null);
  focusedIdRef.current = focusedId;
  const neighborsRef = useRef<Neighbor[]>([]);
  neighborsRef.current = neighbors;
  const modeRef = useRef<Mode>(mode);
  modeRef.current = mode;
  const endpointARef = useRef<string | null>(endpointA);
  endpointARef.current = endpointA;

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

  // ---- latent-space slider -------------------------------------------------

  // Apply a slider position: slerp A→B at t, retrieve the nearest REAL image on
  // the full embedding, and park the 3D cursor on it. Endpoints are pinned
  // (t=0 → A, t=1 → B); interior points take the nearest EXCLUDING A and B so a
  // genuinely different image surfaces (they otherwise dominate the whole arc).
  const applyT = useCallback(
    (tv: number) => {
      const a = endpointA ? byId.get(endpointA) : undefined;
      const b = endpointB ? byId.get(endpointB) : undefined;
      setT(tv);
      if (!a || !b) return;
      const q = slerp(a.vec, b.vec, tv);
      const exclude = new Set([endpointA!, endpointB!]);
      let nb: Neighbor | null;
      if (tv <= 0) nb = { id: endpointA!, score: 1 };
      else if (tv >= 1) nb = { id: endpointB!, score: 1 };
      else nb = nearestImage(q, items, exclude);
      if (nb) {
        setCurrentId(nb.id);
        setCurrentScore(nb.score);
        sceneRef.current?.setPathCursor(nb.id);
      }
      // Live ranked neighbors of this exact arc point (endpoints excluded).
      setArcNeighbors(topNeighbors(q, items, SLIDER_K, exclude));
    },
    [endpointA, endpointB, byId, items]
  );

  const startInterpolate = useCallback(() => {
    resetView(); // drop any focused image so we start from a clean cloud
    setMode("pickA");
    setEndpointA(null);
    setEndpointB(null);
    setPathStops([]);
    setCurrentId(null);
    setArcNeighbors([]);
    setT(0);
  }, [resetView]);

  const exitInterpolate = useCallback(() => {
    setMode("explore");
    setEndpointA(null);
    setEndpointB(null);
    setPathStops([]);
    setCurrentId(null);
    setArcNeighbors([]);
    setT(0);
    sceneRef.current?.clearPath();
  }, []);

  // Click handler while picking endpoints (wired into the scene's onPick).
  const handlePick = useCallback(
    (id: string) => {
      const m = modeRef.current;
      if (m === "pickA") {
        setEndpointA(id);
        setMode("pickB");
      } else if (m === "pickB") {
        const aId = endpointARef.current;
        if (!aId || aId === id) return; // need a distinct endpoint B
        const a = byId.get(aId);
        const b = byId.get(id);
        if (!a || !b) return;
        setEndpointB(id);
        const stops = computePath(a.vec, b.vec, items, aId, id);
        setPathStops(stops);
        sceneRef.current?.setPath(stops.map((s) => s.id));
        setMode("slider");
        // Anchor the slider at t=0 → A.
        setT(0);
        setCurrentId(aId);
        setCurrentScore(1);
        setArcNeighbors(topNeighbors(a.vec, items, SLIDER_K, new Set([aId, id])));
        sceneRef.current?.setPathCursor(aId);
      }
    },
    [byId, items]
  );
  const handlePickRef = useRef(handlePick);
  handlePickRef.current = handlePick;

  // Build the scene once on mount.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const scene = new NavigatorScene(canvas, items, {
      onPick: (id) => {
        const m = modeRef.current;
        if (m === "pickA" || m === "pickB") {
          if (id !== null) handlePickRef.current(id);
          return;
        }
        if (m === "slider") return; // scrub with the slider, not by clicking
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
        if (modeRef.current !== "explore") exitInterpolate();
        else if (focusedIdRef.current) resetView();
        else onExit();
        return;
      }
      // In slider mode, arrows nudge the position along the arc.
      if (modeRef.current === "slider") {
        if (e.key === "ArrowRight") {
          e.preventDefault();
          applyT(Math.min(1, t + 0.02));
        } else if (e.key === "ArrowLeft") {
          e.preventDefault();
          applyT(Math.max(0, t - 0.02));
        }
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
  }, [onExit, resetView, stepForward, stepBack, exitInterpolate, applyT, t]);

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

      {/* Top toolbar: enter / leave the latent-space slider. */}
      {!loading && (
        <div className="navigator-toolbar">
          {mode === "explore" ? (
            <button
              className="navigator-interp-btn"
              onClick={startInterpolate}
              title="Slide along the arc between two images"
            >
              ⇄ Interpolate
            </button>
          ) : (
            <button
              className="navigator-interp-btn active"
              onClick={exitInterpolate}
              title="Exit interpolate mode (Esc)"
            >
              ✕ Exit interpolate
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

      {!loading && (mode === "pickA" || mode === "pickB") && (
        <div className="navigator-hint">
          {mode === "pickA"
            ? "Pick a start image — click anywhere in the cloud"
            : "Pick an end image — the slider walks the arc between them"}
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

      {/* Latent-space slider panel. */}
      {mode === "slider" && endpointA && endpointB && (() => {
        const aImg = byId.get(endpointA);
        const bImg = byId.get(endpointB);
        const curImg = currentId ? byId.get(currentId) : null;
        // The active stop = last stop whose arc segment has started.
        let activeStop = 0;
        for (let i = 0; i < pathStops.length; i++) {
          if (pathStops[i].tStart <= t) activeStop = i;
        }
        return (
          <div className="navigator-slider-panel">
            <div className="navigator-slider-readout">
              {curImg && <span className="dot" />}
              t = {t.toFixed(2)} · nearest similarity {currentScore.toFixed(3)} ·{" "}
              {pathStops.length} images on the arc
            </div>

            {/* Live: the images sitting closest to THIS point on the arc, ranked
                (endpoints excluded). The #1 is the headline shown big in 3D. */}
            {arcNeighbors.length > 0 && (
              <div className="navigator-arcstrip">
                {arcNeighbors.map((nb) => {
                  const img = byId.get(nb.id);
                  if (!img) return null;
                  return (
                    <div
                      key={nb.id}
                      className={`navigator-arcstrip-item ${nb.id === currentId ? "active" : ""}`}
                      title={`similarity ${nb.score.toFixed(3)}`}
                    >
                      <img src={img.thumb} alt="" draggable={false} />
                    </div>
                  );
                })}
              </div>
            )}

            <div className="navigator-slider-row">
              {aImg && (
                <button
                  className={`navigator-endpoint ${t === 0 ? "live" : ""}`}
                  onClick={() => applyT(0)}
                  title="Endpoint A (t=0)"
                >
                  <img src={aImg.thumb} alt="" draggable={false} />
                  <span>A</span>
                </button>
              )}

              <input
                className="navigator-slider"
                type="range"
                min={0}
                max={1000}
                value={Math.round(t * 1000)}
                onChange={(e) => applyT(Number(e.target.value) / 1000)}
              />

              {bImg && (
                <button
                  className={`navigator-endpoint ${t === 1 ? "live" : ""}`}
                  onClick={() => applyT(1)}
                  title="Endpoint B (t=1)"
                >
                  <img src={bImg.thumb} alt="" draggable={false} />
                  <span>B</span>
                </button>
              )}
            </div>

            {pathStops.length > 0 && (
              <div className="navigator-pathstrip">
                {pathStops.map((stop, i) => {
                  const img = byId.get(stop.id);
                  if (!img) return null;
                  return (
                    <button
                      key={`${stop.id}-${i}`}
                      className={`navigator-pathstrip-item ${i === activeStop ? "active" : ""}`}
                      onClick={() => applyT(stop.tStart)}
                      title={`t≈${stop.tStart.toFixed(2)}`}
                    >
                      <img src={img.thumb} alt="" draggable={false} />
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        );
      })()}
    </div>
  );
}
