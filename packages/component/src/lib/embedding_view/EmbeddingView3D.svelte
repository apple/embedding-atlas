<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import * as THREE from "three";
  import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

  import { defaultCategoryColors } from "../colors.js";
  import { pickVisibleIndices } from "./downsample_3d.js";
  import StatusBar3D from "./StatusBar3D.svelte";

  interface Props {
    data: {
      x: Float32Array | number[];
      y: Float32Array | number[];
      z: Float32Array | number[];
      category?: Uint8Array | number[] | null;
    };
    categoryColors?: string[] | null;
    width?: number;
    height?: number;
    /** Point size in the same units as the data (world-space), not screen pixels.
     *  If not specified, a size is picked automatically relative to the data's bounding sphere. */
    pointSize?: number | null;
    /** Row indices (into the `data` arrays) to render as highlighted/selected. */
    selectedIndices?: number[] | null;
    /** Called with the row index (into the `data` arrays) of the point under the cursor when the
     *  user clicks a point (a press-and-release with negligible pointer movement, so it doesn't
     *  fire on orbit/pan drags). Called with `null` when a click doesn't hit any point. */
    onPointClick?: ((index: number | null) => void) | null;
    /** Called (throttled) with the row index under the cursor as the pointer moves, for hover
     *  tooltips. Called with `null` when the cursor isn't over any point. */
    onPointHover?: ((index: number | null) => void) | null;
    /** Approximate maximum number of points to render at once. When the dataset (after
     *  cone-of-view culling) exceeds this, points are downsampled - see `downsample_3d.ts`.
     *  null/Infinity disables downsampling. */
    downsampleMaxPoints?: number | null;
    /** Whether a range-selection/cross-filter is currently active elsewhere in the app. 3D has
     *  no persistent brush outline to draw for it (unlike the 2D lasso/marquee overlay), so this
     *  drives a status-bar badge - an active filter should never be silently invisible. */
    activeFilterHidden?: boolean;
    /** Set to false to hide the built-in status bar (camera reset, point count, filter badge). */
    showStatusBar?: boolean;
  }

  let {
    data,
    categoryColors = null,
    width = 800,
    height = 800,
    pointSize = null,
    selectedIndices = null,
    onPointClick = null,
    onPointHover = null,
    downsampleMaxPoints = 500000,
    activeFilterHidden = false,
    showStatusBar = true,
  }: Props = $props();

  let canvas: HTMLCanvasElement | null = $state(null);
  let scene: THREE.Scene | null = null;
  let camera: THREE.PerspectiveCamera | null = null;
  let renderer: THREE.WebGLRenderer | null = null;
  let controls: OrbitControls | null = null;
  let pointCloud: THREE.Points | null = null;
  let pointGeometry: THREE.BufferGeometry | null = null;
  let highlightPoints: THREE.Points | null = null;
  let animationId: number | null = null;
  let raycaster = new THREE.Raycaster();
  let pointerDownPos: { x: number; y: number } | null = null;
  let currentPointSize = 1;
  let renderedCount = $state.raw(0);
  let totalCount = $state.raw(0);

  // Full (non-downsampled) position/color buffers for the current `data`, rebuilt only when the
  // data itself changes - downsampling then just swaps a lightweight index buffer, so panning to
  // trigger a re-downsample doesn't require re-uploading the whole point cloud to the GPU.
  let fullPositions: Float32Array = new Float32Array();
  let fullColors: Float32Array = new Float32Array();

  // The camera fit computed the first time non-empty data arrives, used by the "reset camera"
  // control. Deliberately captured only once (not re-applied on every data/filter change), so
  // that e.g. narrowing an active filter doesn't yank the user's current view back to a fit.
  let defaultCameraState: { target: THREE.Vector3; position: THREE.Vector3 } | null = null;
  let didInitialFit = false;

  // Idle-scheduled, cancellable downsampling (see downsample_3d.ts). `downsampleGeneration` is
  // bumped whenever the camera moves or the data changes, so any in-flight idle-scheduled
  // computation notices it's stale and abandons itself instead of racing a fresher one.
  let downsampleGeneration = 0;
  let downsampleIndices: Uint32Array | null = null;
  let lastHoverTime = 0;
  // Built-in hover readout (independent of `onPointHover`, which callers use for their own
  // richer tooltips): a minimal, self-contained display of the hovered point's raw coordinates,
  // since a screen-projected label anchored to a moving 3D point needs no external data.
  let hoverIndex: number | null = $state.raw(null);
  let hoverScreenPos: { x: number; y: number } | null = $state.raw(null);

  function initScene() {
    if (!canvas) return;
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 1e6);
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // Cone-of-view downsampling only needs to re-run when the camera actually finishes moving
    // (dragging every intermediate frame would be wasteful); a cheap generation bump on "start"
    // cancels any stale idle-scheduled work immediately so it doesn't finish and stomp a newer
    // request once the user starts moving again.
    controls.addEventListener("start", () => {
      downsampleGeneration++;
    });
    controls.addEventListener("end", () => {
      scheduleDownsample();
    });

    renderer.domElement.addEventListener("pointerdown", (e) => {
      pointerDownPos = { x: e.clientX, y: e.clientY };
    });
    renderer.domElement.addEventListener("pointerup", (e) => {
      if (pointerDownPos == null) return;
      let dx = e.clientX - pointerDownPos.x;
      let dy = e.clientY - pointerDownPos.y;
      pointerDownPos = null;
      // Only treat this as a "click" (point selection) if the pointer barely moved between
      // down/up - otherwise it's an orbit/pan drag that OrbitControls is already handling.
      if (Math.hypot(dx, dy) > 4) return;
      handlePointerSelect(e, onPointClick);
    });
    renderer.domElement.addEventListener("pointermove", (e) => {
      let now = performance.now();
      if (now - lastHoverTime < 60) return;
      lastHoverTime = now;
      let index = pickIndexAt(e);
      hoverIndex = index;
      hoverScreenPos = index != null ? { x: e.clientX, y: e.clientY } : null;
      onPointHover?.(index);
    });
    renderer.domElement.addEventListener("pointerleave", () => {
      hoverIndex = null;
      hoverScreenPos = null;
      onPointHover?.(null);
    });
  }

  /** Resolve the row index (into `data`) of the point under the given pointer event, accounting
   *  for the current downsampling index buffer, or null if no point is under the cursor. */
  function pickIndexAt(e: PointerEvent): number | null {
    if (!canvas || !camera || !pointCloud) return null;
    let rect = canvas.getBoundingClientRect();
    let ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.params.Points!.threshold = Math.max(currentPointSize * 1.5, 1e-4);
    raycaster.setFromCamera(ndc, camera);
    let intersections = raycaster.intersectObject(pointCloud, false);
    if (intersections.length === 0) return null;
    // `intersections[0].index` is the index into the geometry's (possibly downsampled) index
    // buffer, which is itself a list of row indices into `data` - resolve back to the row index.
    let hitIndex = intersections[0].index ?? null;
    if (hitIndex == null) return null;
    return downsampleIndices != null ? (downsampleIndices[hitIndex] ?? null) : hitIndex;
  }

  function handlePointerSelect(e: PointerEvent, callback: ((index: number | null) => void) | null) {
    if (callback == null) return;
    callback(pickIndexAt(e));
  }

  /** Rebuild the full (non-downsampled) position/color buffers from `data`. */
  function rebuildFullBuffers() {
    let count = data.x.length;
    totalCount = count;
    fullPositions = new Float32Array(count * 3);
    fullColors = new Float32Array(count * 3);
    let categoryCount = 1;
    if (data.category != null) {
      for (let i = 0; i < data.category.length; i++) {
        categoryCount = Math.max(categoryCount, data.category[i] + 1);
      }
    }
    let palette = (categoryColors ?? defaultCategoryColors(categoryCount)).map((c) => new THREE.Color(c));
    if (palette.length === 0) {
      palette = [new THREE.Color("#4c78a8")];
    }
    for (let i = 0; i < count; i++) {
      fullPositions[i * 3] = data.x[i];
      fullPositions[i * 3 + 1] = data.y[i];
      fullPositions[i * 3 + 2] = data.z[i];
      let categoryIndex = data.category != null ? (data.category[i] ?? 0) : 0;
      let c = palette[categoryIndex % palette.length];
      fullColors[i * 3] = c.r;
      fullColors[i * 3 + 1] = c.g;
      fullColors[i * 3 + 2] = c.b;
    }
  }

  /** (Re)build the point cloud geometry/material and, on first load only, frame the camera on
   *  the data's bounding sphere. */
  function updatePointCloud() {
    if (!scene || !camera || !controls) return;
    rebuildFullBuffers();

    if (pointCloud) {
      scene.remove(pointCloud);
      pointCloud.geometry.dispose();
      (pointCloud.material as THREE.Material).dispose();
      pointCloud = null;
    }

    let geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(fullPositions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(fullColors, 3));
    geometry.computeBoundingSphere();
    let sphere = geometry.boundingSphere;
    let radius = sphere && sphere.radius > 0 ? sphere.radius : 1;

    // `pointSize` is a world-space size (same units as the data), not a screen-pixel size, so it
    // must scale with the data's extent: a fixed pixel-ish default would render as either
    // invisible specks or giant overlapping squares depending on the data's own coordinate range.
    let size = pointSize ?? radius * 0.02;
    currentPointSize = size;

    let material = new THREE.PointsMaterial({
      size: size,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
    });
    pointGeometry = geometry;
    pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);
    renderedCount = data.x.length;
    updateHighlight();
    downsampleGeneration++;
    downsampleIndices = null;

    if (data.x.length > 0 && sphere != null) {
      let distance = radius / Math.sin((camera.fov * Math.PI) / 360);
      let target = sphere.center.clone();
      let position = new THREE.Vector3(sphere.center.x, sphere.center.y, sphere.center.z + distance * 1.4);
      camera.near = Math.max(distance / 1000, 0.001);
      camera.far = distance * 1000;
      camera.updateProjectionMatrix();

      if (!didInitialFit) {
        // Only snap the camera to the data's bounds the first time this view sees data - once
        // the user has oriented themselves, subsequent data updates (e.g. a filter narrowing the
        // point set) shouldn't yank the camera around; use the reset button for that.
        didInitialFit = true;
        controls.target.copy(target);
        camera.position.copy(position);
        controls.update();
      }
      defaultCameraState = { target, position };
    }

    scheduleDownsample();
  }

  /** Idle-scheduled, chunked, cancellable cone-of-view downsampling. Cheap no-op when the point
   *  count is already within `downsampleMaxPoints` or downsampling is disabled. */
  function scheduleDownsample() {
    if (!camera || !pointGeometry) return;
    let count = data.x.length;
    let limit = downsampleMaxPoints;
    let myGeneration = ++downsampleGeneration;

    if (limit == null || !Number.isFinite(limit) || count <= limit) {
      if (downsampleIndices != null) {
        downsampleIndices = null;
        pointGeometry.setIndex(null);
        pointGeometry.setDrawRange(0, count);
        renderedCount = count;
      }
      return;
    }

    camera.updateMatrixWorld();
    let frustumMatrix = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    let frustum = new THREE.Frustum().setFromProjectionMatrix(frustumMatrix);

    pickVisibleIndices({
      positions: fullPositions,
      count,
      frustum,
      maxPoints: limit,
      isStale: () => myGeneration !== downsampleGeneration,
      onComplete: (indices) => {
        if (myGeneration !== downsampleGeneration || pointGeometry == null) return;
        downsampleIndices = indices;
        pointGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
        pointGeometry.setDrawRange(0, indices.length);
        renderedCount = indices.length;
        updateHighlight();
      },
    });
  }

  // Draw an overlay of enlarged, bright points at `selectedIndices` on top of the main cloud. */
  function updateHighlight() {
    if (!scene) return;
    if (highlightPoints) {
      scene.remove(highlightPoints);
      highlightPoints.geometry.dispose();
      (highlightPoints.material as THREE.Material).dispose();
      highlightPoints = null;
    }
    if (selectedIndices == null || selectedIndices.length === 0) return;

    let positions = new Float32Array(selectedIndices.length * 3);
    for (let i = 0; i < selectedIndices.length; i++) {
      let idx = selectedIndices[i];
      positions[i * 3] = data.x[idx];
      positions[i * 3 + 1] = data.y[idx];
      positions[i * 3 + 2] = data.z[idx];
    }
    let geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    let material = new THREE.PointsMaterial({
      size: currentPointSize * 2.5,
      color: 0xff3b30,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.95,
      depthTest: false,
    });
    highlightPoints = new THREE.Points(geometry, material);
    highlightPoints.renderOrder = 999;
    scene.add(highlightPoints);

    if (selectedIndices.length === 1) {
      focusOnPoint(selectedIndices[0]);
    }
  }

  // --- Camera animation (reset / focus) ---------------------------------------------------

  let cameraAnimationId: number | null = null;

  function animateCameraTo(target: THREE.Vector3, position: THREE.Vector3, duration = 700) {
    if (!camera || !controls) return;
    if (cameraAnimationId != null) cancelAnimationFrame(cameraAnimationId);
    let startTarget = controls.target.clone();
    let startPosition = camera.position.clone();
    let t0 = performance.now();
    let step = () => {
      let t = Math.min(1, (performance.now() - t0) / duration);
      let eased = 1 - Math.pow(1 - t, 3); // cubic ease-out
      controls!.target.lerpVectors(startTarget, target, eased);
      camera!.position.lerpVectors(startPosition, position, eased);
      controls!.update();
      if (t < 1) {
        cameraAnimationId = requestAnimationFrame(step);
      } else {
        cameraAnimationId = null;
      }
    };
    cameraAnimationId = requestAnimationFrame(step);
  }

  /** Reset the camera to the default fit computed when this dataset first loaded. */
  export function resetCamera() {
    if (defaultCameraState == null) return;
    animateCameraTo(defaultCameraState.target, defaultCameraState.position);
  }

  /** Smoothly re-center the view on a point without changing the current viewing angle/distance
   *  (translates both the orbit target and the camera position by the same delta). */
  function focusOnPoint(index: number) {
    if (!camera || !controls) return;
    let newTarget = new THREE.Vector3(data.x[index], data.y[index], data.z[index]);
    let delta = newTarget.clone().sub(controls.target);
    if (delta.lengthSq() < 1e-10) return; // already centered here
    let newPosition = camera.position.clone().add(delta);
    animateCameraTo(newTarget, newPosition, 500);
  }

  function resize() {
    if (!renderer || !camera) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function animate() {
    animationId = requestAnimationFrame(animate);
    controls?.update();
    if (renderer && scene && camera) {
      renderer.render(scene, camera);
    }
  }

  onMount(() => {
    initScene();
    updatePointCloud();
    resize();
    animate();
  });

  onDestroy(() => {
    if (animationId != null) {
      cancelAnimationFrame(animationId);
    }
    if (cameraAnimationId != null) {
      cancelAnimationFrame(cameraAnimationId);
    }
    downsampleGeneration++; // cancel any in-flight idle-scheduled downsample work
    controls?.dispose();
    if (pointCloud) {
      pointCloud.geometry.dispose();
      (pointCloud.material as THREE.Material).dispose();
    }
    if (highlightPoints) {
      highlightPoints.geometry.dispose();
      (highlightPoints.material as THREE.Material).dispose();
    }
    renderer?.dispose();
  });

  $effect(() => {
    // Re-track dependencies explicitly so Svelte 5's effect reruns when the data arrays or
    // category colors change (e.g. star data arriving asynchronously after mount).
    data.x;
    data.y;
    data.z;
    data.category;
    categoryColors;
    if (scene) {
      updatePointCloud();
    }
  });

  $effect(() => {
    selectedIndices;
    if (scene) {
      updateHighlight();
    }
  });

  $effect(() => {
    width;
    height;
    resize();
  });
</script>

<div style:position="relative" style:width="{width}px" style:height="{height}px">
  <canvas bind:this={canvas} style:display="block" style:width="{width}px" style:height="{height}px"></canvas>
  {#if showStatusBar}
    <StatusBar3D
      pointCount={renderedCount}
      totalCount={totalCount}
      activeFilterHidden={activeFilterHidden}
      onResetCamera={resetCamera}
    />
  {/if}
  {#if hoverIndex != null && hoverScreenPos != null}
    <div
      style:position="fixed"
      style:left="{hoverScreenPos.x + 12}px"
      style:top="{hoverScreenPos.y + 12}px"
      style:pointer-events="none"
      style:background="rgba(0,0,0,0.75)"
      style:color="white"
      style:font-size="11px"
      style:line-height="16px"
      style:padding="4px 6px"
      style:border-radius="3px"
      style:white-space="nowrap"
      style:z-index="10"
    >
      x: {data.x[hoverIndex]?.toFixed(3)}, y: {data.y[hoverIndex]?.toFixed(3)}, z: {data.z[hoverIndex]?.toFixed(3)}
      {#if data.category != null}
        &middot; category {data.category[hoverIndex]}
      {/if}
    </div>
  {/if}
</div>
