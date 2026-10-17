// The Moon as a globe made of its own pictures, and the morph from that globe into embedding space.
// Only for a bundle whose meta.json has a "globe" block (the separate Moon dataset). Each item is one
// web-mercator tile, hung flat on the sphere: centred on the ray from the Moon's centre through the
// middle of its patch and square to that ray, so the pictures together make a faceted ball.
// One uniform, uT, carries every picture from its place on the globe (0) to its embedding spot (1).
import * as THREE from 'three';
import { tween, ease } from './anim.js';

const WORLD = 10;           // must match cloud.js: the globe is the unit sphere in layout units
const GAP = 0.975;          // a picture covers this share of its patch: a hairline between neighbours
const TILE_END = 0.22;      // patch size (world units) as it arrives in the cloud, before it fades to its dot
const SPREAD = 0.9;         // stagger: share of the morph by which the last patch starts after the first
const LIFT = 0.28;          // patches rise off the surface (x radius) mid-flight
const MORPH_MS = 3200;
const POLE_DEG = 85.0511;   // web mercator's edge: beyond this the map has no data (a small opening at each pole)

const VERT = /* glsl */`
  #define PI 3.141592653589793
  attribute vec2 aTile;
  attribute vec3 aEmb;
  attribute float aSeed;
  uniform float uT, uN, uR;
  uniform vec3 uRight, uUp;
  varying vec2 vUv;
  varying float vK;
  varying float vShade;
  float mercLat(float gy) { float s = PI * (1.0 - 2.0 * gy); return atan(0.5 * (exp(s) - exp(-s))); }
  vec3 onGlobe(float gx, float gy) {
    float lon = (gx * 2.0 - 1.0) * PI, lat = mercLat(gy);
    return vec3(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));   // y up, longitude 0 toward +z
  }
  void main() {
    float u = uv.x, v = 1.0 - uv.y;                       // v = 0 at the picture's north edge
    float lonc = ((aTile.x + 0.5) / uN * 2.0 - 1.0) * PI, latc = mercLat((aTile.y + 0.5) / uN);
    vec3 nc = onGlobe((aTile.x + 0.5) / uN, (aTile.y + 0.5) / uN);   // the ray through the picture's centre
    vec3 east = vec3(cos(lonc), 0.0, -sin(lonc));
    vec3 north = vec3(-sin(latc) * sin(lonc), cos(latc), -sin(latc) * cos(lonc));
    // the patch's size on the sphere (mercator is conformal, so it is near square, like the picture)
    float w = uR * (2.0 * PI / uN) * cos(latc);
    float h = uR * (mercLat(aTile.y / uN) - mercLat((aTile.y + 1.0) / uN));
    float k = clamp(uT * (1.0 + ${SPREAD.toFixed(2)}) - aSeed * ${SPREAD.toFixed(2)}, 0.0, 1.0);
    float e = k < 0.5 ? 4.0 * k * k * k : 1.0 - pow(-2.0 * k + 2.0, 3.0) / 2.0;
    vec3 pg = uR * nc + (east * (u - 0.5) * w + north * (0.5 - v) * h) * ${GAP.toFixed(2)};
    vec3 pe = aEmb + (uRight * (u - 0.5) + uUp * (0.5 - v)) * ${TILE_END.toFixed(2)};
    vec3 p = mix(pg, pe, e) + nc * (sin(PI * e) * uR * ${LIFT.toFixed(2)});
    vK = e;
    vUv = vec2((aTile.x + u) / uN, 1.0 - (aTile.y + v) / uN);
    // each facet is lit by how squarely it faces the viewer, so the ball reads as round
    vShade = 0.6 + 0.4 * max(dot(nc, normalize(cameraPosition - pg)), 0.0);
    gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
  }`;

const FRAG = /* glsl */`
  uniform sampler2D uAtlas;
  varying vec2 vUv;
  varying float vK;
  varying float vShade;
  void main() {
    float a = 1.0 - smoothstep(0.82, 1.0, vK);
    if (a < 0.01) discard;
    vec3 c = texture2D(uAtlas, vUv).rgb * mix(vShade, 1.0, vK);
    gl_FragColor = vec4(c, a);
  }`;

function globeXYZ(lonDeg, latDeg) {
  const lon = THREE.MathUtils.degToRad(lonDeg), lat = THREE.MathUtils.degToRad(latDeg);
  return new THREE.Vector3(Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon));
}

export class Globe {
  // cloud: the Cloud; info: meta.globe {z, cell, atlas}
  constructor(cloud, info) {
    this.cloud = cloud;
    this.n = 2 ** info.z;
    this.t = 0;                    // 0 = globe, 1 = embedding cloud
    this.busy = false;
    const items = cloud.items, count = items.length;
    this.indexAt = new Map();      // "x,y" -> item index
    const tile = new Float32Array(count * 2), emb = new Float32Array(count * 3);
    this.seed = new Float32Array(count);
    this.centres = [];
    items.forEach((it, i) => {
      const [x, y] = it.tile;
      this.indexAt.set(`${x},${y}`, i);
      tile[i * 2] = x; tile[i * 2 + 1] = y;
      emb.set(cloud.positions.subarray(i * 3, i * 3 + 3), i * 3);
      const lon = (x + 0.5) / this.n * 360 - 180;
      const lat = THREE.MathUtils.radToDeg(Math.atan(Math.sinh(Math.PI * (1 - 2 * (y + 0.5) / this.n))));
      this.centres.push(globeXYZ(lon, lat));
    });

    const base = new THREE.PlaneGeometry(1, 1, 1, 1);
    const geo = new THREE.InstancedBufferGeometry();
    geo.index = base.index;
    geo.setAttribute('position', base.attributes.position);
    geo.setAttribute('uv', base.attributes.uv);
    geo.setAttribute('aTile', new THREE.InstancedBufferAttribute(tile, 2));
    geo.setAttribute('aEmb', new THREE.InstancedBufferAttribute(emb, 3));
    this.seedAttr = new THREE.InstancedBufferAttribute(this.seed, 1);
    geo.setAttribute('aSeed', this.seedAttr);
    geo.instanceCount = count;

    const atlas = new THREE.TextureLoader().load(info.atlas, () => { this.loaded = true; });
    atlas.colorSpace = THREE.NoColorSpace;   // drawn as stored, like the dots' colours
    atlas.anisotropy = 8;
    this.uniforms = {
      uT: { value: 0 }, uN: { value: this.n }, uR: { value: WORLD },
      uRight: { value: new THREE.Vector3(1, 0, 0) }, uUp: { value: new THREE.Vector3(0, 1, 0) },
      uAtlas: { value: atlas },
    };
    this.mesh = new THREE.Mesh(geo, new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uniforms,
      side: THREE.FrontSide, transparent: true, depthWrite: true,   // see apply()
    }));
    this.mesh.frustumCulled = false;   // instances move far from the base plane's bounds
    this.mesh.renderOrder = 1;

    cloud.scene.add(this.mesh);
    cloud.globe = this;
    this.apply(0);
  }

  get active() { return this.t < 1; }        // anything of the globe on screen
  get whole() { return this.t === 0 && !this.busy; }

  // Camera pose that shows the whole Moon left of centre-ish, the same viewing direction as now.
  pose() {
    const c = this.cloud;
    const dir = c.camera.position.clone().sub(c.controls.target).normalize();
    const tanV = Math.tan(THREE.MathUtils.degToRad(c.camera.fov) / 2);
    const dist = WORLD / Math.sin(Math.atan(tanV * 0.78));
    return { target: new THREE.Vector3(), position: dir.multiplyScalar(dist) };
  }

  apply(t) {
    this.t = t;
    this.uniforms.uT.value = t;
    this.mesh.visible = t < 1;
    // whole globe: front faces only, so the hairlines show the wall, not the far side's backs;
    // in flight both, as the pictures turn edge-on on their way to face the viewer
    this.mesh.material.side = t === 0 ? THREE.FrontSide : THREE.DoubleSide;
    // the dots arrive as the patches land
    this.cloud.uniforms.uAlpha.value = THREE.MathUtils.smoothstep(t, 0.55, 1.0);
  }

  // Each frame: keep the arriving patches facing the camera.
  step() {
    if (!this.mesh.visible) return;
    const m = this.cloud.camera.matrixWorld.elements;
    this.uniforms.uRight.value.set(m[0], m[1], m[2]);
    this.uniforms.uUp.value.set(m[4], m[5], m[6]);
  }

  // Patches nearest the viewer leave first, with a little jitter so it reads as breaking up, not peeling.
  restagger() {
    const view = this.cloud.camera.position.clone().normalize();
    let s = 1234567;
    const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    this.centres.forEach((c, i) => { this.seed[i] = 0.7 * (1 - c.dot(view)) / 2 + 0.3 * rand(); });
    this.seedAttr.needsUpdate = true;
  }

  async morph(to, run = 0) {
    if (this.busy || this.t === to) return this.t === to;
    const c = this.cloud;
    this.busy = true;
    if (to === 1) this.restagger();
    const from = this.t;
    const p0 = c.camera.position.clone(), q0 = c.controls.target.clone(), s0 = c.shift;
    c.shift = 0;   // both poses are framed without the wall panel's offset
    const goal = to === 1 ? c.idlePose() : this.pose();
    c.shift = s0;
    c.flying = true;
    c.controls.enabled = false;
    c.controls.autoRotate = false;
    c.mode = 'globe';
    const done = await tween(MORPH_MS, (e, raw) => {
      this.apply(from + (to - from) * raw);
      c.camera.position.lerpVectors(p0, goal.position, e);
      c.controls.target.lerpVectors(q0, goal.target, e);
      c.shift = s0 * (1 - e);
      c.applyShift();
    }, { run, easing: ease.inOutCubic });
    this.apply(done ? to : this.t);
    c.settleControls();
    c.flying = false;
    c.controls.enabled = true;
    c.lastInteract = performance.now();
    c.mode = this.t >= 1 ? 'idle' : 'globe';
    this.busy = false;
    return done;
  }

  // Start on the globe, framed.
  showWhole() {
    const c = this.cloud, g = this.pose();
    c.camera.position.copy(g.position);
    c.controls.target.copy(g.target);
    c.mode = 'globe';
    this.apply(0);
  }

  // The tile under a screen point (whole globe only): ray against the sphere, then lat/lon -> tile.
  pick(x, y) {
    if (!this.whole) return null;
    const c = this.cloud;
    const ndc = new THREE.Vector2(x / window.innerWidth * 2 - 1, 1 - y / window.innerHeight * 2);
    c.raycaster.setFromCamera(ndc, c.camera);
    const hit = c.raycaster.ray.intersectSphere(new THREE.Sphere(new THREE.Vector3(), WORLD), new THREE.Vector3());
    if (!hit) return null;
    const p = hit.divideScalar(WORLD);
    const lat = Math.asin(THREE.MathUtils.clamp(p.y, -1, 1));
    if (Math.abs(THREE.MathUtils.radToDeg(lat)) > POLE_DEG) return null;
    const lon = Math.atan2(p.x, p.z);
    const tx = Math.min(this.n - 1, Math.floor((lon / Math.PI + 1) / 2 * this.n));
    const ty = Math.min(this.n - 1, Math.floor((1 - Math.log(Math.tan(Math.PI / 4 + lat / 2)) / Math.PI) / 2 * this.n));
    return this.indexAt.get(`${tx},${ty}`) ?? null;
  }
}
