import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { clone as cloneSkeleton } from 'three/addons/utils/SkeletonUtils.js'

const loader = new GLTFLoader()
const templateCache = new Map()

const ASSET_LIBRARY = {
  ego: {
    key: 'ego',
    urls: ['/models/sport_car.glb'],
    yawOffset: Math.PI,
    forceColor: 0xffffff,
  },
  vehicle: {
    key: 'vehicle',
    urls: ['/models/other_car.glb'],
    yawOffset: Math.PI,
  },
  cone: {
    key: 'cone',
    urls: ['/models/traffic_cone.glb'],
    yawOffset: 0,
  },
  barrier: {
    key: 'barrier',
    urls: ['/models/traffic_barrier.glb', '/models/concrete_barrier.glb'],
    yawOffset: 0,
  },
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value))
}

function sleep(milliseconds) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds)
  })
}

function setShadowProps(root) {
  root.traverse((node) => {
    node.frustumCulled = false
    if (!node.isMesh) {
      return
    }
    node.castShadow = true
    node.receiveShadow = true
    node.material = Array.isArray(node.material)
      ? node.material.map((material) => material.clone())
      : node.material?.clone?.() || node.material
  })
  return root
}

function collectMaterials(root) {
  const materials = []
  root.traverse((node) => {
    if (!node.material) {
      return
    }
    const list = Array.isArray(node.material) ? node.material : [node.material]
    list.forEach((material) => {
      if (material && !materials.includes(material)) {
        materials.push(material)
      }
    })
  })
  return materials
}

function applyColorOverride(root, colorHex) {
  root.traverse((node) => {
    if (!node.material) {
      return
    }
    const materials = Array.isArray(node.material) ? node.material : [node.material]
    materials.forEach((material) => {
      if (material.color) {
        material.color.setHex(colorHex)
      }
      if (material.emissive) {
        material.emissive.setHex(0x030507)
      }
      if (typeof material.metalness === 'number') {
        material.metalness = Math.max(material.metalness, 0.42)
      }
      if (typeof material.roughness === 'number') {
        material.roughness = clamp(material.roughness, 0.18, 0.58)
      }
      material.needsUpdate = true
    })
  })
}

function selectPrimarySubtree(scene) {
  if (!scene) {
    return new THREE.Group()
  }

  const rootChildren = scene.children.filter((child) => child.visible !== false)
  if (rootChildren.length <= 1) {
    return scene
  }

  let largestChild = null
  let largestVolume = -1

  rootChildren.forEach((child) => {
    const box = new THREE.Box3().setFromObject(child)
    const size = box.getSize(new THREE.Vector3())
    const volume = size.x * size.y * size.z
    if (volume > largestVolume) {
      largestVolume = volume
      largestChild = child
    }
  })

  if (!largestChild) {
    return scene
  }

  const wrapper = new THREE.Group()
  wrapper.add(largestChild.clone())
  return wrapper
}

async function loadUrlWithRetry(url, attempts = 3) {
  let lastError = null
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const gltf = await loader.loadAsync(url)
      const rawScene = gltf.scene || gltf.scenes?.[0]
      return selectPrimarySubtree(rawScene)
    } catch (error) {
      lastError = error
      if (attempt < attempts) {
        await sleep(180 * attempt)
      }
    }
  }
  throw lastError
}

async function loadTemplate(assetConfig) {
  for (const url of assetConfig.urls) {
    try {
      const scene = await loadUrlWithRetry(url, 3)
      return setShadowProps(scene)
    } catch {
      continue
    }
  }
  throw new Error(`Unable to load model asset: ${assetConfig.key}`)
}

function getAssetConfig(className, role = 'object') {
  if (role === 'ego') {
    return ASSET_LIBRARY.ego
  }
  if (['car', 'truck', 'bus', 'trailer', 'construction_vehicle'].includes(className)) {
    return ASSET_LIBRARY.vehicle
  }
  if (className === 'traffic_cone') {
    return ASSET_LIBRARY.cone
  }
  if (['barrier', 'traffic_barrier', 'concrete_barrier'].includes(className)) {
    return ASSET_LIBRARY.barrier
  }
  return null
}

function getTargetSize(box3d = {}) {
  const size = box3d.size || {}
  return {
    width: Math.max(0.2, Number(size.width) || 1),
    height: Math.max(0.2, Number(size.height) || 1),
    length: Math.max(0.2, Number(size.length) || 1),
  }
}

function fitObjectToPhysicalBox(root, box3d, yawOffset = 0) {
  root.rotation.set(0, yawOffset, 0)
  root.position.set(0, 0, 0)
  root.scale.set(1, 1, 1)
  root.updateMatrixWorld(true)

  const baseBox = new THREE.Box3().setFromObject(root)
  const baseSize = baseBox.getSize(new THREE.Vector3())
  const target = getTargetSize(box3d)
  const uniformScale = Math.min(
    target.width / Math.max(baseSize.x, 1e-3),
    target.height / Math.max(baseSize.y, 1e-3),
    target.length / Math.max(baseSize.z, 1e-3),
  )

  root.scale.setScalar(uniformScale)
  root.updateMatrixWorld(true)

  const fittedBox = new THREE.Box3().setFromObject(root)
  const center = fittedBox.getCenter(new THREE.Vector3())
  root.position.x -= center.x
  root.position.z -= center.z
  root.position.y -= fittedBox.min.y
  root.updateMatrixWorld(true)
  return root
}

function makeMaterial(options) {
  return new THREE.MeshStandardMaterial({
    color: options.color,
    metalness: options.metalness ?? 0.3,
    roughness: options.roughness ?? 0.5,
    emissive: options.emissive ?? 0x000000,
    emissiveIntensity: options.emissiveIntensity ?? 0,
  })
}

function setCommonMeshProps(mesh) {
  mesh.castShadow = true
  mesh.receiveShadow = true
  return mesh
}

function createVehicleFallback(box3d, isEgo = false) {
  const group = new THREE.Group()
  const size = getTargetSize(box3d)
  const width = size.width
  const height = size.height
  const length = size.length

  const bodyMaterial = makeMaterial({
    color: isEgo ? 0xffffff : 0xcfd4dc,
    metalness: 0.64,
    roughness: 0.22,
    emissive: isEgo ? 0x091521 : 0x020304,
    emissiveIntensity: isEgo ? 0.12 : 0.03,
  })
  const glassMaterial = makeMaterial({
    color: 0x9caabd,
    metalness: 0.14,
    roughness: 0.08,
  })
  glassMaterial.transparent = true
  glassMaterial.opacity = 0.84

  const lowerBody = setCommonMeshProps(new THREE.Mesh(
    new THREE.BoxGeometry(width * 0.96, height * 0.3, length * 0.88),
    bodyMaterial,
  ))
  lowerBody.position.y = height * 0.22
  group.add(lowerBody)

  const shoulder = setCommonMeshProps(new THREE.Mesh(
    new THREE.BoxGeometry(width * 0.9, height * 0.18, length * 0.7),
    bodyMaterial,
  ))
  shoulder.position.set(0, height * 0.38, -length * 0.02)
  group.add(shoulder)

  const cabin = setCommonMeshProps(new THREE.Mesh(
    new THREE.BoxGeometry(width * 0.68, height * 0.24, length * 0.4),
    glassMaterial,
  ))
  cabin.position.set(0, height * 0.58, -length * 0.02)
  group.add(cabin)

  const hood = setCommonMeshProps(new THREE.Mesh(
    new THREE.BoxGeometry(width * 0.78, height * 0.08, length * 0.22),
    bodyMaterial,
  ))
  hood.position.set(0, height * 0.46, -length * 0.3)
  hood.rotation.x = -0.14
  group.add(hood)

  const rearDeck = setCommonMeshProps(new THREE.Mesh(
    new THREE.BoxGeometry(width * 0.74, height * 0.08, length * 0.18),
    bodyMaterial,
  ))
  rearDeck.position.set(0, height * 0.47, length * 0.24)
  rearDeck.rotation.x = 0.1
  group.add(rearDeck)

  const wheelMaterial = makeMaterial({ color: 0x111317, roughness: 0.9, metalness: 0.06 })
  const rimMaterial = makeMaterial({ color: 0xb7bec8, roughness: 0.36, metalness: 0.72 })
  const wheelRadius = clamp(Math.min(width * 0.16, height * 0.22), 0.18, 0.46)
  const wheelThickness = clamp(width * 0.12, 0.12, 0.26)
  const wheelGeometry = new THREE.CylinderGeometry(wheelRadius, wheelRadius, wheelThickness, 22)
  wheelGeometry.rotateZ(Math.PI / 2)
  const rimGeometry = new THREE.CylinderGeometry(wheelRadius * 0.58, wheelRadius * 0.58, wheelThickness * 1.04, 18)
  rimGeometry.rotateZ(Math.PI / 2)
  const wheelOffsets = [
    [-width * 0.42, wheelRadius, -length * 0.28],
    [width * 0.42, wheelRadius, -length * 0.28],
    [-width * 0.42, wheelRadius, length * 0.24],
    [width * 0.42, wheelRadius, length * 0.24],
  ]
  wheelOffsets.forEach(([x, y, z]) => {
    const tire = setCommonMeshProps(new THREE.Mesh(wheelGeometry, wheelMaterial))
    tire.position.set(x, y, z)
    group.add(tire)

    const rim = setCommonMeshProps(new THREE.Mesh(rimGeometry, rimMaterial))
    rim.position.set(x, y, z)
    group.add(rim)
  })

  setShadowProps(group)
  return group
}

function createPedestrianFallback(box3d) {
  const group = new THREE.Group()
  const size = getTargetSize(box3d)
  const height = clamp(size.height, 1.45, 2.2)
  const shoulderWidth = clamp(size.width * 0.55, 0.28, 0.56)
  const hipWidth = shoulderWidth * 0.75
  const headRadius = clamp(height * 0.11, 0.12, 0.2)

  const skinMaterial = makeMaterial({ color: 0xdcbba6, roughness: 0.78, metalness: 0.02 })
  const coatMaterial = makeMaterial({
    color: 0x6c87a8,
    roughness: 0.72,
    metalness: 0.08,
    emissive: 0x111a25,
    emissiveIntensity: 0.04,
  })
  const pantsMaterial = makeMaterial({ color: 0x303943, roughness: 0.84, metalness: 0.04 })
  const shoeMaterial = makeMaterial({ color: 0x171b20, roughness: 0.9, metalness: 0.04 })

  const head = setCommonMeshProps(new THREE.Mesh(new THREE.SphereGeometry(headRadius, 20, 18), skinMaterial))
  head.position.y = height * 0.9
  group.add(head)

  const torso = setCommonMeshProps(new THREE.Mesh(
    new THREE.CapsuleGeometry(shoulderWidth * 0.34, height * 0.28, 8, 16),
    coatMaterial,
  ))
  torso.position.y = height * 0.58
  group.add(torso)

  const pelvis = setCommonMeshProps(new THREE.Mesh(
    new THREE.BoxGeometry(hipWidth, height * 0.09, size.length * 0.18),
    coatMaterial,
  ))
  pelvis.position.y = height * 0.37
  group.add(pelvis)

  const armGeometry = new THREE.CapsuleGeometry(0.05, height * 0.2, 6, 12)
  const leftArm = setCommonMeshProps(new THREE.Mesh(armGeometry, coatMaterial))
  leftArm.position.set(-shoulderWidth * 0.55, height * 0.6, 0)
  leftArm.rotation.z = 0.22
  group.add(leftArm)
  const rightArm = setCommonMeshProps(new THREE.Mesh(armGeometry, coatMaterial))
  rightArm.position.set(shoulderWidth * 0.55, height * 0.6, 0)
  rightArm.rotation.z = -0.22
  group.add(rightArm)

  const legGeometry = new THREE.CapsuleGeometry(0.055, height * 0.24, 6, 12)
  const leftLeg = setCommonMeshProps(new THREE.Mesh(legGeometry, pantsMaterial))
  leftLeg.position.set(-hipWidth * 0.2, height * 0.16, 0)
  leftLeg.rotation.z = 0.04
  group.add(leftLeg)
  const rightLeg = setCommonMeshProps(new THREE.Mesh(legGeometry, pantsMaterial))
  rightLeg.position.set(hipWidth * 0.2, height * 0.16, 0)
  rightLeg.rotation.z = -0.04
  group.add(rightLeg)

  const footGeometry = new THREE.BoxGeometry(0.09, 0.05, 0.2)
  const leftFoot = setCommonMeshProps(new THREE.Mesh(footGeometry, shoeMaterial))
  leftFoot.position.set(-hipWidth * 0.2, 0.025, 0.03)
  group.add(leftFoot)
  const rightFoot = setCommonMeshProps(new THREE.Mesh(footGeometry, shoeMaterial))
  rightFoot.position.set(hipWidth * 0.2, 0.025, 0.03)
  group.add(rightFoot)

  setShadowProps(group)
  return group
}

function createTwoWheelerFallback(box3d) {
  const group = new THREE.Group()
  const size = getTargetSize(box3d)
  const wheelRadius = clamp(size.height * 0.22, 0.18, 0.42)
  const wheelGap = clamp(size.length * 0.42, 0.28, 0.72)
  const wheelMaterial = makeMaterial({ color: 0x101216, roughness: 0.92, metalness: 0.05 })
  const frameMaterial = makeMaterial({ color: 0xe0b94c, roughness: 0.46, metalness: 0.22 })
  const wheelGeometry = new THREE.TorusGeometry(wheelRadius, 0.035, 12, 24)
  const frameTube = new THREE.CylinderGeometry(0.025, 0.025, size.length * 0.54, 10)

  ;[-wheelGap, wheelGap].forEach((offset) => {
    const wheel = setCommonMeshProps(new THREE.Mesh(wheelGeometry, wheelMaterial))
    wheel.position.set(0, wheelRadius, offset * 0.5)
    wheel.rotation.y = Math.PI / 2
    group.add(wheel)
  })

  const frame = setCommonMeshProps(new THREE.Mesh(frameTube, frameMaterial))
  frame.position.y = wheelRadius + size.height * 0.16
  frame.rotation.x = Math.PI / 2
  group.add(frame)

  const seat = setCommonMeshProps(new THREE.Mesh(
    new THREE.BoxGeometry(size.width * 0.2, 0.05, size.length * 0.18),
    makeMaterial({ color: 0x2d333c, roughness: 0.72, metalness: 0.08 }),
  ))
  seat.position.set(0, wheelRadius + size.height * 0.3, 0)
  group.add(seat)

  const handle = setCommonMeshProps(new THREE.Mesh(
    new THREE.CylinderGeometry(0.02, 0.02, size.width * 0.42, 10),
    frameMaterial,
  ))
  handle.position.set(0, wheelRadius + size.height * 0.32, -wheelGap * 0.45)
  handle.rotation.z = Math.PI / 2
  group.add(handle)

  setShadowProps(group)
  return group
}

function createConeFallback(box3d) {
  const group = new THREE.Group()
  const size = getTargetSize(box3d)
  const coneMaterial = makeMaterial({
    color: 0xff8e2a,
    roughness: 0.72,
    metalness: 0.08,
    emissive: 0x7a2f00,
    emissiveIntensity: 0.12,
  })
  const stripeMaterial = makeMaterial({ color: 0xf5f7fb, roughness: 0.28, metalness: 0.08 })
  const baseMaterial = makeMaterial({ color: 0x262a31, roughness: 0.9, metalness: 0.05 })
  const base = setCommonMeshProps(new THREE.Mesh(
    new THREE.CylinderGeometry(size.width * 0.48, size.width * 0.62, 0.08, 20),
    baseMaterial,
  ))
  base.position.y = 0.04
  group.add(base)
  const cone = setCommonMeshProps(new THREE.Mesh(
    new THREE.ConeGeometry(size.width * 0.26, size.height, 18),
    coneMaterial,
  ))
  cone.position.y = size.height * 0.5
  group.add(cone)
  const stripe = setCommonMeshProps(new THREE.Mesh(
    new THREE.CylinderGeometry(size.width * 0.19, size.width * 0.24, size.height * 0.1, 18),
    stripeMaterial,
  ))
  stripe.position.y = size.height * 0.42
  group.add(stripe)
  setShadowProps(group)
  return group
}

function createBarrierFallback(box3d) {
  const group = new THREE.Group()
  const size = getTargetSize(box3d)
  const barrierMaterial = makeMaterial({ color: 0xd7dce4, roughness: 0.64, metalness: 0.12 })
  const body = setCommonMeshProps(new THREE.Mesh(
    new THREE.BoxGeometry(size.width, size.height * 0.72, size.length),
    barrierMaterial,
  ))
  body.position.y = size.height * 0.36
  group.add(body)
  setShadowProps(group)
  return group
}

export function createFallbackRenderable(objectData, role = 'object') {
  if (role === 'ego') {
    return createVehicleFallback(objectData.box3d, true)
  }
  if (objectData.class === 'pedestrian') {
    return createPedestrianFallback(objectData.box3d)
  }
  if (['motorcycle', 'bicycle'].includes(objectData.class)) {
    return createTwoWheelerFallback(objectData.box3d)
  }
  if (objectData.class === 'traffic_cone') {
    return createConeFallback(objectData.box3d)
  }
  if (['barrier', 'traffic_barrier', 'concrete_barrier'].includes(objectData.class)) {
    return createBarrierFallback(objectData.box3d)
  }
  return createVehicleFallback(objectData.box3d, false)
}

export async function loadRenderableAsset(objectData, role = 'object') {
  const config = getAssetConfig(objectData.class, role)
  if (!config) {
    return null
  }

  if (!templateCache.has(config.key)) {
    templateCache.set(config.key, loadTemplate(config))
  }

  const template = await templateCache.get(config.key)
  const instance = cloneSkeleton(template)
  setShadowProps(instance)
  fitObjectToPhysicalBox(instance, objectData.box3d, config.yawOffset || 0)
  if (typeof config.forceColor === 'number') {
    applyColorOverride(instance, config.forceColor)
  }
  return instance
}

export function disposeRenderable(root) {
  root.traverse((node) => {
    if (node.geometry) {
      node.geometry.dispose()
    }
    if (!node.material) {
      return
    }
    const materials = Array.isArray(node.material) ? node.material : [node.material]
    materials.forEach((material) => material?.dispose?.())
  })
}

export { collectMaterials }
