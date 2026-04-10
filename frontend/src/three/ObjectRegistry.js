import * as THREE from 'three'

const Y_AXIS = new THREE.Vector3(0, 1, 0)
const THREAT_RANK = { low: 1, medium: 2, high: 3 }

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function setMaterialOpacity(material, opacity) {
  material.transparent = opacity < 0.999
  material.opacity = opacity
  material.depthWrite = opacity > 0.65
}

function disposeGroup(group) {
  group.traverse((node) => {
    if (node.geometry) {
      node.geometry.dispose()
    }
    if (node.material) {
      const materials = Array.isArray(node.material) ? node.material : [node.material]
      materials.forEach((material) => material.dispose())
    }
  })
}

function createMaterial(color, options = {}) {
  return new THREE.MeshStandardMaterial({
    color,
    metalness: options.metalness ?? 0.45,
    roughness: options.roughness ?? 0.4,
    emissive: options.emissive ?? 0x000000,
    emissiveIntensity: options.emissiveIntensity ?? 0,
    transparent: true,
    opacity: 1,
  })
}

function createBoxMesh(size, material) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(size.x, size.y, size.z), material)
  mesh.castShadow = true
  mesh.receiveShadow = true
  return mesh
}

function createVehicleModel(size, palette) {
  const group = new THREE.Group()
  const bodyMaterial = createMaterial(palette.body, { metalness: 0.62, roughness: 0.28, emissive: palette.emissive, emissiveIntensity: 0.18 })
  const cabinMaterial = createMaterial(palette.cabin, { metalness: 0.4, roughness: 0.18, emissive: palette.emissive, emissiveIntensity: 0.1 })
  const trimMaterial = createMaterial(palette.trim, { metalness: 0.25, roughness: 0.55 })

  const length = clamp(size.length, 1.8, 13.5)
  const width = clamp(size.width, 0.8, 3.2)
  const height = clamp(size.height, 1.0, 4.2)

  const body = createBoxMesh(new THREE.Vector3(width * 0.94, height * 0.42, length * 0.9), bodyMaterial)
  body.position.y = height * 0.34
  group.add(body)

  const cabin = createBoxMesh(new THREE.Vector3(width * 0.68, height * 0.28, length * 0.44), cabinMaterial)
  cabin.position.set(0, height * 0.62, -length * 0.06)
  group.add(cabin)

  const hood = createBoxMesh(new THREE.Vector3(width * 0.74, height * 0.12, length * 0.2), trimMaterial)
  hood.position.set(0, height * 0.48, -length * 0.29)
  group.add(hood)

  const rearDeck = createBoxMesh(new THREE.Vector3(width * 0.74, height * 0.1, length * 0.18), trimMaterial)
  rearDeck.position.set(0, height * 0.48, length * 0.26)
  group.add(rearDeck)

  const wheelCoverGeometry = new THREE.BoxGeometry(width * 0.18, height * 0.18, length * 0.18)
  const wheelMaterial = createMaterial(0x10141f, { metalness: 0.1, roughness: 0.85 })
  const wheelOffsets = [
    [-width * 0.38, height * 0.12, -length * 0.28],
    [width * 0.38, height * 0.12, -length * 0.28],
    [-width * 0.38, height * 0.12, length * 0.26],
    [width * 0.38, height * 0.12, length * 0.26],
  ]
  wheelOffsets.forEach(([x, y, z]) => {
    const wheel = new THREE.Mesh(wheelCoverGeometry, wheelMaterial)
    wheel.position.set(x, y, z)
    wheel.castShadow = true
    wheel.receiveShadow = true
    group.add(wheel)
  })

  return {
    root: group,
    materials: [bodyMaterial, cabinMaterial, trimMaterial, wheelMaterial],
  }
}

function createPedestrianModel(size, palette) {
  const group = new THREE.Group()
  const bodyMaterial = createMaterial(palette.body, { metalness: 0.1, roughness: 0.65, emissive: palette.emissive, emissiveIntensity: 0.12 })
  const accentMaterial = createMaterial(palette.trim, { metalness: 0.1, roughness: 0.55 })
  const height = clamp(size.height, 1.4, 2.1)

  const torso = createBoxMesh(new THREE.Vector3(0.42, height * 0.42, 0.26), bodyMaterial)
  torso.position.y = height * 0.53
  group.add(torso)

  const hips = createBoxMesh(new THREE.Vector3(0.38, height * 0.14, 0.24), accentMaterial)
  hips.position.y = height * 0.3
  group.add(hips)

  const head = createBoxMesh(new THREE.Vector3(0.24, 0.24, 0.24), bodyMaterial)
  head.position.y = height * 0.86
  group.add(head)

  const leftLeg = createBoxMesh(new THREE.Vector3(0.12, height * 0.34, 0.12), accentMaterial)
  leftLeg.position.set(-0.08, height * 0.12, 0)
  group.add(leftLeg)

  const rightLeg = createBoxMesh(new THREE.Vector3(0.12, height * 0.34, 0.12), accentMaterial)
  rightLeg.position.set(0.08, height * 0.12, 0)
  group.add(rightLeg)

  return {
    root: group,
    materials: [bodyMaterial, accentMaterial],
  }
}

function createConeModel(size, palette) {
  const group = new THREE.Group()
  const baseMaterial = createMaterial(palette.trim, { metalness: 0.1, roughness: 0.7 })
  const bodyMaterial = createMaterial(palette.body, { metalness: 0.08, roughness: 0.72, emissive: palette.emissive, emissiveIntensity: 0.1 })
  const height = clamp(size.height, 0.45, 1.1)

  const base = createBoxMesh(new THREE.Vector3(0.42, 0.08, 0.42), baseMaterial)
  base.position.y = 0.04
  group.add(base)

  const body = createBoxMesh(new THREE.Vector3(0.24, height * 0.82, 0.24), bodyMaterial)
  body.position.y = height * 0.42
  group.add(body)

  const top = createBoxMesh(new THREE.Vector3(0.14, 0.12, 0.14), baseMaterial)
  top.position.y = height * 0.84
  group.add(top)

  return {
    root: group,
    materials: [baseMaterial, bodyMaterial],
  }
}

function createBarrierModel(size, palette) {
  const group = new THREE.Group()
  const bodyMaterial = createMaterial(palette.body, { metalness: 0.18, roughness: 0.64, emissive: palette.emissive, emissiveIntensity: 0.12 })
  const supportMaterial = createMaterial(palette.trim, { metalness: 0.12, roughness: 0.72 })
  const length = clamp(size.length, 0.9, 4.2)

  const rail = createBoxMesh(new THREE.Vector3(clamp(size.width, 0.35, 0.8), clamp(size.height, 0.45, 1.0), length), bodyMaterial)
  rail.position.y = clamp(size.height, 0.45, 1.0) * 0.5
  group.add(rail)

  const feetOffset = length * 0.32
  ;[-feetOffset, feetOffset].forEach((offset) => {
    const foot = createBoxMesh(new THREE.Vector3(0.34, 0.08, 0.42), supportMaterial)
    foot.position.set(0, 0.04, offset)
    group.add(foot)
  })

  return {
    root: group,
    materials: [bodyMaterial, supportMaterial],
  }
}

function createSemanticModel(objectData) {
  const palette = objectData.source === 'gt'
    ? { body: 0x46d98a, cabin: 0x96f5c0, trim: 0x173d2d, emissive: 0x0b3b25 }
    : { body: 0x7ecbff, cabin: 0xddefff, trim: 0x244a74, emissive: 0x0e2b52 }

  if (['car', 'truck', 'bus', 'trailer', 'construction_vehicle'].includes(objectData.class)) {
    return createVehicleModel(objectData.box3d.size, palette)
  }
  if (['pedestrian', 'bicycle', 'motorcycle'].includes(objectData.class)) {
    return createPedestrianModel(objectData.box3d.size, palette)
  }
  if (objectData.class === 'traffic_cone') {
    return createConeModel(objectData.box3d.size, { ...palette, body: 0xff9c4b, trim: 0x5d3b11 })
  }
  return createBarrierModel(objectData.box3d.size, { ...palette, body: 0xc8d0dc, trim: 0x3b4a60 })
}

function createThreatRing() {
  const geometry = new THREE.RingGeometry(0.82, 1.04, 48)
  const material = new THREE.MeshBasicMaterial({
    color: 0xff7045,
    transparent: true,
    opacity: 0,
    side: THREE.DoubleSide,
    depthWrite: false,
  })
  const mesh = new THREE.Mesh(geometry, material)
  mesh.rotation.x = -Math.PI / 2
  mesh.position.y = 0.06
  mesh.visible = false
  return mesh
}

function createGroundShadow(radius) {
  const material = new THREE.MeshBasicMaterial({
    color: 0x05070c,
    transparent: true,
    opacity: 0.2,
    depthWrite: false,
  })
  const mesh = new THREE.Mesh(new THREE.CircleGeometry(clamp(radius, 0.35, 2.6), 32), material)
  mesh.rotation.x = -Math.PI / 2
  mesh.position.y = 0.03
  return mesh
}

function toWorldPosition(center) {
  return new THREE.Vector3(-center.y, center.z, -center.x)
}

function toWorldQuaternion(yaw) {
  return new THREE.Quaternion().setFromAxisAngle(Y_AXIS, yaw)
}

function setGroupOpacity(group, opacity) {
  group.traverse((node) => {
    if (!node.material) {
      return
    }
    const materials = Array.isArray(node.material) ? node.material : [node.material]
    materials.forEach((material) => setMaterialOpacity(material, opacity))
  })
}

export class ObjectRegistry {
  constructor(scene) {
    this.scene = scene
    this.items = new Map()
  }

  update(objects) {
    const seenIds = new Set()

    objects.forEach((objectData) => {
      seenIds.add(objectData.id)
      const existing = this.items.get(objectData.id)
      if (existing) {
        this.syncTracked(existing, objectData)
        return
      }

      const tracked = this.createTracked(objectData)
      this.items.set(objectData.id, tracked)
      this.scene.add(tracked.group)
    })

    this.items.forEach((tracked, id) => {
      tracked.stale = !seenIds.has(id)
    })
  }

  createTracked(objectData) {
    const group = new THREE.Group()
    const model = createSemanticModel(objectData)
    const ring = createThreatRing()
    const shadow = createGroundShadow(Math.max(objectData.box3d.size.length, objectData.box3d.size.width) * 0.35)
    group.add(shadow)
    group.add(model.root)
    group.add(ring)

    const tracked = {
      id: objectData.id,
      group,
      ring,
      shadow,
      materials: model.materials,
      targetPosition: toWorldPosition(objectData.box3d.center),
      targetQuaternion: toWorldQuaternion(objectData.box3d.yaw),
      meta: {},
      opacity: 0.04,
      stale: false,
      pulseOffset: Math.random(),
    }

    group.position.copy(tracked.targetPosition)
    group.quaternion.copy(tracked.targetQuaternion)
    this.syncTracked(tracked, objectData, true)
    setGroupOpacity(group, tracked.opacity)
    return tracked
  }

  syncTracked(tracked, objectData, snap = false) {
    tracked.meta = {
      id: objectData.id,
      source: objectData.source,
      className: objectData.class,
      label: objectData.label,
      score: objectData.score,
      threatLevel: objectData.threat_level,
      height: objectData.box3d.size.height,
      longitudinal: objectData.box3d.center.x,
    }

    tracked.targetPosition.copy(toWorldPosition(objectData.box3d.center))
    tracked.targetQuaternion.copy(toWorldQuaternion(objectData.box3d.yaw))
    tracked.stale = false

    if (snap) {
      tracked.group.position.copy(tracked.targetPosition)
      tracked.group.quaternion.copy(tracked.targetQuaternion)
    }
  }

  animate(deltaSeconds, elapsedSeconds) {
    const positionAlpha = 1 - Math.exp(-deltaSeconds * 7.5)
    const rotationAlpha = 1 - Math.exp(-deltaSeconds * 9.5)

    this.items.forEach((tracked, id) => {
      if (tracked.stale) {
        tracked.opacity = Math.max(0, tracked.opacity - deltaSeconds * 2.8)
      } else {
        tracked.opacity = Math.min(1, tracked.opacity + deltaSeconds * 3.2)
      }

      tracked.group.position.lerp(tracked.targetPosition, positionAlpha)
      tracked.group.quaternion.slerp(tracked.targetQuaternion, rotationAlpha)

      setGroupOpacity(tracked.group, tracked.opacity)
      tracked.shadow.material.opacity = tracked.opacity * 0.18

      const threatRank = THREAT_RANK[tracked.meta.threatLevel] || 1
      const isHighlighted = tracked.meta.source === 'pred' && threatRank >= 2 && !tracked.stale
      tracked.ring.visible = isHighlighted && tracked.opacity > 0.08
      if (tracked.ring.visible) {
        const pulse = (elapsedSeconds * 0.9 + tracked.pulseOffset) % 1
        const scale = 1 + pulse * (1.35 + threatRank * 0.2)
        tracked.ring.scale.setScalar(scale)
        tracked.ring.material.color.setHex(tracked.meta.threatLevel === 'high' ? 0xff5d4f : 0xffb34d)
        tracked.ring.material.opacity = (1 - pulse) * (tracked.meta.threatLevel === 'high' ? 0.28 : 0.18) * tracked.opacity
      }

      tracked.materials.forEach((material) => {
        material.emissiveIntensity = tracked.meta.source === 'pred'
          ? tracked.meta.threatLevel === 'high' ? 0.55 : tracked.meta.threatLevel === 'medium' ? 0.26 : 0.12
          : 0.08
      })

      if (tracked.stale && tracked.opacity <= 0.02) {
        this.scene.remove(tracked.group)
        disposeGroup(tracked.group)
        this.items.delete(id)
      }
    })
  }

  getOverlayAnchors(camera, viewport) {
    const anchors = []
    this.items.forEach((tracked) => {
      if (tracked.stale || tracked.opacity < 0.2 || tracked.meta.source !== 'pred') {
        return
      }

      const threatRank = THREAT_RANK[tracked.meta.threatLevel] || 1
      if (threatRank < 2) {
        return
      }

      const world = tracked.group.position.clone()
      world.y += Math.max(1.2, tracked.meta.height + 0.7)
      const projected = world.project(camera)
      if (projected.z < -1 || projected.z > 1) {
        return
      }
      if (Math.abs(projected.x) > 1.1 || Math.abs(projected.y) > 1.1) {
        return
      }

      anchors.push({
        id: tracked.id,
        threatLevel: tracked.meta.threatLevel,
        className: tracked.meta.className,
        score: tracked.meta.score,
        distanceM: Math.max(0, tracked.meta.longitudinal),
        x: (projected.x * 0.5 + 0.5) * viewport.width,
        y: (-projected.y * 0.5 + 0.5) * viewport.height,
      })
    })

    return anchors
      .sort((left, right) => {
        const threatGap = (THREAT_RANK[right.threatLevel] || 0) - (THREAT_RANK[left.threatLevel] || 0)
        if (threatGap !== 0) {
          return threatGap
        }
        return left.distanceM - right.distanceM
      })
      .slice(0, 4)
  }

  clear() {
    this.items.forEach((tracked) => {
      this.scene.remove(tracked.group)
      disposeGroup(tracked.group)
    })
    this.items.clear()
  }
}
