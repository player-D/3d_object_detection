import * as THREE from 'three'

import { ObjectRegistry } from './ObjectRegistry'

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

function disposeNode(node) {
  node.traverse((child) => {
    if (child.geometry) {
      child.geometry.dispose()
    }
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material]
      materials.forEach((material) => material.dispose())
    }
  })
}

function createLaneLine(points, color, dashed = false) {
  const geometry = new THREE.BufferGeometry().setFromPoints(
    points.map((point) => new THREE.Vector3(-point.y, 0.04, -point.x)),
  )
  const material = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: 1.8, gapSize: 1.3, transparent: true, opacity: 0.72 })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.82 })
  const line = new THREE.Line(geometry, material)
  if (dashed) {
    line.computeLineDistances()
  }
  return line
}

function createCorridorMesh(corridor) {
  if (!Array.isArray(corridor) || corridor.length < 2) {
    return null
  }

  const positions = []
  corridor.forEach((point, index) => {
    if (index === 0) {
      return
    }
    const prev = corridor[index - 1]
    const quad = [
      [-prev.left_y, 0.02, -prev.x],
      [-prev.right_y, 0.02, -prev.x],
      [-point.right_y, 0.02, -point.x],
      [-point.left_y, 0.02, -point.x],
    ]
    positions.push(...quad[0], ...quad[1], ...quad[2], ...quad[0], ...quad[2], ...quad[3])
  })

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.computeVertexNormals()

  return new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      color: 0x2455a5,
      transparent: true,
      opacity: 0.14,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  )
}

function createEgoVehicle() {
  const group = new THREE.Group()
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: 0xf4f7fb,
    metalness: 0.78,
    roughness: 0.18,
    emissive: 0x0f3f73,
    emissiveIntensity: 0.28,
  })
  const glassMaterial = new THREE.MeshStandardMaterial({
    color: 0xc7dfff,
    metalness: 0.2,
    roughness: 0.08,
    transparent: true,
    opacity: 0.92,
  })
  const trimMaterial = new THREE.MeshStandardMaterial({
    color: 0x1a2330,
    metalness: 0.2,
    roughness: 0.64,
  })

  const body = new THREE.Mesh(new THREE.BoxGeometry(1.84, 0.68, 4.68), bodyMaterial)
  body.position.y = 0.52
  body.castShadow = true
  body.receiveShadow = true
  group.add(body)

  const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.36, 0.54, 2.12), glassMaterial)
  cabin.position.set(0, 0.96, -0.12)
  cabin.castShadow = true
  cabin.receiveShadow = true
  group.add(cabin)

  const nose = new THREE.Mesh(new THREE.BoxGeometry(1.46, 0.12, 0.96), trimMaterial)
  nose.position.set(0, 0.66, -1.54)
  nose.castShadow = true
  group.add(nose)

  const rear = new THREE.Mesh(new THREE.BoxGeometry(1.52, 0.12, 0.82), trimMaterial)
  rear.position.set(0, 0.66, 1.52)
  rear.castShadow = true
  group.add(rear)

  const halo = new THREE.Mesh(
    new THREE.RingGeometry(1.18, 1.44, 64),
    new THREE.MeshBasicMaterial({
      color: 0x7dd0ff,
      transparent: true,
      opacity: 0.22,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  )
  halo.rotation.x = -Math.PI / 2
  halo.position.y = 0.06
  group.add(halo)
  group.userData.halo = halo
  return group
}

export class SRSceneEngine {
  constructor({ canvas, container, onOverlayUpdate }) {
    this.canvas = canvas
    this.container = container
    this.onOverlayUpdate = onOverlayUpdate
    this.scenePacket = null
    this.animationFrame = 0
    this.clock = new THREE.Clock()
    this.lastOverlayFrame = 0

    this.cameraTarget = new THREE.Vector3(0, 1.4, -18)
    this.cameraTargetWanted = this.cameraTarget.clone()
    this.cameraPosition = new THREE.Vector3(10, 8.8, 16)
    this.cameraPositionWanted = this.cameraPosition.clone()
    this.size = { width: 1, height: 1 }

    this.handleWindowResize = () => this.resize()

    this.init()
  }

  init() {
    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(0x08111d)
    this.scene.fog = new THREE.FogExp2(0x08111d, 0.03)

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 240)
    this.camera.position.copy(this.cameraPosition)
    this.camera.lookAt(this.cameraTarget)

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    })
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.02
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap

    const ambientLight = new THREE.AmbientLight(0xc7d6e5, 0.9)
    this.scene.add(ambientLight)

    const sunLight = new THREE.DirectionalLight(0xffffff, 1.45)
    sunLight.position.set(18, 24, 10)
    sunLight.castShadow = true
    sunLight.shadow.mapSize.set(2048, 2048)
    sunLight.shadow.camera.near = 1
    sunLight.shadow.camera.far = 80
    sunLight.shadow.camera.left = -35
    sunLight.shadow.camera.right = 35
    sunLight.shadow.camera.top = 35
    sunLight.shadow.camera.bottom = -35
    this.scene.add(sunLight)

    this.scene.add(new THREE.HemisphereLight(0x8fbfe2, 0x0c1220, 0.42))

    this.groundGroup = new THREE.Group()
    this.scene.add(this.groundGroup)
    this.createGround()

    this.laneGroup = new THREE.Group()
    this.scene.add(this.laneGroup)

    this.egoVehicle = createEgoVehicle()
    this.scene.add(this.egoVehicle)

    this.registry = new ObjectRegistry(this.scene)

    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.resize())
      this.resizeObserver.observe(this.container)
    }
    window.addEventListener('resize', this.handleWindowResize)

    this.resize()
    this.animate()
  }

  createGround() {
    const base = new THREE.Mesh(
      new THREE.PlaneGeometry(150, 150),
      new THREE.MeshStandardMaterial({
        color: 0x0b1727,
        roughness: 0.86,
        metalness: 0.28,
      }),
    )
    base.rotation.x = -Math.PI / 2
    base.receiveShadow = true
    this.groundGroup.add(base)

    const sheen = new THREE.Mesh(
      new THREE.PlaneGeometry(110, 110),
      new THREE.MeshBasicMaterial({
        color: 0x133154,
        transparent: true,
        opacity: 0.08,
        depthWrite: false,
      }),
    )
    sheen.rotation.x = -Math.PI / 2
    sheen.position.y = 0.01
    this.groundGroup.add(sheen)

    const grid = new THREE.GridHelper(120, 60, 0x2b4d74, 0x16314b)
    grid.position.y = 0.015
    grid.material.transparent = true
    grid.material.opacity = 0.48
    this.groundGroup.add(grid)
  }

  updateLanes(lanes) {
    while (this.laneGroup.children.length) {
      const child = this.laneGroup.children[0]
      this.laneGroup.remove(child)
      disposeNode(child)
    }

    const corridorMesh = createCorridorMesh(lanes?.corridor)
    if (corridorMesh) {
      this.laneGroup.add(corridorMesh)
    }

    const centerlines = Array.isArray(lanes?.centerlines) ? lanes.centerlines : []
    centerlines.forEach((lane) => {
      this.laneGroup.add(createLaneLine(lane.points, 0xffffff, true))
    })

    const boundaries = Array.isArray(lanes?.boundaries) ? lanes.boundaries : []
    boundaries.forEach((lane) => {
      this.laneGroup.add(createLaneLine(lane.points, 0x4f6b84, false))
    })
  }

  setScenePacket(scenePacket) {
    this.scenePacket = scenePacket
    this.updateLanes(scenePacket?.lanes || {})
    this.registry.update(Array.isArray(scenePacket?.objects) ? scenePacket.objects : [])
    this.updateCameraIntent()
  }

  updateCameraIntent() {
    const objects = Array.isArray(this.scenePacket?.objects) ? this.scenePacket.objects : []
    const primaryThreat = objects
      .filter((objectItem) => objectItem.source === 'pred')
      .sort((left, right) => {
        const threatWeight = { high: 3, medium: 2, low: 1 }
        const threatGap = (threatWeight[right.threat_level] || 0) - (threatWeight[left.threat_level] || 0)
        if (threatGap !== 0) {
          return threatGap
        }
        return left.box3d.center.x - right.box3d.center.x
      })[0]

    const lateralBias = primaryThreat ? clamp(primaryThreat.box3d.center.y * -0.28, -3.2, 3.2) : 0
    const forwardLook = primaryThreat ? clamp(primaryThreat.box3d.center.x * 0.72, 16, 34) : 18
    this.cameraTargetWanted.set(lateralBias, 1.35, -forwardLook)
    this.cameraPositionWanted.set(10 + lateralBias * 0.28, 8.8, 16.5)
  }

  resize() {
    const bounds = this.container.getBoundingClientRect()
    const width = Math.max(1, Math.floor(bounds.width))
    const height = Math.max(1, Math.floor(bounds.height))
    this.size = { width, height }
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    this.renderer.setSize(width, height, false)
  }

  animate() {
    this.animationFrame = window.requestAnimationFrame(() => this.animate())

    const deltaSeconds = Math.min(this.clock.getDelta(), 0.05)
    const elapsedSeconds = this.clock.elapsedTime

    const cameraAlpha = 1 - Math.exp(-deltaSeconds * 2.8)
    this.cameraTarget.lerp(this.cameraTargetWanted, cameraAlpha)
    this.cameraPosition.lerp(this.cameraPositionWanted, cameraAlpha)
    this.camera.position.copy(this.cameraPosition)
    this.camera.lookAt(this.cameraTarget)

    const halo = this.egoVehicle.userData.halo
    if (halo) {
      const pulse = (Math.sin(elapsedSeconds * 1.8) + 1) * 0.5
      halo.material.opacity = 0.14 + pulse * 0.12
      const scale = 1 + pulse * 0.08
      halo.scale.setScalar(scale)
    }

    this.registry.animate(deltaSeconds, elapsedSeconds)
    this.renderer.render(this.scene, this.camera)

    if (this.onOverlayUpdate && elapsedSeconds - this.lastOverlayFrame > 1 / 20) {
      this.lastOverlayFrame = elapsedSeconds
      this.onOverlayUpdate(this.registry.getOverlayAnchors(this.camera, this.size))
    }
  }

  dispose() {
    window.cancelAnimationFrame(this.animationFrame)
    window.removeEventListener('resize', this.handleWindowResize)
    if (this.resizeObserver) {
      this.resizeObserver.disconnect()
    }
    this.registry.clear()
    disposeNode(this.groundGroup)
    disposeNode(this.laneGroup)
    disposeNode(this.egoVehicle)
    this.renderer.dispose()
  }
}
