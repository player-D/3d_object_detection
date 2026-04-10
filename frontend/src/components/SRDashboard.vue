<template>
  <div class="sr-dashboard">
    <!-- Three.js Canvas -->
    <canvas ref="canvasRef" class="three-canvas"></canvas>
    
    <!-- HUD UI Overlay -->
    <div class="hud-overlay">
      <!-- 左上角：车速和状态图标 -->
      <div class="hud-left-top">
        <div class="speed-display">
          <span class="speed-value">{{ egoVelocity }}</span>
          <span class="speed-unit">km/h</span>
        </div>
        <div class="status-icons">
          <div class="status-icon active">ACC</div>
          <div class="status-icon active">LCC</div>
          <div class="status-icon">NGP</div>
        </div>
      </div>
      
      <!-- 中上方：提示气泡 -->
      <div class="hud-center-top">
        <div class="alert-bubble">
          <span class="alert-text">{{ alertMessage }}</span>
        </div>
      </div>
      
      <!-- 右下角：装饰图标 -->
      <div class="hud-right-bottom">
        <div class="decoration-icon">🚗</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import * as THREE from 'three'

const canvasRef = ref(null)
const egoVelocity = ref(47)
const alertMessage = ref('即将超车')

let scene, camera, renderer, animationId
let objects3D = []

// 加载检测结果的 JSON 数据
const loadDetectionResults = async () => {
  try {
    const response = await fetch('/api/detection-results')
    const data = await response.json()
    egoVelocity.value = data.ego_velocity || 47
    return data.objects || []
  } catch (error) {
    console.error('Failed to load detection results:', error)
    return []
  }
}

// 初始化 Three.js 场景
const initThreeScene = () => {
  const canvas = canvasRef.value
  const width = window.innerWidth
  const height = window.innerHeight

  // 场景
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0F172A) // 深邃的夜空蓝
  
  // 雾化效果
  scene.fog = new THREE.FogExp2(0x0F172A, 0.02)

  // 相机
  camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000)
  camera.position.set(0, 15, 20)
  camera.lookAt(0, 0, 0)

  // 渲染器
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
  renderer.setSize(width, height)
  renderer.setPixelRatio(window.devicePixelRatio)
  renderer.shadowMap.enabled = true

  // 灯光
  const ambientLight = new THREE.AmbientLight(0x404040, 0.5)
  scene.add(ambientLight)

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
  directionalLight.position.set(10, 20, 10)
  directionalLight.castShadow = true
  scene.add(directionalLight)

  // 地面
  createGround()
  
  // 自车
  createEgoVehicle()
  
  // 车道引导线
  createLaneGuide()
}

// 创建地面
const createGround = () => {
  // 深灰色地面
  const groundGeometry = new THREE.PlaneGeometry(200, 200)
  const groundMaterial = new THREE.MeshStandardMaterial({ 
    color: 0x1a1a2e,
    roughness: 0.8,
    metalness: 0.2
  })
  const ground = new THREE.Mesh(groundGeometry, groundMaterial)
  ground.rotation.x = -Math.PI / 2
  ground.position.y = -0.01
  ground.receiveShadow = true
  scene.add(ground)

  // 网格辅助线（幽蓝色）
  const gridHelper = new THREE.GridHelper(200, 100, 0x4a5568, 0x2d3748)
  gridHelper.position.y = 0
  scene.add(gridHelper)
}

// 创建自车
const createEgoVehicle = () => {
  const carGeometry = new THREE.BoxGeometry(4.5, 1.5, 1.8)
  const carMaterial = new THREE.MeshStandardMaterial({
    color: 0xc0c0c0, // 银色
    metalness: 0.8,
    roughness: 0.2
  })
  const egoCar = new THREE.Mesh(carGeometry, carMaterial)
  egoCar.position.set(0, 0.75, 0)
  egoCar.castShadow = true
  scene.add(egoCar)
}

// 创建车道引导线
const createLaneGuide = () => {
  const points = []
  for (let i = 0; i < 50; i++) {
    const x = i * 0.5
    const width = 2 + i * 0.02
    points.push(new THREE.Vector3(x, 0.05, -width))
    points.push(new THREE.Vector3(x, 0.05, width))
  }
  
  const guideGeometry = new THREE.BufferGeometry().setFromPoints(points)
  const guideMaterial = new THREE.LineBasicMaterial({ 
    color: 0x3b82f6,
    transparent: true,
    opacity: 0.6
  })
  const guideLine = new THREE.Line(guideGeometry, guideMaterial)
  scene.add(guideLine)
}

// 创建 3D 目标物体
const createObject3D = (objData) => {
  const { class: className, x, y, z, w, l, h, yaw } = objData
  
  // 坐标转换：自动驾驶 Z 朝上，Three.js Y 朝上
  const posX = x
  const posY = z
  const posZ = -y
  
  let geometry, material
  
  if (className === 'car' || className === 'truck' || className === 'bus') {
    // 车辆：使用金属质感
    geometry = new THREE.BoxGeometry(l, h, w)
    material = new THREE.MeshStandardMaterial({
      color: 0xe8e8e8, // 极光银
      metalness: 0.6,
      roughness: 0.2
    })
  } else if (className === 'pedestrian') {
    // 行人：使用蓝色发光圆柱体
    geometry = new THREE.CylinderGeometry(0.3, 0.3, h, 16)
    material = new THREE.MeshBasicMaterial({
      color: 0x3b82f6,
      transparent: true,
      opacity: 0.8
    })
  } else if (className === 'traffic_cone') {
    // 交通锥
    geometry = new THREE.ConeGeometry(0.3, h, 8)
    material = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.9
    })
  } else {
    // 其他物体
    geometry = new THREE.BoxGeometry(l, h, w)
    material = new THREE.MeshStandardMaterial({
      color: 0x64748b,
      metalness: 0.4,
      roughness: 0.6
    })
  }
  
  const mesh = new THREE.Mesh(geometry, material)
  mesh.position.set(posX, posY, posZ)
  mesh.rotation.y = -yaw // Three.js 旋转方向
  mesh.castShadow = true
  mesh.receiveShadow = true
  
  return mesh
}

// 更新 3D 场景中的物体
const updateObjects3D = (objectsData) => {
  // 清除旧物体
  objects3D.forEach(obj => scene.remove(obj))
  objects3D = []
  
  // 添加新物体
  objectsData.forEach(objData => {
    const mesh = createObject3D(objData)
    scene.add(mesh)
    objects3D.push(mesh)
  })
}

// 动画循环
const animate = () => {
  animationId = requestAnimationFrame(animate)
  renderer.render(scene, camera)
}

// 窗口大小调整
const handleResize = () => {
  const width = window.innerWidth
  const height = window.innerHeight
  camera.aspect = width / height
  camera.updateProjectionMatrix()
  renderer.setSize(width, height)
}

onMounted(async () => {
  initThreeScene()
  
  // 加载并显示检测结果
  const objectsData = await loadDetectionResults()
  updateObjects3D(objectsData)
  
  // 启动动画
  animate()
  
  // 监听窗口大小变化
  window.addEventListener('resize', handleResize)
  
  // 定期刷新数据（模拟实时更新）
  setInterval(async () => {
    const objectsData = await loadDetectionResults()
    updateObjects3D(objectsData)
  }, 1000)
})

onUnmounted(() => {
  cancelAnimationFrame(animationId)
  window.removeEventListener('resize', handleResize)
  renderer.dispose()
})
</script>

<style scoped>
.sr-dashboard {
  position: relative;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background: #0F172A;
}

.three-canvas {
  display: block;
  width: 100%;
  height: 100%;
}

.hud-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.hud-left-top {
  position: absolute;
  top: 30px;
  left: 30px;
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.speed-display {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.speed-value {
  font-size: 48px;
  font-weight: bold;
  color: #ffffff;
  font-family: 'Arial', sans-serif;
}

.speed-unit {
  font-size: 18px;
  color: #94a3b8;
  font-family: 'Arial', sans-serif;
}

.status-icons {
  display: flex;
  gap: 10px;
}

.status-icon {
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  color: #64748b;
  font-weight: bold;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.status-icon.active {
  background: rgba(59, 130, 246, 0.3);
  color: #3b82f6;
  border-color: #3b82f6;
  box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
}

.hud-center-top {
  position: absolute;
  top: 30px;
  left: 50%;
  transform: translateX(-50%);
}

.alert-bubble {
  padding: 12px 24px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.alert-text {
  font-size: 16px;
  color: #ffffff;
  font-weight: 500;
  font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
}

.hud-right-bottom {
  position: absolute;
  bottom: 30px;
  right: 30px;
}

.decoration-icon {
  font-size: 40px;
  opacity: 0.6;
}
</style>
