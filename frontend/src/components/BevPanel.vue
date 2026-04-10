<template>
  <div ref="rootRef" class="bev-panel">
    <canvas ref="canvasRef" class="bev-canvas"></canvas>

    <div class="bev-overlay">
      <div class="legend-card">
        <div class="legend-item">
          <span class="legend-swatch gt"></span>
          <span>GT 轮廓</span>
        </div>
        <div class="legend-item">
          <span class="legend-swatch pred"></span>
          <span>Pred 填充</span>
        </div>
        <div class="legend-item">
          <span class="legend-swatch threat"></span>
          <span>风险波纹</span>
        </div>
        <div class="legend-item">
          <span class="legend-swatch ego"></span>
          <span>自车</span>
        </div>
      </div>

      <div class="summary-card">
        <span>单图叠加对照</span>
        <strong>更适合量产调试，图例直接区分 GT / Pred</strong>
      </div>
    </div>
  </div>
</template>

<script setup>
import { nextTick, onMounted, onUnmounted, ref, watch } from 'vue'

const props = defineProps({
  scene: {
    type: Object,
    required: true,
  },
})

const rootRef = ref(null)
const canvasRef = ref(null)

let resizeObserver = null

function toCanvasPoint(xForward, yLeft, width, height) {
  const xRange = 20
  const zMin = -12
  const zMax = 72
  const usableWidth = width - 56
  const usableHeight = height - 56
  const x = width / 2 - (yLeft / xRange) * (usableWidth / 2)
  const y = height - 28 - ((xForward - zMin) / (zMax - zMin)) * usableHeight
  return { x, y }
}

function yawToCorners(box3d) {
  const { center, size, yaw } = box3d
  const halfL = size.length / 2
  const halfW = size.width / 2
  const baseCorners = [
    [halfL, halfW],
    [halfL, -halfW],
    [-halfL, -halfW],
    [-halfL, halfW],
  ]

  return baseCorners.map(([localX, localY]) => ({
    x: center.x + localX * Math.cos(yaw) - localY * Math.sin(yaw),
    y: center.y + localX * Math.sin(yaw) + localY * Math.cos(yaw),
  }))
}

function drawGrid(ctx, width, height) {
  ctx.save()
  ctx.strokeStyle = 'rgba(152, 184, 214, 0.08)'
  ctx.lineWidth = 1

  for (let lane = -12; lane <= 12; lane += 3) {
    const start = toCanvasPoint(-12, lane, width, height)
    const end = toCanvasPoint(72, lane, width, height)
    ctx.beginPath()
    ctx.moveTo(start.x, start.y)
    ctx.lineTo(end.x, end.y)
    ctx.stroke()
  }

  for (let meter = 0; meter <= 70; meter += 10) {
    const left = toCanvasPoint(meter, -12, width, height)
    const right = toCanvasPoint(meter, 12, width, height)
    ctx.beginPath()
    ctx.moveTo(left.x, left.y)
    ctx.lineTo(right.x, right.y)
    ctx.stroke()

    ctx.fillStyle = 'rgba(198, 214, 230, 0.62)'
    ctx.font = '12px Bahnschrift, Segoe UI, sans-serif'
    ctx.fillText(`${meter}m`, right.x - 6, right.y - 8)
  }
  ctx.restore()
}

function averageLaneOffset(lane) {
  if (!Array.isArray(lane?.points) || !lane.points.length) {
    return 0
  }
  return lane.points.reduce((sum, point) => sum + point.y, 0) / lane.points.length
}

function buildRoadEnvelope(scene) {
  const boundaries = Array.isArray(scene?.lanes?.boundaries) ? scene.lanes.boundaries : []
  const corridor = Array.isArray(scene?.lanes?.corridor) ? scene.lanes.corridor : []
  const leftBoundary = boundaries
    .filter((lane) => averageLaneOffset(lane) >= 0)
    .sort((left, right) => averageLaneOffset(right) - averageLaneOffset(left))[0]
  const rightBoundary = boundaries
    .filter((lane) => averageLaneOffset(lane) <= 0)
    .sort((left, right) => averageLaneOffset(left) - averageLaneOffset(right))[0]

  if (leftBoundary && rightBoundary) {
    return {
      left: leftBoundary.points,
      right: rightBoundary.points,
      leftKind: leftBoundary.kind,
      rightKind: rightBoundary.kind,
    }
  }

  if (corridor.length) {
    return {
      left: corridor.map((point) => ({ x: point.x, y: point.left_y })),
      right: corridor.map((point) => ({ x: point.x, y: point.right_y })),
      leftKind: 'boundary',
      rightKind: 'boundary',
    }
  }

  return null
}

function drawRoadBody(ctx, scene, width, height) {
  const envelope = buildRoadEnvelope(scene)
  if (!envelope || envelope.left.length < 2 || envelope.right.length < 2) {
    return
  }

  ctx.save()
  ctx.beginPath()
  envelope.left.forEach((point, index) => {
    const pixel = toCanvasPoint(point.x, point.y, width, height)
    if (index === 0) {
      ctx.moveTo(pixel.x, pixel.y)
    } else {
      ctx.lineTo(pixel.x, pixel.y)
    }
  })
  for (let index = envelope.right.length - 1; index >= 0; index -= 1) {
    const point = envelope.right[index]
    const pixel = toCanvasPoint(point.x, point.y, width, height)
    ctx.lineTo(pixel.x, pixel.y)
  }
  ctx.closePath()
  ctx.fillStyle = 'rgba(52, 52, 54, 0.88)'
  ctx.fill()

  if (envelope.leftKind === 'curb') {
    ctx.strokeStyle = 'rgba(236, 237, 239, 0.85)'
    ctx.lineWidth = 4
    ctx.beginPath()
    envelope.left.forEach((point, index) => {
      const pixel = toCanvasPoint(point.x, point.y, width, height)
      if (index === 0) {
        ctx.moveTo(pixel.x, pixel.y)
      } else {
        ctx.lineTo(pixel.x, pixel.y)
      }
    })
    ctx.stroke()
  }

  if (envelope.rightKind === 'curb') {
    ctx.strokeStyle = 'rgba(236, 237, 239, 0.85)'
    ctx.lineWidth = 4
    ctx.beginPath()
    envelope.right.forEach((point, index) => {
      const pixel = toCanvasPoint(point.x, point.y, width, height)
      if (index === 0) {
        ctx.moveTo(pixel.x, pixel.y)
      } else {
        ctx.lineTo(pixel.x, pixel.y)
      }
    })
    ctx.stroke()
  }
  ctx.restore()
}

function drawCorridor(ctx, corridor, width, height) {
  if (!Array.isArray(corridor) || corridor.length < 2) {
    return
  }

  ctx.save()
  ctx.fillStyle = 'rgba(60, 123, 214, 0.12)'
  ctx.beginPath()
  corridor.forEach((point, index) => {
    const pixel = toCanvasPoint(point.x, point.left_y, width, height)
    if (index === 0) {
      ctx.moveTo(pixel.x, pixel.y)
    } else {
      ctx.lineTo(pixel.x, pixel.y)
    }
  })
  for (let index = corridor.length - 1; index >= 0; index -= 1) {
    const point = corridor[index]
    const pixel = toCanvasPoint(point.x, point.right_y, width, height)
    ctx.lineTo(pixel.x, pixel.y)
  }
  ctx.closePath()
  ctx.fill()
  ctx.restore()
}

function drawLane(ctx, lane, width, height, dashed = false) {
  if (!Array.isArray(lane.points) || lane.points.length < 2) {
    return
  }

  ctx.save()
  const laneKind = lane.kind || 'boundary'
  const isCurb = laneKind === 'curb'
  ctx.strokeStyle = isCurb
    ? 'rgba(245, 246, 248, 0.92)'
    : dashed
      ? 'rgba(255, 255, 255, 0.6)'
      : 'rgba(114, 145, 175, 0.7)'
  ctx.lineWidth = isCurb ? 3.2 : dashed ? 1.4 : 1.2
  ctx.setLineDash(isCurb ? [] : dashed ? [10, 10] : [])
  ctx.beginPath()
  lane.points.forEach((point, index) => {
    const pixel = toCanvasPoint(point.x, point.y, width, height)
    if (index === 0) {
      ctx.moveTo(pixel.x, pixel.y)
    } else {
      ctx.lineTo(pixel.x, pixel.y)
    }
  })
  ctx.stroke()
  ctx.restore()
}

function drawObject(ctx, objectItem, width, height, labelAnchorY = null, showLabel = true) {
  const corners = yawToCorners(objectItem.box3d).map((corner) =>
    toCanvasPoint(corner.x, corner.y, width, height),
  )
  const center = toCanvasPoint(objectItem.box3d.center.x, objectItem.box3d.center.y, width, height)
  const heading = toCanvasPoint(
    objectItem.box3d.center.x + Math.cos(objectItem.box3d.yaw) * objectItem.box3d.size.length * 0.38,
    objectItem.box3d.center.y + Math.sin(objectItem.box3d.yaw) * objectItem.box3d.size.length * 0.38,
    width,
    height,
  )

  ctx.save()
  ctx.beginPath()
  corners.forEach((point, index) => {
    if (index === 0) {
      ctx.moveTo(point.x, point.y)
    } else {
      ctx.lineTo(point.x, point.y)
    }
  })
  ctx.closePath()

  if (objectItem.source === 'ego') {
    ctx.fillStyle = 'rgba(244, 247, 251, 0.34)'
    ctx.strokeStyle = '#f4f7fb'
    ctx.lineWidth = 2
    ctx.fill()
    ctx.stroke()
  } else if (objectItem.source === 'gt') {
    ctx.strokeStyle = '#42dd88'
    ctx.lineWidth = 2
    ctx.stroke()
  } else {
    ctx.fillStyle = objectItem.threat_level === 'high'
      ? 'rgba(255, 102, 93, 0.4)'
      : objectItem.threat_level === 'medium'
        ? 'rgba(255, 181, 78, 0.34)'
        : 'rgba(110, 201, 255, 0.32)'
    ctx.strokeStyle = objectItem.threat_level === 'high' ? '#ff6558' : '#8ed6ff'
    ctx.lineWidth = 2
    ctx.fill()
    ctx.stroke()
  }

  ctx.beginPath()
  ctx.moveTo(center.x, center.y)
  ctx.lineTo(heading.x, heading.y)
  ctx.strokeStyle = objectItem.source === 'gt' ? '#42dd88' : objectItem.source === 'ego' ? '#f4f7fb' : '#ffffff'
  ctx.lineWidth = 1.5
  ctx.stroke()

  if (objectItem.source === 'pred' && objectItem.threat_level !== 'low') {
    ctx.beginPath()
    ctx.arc(center.x, center.y, objectItem.threat_level === 'high' ? 20 : 14, 0, Math.PI * 2)
    ctx.strokeStyle = objectItem.threat_level === 'high' ? 'rgba(255, 95, 77, 0.42)' : 'rgba(255, 181, 78, 0.36)'
    ctx.lineWidth = 1.6
    ctx.stroke()
  }

  if (showLabel) {
    ctx.fillStyle = objectItem.source === 'gt' ? '#9df3c7' : '#eff7ff'
    ctx.font = '12px Bahnschrift, Segoe UI, sans-serif'
    const labelY = labelAnchorY ?? center.y - 8
    ctx.fillText(objectItem.label, center.x + 8, labelY)
  }
  ctx.restore()
}

function drawEgo(ctx, ego, width, height) {
  drawObject(
    ctx,
    {
      source: 'ego',
      label: '自车',
      threat_level: 'low',
      box3d: ego.box3d,
    },
    width,
    height,
  )
}

function render() {
  const root = rootRef.value
  const canvas = canvasRef.value
  if (!root || !canvas) {
    return
  }

  const width = Math.max(1, Math.floor(root.clientWidth))
  const height = Math.max(1, Math.floor(root.clientHeight))
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  canvas.width = width * dpr
  canvas.height = height * dpr
  canvas.style.width = `${width}px`
  canvas.style.height = `${height}px`

  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, width, height)

  const gradient = ctx.createLinearGradient(0, 0, 0, height)
  gradient.addColorStop(0, '#0f1f31')
  gradient.addColorStop(1, '#09131f')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, width, height)

  drawRoadBody(ctx, props.scene, width, height)
  drawGrid(ctx, width, height)
  drawCorridor(ctx, props.scene?.lanes?.corridor, width, height)
  ;(props.scene?.lanes?.centerlines || []).forEach((lane) => drawLane(ctx, lane, width, height, true))
  ;(props.scene?.lanes?.boundaries || []).forEach((lane) => drawLane(ctx, lane, width, height, false))

  drawEgo(ctx, props.scene.ego, width, height)
  const labelRows = []
  let gtLabels = 0
  let predLabels = 0
  ;(props.scene?.objects || [])
    .slice()
    .sort((left, right) => {
      const leftRank = left.source === 'pred' ? 1 : 0
      const rightRank = right.source === 'pred' ? 1 : 0
      if (leftRank !== rightRank) {
        return leftRank - rightRank
      }
      return left.box3d.center.x - right.box3d.center.x
    })
    .forEach((objectItem) => {
      const basePoint = toCanvasPoint(objectItem.box3d.center.x, objectItem.box3d.center.y, width, height)
      let labelY = basePoint.y - 8
      while (labelRows.some((existing) => Math.abs(existing - labelY) < 16)) {
        labelY -= 16
      }
      const allowLabel = objectItem.source === 'pred'
        ? predLabels < 8 && (objectItem.threat_level !== 'low' || objectItem.box3d.center.x < 22)
        : gtLabels < 4 && objectItem.box3d.center.x < 18
      drawObject(ctx, objectItem, width, height, labelY, allowLabel)
      if (allowLabel) {
        labelRows.push(labelY)
        if (objectItem.source === 'pred') {
          predLabels += 1
        } else {
          gtLabels += 1
        }
      }
    })
}

onMounted(() => {
  nextTick(render)
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(() => render())
    resizeObserver.observe(rootRef.value)
  }
})

watch(
  () => props.scene,
  () => nextTick(render),
  { deep: true, immediate: true },
)

onUnmounted(() => {
  if (resizeObserver) {
    resizeObserver.disconnect()
  }
})
</script>

<style scoped>
.bev-panel {
  position: relative;
  min-height: 420px;
  border-radius: 26px;
  overflow: hidden;
  border: 1px solid rgba(143, 177, 206, 0.16);
  background:
    radial-gradient(circle at top, rgba(74, 143, 255, 0.16), transparent 45%),
    linear-gradient(180deg, rgba(12, 27, 44, 0.9), rgba(6, 15, 25, 0.96));
}

.bev-canvas {
  width: 100%;
  height: 100%;
  display: block;
}

.bev-overlay {
  position: absolute;
  inset: 14px;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  pointer-events: none;
}

.legend-card,
.summary-card {
  border-radius: 18px;
  padding: 12px 14px;
  background: rgba(9, 20, 34, 0.64);
  border: 1px solid rgba(143, 177, 206, 0.18);
  backdrop-filter: blur(12px);
}

.legend-card {
  display: grid;
  gap: 8px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-secondary);
  font-size: 12px;
}

.legend-swatch {
  width: 14px;
  height: 14px;
  border-radius: 4px;
  border: 1px solid rgba(255, 255, 255, 0.16);
}

.legend-swatch.gt {
  background: rgba(66, 221, 136, 0.2);
  border-color: rgba(66, 221, 136, 0.88);
}

.legend-swatch.pred {
  background: rgba(110, 201, 255, 0.32);
  border-color: rgba(142, 214, 255, 0.88);
}

.legend-swatch.threat {
  background: rgba(255, 101, 88, 0.28);
  border-color: rgba(255, 101, 88, 0.84);
}

.legend-swatch.ego {
  background: rgba(244, 247, 251, 0.36);
  border-color: rgba(244, 247, 251, 0.92);
}

.summary-card {
  max-width: 240px;
  display: grid;
  gap: 4px;
  color: var(--text-secondary);
  font-size: 12px;
}

.summary-card strong {
  color: var(--text-primary);
  font-size: 13px;
  font-weight: 600;
}

@media (max-width: 768px) {
  .bev-panel {
    min-height: 340px;
  }

  .bev-overlay {
    flex-direction: column;
    gap: 10px;
  }

  .summary-card {
    max-width: none;
  }
}
</style>
