<template>
  <div ref="rootRef" class="sr-stage">
    <canvas ref="canvasRef" class="sr-canvas"></canvas>

    <div class="hud-layer">
      <div class="hud-top">
        <div class="speed-card">
          <span class="speed-value">{{ speedKph }}</span>
          <span class="speed-unit">km/h</span>
        </div>

        <div class="status-cluster">
          <div class="status-chip active">SR ONLINE</div>
          <div class="status-chip">Frame {{ props.scene.frame_id }}</div>
          <div class="status-chip" :class="maxThreat">
            {{ threatBadge }}
          </div>
        </div>
      </div>

      <div class="hud-banner">
        <span>{{ bannerText }}</span>
      </div>

      <div
        v-for="item in overlayItems"
        :key="item.id"
        class="target-label"
        :class="item.threatLevel"
        :style="{ left: `${item.x}px`, top: `${item.y}px` }"
      >
        <div class="label-card">
          <span class="label-title">{{ item.threatText }}</span>
          <strong>{{ item.classText }}</strong>
          <span class="label-meta">
            {{ item.distanceM.toFixed(1) }}m
            <template v-if="item.score !== null"> | {{ item.score.toFixed(2) }}</template>
          </span>
        </div>
        <div class="label-stem"></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'

import { formatClassLabel, formatThreatLabel } from '../data/scenePacket'
import { SRSceneEngine } from '../three/SRSceneEngine'

const props = defineProps({
  scene: {
    type: Object,
    required: true,
  },
})

const rootRef = ref(null)
const canvasRef = ref(null)
const overlayItems = ref([])

let engine = null

const speedKph = computed(() => Math.round(props.scene?.ego?.speed_kph ?? 0))
const maxThreat = computed(() => props.scene?.summary?.max_threat_level || 'low')
const threatBadge = computed(() => `${formatThreatLabel(maxThreat.value)} · ${props.scene?.summary?.pred_count ?? 0} Pred`)
const bannerText = computed(() => {
  if (maxThreat.value === 'high') {
    return 'SR 正在高亮高风险目标，标签与波纹均绑定预测结果。'
  }
  if (maxThreat.value === 'medium') {
    return '当前存在中风险目标，已保持稳定跟踪与标签投影。'
  }
  return '场景稳定，Bird Eye 与 SR 共享同一份结构化场景数据。'
})

function handleOverlayUpdate(items) {
  overlayItems.value = items.map((item) => ({
    ...item,
    classText: formatClassLabel(item.className),
    threatText: formatThreatLabel(item.threatLevel),
  }))
}

onMounted(() => {
  engine = new SRSceneEngine({
    canvas: canvasRef.value,
    container: rootRef.value,
    onOverlayUpdate: handleOverlayUpdate,
  })
  engine.setScenePacket(props.scene)
})

watch(
  () => props.scene,
  (scene) => {
    if (engine) {
      engine.setScenePacket(scene)
    }
  },
  { deep: true },
)

onUnmounted(() => {
  if (engine) {
    engine.dispose()
    engine = null
  }
})
</script>

<style scoped>
.sr-stage {
  position: relative;
  min-height: 480px;
  border-radius: 28px;
  overflow: hidden;
  border: 1px solid rgba(143, 177, 206, 0.16);
  background:
    radial-gradient(circle at top, rgba(57, 119, 204, 0.18), transparent 42%),
    linear-gradient(180deg, rgba(11, 23, 37, 0.9), rgba(5, 12, 21, 0.98));
}

.sr-canvas {
  display: block;
  width: 100%;
  height: 100%;
}

.hud-layer {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.hud-top {
  position: absolute;
  top: 18px;
  left: 18px;
  right: 18px;
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-start;
}

.speed-card {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding: 14px 18px;
  border-radius: 22px;
  background: rgba(8, 18, 30, 0.48);
  border: 1px solid rgba(143, 177, 206, 0.18);
  backdrop-filter: blur(14px);
}

.speed-value {
  font-size: 42px;
  line-height: 1;
  font-weight: 700;
  color: #f1f6fb;
}

.speed-unit {
  font-size: 15px;
  color: var(--text-muted);
}

.status-cluster {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.status-chip {
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(8, 18, 30, 0.48);
  border: 1px solid rgba(143, 177, 206, 0.16);
  color: var(--text-secondary);
  font-size: 12px;
  backdrop-filter: blur(14px);
}

.status-chip.active {
  color: #d8ecff;
  border-color: rgba(125, 208, 255, 0.36);
}

.status-chip.high {
  color: #ffd6d1;
  border-color: rgba(255, 101, 88, 0.45);
}

.status-chip.medium {
  color: #ffe2bc;
  border-color: rgba(255, 181, 78, 0.45);
}

.hud-banner {
  position: absolute;
  top: 92px;
  left: 50%;
  transform: translateX(-50%);
  min-width: 280px;
  padding: 12px 18px;
  border-radius: 999px;
  background: rgba(8, 18, 30, 0.44);
  border: 1px solid rgba(143, 177, 206, 0.16);
  backdrop-filter: blur(14px);
  color: var(--text-secondary);
  font-size: 13px;
  text-align: center;
}

.target-label {
  position: absolute;
  transform: translate(-50%, calc(-100% - 18px));
  display: flex;
  flex-direction: column;
  align-items: center;
}

.label-card {
  min-width: 108px;
  padding: 10px 12px;
  border-radius: 16px;
  background: rgba(7, 16, 28, 0.78);
  border: 1px solid rgba(143, 177, 206, 0.18);
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
  text-align: center;
}

.label-title {
  display: block;
  margin-bottom: 3px;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.label-card strong {
  display: block;
  font-size: 14px;
  color: #f2f7fb;
}

.label-meta {
  display: block;
  margin-top: 4px;
  color: var(--text-secondary);
  font-size: 11px;
}

.label-stem {
  width: 1px;
  height: 18px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0));
}

.target-label.high .label-card {
  border-color: rgba(255, 101, 88, 0.48);
}

.target-label.medium .label-card {
  border-color: rgba(255, 181, 78, 0.42);
}

@media (max-width: 768px) {
  .sr-stage {
    min-height: 380px;
  }

  .hud-top {
    flex-direction: column;
  }

  .status-cluster {
    justify-content: flex-start;
  }

  .hud-banner {
    left: 18px;
    right: 18px;
    transform: none;
    min-width: 0;
  }
}
</style>
