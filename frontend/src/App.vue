<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand-block">
        <div class="brand-kicker">Autonomous Perception Workbench</div>
        <h1>TDR-QAF 3D Detection + Scene Rendering</h1>
        <p>以统一场景数据包驱动 SR、BEV 与相机结果，而不是继续堆静态截图。</p>
      </div>

      <div class="topbar-status">
        <div class="status-chip" :class="backendStatus ? 'online' : 'offline'">
          <span class="status-dot"></span>
          {{ backendStatus ? '后端在线' : '后端离线' }}
        </div>
        <div class="status-chip">设备 {{ deviceType }}</div>
        <div class="status-chip">样本 {{ currentSampleId || '--' }}</div>
      </div>
    </header>

    <div class="workspace">
      <aside class="control-column">
        <section class="glass-panel control-panel">
          <div class="panel-heading">
            <div>
              <p class="eyebrow">Inference Control</p>
              <h2>推理配置</h2>
            </div>
            <span class="panel-note">支持结构化 scene stream</span>
          </div>

          <el-form label-position="top" class="control-form">
            <el-form-item label="置信度阈值">
              <el-slider
                v-model="confidenceThreshold"
                :min="0.01"
                :max="1"
                :step="0.01"
                class="control-slider"
              />
              <div class="range-readout">{{ confidenceThreshold.toFixed(2) }}</div>
            </el-form-item>

            <el-form-item label="样本模式">
              <el-radio-group v-model="mode" class="mode-switch">
                <el-radio-button value="random">随机抽取</el-radio-button>
                <el-radio-button value="specific">指定索引</el-radio-button>
              </el-radio-group>
            </el-form-item>

            <el-form-item v-if="mode === 'specific'" label="样本索引">
              <el-input-number
                v-model="sampleIndex"
                :min="0"
                :max="Math.max(0, datasetSize - 1)"
                controls-position="right"
                class="index-input"
              />
              <div class="input-hint">可用范围：0 - {{ Math.max(0, datasetSize - 1) }}（共 {{ datasetSize }} 个样本）</div>
            </el-form-item>

            <el-button
              type="primary"
              class="run-button"
              :loading="isLoading"
              :disabled="!backendStatus"
              @click="startDetection"
            >
              开始推理并刷新 SR
            </el-button>
          </el-form>
        </section>

        <section class="glass-panel stats-panel">
          <div class="panel-heading compact">
            <div>
              <p class="eyebrow">Telemetry</p>
              <h2>实时指标</h2>
            </div>
          </div>

          <div class="stats-grid">
            <div v-for="metric in metrics" :key="metric.label" class="metric-card">
              <span class="metric-label">{{ metric.label }}</span>
              <strong>{{ metric.value }}</strong>
            </div>
          </div>
        </section>

        <section class="glass-panel insight-panel">
          <div class="panel-heading compact">
            <div>
              <p class="eyebrow">Perception Summary</p>
              <h2>目标与风险</h2>
            </div>
          </div>

          <div class="threat-row">
            <div class="threat-card high">
              <span>高风险</span>
              <strong>{{ threatCounts.high }}</strong>
            </div>
            <div class="threat-card medium">
              <span>中风险</span>
              <strong>{{ threatCounts.medium }}</strong>
            </div>
            <div class="threat-card low">
              <span>低风险</span>
              <strong>{{ threatCounts.low }}</strong>
            </div>
          </div>

          <div class="class-breakdown">
            <div v-for="item in predBreakdown" :key="item.className" class="breakdown-row">
              <span>{{ item.label }}</span>
              <strong>{{ item.count }}</strong>
            </div>
            <div v-if="predBreakdown.length === 0" class="empty-breakdown">
              当前预测为空，SR 仅保留自车与车道。
            </div>
          </div>
        </section>
      </aside>

      <main class="content-column">
        <section class="hero-grid">
          <article class="glass-panel sr-card">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Scene Rendering</p>
                <h2>量产风格 SR 主视图</h2>
                <p class="section-copy">Three.js 实时渲染，自车高亮、目标平滑插值、3D 投影标签与风险波纹全部在前端完成。</p>
              </div>
            </div>
            <SceneRenderingPanel :scene="scenePacket" />
          </article>

          <article class="glass-panel bev-card">
            <div class="section-heading">
              <div>
                <p class="eyebrow">Bird Eye View</p>
                <h2>单图叠加式 BEV</h2>
                <p class="section-copy">这里我选一张图叠加 GT 与 Pred，并配清晰图例。工程调试时比拆两张来回对照更高效。</p>
              </div>
            </div>
            <BevPanel :scene="scenePacket" />
          </article>
        </section>

        <section class="camera-grid">
          <article class="glass-panel camera-card fusion-card">
            <CameraPanel
              title="Camera Fusion"
              subtitle="多视角 GT + Pred 总览"
              :image-src="imageCombined"
              status="GT + Pred"
            />
          </article>

          <article class="glass-panel camera-card focus-card">
            <CameraPanel
              title="Target Focus"
              subtitle="关键目标对焦与原图关联"
              :image-src="imagePred"
              status="Focus Pair"
            />
          </article>

          <article class="glass-panel camera-card front-card">
            <CameraPanel
              title="Front Camera"
              subtitle="前视相机原图"
              :image-src="imageFront"
              status="Original"
            />
          </article>
        </section>

        <section class="review-grid">
          <article class="glass-panel camera-card">
            <CameraPanel
              title="Backend SR"
              subtitle="Prediction render from backend"
              :image-src="imageSrPred"
              status="Pred SR"
            />
          </article>

          <article class="glass-panel camera-card">
            <CameraPanel
              title="Backend SR"
              subtitle="Ground truth render from backend"
              :image-src="imageSrGt"
              status="GT SR"
            />
          </article>

          <article class="glass-panel camera-card">
            <CameraPanel
              title="Backend BEV"
              subtitle="Rendered BEV verification image"
              :image-src="imageBev"
              status="BEV"
            />
          </article>
        </section>
      </main>
    </div>

    <transition name="fade">
      <div v-if="isLoading" class="loading-overlay">
        <div class="loader-ring"></div>
        <span>后端正在解码 3D 结果并刷新结构化场景...</span>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { ElMessageBox, ElNotification } from 'element-plus'

import { fetchHealth, requestPrediction } from './api/client'
import CameraPanel from './components/CameraPanel.vue'
import BevPanel from './components/BevPanel.vue'
import SceneRenderingPanel from './components/SceneRenderingPanel.vue'
import { createFallbackScenePacket, formatClassLabel, normalizeScenePacket } from './data/scenePacket'

const confidenceThreshold = ref(0.45)
const mode = ref('random')
const sampleIndex = ref(0)
const isLoading = ref(false)
const imageCombined = ref('')
const imagePred = ref('')
const imageBev = ref('')
const imageSrGt = ref('')
const imageSrPred = ref('')
const imageFront = ref('')
const backendStatus = ref(false)
const deviceType = ref('CPU')
const datasetSize = ref(0)
const gtTotal = ref(0)
const detectionCount = ref(0)
const inferenceTime = ref(0)
const currentSampleId = ref('')
const predDetails = ref({})
const scenePacket = ref(createFallbackScenePacket())

let healthTimer = null

const metrics = computed(() => [
  { label: 'GT 总数', value: gtTotal.value },
  { label: 'Pred 总数', value: detectionCount.value },
  { label: '推理耗时', value: `${inferenceTime.value} ms` },
  { label: '车速', value: `${Math.round(scenePacket.value.ego.speed_kph)} km/h` },
])

const threatCounts = computed(() => scenePacket.value?.summary?.threat_counts || { high: 0, medium: 0, low: 0 })

const predBreakdown = computed(() =>
  Object.entries(predDetails.value)
    .map(([className, count]) => ({
      className,
      label: formatClassLabel(className),
      count,
    }))
    .sort((left, right) => right.count - left.count),
)

async function checkBackendStatus() {
  try {
    const data = await fetchHealth()
    backendStatus.value = data.status === 'healthy'
    deviceType.value = data.device || 'CPU'
    datasetSize.value = data.dataset_size || 0
  } catch {
    backendStatus.value = false
  }
}

async function startDetection() {
  isLoading.value = true

  try {
    const payload = {
      mode: mode.value,
      confidence_threshold: confidenceThreshold.value,
    }

    if (mode.value === 'specific') {
      payload.sample_index = sampleIndex.value
    }

    const data = await requestPrediction(payload)
    imageCombined.value = data.image_combined || ''
    imagePred.value = data.image_pred || ''
    imageBev.value = data.image_bev || ''
    imageSrGt.value = data.image_sr_gt || ''
    imageSrPred.value = data.image_sr_pred || ''
    imageFront.value = data.image_front || ''
    scenePacket.value = normalizeScenePacket(data.scene_stream)

    if (data.stats) {
      gtTotal.value = data.stats.gt_total || 0
      detectionCount.value = data.stats.pred_total || 0
      predDetails.value = data.stats.pred_details || {}
      inferenceTime.value = data.stats.latency_ms || 0
      currentSampleId.value = data.stats.sample_token || ''
    }

    ElNotification({
      title: '场景已更新',
      message: detectionCount.value > 0 ? 'SR、BEV 与相机结果已同步刷新。' : '当前样本没有预测目标，已保留稳定 SR 基础场景。',
      type: 'success',
      duration: 2600,
    })
  } catch (error) {
    const message = error?.response?.data?.detail || error?.message || '检测失败'
    await ElMessageBox.alert(message, '推理失败', {
      confirmButtonText: '确定',
      type: 'error',
    })
  } finally {
    isLoading.value = false
  }
}

onMounted(() => {
  checkBackendStatus()
  healthTimer = window.setInterval(checkBackendStatus, 10000)
})

onUnmounted(() => {
  if (healthTimer) {
    window.clearInterval(healthTimer)
    healthTimer = null
  }
})
</script>

<style scoped>
.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  padding: 24px;
  gap: 18px;
  background:
    radial-gradient(circle at top left, rgba(66, 126, 219, 0.18), transparent 32%),
    radial-gradient(circle at top right, rgba(117, 194, 255, 0.08), transparent 28%),
    linear-gradient(180deg, #050b14, #09111d 45%, #060b13 100%);
}

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 20px;
  padding: 22px 24px;
  border-radius: 28px;
  border: 1px solid rgba(143, 177, 206, 0.14);
  background: rgba(6, 15, 26, 0.78);
  backdrop-filter: blur(14px);
}

.brand-kicker,
.eyebrow {
  margin: 0 0 8px;
  font-size: 11px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.brand-block h1,
.panel-heading h2,
.section-heading h2 {
  margin: 0;
  font-size: 26px;
  font-weight: 650;
  color: var(--text-primary);
}

.brand-block p,
.section-copy {
  margin: 8px 0 0;
  max-width: 760px;
  color: var(--text-secondary);
  font-size: 14px;
}

.topbar-status {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px;
}

.status-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(11, 23, 38, 0.68);
  border: 1px solid rgba(143, 177, 206, 0.14);
  color: var(--text-secondary);
  font-size: 13px;
}

.status-chip.online {
  color: #c8f4da;
}

.status-chip.offline {
  color: #ffc5bd;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
  box-shadow: 0 0 12px currentColor;
}

.workspace {
  display: grid;
  grid-template-columns: 340px minmax(0, 1fr);
  gap: 18px;
  min-height: 0;
  flex: 1;
}

.control-column,
.content-column {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.glass-panel {
  border-radius: 28px;
  border: 1px solid rgba(143, 177, 206, 0.14);
  background:
    linear-gradient(180deg, rgba(11, 22, 36, 0.86), rgba(5, 12, 21, 0.92)),
    radial-gradient(circle at top, rgba(68, 130, 219, 0.08), transparent 50%);
  backdrop-filter: blur(14px);
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
}

.control-panel,
.stats-panel,
.insight-panel,
.sr-card,
.bev-card,
.camera-card {
  padding: 22px;
}

.panel-heading,
.section-heading {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
  margin-bottom: 18px;
}

.panel-heading.compact {
  margin-bottom: 16px;
}

.panel-note {
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(117, 194, 255, 0.12);
  border: 1px solid rgba(117, 194, 255, 0.2);
  color: var(--accent-ice);
  font-size: 12px;
}

.control-form :deep(.el-form-item__label) {
  color: var(--text-secondary);
  font-size: 13px;
}

.control-slider {
  margin-top: 6px;
}

.range-readout,
.input-hint {
  margin-top: 8px;
  color: var(--text-muted);
  font-size: 12px;
}

.mode-switch,
.index-input {
  width: 100%;
}

.run-button {
  width: 100%;
  height: 46px;
  border-radius: 16px;
  border: none;
  font-size: 15px;
  font-weight: 600;
  background: linear-gradient(135deg, #1d7bff, #53b7ff);
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.metric-card {
  padding: 16px;
  border-radius: 20px;
  background: rgba(11, 24, 40, 0.72);
  border: 1px solid rgba(143, 177, 206, 0.1);
}

.metric-label {
  display: block;
  margin-bottom: 8px;
  font-size: 12px;
  color: var(--text-muted);
}

.metric-card strong {
  font-size: 24px;
  color: var(--text-primary);
}

.threat-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}

.threat-card {
  padding: 14px;
  border-radius: 18px;
  border: 1px solid transparent;
}

.threat-card span {
  display: block;
  margin-bottom: 8px;
  font-size: 12px;
  color: var(--text-muted);
}

.threat-card strong {
  font-size: 24px;
}

.threat-card.high {
  background: rgba(255, 101, 88, 0.12);
  border-color: rgba(255, 101, 88, 0.2);
  color: #ffd4ce;
}

.threat-card.medium {
  background: rgba(255, 181, 78, 0.12);
  border-color: rgba(255, 181, 78, 0.2);
  color: #ffe1bc;
}

.threat-card.low {
  background: rgba(110, 201, 255, 0.12);
  border-color: rgba(110, 201, 255, 0.18);
  color: #d7efff;
}

.class-breakdown {
  display: grid;
  gap: 10px;
}

.breakdown-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(11, 24, 40, 0.64);
  border: 1px solid rgba(143, 177, 206, 0.08);
  color: var(--text-secondary);
}

.breakdown-row strong {
  color: var(--text-primary);
}

.empty-breakdown {
  padding: 14px;
  border-radius: 16px;
  background: rgba(11, 24, 40, 0.4);
  color: var(--text-muted);
  font-size: 13px;
}

.hero-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.8fr) minmax(360px, 0.95fr);
  gap: 18px;
}

.camera-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.review-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

.fusion-card {
  grid-column: 1 / -1;
}

.loading-overlay {
  position: fixed;
  inset: 0;
  display: grid;
  place-items: center;
  gap: 16px;
  background: rgba(3, 8, 16, 0.76);
  backdrop-filter: blur(12px);
  z-index: 20;
  color: var(--text-primary);
}

.loader-ring {
  width: 68px;
  height: 68px;
  border-radius: 50%;
  border: 4px solid rgba(111, 163, 211, 0.14);
  border-top-color: #66b7ff;
  animation: spin 0.9s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 1360px) {
  .hero-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 1080px) {
  .workspace {
    grid-template-columns: 1fr;
  }

  .camera-grid {
    grid-template-columns: 1fr;
  }

  .review-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .app-shell {
    padding: 14px;
  }

  .topbar {
    flex-direction: column;
  }

  .topbar-status {
    justify-content: flex-start;
  }

  .stats-grid {
    grid-template-columns: 1fr;
  }
}
</style>
