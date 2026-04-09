<template>
  <el-container class="app-container">
    <!-- 顶部导航栏 -->
    <el-header class="header">
      <div class="header-left">
        <el-icon class="header-icon"><TrendCharts /></el-icon>
        <h1 class="header-title">TDR-QAF 道路场景多视角 3D 目标检测系统</h1>
      </div>
      <div class="header-right">
        <div class="status-item">
          <span :class="['status-indicator', backendStatus ? 'status-online' : 'status-offline']"></span>
          <span>后端状态：{{ backendStatus ? '在线' : '离线' }}</span>
        </div>
        <div class="status-item">
          <span>设备：{{ deviceType }}</span>
        </div>
      </div>
    </el-header>
    
    <el-container>
      <!-- 左侧交互控制台 -->
      <el-aside width="380px" class="sidebar">
        <el-card class="custom-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <span>推理配置</span>
            </div>
          </template>
          <el-form label-width="100px">
            <el-form-item label="置信度阈值">
              <el-slider 
                v-model="confidenceThreshold" 
                :min="0.01" 
                :max="1.0" 
                :step="0.01" 
                class="custom-slider"
              />
              <div class="slider-value">{{ confidenceThreshold.toFixed(2) }}</div>
            </el-form-item>
            
            <el-form-item label="模式切换">
              <el-radio-group v-model="mode" class="mode-radio-group">
                <el-radio-button label="random" class="mode-button">随机抽取</el-radio-button>
                <el-radio-button label="specific" class="mode-button">指定索引</el-radio-button>
              </el-radio-group>
            </el-form-item>
            
            <el-form-item label="索引选择" v-if="mode === 'specific'">
              <el-input-number 
                v-model="sampleIndex" 
                :min="0" 
                :max="Math.max(0, datasetSize - 1)" 
                class="custom-input"
              />
              <div style="font-size: 12px; color: #909399; margin-top: 5px;">
                当前可选样本数：{{ datasetSize }}
              </div>
            </el-form-item>
            
            <el-form-item>
              <el-button 
                type="primary" 
                size="large" 
                :loading="isLoading" 
                @click="startDetection" 
                class="custom-button start-button"
              >
                <el-icon><Refresh /></el-icon>
                开始检测
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>
        
        <!-- 实时指标卡片 -->
        <div class="stats-container">
          <h3 class="stats-title">实时指标</h3>
          <el-row :gutter="20">
            <el-col :span="24">
              <div class="stat-card">
                <div class="stat-value">{{ gtTotal }}</div>
                <div class="stat-label">真实标注 (GT) 总数</div>
              </div>
            </el-col>
            <el-col :span="24">
              <div class="stat-card">
                <div class="stat-value">{{ detectionCount }}</div>
                <div class="stat-label">模型预测 (Pred) 总数</div>
              </div>
            </el-col>
            <el-col :span="24">
              <div class="stat-card">
                <div class="stat-value">{{ inferenceTime }} ms</div>
                <div class="stat-label">推理耗时</div>
              </div>
            </el-col>
            <el-col :span="24">
              <div class="stat-card">
                <div class="stat-value">{{ currentSampleId }}</div>
                <div class="stat-label">当前样本 ID</div>
              </div>
            </el-col>
          </el-row>
          
          <!-- 模型预测类别明细 -->
          <div class="pred-details-container">
            <h4 class="pred-details-title">模型预测类别明细</h4>
            <el-collapse v-model="activeNames">
              <el-collapse-item title="类别分布" name="1">
                <div class="pred-tags">
                  <div v-for="(count, category) in predDetails" :key="category" class="pred-tag">
                    <span class="tag-icon">{{ getCategoryIcon(category) }}</span>
                    <span class="tag-name">{{ category }}</span>
                    <span class="tag-count">{{ count }} 个</span>
                  </div>
                  <div v-if="Object.keys(predDetails).length === 0" class="no-preds">
                    未检出预测目标
                  </div>
                </div>
              </el-collapse-item>
            </el-collapse>
          </div>
        </div>
      </el-aside>
      
      <!-- 右侧主展示区 -->
      <el-main class="main-content">
        <el-card class="custom-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <span>3D 目标检测结果</span>
            </div>
          </template>
          
          <!-- 图片显示区域 -->
          <div v-if="imageCombined && imagePred && imageBev" class="image-container">
            <el-card class="image-card" shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>真值与预测 (GT + Pred)</span>
                </div>
              </template>
              <el-image
                :src="imageCombined"
                :preview-src-list="[imageCombined]"
                :preview-teleported="true"
                fit="contain"
                hide-on-click-modal
                class="result-image"
              />
            </el-card>
            
            <el-card class="image-card" shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>仅预测 (Pred Only)</span>
                </div>
              </template>
              <el-image
                :src="imagePred"
                :preview-src-list="[imagePred]"
                :preview-teleported="true"
                fit="contain"
                hide-on-click-modal
                class="result-image"
              />
              <div style="padding-top: 8px; font-size: 12px; color: #909399;">
                左图为最匹配 GT，右图为最高分 Pred（放大显示）
              </div>
            </el-card>
            
            <el-card class="image-card" shadow="hover">
              <template #header>
                <div class="card-header">
                  <span>上帝视角鸟瞰图 (BEV)</span>
                </div>
              </template>
              <el-image
                :src="imageBev"
                :preview-src-list="[imageBev]"
                :preview-teleported="true"
                fit="contain"
                hide-on-click-modal
                class="result-image"
              />
            </el-card>
          </div>
          
          <!-- 空状态 -->
          <div v-else class="empty-state">
            <el-empty description="等待指令：请选择场景并点击开始检测" />
          </div>
        </el-card>
      </el-main>
    </el-container>
    
    <!-- 全局加载状态 -->
    <div v-if="isLoading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <div>AI 模型正在进行多视角特征融合与 3D 投影...</div>
    </div>
  </el-container>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import { ElNotification, ElMessageBox } from 'element-plus'

// 状态管理
const confidenceThreshold = ref(0.5)
const mode = ref('random')
const sampleIndex = ref(0)
const isLoading = ref(false)
const imageCombined = ref('')
const imagePred = ref('')
const imageBev = ref('')
const detectionCount = ref(0)
const inferenceTime = ref(0)
const currentSampleId = ref('')
const backendStatus = ref(false)
const deviceType = ref('CPU')
const gtTotal = ref(0)
const predDetails = ref({})
const activeNames = ref(['1'])
const datasetSize = ref(0)
let healthTimer = null

// 检查后端状态
const checkBackendStatus = async () => {
  try {
    const response = await axios.get('http://127.0.0.1:8000/health')
    const response = await axios.get('http://127.0.0.1:8000/health')
    backendStatus.value = true
    deviceType.value = response.data.device || 'CPU'
    datasetSize.value = response.data.dataset_size || 0
  } catch (error) {
    backendStatus.value = false
  }
}

// 开始检测
const startDetection = async () => {
  isLoading.value = true
  
  try {
    // 构建请求参数
    const params = {
      mode: mode.value,
      confidence_threshold: confidenceThreshold.value
    }
    
    if (mode.value === 'specific') {
      params.sample_index = sampleIndex.value
    }
    
    // 发送请求
    const response = await axios.post('http://127.0.0.1:8000/api/predict', params)
    
    // 处理响应
    if (response.data && response.data.status === 'success' && response.data.image_combined && response.data.image_pred && response.data.image_bev) {
      // 直接使用后端返回的完整 base64 字符串
      imageCombined.value = response.data.image_combined
      imagePred.value = response.data.image_pred
      imageBev.value = response.data.image_bev
      
      // 更新统计信息
      if (response.data.stats) {
        gtTotal.value = response.data.stats.gt_total || 0
        predDetails.value = response.data.stats.pred_details || {}
        detectionCount.value = response.data.stats.pred_total || 0
        inferenceTime.value = response.data.stats.latency_ms || 0
        currentSampleId.value = response.data.stats.sample_token || ''
      } else {
        gtTotal.value = 0
        predDetails.value = {}
        detectionCount.value = 0
        inferenceTime.value = 0
        currentSampleId.value = ''
      }
      
      // 显示通知
      const message = predDetails.value && Object.keys(predDetails.value).length > 0 ? '检测成功' : '未检出预测目标'
      ElNotification({
        title: '成功',
        message: message,
        type: 'success',
        duration: 3000
      })
    } else {
      throw new Error(response.data.error || '检测失败')
    }
  } catch (error) {
    // 显示错误信息
    ElMessageBox.alert(
      error.message || '后端未启动或权重文件丢失，请检查后端服务',
      '检测失败',
      {
        confirmButtonText: '确定',
        type: 'error'
      }
    )
  } finally {
    isLoading.value = false
  }
}

// 获取类别图标
const getCategoryIcon = (category) => {
  const iconMap = {
    car: '🚗',
    truck: '🚛',
    bus: '🚌',
    trailer: '🚚',
    construction_vehicle: '🏗️',
    pedestrian: '🚶',
    motorcycle: '🏍️',
    bicycle: '🚲',
    traffic_cone: '⛑️',
    barrier: '🚧'
  }
  return iconMap[category] || '📦'
}

// 组件挂载时检查后端状态
onMounted(() => {
  checkBackendStatus()
  // 每 10 秒检查一次后端状态
  healthTimer = setInterval(checkBackendStatus, 10000)
})

// 组件卸载时清理
onUnmounted(() => {
  if (healthTimer) {
    clearInterval(healthTimer)
    healthTimer = null
  }
})
</script>

<style scoped>
.app-container {
  height: 100vh;
  width: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.header {
  height: 60px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 30px;
  box-sizing: border-box;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 15px;
}

.header-icon {
  font-size: 28px;
  color: var(--primary-color);
}

.header-title {
  font-size: 20px;
  font-weight: 600;
  margin: 0;
  color: var(--text-primary);
}

.header-right {
  display: flex;
  gap: 30px;
  align-items: center;
}

.status-item {
  display: flex;
  align-items: center;
  font-size: 14px;
  color: var(--text-secondary);
}

.sidebar {
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-color);
  padding: 20px;
  overflow-y: auto;
  height: calc(100vh - 60px);
}

.main-content {
  background: var(--bg-primary);
  padding: 20px;
  overflow-y: auto;
  height: calc(100vh - 60px);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  color: var(--text-primary);
}

.slider-value {
  text-align: center;
  margin-top: 10px;
  color: var(--primary-color);
  font-weight: 500;
}

.mode-radio-group {
  width: 100%;
}

.start-button {
  width: 100%;
  margin-top: 20px;
  font-size: 16px;
  padding: 12px;
  background: var(--primary-color);
  border-color: var(--primary-color);
}

.start-button:hover {
  background: var(--secondary-color);
  border-color: var(--secondary-color);
}

.stats-container {
  margin-top: 30px;
}

.stats-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 15px;
  color: var(--text-primary);
}

.image-container {
  display: flex;
  flex-direction: column;
  gap: 30px;
  padding: 10px;
}

.image-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  box-shadow: var(--shadow);
  transition: var(--transition);
}

.image-card:hover {
  box-shadow: var(--glow), var(--shadow);
  border-color: var(--primary-color);
}

.result-image {
  width: 100%;
  max-height: calc(50vh - 120px);
  cursor: pointer;
  transition: transform 0.3s ease;
}

.result-image:hover {
  transform: scale(1.02);
}

/* 预测类别明细样式 */
.pred-details-container {
  margin-top: 30px;
}

.pred-details-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 15px;
  color: var(--text-primary);
}

.pred-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  padding: 15px;
  background: var(--bg-tertiary);
  border-radius: 8px;
  border: 1px solid var(--border-color);
}

.pred-tag {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 20px;
  font-size: 14px;
  transition: var(--transition);
}

.pred-tag:hover {
  border-color: var(--primary-color);
  box-shadow: var(--glow);
  transform: translateY(-2px);
}

.tag-icon {
  font-size: 16px;
}

.tag-name {
  color: var(--text-primary);
  font-weight: 500;
}

.tag-count {
  color: var(--primary-color);
  font-weight: 600;
}

.no-preds {
  color: var(--text-tertiary);
  font-style: italic;
  padding: 20px;
  text-align: center;
  width: 100%;
}

/* 折叠面板样式 */
.el-collapse {
  border: 1px solid var(--border-color);
  border-radius: 8px;
  overflow: hidden;
}

.el-collapse-item {
  border-bottom: 1px solid var(--border-color);
}

.el-collapse-item:last-child {
  border-bottom: none;
}

.el-collapse-item__header {
  background: var(--bg-secondary);
  color: var(--text-primary);
  border-bottom: 1px solid var(--border-color);
}

.el-collapse-item__content {
  background: var(--bg-tertiary);
  border-top: 1px solid var(--border-color);
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .sidebar {
    width: 320px !important;
  }
}

@media (max-width: 768px) {
  .app-container {
    flex-direction: column;
  }
  
  .header {
    padding: 0 20px;
  }
  
  .header-title {
    font-size: 16px;
  }
  
  .sidebar {
    width: 100% !important;
    height: auto !important;
    border-right: none;
    border-bottom: 1px solid var(--border-color);
  }
  
  .main-content {
    margin-left: 0 !important;
    height: calc(100vh - 60px);
  }
  
  .pred-tags {
    flex-direction: column;
  }
  
  .pred-tag {
    width: 100%;
    justify-content: space-between;
  }
}
</style>
