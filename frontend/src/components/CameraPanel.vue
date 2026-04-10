<template>
  <section class="camera-panel">
    <div class="panel-copy">
      <div>
        <p class="eyebrow">{{ title }}</p>
        <h3>{{ subtitle }}</h3>
      </div>
      <span v-if="status" class="status-pill">{{ status }}</span>
    </div>

    <div class="image-shell" :class="{ empty: !imageSrc }">
      <el-image
        v-if="imageSrc"
        :src="imageSrc"
        :preview-src-list="[imageSrc]"
        fit="contain"
        hide-on-click-modal
        preview-teleported
        class="camera-image"
      />
      <div v-else class="empty-copy">
        <span>等待本次推理结果</span>
      </div>
    </div>
  </section>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    default: '',
  },
  subtitle: {
    type: String,
    default: '',
  },
  imageSrc: {
    type: String,
    default: '',
  },
  status: {
    type: String,
    default: '',
  },
})
</script>

<style scoped>
.camera-panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.panel-copy {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.eyebrow {
  margin: 0 0 6px;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-muted);
}

h3 {
  margin: 0;
  font-size: 17px;
  font-weight: 600;
  color: var(--text-primary);
}

.status-pill {
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(117, 194, 255, 0.12);
  border: 1px solid rgba(117, 194, 255, 0.24);
  color: var(--accent-ice);
  font-size: 12px;
}

.image-shell {
  min-height: 310px;
  border-radius: 22px;
  overflow: hidden;
  border: 1px solid rgba(142, 176, 209, 0.16);
  background:
    linear-gradient(180deg, rgba(15, 31, 51, 0.72), rgba(7, 17, 28, 0.9)),
    radial-gradient(circle at top, rgba(84, 154, 255, 0.12), transparent 42%);
}

.image-shell.empty {
  display: grid;
  place-items: center;
}

.camera-image {
  width: 100%;
  height: 100%;
  min-height: 310px;
}

.empty-copy {
  color: var(--text-muted);
  font-size: 14px;
}
</style>
