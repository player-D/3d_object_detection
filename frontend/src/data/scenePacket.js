const fallbackPacket = {
  schema_version: '1.0.0',
  frame_id: 'demo-frame',
  timestamp_ms: Date.now(),
  ego: {
    id: 'ego',
    speed_mps: 11.8,
    speed_kph: 42.5,
    heading_rad: 0,
    threat_level: 'low',
    box3d: {
      center: { x: 0, y: 0, z: 0.75 },
      size: { length: 4.86, width: 1.92, height: 1.55 },
      yaw: 0,
    },
  },
  lanes: {
    centerlines: [
      laneFromOffset('ego-center', 'centerline', 0),
      laneFromOffset('left-adjacent', 'centerline', 3.6),
      laneFromOffset('right-adjacent', 'centerline', -3.6),
    ],
    boundaries: [
      laneFromOffset('left-boundary', 'boundary', 7.2),
      laneFromOffset('right-boundary', 'boundary', -7.2),
    ],
    corridor: buildCorridor(),
  },
  objects: [
    {
      id: 'gt-car-001',
      source: 'gt',
      class: 'car',
      label: 'GT car',
      score: null,
      threat_level: 'medium',
      box3d: {
        center: { x: 18.4, y: 0.4, z: 0.9 },
        size: { length: 4.72, width: 1.88, height: 1.58 },
        yaw: 0.05,
      },
    },
    {
      id: 'pred-car-001',
      source: 'pred',
      class: 'car',
      label: 'Pred car',
      score: 0.94,
      threat_level: 'high',
      box3d: {
        center: { x: 15.6, y: -0.2, z: 0.9 },
        size: { length: 4.68, width: 1.9, height: 1.6 },
        yaw: 0.03,
      },
    },
    {
      id: 'pred-ped-002',
      source: 'pred',
      class: 'pedestrian',
      label: 'Pred ped',
      score: 0.82,
      threat_level: 'medium',
      box3d: {
        center: { x: 11.4, y: 3.4, z: 0.86 },
        size: { length: 0.7, width: 0.7, height: 1.74 },
        yaw: 1.57,
      },
    },
  ],
  summary: {
    gt_count: 1,
    pred_count: 2,
    threat_counts: { high: 1, medium: 1, low: 0 },
    max_threat_level: 'high',
  },
}

function laneFromOffset(id, kind, offset) {
  return {
    id,
    kind,
    points: Array.from({ length: 20 }, (_, index) => ({
      x: -6 + index * 4,
      y: offset,
      z: 0,
    })),
  }
}

function buildCorridor() {
  return Array.from({ length: 21 }, (_, index) => {
    const x = index * 3
    const width = 1.7 + x * 0.055
    return {
      x,
      left_y: width,
      right_y: -width,
    }
  })
}

function clone(value) {
  return JSON.parse(JSON.stringify(value))
}

function normalizePoint(point = {}) {
  return {
    x: Number.isFinite(point.x) ? point.x : 0,
    y: Number.isFinite(point.y) ? point.y : 0,
    z: Number.isFinite(point.z) ? point.z : 0,
  }
}

function normalizeLane(lane = {}, index = 0) {
  return {
    id: lane.id || `lane-${index}`,
    kind: lane.kind || 'centerline',
    points: Array.isArray(lane.points) ? lane.points.map(normalizePoint) : [],
  }
}

function normalizeObject(object = {}, index = 0) {
  const box3d = object.box3d || {}
  const center = normalizePoint(box3d.center)
  const size = box3d.size || {}
  return {
    id: object.id || `object-${index}`,
    source: object.source === 'gt' ? 'gt' : 'pred',
    class: object.class || 'unknown',
    label: object.label || object.class || `obj-${index}`,
    score: Number.isFinite(object.score) ? object.score : null,
    threat_level: ['high', 'medium', 'low'].includes(object.threat_level) ? object.threat_level : 'low',
    box3d: {
      center,
      size: {
        length: Number.isFinite(size.length) ? size.length : 1,
        width: Number.isFinite(size.width) ? size.width : 1,
        height: Number.isFinite(size.height) ? size.height : 1,
      },
      yaw: Number.isFinite(box3d.yaw) ? box3d.yaw : 0,
    },
  }
}

export function createFallbackScenePacket() {
  return clone(fallbackPacket)
}

export function normalizeScenePacket(packet) {
  if (!packet || typeof packet !== 'object') {
    return createFallbackScenePacket()
  }

  const fallback = createFallbackScenePacket()
  const lanes = packet.lanes || {}

  const normalized = {
    schema_version: packet.schema_version || fallback.schema_version,
    frame_id: packet.frame_id || fallback.frame_id,
    timestamp_ms: Number.isFinite(packet.timestamp_ms) ? packet.timestamp_ms : fallback.timestamp_ms,
    ego: {
      ...fallback.ego,
      ...(packet.ego || {}),
      speed_mps: Number.isFinite(packet?.ego?.speed_mps) ? packet.ego.speed_mps : fallback.ego.speed_mps,
      speed_kph: Number.isFinite(packet?.ego?.speed_kph) ? packet.ego.speed_kph : fallback.ego.speed_kph,
      heading_rad: Number.isFinite(packet?.ego?.heading_rad) ? packet.ego.heading_rad : fallback.ego.heading_rad,
      box3d: {
        center: normalizePoint(packet?.ego?.box3d?.center || fallback.ego.box3d.center),
        size: {
          length: Number.isFinite(packet?.ego?.box3d?.size?.length) ? packet.ego.box3d.size.length : fallback.ego.box3d.size.length,
          width: Number.isFinite(packet?.ego?.box3d?.size?.width) ? packet.ego.box3d.size.width : fallback.ego.box3d.size.width,
          height: Number.isFinite(packet?.ego?.box3d?.size?.height) ? packet.ego.box3d.size.height : fallback.ego.box3d.size.height,
        },
        yaw: Number.isFinite(packet?.ego?.box3d?.yaw) ? packet.ego.box3d.yaw : fallback.ego.box3d.yaw,
      },
    },
    lanes: {
      centerlines: Array.isArray(lanes.centerlines) && lanes.centerlines.length
        ? lanes.centerlines.map(normalizeLane)
        : fallback.lanes.centerlines,
      boundaries: Array.isArray(lanes.boundaries) && lanes.boundaries.length
        ? lanes.boundaries.map(normalizeLane)
        : fallback.lanes.boundaries,
      corridor: Array.isArray(lanes.corridor) && lanes.corridor.length
        ? lanes.corridor.map((item) => ({
            x: Number.isFinite(item.x) ? item.x : 0,
            left_y: Number.isFinite(item.left_y) ? item.left_y : 0,
            right_y: Number.isFinite(item.right_y) ? item.right_y : 0,
          }))
        : fallback.lanes.corridor,
    },
    objects: Array.isArray(packet.objects) && packet.objects.length
      ? packet.objects.map(normalizeObject)
      : fallback.objects,
    summary: {
      gt_count: Number.isFinite(packet?.summary?.gt_count) ? packet.summary.gt_count : fallback.summary.gt_count,
      pred_count: Number.isFinite(packet?.summary?.pred_count) ? packet.summary.pred_count : fallback.summary.pred_count,
      threat_counts: {
        high: Number.isFinite(packet?.summary?.threat_counts?.high) ? packet.summary.threat_counts.high : fallback.summary.threat_counts.high,
        medium: Number.isFinite(packet?.summary?.threat_counts?.medium) ? packet.summary.threat_counts.medium : fallback.summary.threat_counts.medium,
        low: Number.isFinite(packet?.summary?.threat_counts?.low) ? packet.summary.threat_counts.low : fallback.summary.threat_counts.low,
      },
      max_threat_level: ['high', 'medium', 'low'].includes(packet?.summary?.max_threat_level)
        ? packet.summary.max_threat_level
        : fallback.summary.max_threat_level,
    },
  }

  return normalized
}

export function formatThreatLabel(threatLevel) {
  return {
    high: '危险',
    medium: '注意',
    low: '正常',
  }[threatLevel] || '正常'
}

export function formatClassLabel(className) {
  return {
    car: '汽车',
    truck: '卡车',
    bus: '公交',
    trailer: '挂车',
    construction_vehicle: '工程车',
    pedestrian: '行人',
    motorcycle: '摩托车',
    bicycle: '自行车',
    traffic_cone: '锥桶',
    barrier: '隔离栏',
  }[className] || className
}
