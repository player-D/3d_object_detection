import axios from 'axios'

const http = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
  timeout: 120000,
})

export async function fetchHealth() {
  const { data } = await http.get('/health')
  return data
}

export async function requestPrediction(payload) {
  const { data } = await http.post('/api/predict', payload)
  return data
}

export { http }
