import { createApp } from 'vue'
import App from './App.vue'
import './style.css'

import {
  ElButton,
  ElForm,
  ElFormItem,
  ElImage,
  ElInputNumber,
  ElRadioButton,
  ElRadioGroup,
  ElSlider,
} from 'element-plus'

import 'element-plus/es/components/button/style/css'
import 'element-plus/es/components/form/style/css'
import 'element-plus/es/components/form-item/style/css'
import 'element-plus/es/components/image/style/css'
import 'element-plus/es/components/input-number/style/css'
import 'element-plus/es/components/radio-button/style/css'
import 'element-plus/es/components/radio-group/style/css'
import 'element-plus/es/components/slider/style/css'
import 'element-plus/es/components/message-box/style/css'
import 'element-plus/es/components/notification/style/css'

const app = createApp(App)

app.component('ElButton', ElButton)
app.component('ElForm', ElForm)
app.component('ElFormItem', ElFormItem)
app.component('ElImage', ElImage)
app.component('ElInputNumber', ElInputNumber)
app.component('ElRadioButton', ElRadioButton)
app.component('ElRadioGroup', ElRadioGroup)
app.component('ElSlider', ElSlider)

app.mount('#app')
