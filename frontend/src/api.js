import axios from 'axios'

// Configure axios base URL for development
const isDev = import.meta.env.DEV
const baseURL = isDev ? 'http://localhost:8000' : ''

const api = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export default api
