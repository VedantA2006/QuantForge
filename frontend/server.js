const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Proxy /api requests to the backend (bypasses mixed content)
app.use('/api', createProxyMiddleware({
  target: API_URL,
  changeOrigin: true,
}));

// Serve React static build
app.use(express.static(path.join(__dirname, 'build')));

// All other routes → index.html (SPA)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`QuantForge Dashboard running on port ${PORT}`);
  console.log(`Proxying /api → ${API_URL}`);
});
