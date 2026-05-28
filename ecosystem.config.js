module.exports = {
  apps: [{
    name: 'taurus',
    script: 'server.js',
    cwd: '/var/www/taurus-app',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '400M',
    env: { NODE_ENV: 'production' },
    error_file: '/var/log/taurus-error.log',
    out_file: '/var/log/taurus-out.log'
  }]
};
