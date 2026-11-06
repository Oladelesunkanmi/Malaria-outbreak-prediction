document.addEventListener('DOMContentLoaded', function () {
  fetch('/api/dashboard/charts')
    .then(response => response.json())
    .then(data => {
      // 1. Monthly Trend Chart
      const monthlyCtx = document.getElementById('monthlyTrendChart');
      if (monthlyCtx && data.monthly_trend) {
        new Chart(monthlyCtx, {
          type: 'line',
          data: {
            labels: data.monthly_trend.labels,
            datasets: data.monthly_trend.datasets
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: { position: 'top', labels: { boxWidth: 12, font: { size: 11 } } },
              tooltip: {
                callbacks: {
                  label: function (ctx) {
                    return `${ctx.dataset.label}: ${ctx.parsed.y.toLocaleString()} cases`;
                  }
                }
              }
            },
            scales: {
              x: {
                grid: { display: false },
                ticks: {
                  maxTicksLimit: 12,
                  font: { size: 10 }
                }
              },
              y: {
                beginAtZero: true,
                ticks: {
                  callback: function (val) {
                    return (val / 1000).toFixed(0) + 'k';
                  },
                  font: { size: 10 }
                }
              }
            }
          }
        });
      }

      // 2. State Distribution Doughnut Chart
      const stateCtx = document.getElementById('stateDistributionChart');
      if (stateCtx && data.state_distribution) {
        new Chart(stateCtx, {
          type: 'doughnut',
          data: {
            labels: data.state_distribution.labels,
            datasets: [{
              data: data.state_distribution.data,
              backgroundColor: data.state_distribution.backgroundColor,
              borderWidth: 2,
              borderColor: '#ffffff'
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } },
              tooltip: {
                callbacks: {
                  label: function (ctx) {
                    const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                    const val = ctx.parsed;
                    const pct = ((val / total) * 100).toFixed(1);
                    return ` ${ctx.label}: ${val.toLocaleString()} (${pct}%)`;
                  }
                }
              }
            },
            cutout: '65%'
          }
        });
      }

      // 3. Yearly Trend Bar Chart
      const yearlyCtx = document.getElementById('yearlyTrendChart');
      if (yearlyCtx && data.yearly_trend) {
        new Chart(yearlyCtx, {
          type: 'bar',
          data: {
            labels: data.yearly_trend.labels,
            datasets: data.yearly_trend.datasets
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: 'top', labels: { boxWidth: 12, font: { size: 11 } } },
              tooltip: {
                callbacks: {
                  label: function (ctx) {
                    return `${ctx.dataset.label}: ${ctx.parsed.y.toLocaleString()} cases`;
                  }
                }
              }
            },
            scales: {
              x: {
                grid: { display: false },
                ticks: { font: { size: 10 } }
              },
              y: {
                beginAtZero: true,
                ticks: {
                  callback: function (val) {
                    return (val / 1000).toFixed(0) + 'k';
                  },
                  font: { size: 10 }
                }
              }
            }
          }
        });
      }
    })
    .catch(err => {
      console.error('Error loading dashboard charts:', err);
    });
});

