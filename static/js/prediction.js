document.addEventListener('DOMContentLoaded', function () {
  const stateSelect = document.getElementById('stateSelect');
  const yearInput = document.getElementById('yearInput');
  const monthInput = document.getElementById('monthInput');
  const selectedStateLabel = document.getElementById('selectedStateLabel');
  const stateEligibilityText = document.getElementById('stateEligibilityText');
  const nextEligibleLabel = document.getElementById('nextEligibleLabel');
  const form = document.getElementById('predictionForm');
  const submitBtn = document.getElementById('submitBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnIcon = document.getElementById('btnIcon');
  const btnText = document.getElementById('btnText');

  if (stateSelect) {
    stateSelect.addEventListener('change', function () {
      const state = this.value;
      if (selectedStateLabel) selectedStateLabel.textContent = state;

      fetch(`/api/predict/eligible/${encodeURIComponent(state)}`)
        .then(response => response.json())
        .then(data => {
          if (data && data.next_year) {
            if (yearInput) yearInput.value = data.next_year;
            if (monthInput) monthInput.value = data.next_month;
            if (nextEligibleLabel) {
              const moStr = String(data.next_month).padStart(2, '0');
              nextEligibleLabel.textContent = `${data.next_year}-${moStr}`;
            }
            if (stateEligibilityText) {
              const moStr = String(data.latest_month).padStart(2, '0');
              stateEligibilityText.innerHTML = `Latest recorded observation for <strong>${state}</strong>: <code>${data.latest_year}-${moStr}</code>`;
            }
            
            const horizonWarningBox = document.getElementById('horizonWarningBox');
            const horizonWarningText = document.getElementById('horizonWarningText');
            if (horizonWarningBox && horizonWarningText) {
              if (data.horizon_limit_reached) {
                horizonWarningBox.classList.remove('d-none');
                horizonWarningText.innerHTML = `<strong>Maximum Horizon Limit:</strong> Forecasts are capped at 3 months ahead of historical surveillance records.`;
                if (submitBtn) submitBtn.disabled = true;
              } else if (data.horizon_warning) {
                horizonWarningBox.classList.remove('d-none');
                horizonWarningText.innerHTML = `<strong>Chained Forecast Warning:</strong> This forecast is <strong>${data.horizon_months} months ahead</strong> of historical records. Recursive accuracy decreases with horizon length. See <a href="/evaluation#horizonSection" class="alert-link">horizon benchmarks</a>.`;
                if (submitBtn) submitBtn.disabled = false;
              } else {
                horizonWarningBox.classList.add('d-none');
                if (submitBtn) submitBtn.disabled = false;
              }
            }
          }
        })
        .catch(err => {
          console.error('Failed to fetch eligible period for state:', err);
        });
    });
  }

  if (form) {
    form.addEventListener('submit', function () {
      if (submitBtn) {
        submitBtn.disabled = true;
        if (btnSpinner) btnSpinner.classList.remove('d-none');
        if (btnIcon) btnIcon.classList.add('d-none');
        if (btnText) btnText.textContent = 'Generating Forecast...';
      }
    });
  }
});

