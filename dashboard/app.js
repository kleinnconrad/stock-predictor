document.addEventListener('DOMContentLoaded', () => {
  let reportData = [];
  let executionDate = null;
  let parameters = {};
  
  let currentSort = { column: 'final_prediction', asc: true };

  const tableBodyOther = document.getElementById('table-body');
  const tableBodyBuys = document.getElementById('buys-table-body');
  const searchInput = document.getElementById('search-input');
  const filterSelect = document.getElementById('filter-status');
  const ths = document.querySelectorAll('th[data-sort]');

  const kpiTotal = document.getElementById('kpi-total');
  const kpiBuys = document.getElementById('kpi-buys');
  const kpiUp = document.getElementById('kpi-up');
  const kpiNotUp = document.getElementById('kpi-notup');
  const kpiAcc = document.getElementById('kpi-acc');
  const kpiFailedStep2 = document.getElementById('kpi-failed-step2');
  const kpiMissingData = document.getElementById('kpi-missing-data');
  const kpiSuppressed = document.getElementById('kpi-suppressed');
  let suppressedCount = 0;
  
  const metaDate = document.getElementById('meta-date');
  const metaParams = document.getElementById('meta-params');

  // Determine correct data URL based on environment (local vs GitHub Pages)
  const isGitHubPages = window.location.hostname.includes('github.io');
  const dataUrl = isGitHubPages ? './data/full_batch_report.json' : '../data/processed/full_batch_report.json';

  // Fetch data
  fetch(dataUrl)
    .then(response => {
      if (!response.ok) throw new Error("Network response was not ok");
      return response.text();
    })
    .then(text => {
      // Python's json.dump writes float('inf') as Infinity which breaks JSON.parse
      const safeText = text.replace(/(:\s*)(Infinity|-Infinity|NaN)\b/g, '$1"$2"');
      return JSON.parse(safeText);
    })
    .then(data => {
      if (Array.isArray(data)) {
         // Fallback for old data
         reportData = data;
      } else {
         reportData = data.predictions || [];
         executionDate = data.execution_date;
         parameters = data.parameters || {};
      }
      
      const originalLength = reportData.length;
      reportData = reportData.filter(d => {
        if (typeof d.latest_price === 'number' && d.latest_price <= 10) return false;
        return true;
      });
      suppressedCount = originalLength - reportData.length;

      renderMetadata();
      calculateKPIs();
      renderTables();
      renderVariables();
    })
    .catch(error => {
      console.error("Failed to load JSON data", error);
      const errorHtml = `<tr><td colspan="5" style="text-align:center; color: var(--status-red);">Failed to load data. Ensure full_batch_report.json is available.</td></tr>`;
      tableBodyOther.innerHTML = errorHtml;
      tableBodyBuys.innerHTML = errorHtml;
    });

  function renderMetadata() {
    if (executionDate) {
      const d = new Date(executionDate);
      metaDate.textContent = d.toLocaleString() + " UTC";
    } else {
      metaDate.textContent = "N/A (Legacy Data)";
    }
    
    metaParams.innerHTML = '';
    for (const [key, value] of Object.entries(parameters)) {
      const badge = document.createElement('div');
      badge.className = 'param-badge';
      badge.innerHTML = `<strong>${key}:</strong> ${value}`;
      metaParams.appendChild(badge);
    }
  }

  function calculateKPIs() {
    kpiTotal.textContent = reportData.length;
    
    const buys = reportData.filter(d => d.final_prediction === 'UP_FINAL_BUY').length;
    kpiBuys.textContent = buys;
    
    const up = reportData.filter(d => d.final_prediction === 'UP').length;
    kpiUp.textContent = up;
    
    const notUp = reportData.filter(d => d.final_prediction === 'NOT_UP').length;
    kpiNotUp.textContent = notUp;

    const failedStep2 = reportData.filter(d => d.final_prediction === 'UP' && (d.step2_model && d.step2_model.feature_diagnostics)).length;
    if (kpiFailedStep2) kpiFailedStep2.textContent = failedStep2;

    const missingData = reportData.filter(d => d.final_prediction === 'UP' && (!d.step2_model || !d.step2_model.feature_diagnostics)).length;
    kpiMissingData.textContent = missingData;

    if (kpiSuppressed) kpiSuppressed.textContent = suppressedCount;

    const accuracies = reportData
      .filter(d => d.step1_model && d.step1_model.cv_accuracy)
      .map(d => d.step1_model.cv_accuracy);
    
    if (accuracies.length > 0) {
      const avg = accuracies.reduce((a, b) => a + b, 0) / accuracies.length;
      kpiAcc.textContent = (avg * 100).toFixed(1) + '%';
    } else {
      kpiAcc.textContent = 'N/A';
    }
  }

  function getStatusHtml(status) {
    if (status === 'UP_FINAL_BUY') {
      return `<span class="status-pill status-up-final">UP_FINAL_BUY</span>`;
    } else if (status === 'UP') {
      return `<span class="status-pill status-up">UP</span>`;
    } else {
      return `<span class="status-pill status-not-up">${status || 'NOT_UP'}</span>`;
    }
  }

  function createRowHtml(d) {
    const stockName = d.stock_name || 'UNKNOWN';
    const companyName = d.company_name ? d.company_name : '';
    const companyDesc = d.company_description ? d.company_description : '';
    const priceStr = typeof d.latest_price === 'number' ? `€${d.latest_price.toFixed(2)}` : 'N/A';
    const acc = d.step1_model?.cv_accuracy ? (d.step1_model.cv_accuracy * 100).toFixed(1) : 0;
    const ksRaw = d.step1_model?.ks_cutoff;
    const ks = typeof ksRaw === 'number' ? ksRaw.toFixed(3) : (ksRaw || 'N/A');
    const step1Class = d.step1_model?.predicted_class || 'N/A';

    let priceHtml = `<td>${priceStr}</td>`;
    if (typeof d.latest_price === 'number' && d.latest_price <= 30) {
      priceHtml = `<td style="color: var(--status-red); font-weight: bold;" title="Warning: Small price differences change the model outcome completely. Robust prediction not possible for stocks <= 30€.">
        ${priceStr} [!WARN]
      </td>`;
    }

    return `
      <td class="ticker-name">
        <div style="font-weight: bold;">${stockName}</div>
        ${companyName ? `<div style="font-size: 0.85em; color: var(--text-muted); margin-top: 4px; font-weight: normal;">${companyName}</div>` : ''}
        ${companyDesc ? `<div style="font-size: 0.75em; color: var(--text-muted); opacity: 0.8; margin-top: 2px; font-weight: normal; max-width: 300px; white-space: normal;">${companyDesc}</div>` : ''}
      </td>
      ${priceHtml}
      <td>${getStatusHtml(d.final_prediction)}</td>
      <td>${getStatusHtml(step1Class)}</td>
      <td>
        ${acc > 0 ? `${acc}% <div class="acc-bar-bg"><div class="acc-bar-fill" style="width: ${acc}%"></div></div>` : 'N/A'}
      </td>
      <td>${ks}</td>
    `;
  }

  function renderTables() {
    const searchTerm = searchInput.value.toLowerCase();
    const filterStatus = filterSelect.value;

    let filtered = reportData.filter(d => {
      const stockName = d.stock_name || 'UNKNOWN';
      const matchSearch = stockName.toLowerCase().includes(searchTerm);
      const matchStatus = filterStatus === 'ALL' || d.final_prediction === filterStatus;
      return matchSearch && matchStatus;
    });

    // Split into buys and others
    let buysList = filtered.filter(d => d.final_prediction === 'UP_FINAL_BUY');
    let otherList = filtered.filter(d => d.final_prediction !== 'UP_FINAL_BUY');

    // Sorting logic
    const sortFn = (a, b) => {
      let valA, valB;
      const col = currentSort.column;
      
      if (col === 'stock_name') {
        valA = a.stock_name || 'UNKNOWN';
        valB = b.stock_name || 'UNKNOWN';
      } else if (col === 'latest_price') {
        valA = typeof a.latest_price === 'number' ? a.latest_price : -1;
        valB = typeof b.latest_price === 'number' ? b.latest_price : -1;
      } else if (col === 'final_prediction') {
        valA = a.final_prediction || '';
        valB = b.final_prediction || '';
      } else if (col === 'step1_class') {
        valA = a.step1_model?.predicted_class || '';
        valB = b.step1_model?.predicted_class || '';
      } else if (col === 'cv_accuracy') {
        valA = a.step1_model?.cv_accuracy || 0;
        valB = b.step1_model?.cv_accuracy || 0;
      } else if (col === 'ks_cutoff') {
        valA = a.step1_model?.ks_cutoff || 0;
        valB = b.step1_model?.ks_cutoff || 0;
      }

      if (valA < valB) return currentSort.asc ? -1 : 1;
      if (valA > valB) return currentSort.asc ? 1 : -1;
      return 0;
    };

    buysList.sort(sortFn);
    otherList.sort(sortFn);

    // Render Buys Table
    tableBodyBuys.innerHTML = '';
    if (buysList.length === 0) {
      tableBodyBuys.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--text-muted);">No confirmed buy candidates match filters.</td></tr>`;
    } else {
      buysList.forEach(d => {
        const tr = document.createElement('tr');
        tr.innerHTML = createRowHtml(d);
        tableBodyBuys.appendChild(tr);
      });
    }

    // Render Other Table
    tableBodyOther.innerHTML = '';
    if (otherList.length === 0) {
      tableBodyOther.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--text-muted);">No other stocks match filters.</td></tr>`;
    } else {
      otherList.forEach(d => {
        const tr = document.createElement('tr');
        tr.innerHTML = createRowHtml(d);
        tableBodyOther.appendChild(tr);
      });
    }

    // Update sort icons
    ths.forEach(th => {
      const icon = th.querySelector('.sort-icon');
      if (th.dataset.sort === currentSort.column) {
        icon.textContent = currentSort.asc ? '▲' : '▼';
      } else {
        icon.textContent = '';
      }
    });
  }

  // Event Listeners
  searchInput.addEventListener('input', renderTables);
  filterSelect.addEventListener('change', renderTables);

  ths.forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.sort;
      if (currentSort.column === col) {
        currentSort.asc = !currentSort.asc;
      } else {
        currentSort.column = col;
        currentSort.asc = true; // default asc for new column
      }
      renderTables();
    });
  });

  // Event Listeners for variables
  const varSearchInput = document.getElementById('var-search-input');
  if (varSearchInput) {
    varSearchInput.addEventListener('input', renderVariables);
  }

  function renderVariables() {
    const container = document.getElementById('variables-container');
    if (!container) return;
    
    const search = varSearchInput ? varSearchInput.value.toLowerCase() : '';
    
    let html = '';
    reportData.forEach(d => {
      const stockName = d.stock_name || 'UNKNOWN';
      if (search && !stockName.toLowerCase().includes(search)) return;
      
      let step1Html = '';
      if (d.step1_model && d.step1_model.selected_predictors_and_weights) {
        let items = '';
        for (const [k, v] of Object.entries(d.step1_model.selected_predictors_and_weights)) {
          items += `<div class="var-item"><span class="var-key">${k}</span><span class="var-value">${Number(v).toFixed(4)}</span></div>`;
        }
        step1Html = `<h4>Step 1: Macro Predictors & Weights</h4><div class="var-grid">${items}</div>`;
      }
      
      let step2Html = '';
      if (d.step2_model && d.step2_model.feature_diagnostics) {
        let items = '';
        for (const [k, v] of Object.entries(d.step2_model.feature_diagnostics)) {
          if (typeof v === 'object' && v !== null) {
            items += `<div style="grid-column: 1 / -1; margin-top: 1rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; margin-bottom: 0.5rem;"><strong>${k}</strong></div>`;
            for (const [subK, subV] of Object.entries(v)) {
              let disp = typeof subV === 'number' ? Number(subV).toFixed(2) : subV;
              items += `<div class="var-item" style="padding-left: 1rem;"><span class="var-key">${subK}</span><span class="var-value" style="font-weight: bold;">${disp}</span></div>`;
            }
          } else {
            let disp = v;
            if (typeof v === 'boolean') {
              disp = v ? '<span style="color: var(--status-green);">PASS</span>' : '<span style="color: var(--status-red);">FAIL</span>';
            } else if (typeof v === 'number') {
              disp = Number(v).toFixed(2);
            }
            items += `<div class="var-item"><span class="var-key">${k}</span><span class="var-value">${disp}</span></div>`;
          }
        }
        step2Html = `<div style="margin-top: 2rem; border-top: 2px dashed var(--border-color); padding-top: 1rem;">
          <h4 style="color: var(--status-orange); margin-bottom: 1rem; text-transform: uppercase;">Step 2: Fundamental Ruleset Diagnostics</h4>
          <div class="var-grid">${items}</div>
        </div>`;
      } else if (d.final_prediction === 'UP') {
        step2Html = `<div style="margin-top: 2rem; border-top: 2px dashed var(--border-color); padding-top: 1rem;">
          <h4 style="color: var(--status-orange); margin-bottom: 1rem; text-transform: uppercase;">Step 2: Fundamental Ruleset Diagnostics</h4>
          <div style="color: var(--status-red); margin-top: 0.5rem;">FAIL: Insufficient quarterly fundamental data to evaluate ruleset.</div>
        </div>`;
      }
      
      if (step1Html || step2Html) {
        html += `
          <details class="stock-variables">
            <summary>${stockName} ${d.company_name ? '- ' + d.company_name : ''}</summary>
            <div class="variables-content">
              ${step1Html}
              ${step2Html}
            </div>
          </details>
        `;
      }
    });
    
    if (!html) {
      html = '<div style="color: var(--text-muted);">No variables found for the given search.</div>';
    }
    
    container.innerHTML = html;
  }
});
