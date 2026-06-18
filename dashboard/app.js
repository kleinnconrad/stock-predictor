document.addEventListener('DOMContentLoaded', () => {
  let reportData = [];
  let currentSort = { column: 'final_prediction', asc: true };

  const tableBody = document.getElementById('table-body');
  const searchInput = document.getElementById('search-input');
  const filterSelect = document.getElementById('filter-status');
  const ths = document.querySelectorAll('th[data-sort]');

  const kpiTotal = document.getElementById('kpi-total');
  const kpiBuys = document.getElementById('kpi-buys');
  const kpiAcc = document.getElementById('kpi-acc');

  // Determine correct data URL based on environment (local vs GitHub Pages)
  const isGitHubPages = window.location.hostname.includes('github.io');
  const dataUrl = isGitHubPages ? './data/full_batch_report.json' : '../data/processed/full_batch_report.json';

  // Fetch data
  fetch(dataUrl)
    .then(response => {
      if (!response.ok) throw new Error("Network response was not ok");
      return response.json();
    })
    .then(data => {
      reportData = data;
      calculateKPIs();
      renderTable();
    })
    .catch(error => {
      console.error("Failed to load JSON data", error);
      tableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--status-red);">Failed to load data. Ensure full_batch_report.json is available.</td></tr>`;
    });

  function calculateKPIs() {
    kpiTotal.textContent = reportData.length;
    const buys = reportData.filter(d => d.final_prediction === 'UP_FINAL_BUY').length;
    kpiBuys.textContent = buys;

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

  function renderTable() {
    const searchTerm = searchInput.value.toLowerCase();
    const filterStatus = filterSelect.value;

    let filtered = reportData.filter(d => {
      const matchSearch = d.stock_name.toLowerCase().includes(searchTerm);
      const matchStatus = filterStatus === 'ALL' || d.final_prediction === filterStatus;
      return matchSearch && matchStatus;
    });

    // Sorting
    filtered.sort((a, b) => {
      let valA, valB;
      const col = currentSort.column;
      
      if (col === 'stock_name') {
        valA = a.stock_name;
        valB = b.stock_name;
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
    });

    tableBody.innerHTML = '';

    if (filtered.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="5" style="text-align:center;">No stocks match your filters.</td></tr>`;
      return;
    }

    filtered.forEach(d => {
      const tr = document.createElement('tr');
      
      const acc = d.step1_model?.cv_accuracy ? (d.step1_model.cv_accuracy * 100).toFixed(1) : 0;
      const ks = d.step1_model?.ks_cutoff ? d.step1_model.ks_cutoff.toFixed(3) : 'N/A';
      const step1Class = d.step1_model?.predicted_class || 'N/A';

      tr.innerHTML = `
        <td class="ticker-name">${d.stock_name}</td>
        <td>${getStatusHtml(d.final_prediction)}</td>
        <td>${getStatusHtml(step1Class)}</td>
        <td>
          ${acc > 0 ? `${acc}% <div class="acc-bar-bg"><div class="acc-bar-fill" style="width: ${acc}%"></div></div>` : 'N/A'}
        </td>
        <td>${ks}</td>
      `;
      tableBody.appendChild(tr);
    });

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
  searchInput.addEventListener('input', renderTable);
  filterSelect.addEventListener('change', renderTable);

  ths.forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.sort;
      if (currentSort.column === col) {
        currentSort.asc = !currentSort.asc;
      } else {
        currentSort.column = col;
        currentSort.asc = true; // default asc for new column
      }
      renderTable();
    });
  });
});
