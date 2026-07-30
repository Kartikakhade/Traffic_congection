(function () {
  const bootstrap = window.APP_BOOTSTRAP || {};
  const state = {
    map: null,
    selectionMarker: null,
    junctionMarkers: [],
  };

  const elements = {
    statusBadge: document.getElementById("statusBadge"),
    predictionTitle: document.getElementById("predictionTitle"),
    predictionSummary: document.getElementById("predictionSummary"),
    junctionName: document.getElementById("junctionName"),
    junctionDescription: document.getElementById("junctionDescription"),
    junctionDistance: document.getElementById("junctionDistance"),
    liveDelay: document.getElementById("liveDelay"),
    liveDelayText: document.getElementById("liveDelayText"),
    liveTime: document.getElementById("liveTime"),
    liveTimeText: document.getElementById("liveTimeText"),
    forecastTime: document.getElementById("forecastTime"),
    forecastWindowText: document.getElementById("forecastWindowText"),
    predictedVehicles: document.getElementById("predictedVehicles"),
    latestObservedVehicles: document.getElementById("latestObservedVehicles"),
    latestObservedTime: document.getElementById("latestObservedTime"),
    dataWindowText: document.getElementById("dataWindowText"),
    sourceNote: document.getElementById("sourceNote"),
    trendChart: document.getElementById("trendChart"),
  };

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function setLoadingState() {
    elements.statusBadge.className = "status-badge";
    elements.statusBadge.textContent = "Loading";
    elements.predictionTitle.textContent = "Checking live traffic";
    elements.predictionSummary.textContent = "Fetching TomTom live flow data for the clicked point.";
  }

  function clearForecastFields() {
    elements.forecastTime.textContent = "-";
    elements.forecastWindowText.textContent = "-";
    elements.predictedVehicles.textContent = "-";
    elements.latestObservedVehicles.textContent = "-";
    elements.latestObservedTime.textContent = "-";
    elements.trendChart.innerHTML = "";
  }

  async function requestPrediction(lat, lng) {
    setLoadingState();
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lng }),
    });
    if (!response.ok) {
      throw new Error("Prediction request failed.");
    }
    return response.json();
  }

  function updateStatusBadge(tone, label) {
    elements.statusBadge.className = `status-badge ${tone}`;
    elements.statusBadge.textContent = label;
  }

  function updateUi(payload) {
    const liveTraffic = payload.liveTraffic || {};

    if (liveTraffic.available) {
      updateStatusBadge(liveTraffic.tone, liveTraffic.label);
      elements.predictionTitle.textContent = liveTraffic.label;
      elements.predictionSummary.textContent = `${liveTraffic.summary} ${liveTraffic.note || ""}`.trim();
      elements.liveDelay.textContent = `${liveTraffic.delayPercent}%`;
      elements.liveDelayText.textContent = `Current speed ${liveTraffic.currentSpeed} km/h vs free-flow ${liveTraffic.freeFlowSpeed} km/h.`;
      elements.liveTime.textContent = liveTraffic.checkedAt;
      elements.liveTimeText.textContent = `Confidence ${liveTraffic.confidence}. Road closure: ${liveTraffic.roadClosure ? "Yes" : "No"}.`;
    } else {
      updateStatusBadge("medium", "Live traffic unavailable");
      elements.predictionTitle.textContent = "Live TomTom traffic is not ready";
      elements.predictionSummary.textContent = liveTraffic.summary || "Live traffic is unavailable at the moment.";
      elements.liveDelay.textContent = "Unavailable";
      elements.liveDelayText.textContent = "Set TOMTOM_API_KEY to enable live TomTom traffic.";
      elements.liveTime.textContent = "-";
      elements.liveTimeText.textContent = liveTraffic.source || "TomTom Traffic API";
    }

    elements.junctionName.textContent = payload.junction.name;
    elements.junctionDescription.textContent = payload.junction.description || "Nearest configured junction.";
    elements.junctionDistance.textContent = `${payload.junction.distanceKm.toFixed(3)} km`;

    if (payload.forecastReady) {
      const forecast = payload.forecast || {};
      elements.forecastTime.textContent = forecast.forecastTimestamp || "-";
      elements.forecastWindowText.textContent = forecast.forecastWindowEndTimestamp
        ? `Forecast window ends at ${forecast.forecastWindowEndTimestamp}`
        : "Forecast window unavailable.";
      elements.predictedVehicles.textContent = forecast.predictedVehicles !== undefined
        ? `${forecast.predictedVehicles} vehicles`
        : "-";
      elements.latestObservedVehicles.textContent = forecast.latestObservedVehicles !== undefined
        ? `${forecast.latestObservedVehicles} vehicles`
        : "-";
      elements.latestObservedTime.textContent = forecast.latestObservedTimestamp
        ? `Latest observed at ${forecast.latestObservedTimestamp}`
        : "-";
      renderTrendChart(payload.trend || { history: [], forecast: [] });
      highlightNearestJunction(payload.junction.id);
    } else {
      clearForecastFields();
      elements.forecastWindowText.textContent = payload.forecastMessage || "Forecast unavailable.";
    }
  }

  function renderTrendChart(trend) {
    const width = 620;
    const height = 240;
    const padding = { top: 28, right: 24, bottom: 34, left: 36 };
    const series = [
      ...trend.history.map((item) => item.value),
      ...trend.forecast.map((item) => item.value),
    ];
    if (!series.length) {
      elements.trendChart.innerHTML = "";
      return;
    }

    const minValue = Math.min(...series);
    const maxValue = Math.max(...series);
    const span = Math.max(maxValue - minValue, 1);
    const historyStep = trend.history.length > 1 ? (width - padding.left - padding.right) / (trend.history.length - 1) : 0;
    const forecastStep = trend.forecast.length > 1 ? (width - padding.left - padding.right) / (trend.forecast.length - 1) : 0;

    const toY = (value) => height - padding.bottom - ((value - minValue) / span) * (height - padding.top - padding.bottom);
    const historyPoints = trend.history
      .map((item, index) => `${padding.left + index * historyStep},${toY(item.value)}`)
      .join(" ");
    const forecastPoints = trend.forecast
      .map((item, index) => `${padding.left + index * forecastStep},${toY(item.value)}`)
      .join(" ");

    const midY = toY((minValue + maxValue) / 2.0);
    elements.trendChart.innerHTML = `
      <rect x="0" y="0" width="${width}" height="${height}" rx="22" fill="transparent"></rect>
      <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}" stroke="rgba(18,52,77,0.12)" />
      <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="rgba(18,52,77,0.12)" />
      <line x1="${padding.left}" y1="${midY}" x2="${width - padding.right}" y2="${midY}" stroke="rgba(18,52,77,0.08)" stroke-dasharray="6 6" />
      <polyline fill="none" stroke="#14807b" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" points="${historyPoints}"></polyline>
      <polyline fill="none" stroke="#df6d2d" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" points="${forecastPoints}"></polyline>
      <text x="${padding.left}" y="${padding.top - 8}" fill="#3b5c71" font-size="12">${escapeHtml(maxValue.toFixed(1))}</text>
      <text x="${padding.left}" y="${height - 10}" fill="#3b5c71" font-size="12">${escapeHtml(minValue.toFixed(1))}</text>
    `;
  }

  function updateDataWindowText() {
    elements.dataWindowText.textContent = `Historical data available from ${bootstrap.dataWindow.historyStart} to ${bootstrap.dataWindow.historyEnd}.`;
    if (!bootstrap.predictionReady) {
      elements.sourceNote.textContent = "Forecasts are blocked until real junction coordinates are configured.";
      return;
    }
    if (bootstrap.liveTrafficEnabled) {
      elements.sourceNote.textContent = "Live TomTom traffic is enabled. The app reads traffic at the clicked point and shows the model forecast.";
      return;
    }
    elements.sourceNote.textContent = "Live TomTom traffic needs TOMTOM_API_KEY. Without it, only the model forecast is shown.";
  }

  function handleMapSelection(lat, lng) {
    placeSelectionMarker(lat, lng);
    requestPrediction(lat, lng)
      .then(updateUi)
      .catch(() => {
        elements.statusBadge.className = "status-badge high";
        elements.statusBadge.textContent = "Error";
        elements.predictionTitle.textContent = "Traffic could not be loaded";
        elements.predictionSummary.textContent = "Please retry the map selection or restart the app.";
      });
  }

  function createLeafletMarkerHtml(label, selected) {
    const className = selected ? "selected-pin" : "junction-chip";
    const content = selected ? "" : escapeHtml(label);
    return `<div class="${className}">${content}</div>`;
  }

  function highlightNearestJunction(junctionId) {
    state.junctionMarkers.forEach((markerBundle) => {
      const isSelected = markerBundle.id === String(junctionId);
      markerBundle.marker.setIcon(
        L.divIcon({
          className: "",
          html: createLeafletMarkerHtml(markerBundle.label, false),
          iconSize: [isSelected ? 120 : 92, 36],
          iconAnchor: [isSelected ? 60 : 46, 18],
        }),
      );
      markerBundle.marker.setZIndexOffset(isSelected ? 1000 : 0);
    });
  }

  function placeSelectionMarker(lat, lng) {
    if (state.selectionMarker) {
      state.selectionMarker.setLatLng([lat, lng]);
      return;
    }
    state.selectionMarker = L.marker([lat, lng], {
      icon: L.divIcon({
        className: "",
        html: createLeafletMarkerHtml("", true),
        iconSize: [20, 20],
        iconAnchor: [10, 20],
      }),
    }).addTo(state.map);
  }

  function initLeafletMap() {
    state.map = L.map("map", {
      zoomControl: true,
      scrollWheelZoom: true,
    }).setView([bootstrap.mapCenter.lat, bootstrap.mapCenter.lng], bootstrap.mapCenter.zoom || 13);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(state.map);

    bootstrap.junctions.forEach((junction) => {
      const marker = L.marker([junction.lat, junction.lng], {
        icon: L.divIcon({
          className: "",
          html: createLeafletMarkerHtml(junction.name, false),
          iconSize: [92, 36],
          iconAnchor: [46, 18],
        }),
      }).addTo(state.map);
      marker.bindTooltip(`${junction.name}<br>${junction.latestLabel}`, { direction: "top" });
      state.junctionMarkers.push({
        id: junction.id,
        label: junction.name,
        marker,
      });
    });

    state.map.on("click", (event) => {
      handleMapSelection(event.latlng.lat, event.latlng.lng);
    });
  }

  function initApp() {
    updateDataWindowText();
    initLeafletMap();
  }

  document.addEventListener("DOMContentLoaded", initApp);
})();
