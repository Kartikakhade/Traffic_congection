(function () {
  const bootstrap = window.APP_BOOTSTRAP || {};
  const CARTO_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJhIjoiYWNfOGw5ZncxcDMiLCJqdGkiOiJhNDkzODU4MjYwYjM5YmNiZjZjOTlhNDkxMGNhMjRkMSJ9.8NQXxjpc0G6igy9zvf8d3mFF4sydv2btSvjh2tUhvYw";
  const CARTO_USER = "ac_8l9fw1p3";

  // Tile layer definitions (authenticated CARTO + fallbacks)
  const TILE_LAYERS = {
    dark: {
      label: "Dark Matter",
      icon: "🌑",
      url: `https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png?api_key=${CARTO_API_KEY}`,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 20,
    },
    voyager: {
      label: "Street Light",
      icon: "🗺️",
      url: `https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png?api_key=${CARTO_API_KEY}`,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 20,
    },
    satellite: {
      label: "Satellite",
      icon: "🛰️",
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      attribution: "Tiles &copy; Esri — Source: Esri, USGS, NOAA",
      subdomains: "",
      maxZoom: 19,
    },
  };

  const state = {
    map: null,
    selectionMarker: null,
    corridorMarkers: [],
    searchTimeout: null,
    currentTileLayer: null,
    currentStyle: "dark",
  };

  const elements = {
    statusBadge: document.getElementById("statusBadge"),
    predictionTitle: document.getElementById("predictionTitle"),
    predictionSummary: document.getElementById("predictionSummary"),
    roadClassBadge: document.getElementById("roadClassBadge"),
    lanesBadge: document.getElementById("lanesBadge"),
    speedLimitBadge: document.getElementById("speedLimitBadge"),
    weatherBadge: document.getElementById("weatherBadge"),
    liveCurrentSpeed: document.getElementById("liveCurrentSpeed"),
    liveFreeFlowSpeed: document.getElementById("liveFreeFlowSpeed"),
    liveCongestionIndex: document.getElementById("liveCongestionIndex"),
    liveDelayPercent: document.getElementById("liveDelayPercent"),
    liveSpeedText: document.getElementById("liveSpeedText"),
    delayText: document.getElementById("delayText"),
    fcTime15: document.getElementById("fcTime15"),
    fcCi15: document.getElementById("fcCi15"),
    fcPill15: document.getElementById("fcPill15"),
    fcSpeed15: document.getElementById("fcSpeed15"),
    fcTime30: document.getElementById("fcTime30"),
    fcCi30: document.getElementById("fcCi30"),
    fcPill30: document.getElementById("fcPill30"),
    fcSpeed30: document.getElementById("fcSpeed30"),
    fcTime60: document.getElementById("fcTime60"),
    fcCi60: document.getElementById("fcCi60"),
    fcPill60: document.getElementById("fcPill60"),
    fcSpeed60: document.getElementById("fcSpeed60"),
    nearestAnchorText: document.getElementById("nearestAnchorText"),
    h3IndexText: document.getElementById("h3IndexText"),
    activeCoordinates: document.getElementById("activeCoordinates"),
    locationSearchInput: document.getElementById("locationSearchInput"),
    searchResultsDropdown: document.getElementById("searchResultsDropdown"),
    clearSearchBtn: document.getElementById("clearSearchBtn"),
    cityChips: document.getElementById("cityChips"),
  };

  function createCustomIcon(tone = "primary") {
    const color = tone === "high" ? "#ef4444" : tone === "medium" ? "#f59e0b" : tone === "low" ? "#10b981" : "#38bdf8";
    return L.divIcon({
      className: "custom-pin",
      html: `<div style="
        width: 14px;
        height: 14px;
        background: ${color};
        border: 2px solid #ffffff;
        border-radius: 50%;
        box-shadow: 0 0 10px ${color};
      "></div>`,
      iconSize: [16, 16],
      iconAnchor: [8, 8],
    });
  }

  function setLoadingState(lat, lng) {
    elements.statusBadge.className = "status-badge";
    elements.statusBadge.textContent = "Analyzing";
    elements.predictionTitle.textContent = "Querying Live Traffic AI...";
    elements.predictionSummary.textContent = `Extracting road attributes and calling live flow telemetry at (${lat.toFixed(4)}, ${lng.toFixed(4)})...`;
    elements.activeCoordinates.textContent = `Coordinates: ${lat.toFixed(4)}, ${lng.toFixed(4)} • Processing`;
  }

  async function requestPrediction(lat, lng) {
    setLoadingState(lat, lng);
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lng }),
    });
    if (!response.ok) {
      throw new Error("Prediction API call failed.");
    }
    return response.json();
  }

  function updateUi(payload) {
    const live = payload.liveTraffic || {};
    const road = payload.road || {};
    const wx = payload.weather || {};
    const anchor = payload.nearestAnchor || {};
    const forecasts = payload.forecasts || [];

    // Tone & Status Badge
    const tone = payload.overallTone || (live.tone || "low");
    elements.statusBadge.className = `status-badge ${tone}`;
    elements.statusBadge.textContent = live.available ? live.label : "Estimated Flow";

    // Title & Summary
    elements.predictionTitle.textContent = `${road.roadClass || "Road"} • ${anchor.city || "India"}`;
    elements.predictionSummary.textContent = live.available
      ? `${live.summary} Observed on ${road.roadClass.toLowerCase()} road (${road.lanes || 2} lanes).`
      : `Live telemetry fallback: Extrapolating typical congestion profile for ${road.roadClass.toLowerCase()} road.`;

    // Road Attributes Badges
    elements.roadClassBadge.textContent = `ROAD: ${road.roadClass || "PRIMARY"}`;
    elements.lanesBadge.textContent = `LANES: ${road.lanes || 2}`;
    elements.speedLimitBadge.textContent = `LIMIT: ${road.speedLimitKmh || 50} KM/H`;
    elements.weatherBadge.textContent = `WEATHER: ${wx.temperatureC || 26}°C • ${wx.summary || "Clear"}`;

    // Live Metrics
    if (live.available) {
      elements.liveCurrentSpeed.textContent = `${live.currentSpeed} km/h`;
      elements.liveFreeFlowSpeed.textContent = `${live.freeFlowSpeed} km/h`;
      elements.liveCongestionIndex.textContent = `${live.congestionIndex}`;
      elements.liveDelayPercent.textContent = `+${live.delayPercent}%`;
      elements.liveSpeedText.textContent = `Live TomTom flow at ${live.checkedAt.split(" ")[1]}`;
      elements.delayText.textContent = `${live.currentTravelTimeSeconds}s travel vs ${live.freeFlowTravelTimeSeconds}s normal`;
    } else {
      elements.liveCurrentSpeed.textContent = "-- km/h";
      elements.liveFreeFlowSpeed.textContent = `${road.speedLimitKmh || 50} km/h`;
      elements.liveCongestionIndex.textContent = "0.200";
      elements.liveDelayPercent.textContent = "0%";
      elements.liveSpeedText.textContent = "TomTom API rate limit or key unavailable";
      elements.delayText.textContent = "Normal typical baseline";
    }

    // Forecast Cards
    if (forecasts.length >= 3) {
      const f15 = forecasts[0];
      elements.fcTime15.textContent = f15.targetTime;
      elements.fcCi15.textContent = `${f15.congestionIndex}`;
      elements.fcPill15.className = `fc-pill ${f15.tone}`;
      elements.fcPill15.textContent = f15.tone.toUpperCase();
      elements.fcSpeed15.textContent = `Est. Speed: ${f15.estimatedSpeedKmh} km/h`;

      const f30 = forecasts[1];
      elements.fcTime30.textContent = f30.targetTime;
      elements.fcCi30.textContent = `${f30.congestionIndex}`;
      elements.fcPill30.className = `fc-pill ${f30.tone}`;
      elements.fcPill30.textContent = f30.tone.toUpperCase();
      elements.fcSpeed30.textContent = `Est. Speed: ${f30.estimatedSpeedKmh} km/h`;

      const f60 = forecasts[2];
      elements.fcTime60.textContent = f60.targetTime;
      elements.fcCi60.textContent = `${f60.congestionIndex}`;
      elements.fcPill60.className = `fc-pill ${f60.tone}`;
      elements.fcPill60.textContent = f60.tone.toUpperCase();
      elements.fcSpeed60.textContent = `Est. Speed: ${f60.estimatedSpeedKmh} km/h`;
    }

    // Anchor & H3 Hex
    elements.nearestAnchorText.textContent = `${anchor.corridorName || "Corridor"}, ${anchor.city || "City"} (${anchor.distanceKm || 0} km away)`;
    elements.h3IndexText.textContent = payload.h3Index || "N/A";
    elements.activeCoordinates.textContent = `Coordinates: ${payload.point.lat}, ${payload.point.lng} • Checked at ${payload.checkedAt.split(" ")[1]}`;
  }

  function handleMapClick(lat, lng) {
    if (state.selectionMarker) {
      state.map.removeLayer(state.selectionMarker);
    }
    state.selectionMarker = L.marker([lat, lng], {
      icon: createCustomIcon("primary"),
    }).addTo(state.map);

    requestPrediction(lat, lng)
      .then(updateUi)
      .catch((err) => {
        elements.predictionTitle.textContent = "Error Querying Point";
        elements.predictionSummary.textContent = err.message || "Failed to fetch traffic metrics.";
        elements.statusBadge.className = "status-badge high";
        elements.statusBadge.textContent = "Error";
      });
  }

  function switchTileLayer(styleKey) {
    if (!TILE_LAYERS[styleKey]) return;
    if (state.currentTileLayer) {
      state.map.removeLayer(state.currentTileLayer);
    }
    const cfg = TILE_LAYERS[styleKey];
    state.currentTileLayer = L.tileLayer(cfg.url, {
      attribution: cfg.attribution,
      subdomains: cfg.subdomains || "abcd",
      maxZoom: cfg.maxZoom || 19,
    }).addTo(state.map);
    state.currentStyle = styleKey;

    // Update switcher button states
    document.querySelectorAll(".tile-switcher-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-style") === styleKey);
    });
  }

  function initTileSwitcher() {
    const container = document.createElement("div");
    container.className = "tile-switcher";
    container.innerHTML = Object.entries(TILE_LAYERS)
      .map(
        ([key, cfg]) =>
          `<button class="tile-switcher-btn${key === state.currentStyle ? " active" : ""}" data-style="${key}" title="${cfg.label}">${cfg.icon} ${cfg.label}</button>`
      )
      .join("");
    container.querySelectorAll(".tile-switcher-btn").forEach((btn) => {
      btn.addEventListener("click", () => switchTileLayer(btn.getAttribute("data-style")));
    });
    document.getElementById("map").appendChild(container);
  }

  function initMap() {
    const center = bootstrap.mapCenter || { lat: 20.5937, lng: 78.9629, zoom: 5 };
    state.map = L.map("map", {
      zoomControl: true,
      minZoom: 4,
      maxZoom: 20,
    }).setView([center.lat, center.lng], center.zoom);

    // Authenticated CARTO Dark Matter tiles
    switchTileLayer("dark");
    initTileSwitcher();

    // Render 30 Indian monitored corridor anchor markers
    const corridors = bootstrap.corridors || [];
    corridors.forEach((c) => {
      const marker = L.circleMarker([c.lat, c.lng], {
        radius: 6,
        fillColor: "#38bdf8",
        color: "#ffffff",
        weight: 1.5,
        opacity: 0.9,
        fillOpacity: 0.8,
      }).addTo(state.map);

      marker.bindTooltip(`<strong>${c.city.toUpperCase()}:</strong> ${c.corridor_name}<br><small>${c.road_class.toUpperCase()} ROAD</small>`, {
        direction: "top",
        offset: [0, -5],
      });

      marker.on("click", (e) => {
        L.DomEvent.stopPropagation(e);
        state.map.setView([c.lat, c.lng], 14, { animate: true });
        handleMapClick(c.lat, c.lng);
      });

      state.corridorMarkers.push(marker);
    });

    state.map.on("click", (e) => {
      handleMapClick(e.latlng.lat, e.latlng.lng);
    });

    // Auto-select Silk Board on load if available
    const silkBoard = corridors.find((c) => c.corridor_name.includes("Silk Board"));
    if (silkBoard) {
      handleMapClick(silkBoard.lat, silkBoard.lng);
    }
  }

  function initCityChips() {
    const chips = elements.cityChips.querySelectorAll(".city-chip");
    const cityCoords = {
      all: { lat: 20.5937, lng: 78.9629, zoom: 5 },
      "Delhi-NCR": { lat: 28.6139, lng: 77.2090, zoom: 12 },
      Mumbai: { lat: 19.0760, lng: 72.8777, zoom: 12 },
      Bengaluru: { lat: 12.9716, lng: 77.5946, zoom: 12 },
      Hyderabad: { lat: 17.3850, lng: 78.4867, zoom: 12 },
      Chennai: { lat: 13.0827, lng: 80.2707, zoom: 12 },
      Kolkata: { lat: 22.5726, lng: 88.3639, zoom: 12 },
      Pune: { lat: 18.5204, lng: 73.8567, zoom: 12 },
      Ahmedabad: { lat: 23.0225, lng: 72.5714, zoom: 12 },
      Jaipur: { lat: 26.9124, lng: 75.7873, zoom: 12 },
      Lucknow: { lat: 26.8467, lng: 80.9462, zoom: 12 },
      Kochi: { lat: 9.9312, lng: 76.2673, zoom: 12 },
    };

    chips.forEach((chip) => {
      chip.addEventListener("click", () => {
        chips.forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");

        const cityName = chip.getAttribute("data-city");
        const target = cityCoords[cityName];
        if (target && state.map) {
          state.map.flyTo([target.lat, target.lng], target.zoom, { duration: 1.2 });
        }
      });
    });
  }

  function initSearchBar() {
    const input = elements.locationSearchInput;
    const dropdown = elements.searchResultsDropdown;
    const clearBtn = elements.clearSearchBtn;

    input.addEventListener("input", () => {
      const q = input.value.trim();
      clearBtn.style.display = q ? "block" : "none";

      if (state.searchTimeout) {
        clearTimeout(state.searchTimeout);
      }

      if (q.length < 2) {
        dropdown.style.display = "none";
        dropdown.innerHTML = "";
        return;
      }

      state.searchTimeout = setTimeout(async () => {
        try {
          const resp = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
          const results = await resp.json();
          if (results.length === 0) {
            dropdown.innerHTML = `<div class="search-result-item"><span>No matching places found in India</span></div>`;
            dropdown.style.display = "block";
            return;
          }

          dropdown.innerHTML = results
            .map(
              (r) => `
            <div class="search-result-item" data-lat="${r.lat}" data-lng="${r.lng}">
              <strong>${r.name}</strong>
              <span>${r.displayName}</span>
            </div>`
            )
            .join("");
          dropdown.style.display = "block";

          dropdown.querySelectorAll(".search-result-item").forEach((item) => {
            item.addEventListener("click", () => {
              const lat = parseFloat(item.getAttribute("data-lat"));
              const lng = parseFloat(item.getAttribute("data-lng"));
              dropdown.style.display = "none";
              input.value = item.querySelector("strong").textContent;
              state.map.flyTo([lat, lng], 15, { duration: 1.2 });
              handleMapClick(lat, lng);
            });
          });
        } catch {
          dropdown.style.display = "none";
        }
      }, 300);
    });

    clearBtn.addEventListener("click", () => {
      input.value = "";
      clearBtn.style.display = "none";
      dropdown.style.display = "none";
    });

    document.addEventListener("click", (e) => {
      if (!e.target.closest(".search-bar-container")) {
        dropdown.style.display = "none";
      }
    });
  }

  // Initialization
  document.addEventListener("DOMContentLoaded", () => {
    initMap();
    initCityChips();
    initSearchBar();
  });
})();
