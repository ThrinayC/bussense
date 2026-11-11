/**
 * BusSense Incident & VRU Intelligence Dashboard Script
 */

const API_BASE = "http://127.0.0.1:8000";

// Karlsruhe / KITTI Drive Coordinates
const DEFAULT_LAT = 49.0146;
const DEFAULT_LON = 8.4322;
const DEFAULT_ZOOM = 16;

let map;
let geoJsonLayer;

// Embedded Fallback GeoJSON if API is offline
const fallbackGeoJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "segment_id": "SEG_49.01460_8.43220",
                "center_lat": 49.0146,
                "center_lon": 8.4322,
                "total_observations": 12,
                "crossings": 4,
                "conflicts": 2,
                "near_misses": 2,
                "bus_passes": 1,
                "vai_score": 18.0,
                "risk_level": "HIGH"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[8.43206, 49.01451], [8.43234, 49.01451], [8.43234, 49.01469], [8.43206, 49.01469], [8.43206, 49.01451]]
                ]
            }
        },
        {
            "type": "Feature",
            "properties": {
                "segment_id": "SEG_49.01480_8.43200",
                "center_lat": 49.0148,
                "center_lon": 8.4320,
                "total_observations": 5,
                "crossings": 2,
                "conflicts": 1,
                "near_misses": 0,
                "bus_passes": 1,
                "vai_score": 4.0,
                "risk_level": "LOW"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[8.43186, 49.01471], [8.43214, 49.01471], [8.43214, 49.01489], [8.43186, 49.01489], [8.43186, 49.01471]]
                ]
            }
        },
        {
            "type": "Feature",
            "properties": {
                "segment_id": "SEG_49.01500_8.43180",
                "center_lat": 49.0150,
                "center_lon": 8.4318,
                "total_observations": 8,
                "crossings": 3,
                "conflicts": 2,
                "near_misses": 0,
                "bus_passes": 1,
                "vai_score": 7.0,
                "risk_level": "MEDIUM"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[8.43166, 49.01491], [8.43194, 49.01491], [8.43194, 49.01509], [8.43166, 49.01509], [8.43166, 49.01491]]
                ]
            }
        },
        {
            "type": "Feature",
            "properties": {
                "segment_id": "SEG_49.01440_8.43240",
                "center_lat": 49.0144,
                "center_lon": 8.4324,
                "total_observations": 3,
                "crossings": 1,
                "conflicts": 0,
                "near_misses": 0,
                "bus_passes": 1,
                "vai_score": 1.0,
                "risk_level": "LOW"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[8.43226, 49.01431], [8.43254, 49.01431], [8.43254, 49.01449], [8.43226, 49.01449], [8.43226, 49.01431]]
                ]
            }
        }
    ]
};

// Fallback Incident Events
const fallbackEvents = [
    {
        "event_id": "EVT_0076_0002_HARD_BRAKING",
        "event_type": "HARD_BRAKING",
        "state": "UNVERIFIED",
        "timestamp": 7.6,
        "frame_id": 76,
        "metrics": {
            "distance_m": 8.42,
            "closing_rate_mps": 4.15,
            "relative_speed_mps": -4.15,
            "ttc_kinematic_s": 2.03
        },
        "uncertainty": { "range_qualified": true }
    },
    {
        "event_id": "EVT_0075_0001_HARD_BRAKING",
        "event_type": "HARD_BRAKING",
        "state": "UNVERIFIED",
        "timestamp": 7.5,
        "frame_id": 75,
        "metrics": {
            "distance_m": 8.84,
            "closing_rate_mps": 3.82,
            "relative_speed_mps": -3.82,
            "ttc_kinematic_s": 2.31
        },
        "uncertainty": { "range_qualified": true }
    },
    {
        "event_id": "EVT_0042_0003_RAPID_CLOSING",
        "event_type": "RAPID_CLOSING",
        "state": "UNVERIFIED",
        "timestamp": 4.2,
        "frame_id": 42,
        "metrics": {
            "distance_m": 12.50,
            "closing_rate_mps": 4.80,
            "relative_speed_mps": -4.80,
            "ttc_kinematic_s": 2.60
        },
        "uncertainty": { "range_qualified": true }
    },
    {
        "event_id": "EVT_0058_0004_CUT_IN_CANDIDATE",
        "event_type": "CUT_IN_CANDIDATE",
        "state": "UNVERIFIED",
        "timestamp": 5.8,
        "frame_id": 58,
        "metrics": {
            "distance_m": 9.20,
            "closing_rate_mps": 1.40,
            "relative_speed_mps": -1.40,
            "ttc_kinematic_s": null
        },
        "uncertainty": { "range_qualified": true }
    }
];

function initMap() {
    map = L.map('map', {
        center: [DEFAULT_LAT, DEFAULT_LON],
        zoom: DEFAULT_ZOOM,
        zoomControl: true
    });

    // DarkMatter tiles (CartoDB)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap',
        maxZoom: 19
    }).addTo(map);

    loadGeoJSON(fallbackGeoJSON);
}

function getFeatureColor(vai) {
    if (vai >= 15.0) return '#da3633';
    if (vai >= 5.0) return '#d29922';
    return '#58a6ff';
}

function loadGeoJSON(geojson) {
    if (geoJsonLayer) {
        map.removeLayer(geoJsonLayer);
    }

    geoJsonLayer = L.geoJSON(geojson, {
        style: function (feature) {
            const vai = feature.properties.vai_score || 0;
            const col = getFeatureColor(vai);
            return {
                color: col,
                weight: 2,
                opacity: 0.9,
                fillColor: col,
                fillOpacity: 0.45
            };
        },
        onEachFeature: function (feature, layer) {
            const p = feature.properties;
            const popupContent = `
                <div style="font-family: 'Outfit', sans-serif; font-size: 13px; color: #161b22; line-height: 1.4;">
                    <b style="font-size: 14px;">${p.segment_id}</b><br/>
                    <b>VAI Score:</b> <span style="color: ${getFeatureColor(p.vai_score)}; font-weight: bold;">${p.vai_score}</span> (${p.risk_level})<br/>
                    <b>Crossings:</b> ${p.crossings}<br/>
                    <b>Conflicts:</b> ${p.conflicts}<br/>
                    <b>Near Misses:</b> ${p.near_misses}<br/>
                    <b>Observations:</b> ${p.total_observations}
                </div>
            `;
            layer.bindPopup(popupContent);
        }
    }).addTo(map);

    renderVRUTable(geojson.features);
    document.getElementById('vru-count-chip').textContent = `${geojson.features.length} Segments`;
}

function renderVRUTable(features) {
    const tbody = document.querySelector('#vru-table tbody');
    tbody.innerHTML = '';

    features.forEach(f => {
        const p = f.properties;
        const row = document.createElement('tr');
        const col = getFeatureColor(p.vai_score);
        row.innerHTML = `
            <td>${p.segment_id}</td>
            <td>${p.crossings}</td>
            <td>${p.conflicts}</td>
            <td>${p.near_misses}</td>
            <td style="color: ${col}; font-weight: bold;">${p.vai_score}</td>
            <td><span class="badge" style="background: ${col}22; color: ${col}; border-color: ${col}44;">${p.risk_level}</span></td>
        `;
        row.addEventListener('click', () => {
            map.flyTo([p.center_lat, p.center_lon], 18);
        });
        tbody.appendChild(row);
    });
}

function renderEventFeed(events) {
    const feed = document.getElementById('event-feed');
    feed.innerHTML = '';

    events.forEach(ev => {
        const card = document.createElement('div');
        card.className = 'event-card';

        const dist = ev.metrics.distance_m ? `${ev.metrics.distance_m.toFixed(1)}m` : 'N/A';
        const closing = ev.metrics.closing_rate_mps ? `${ev.metrics.closing_rate_mps.toFixed(1)} m/s` : 'N/A';
        const ttc = ev.metrics.ttc_kinematic_s ? `${ev.metrics.ttc_kinematic_s.toFixed(2)}s` : 'N/A';

        card.innerHTML = `
            <div class="event-info">
                <div class="event-id-row">
                    <span class="event-id">${ev.event_id}</span>
                    <span class="event-type-badge">${ev.event_type}</span>
                </div>
                <div class="event-details">
                    Frame #${ev.frame_id} &bull; t = ${ev.timestamp.toFixed(2)}s &bull; Range ${dist}
                </div>
            </div>
            <div class="event-metrics">
                <span class="metric-highlight">TTC: ${ttc}</span>
                <span class="metric-sub">Closing: ${closing}</span>
            </div>
        `;
        feed.appendChild(card);
    });

    document.getElementById('evt-counter').textContent = `${events.length} Events`;
}

async function fetchTelemetry() {
    try {
        const healthRes = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(2000) });
        if (healthRes.ok) {
            document.getElementById('hdr-api-status').textContent = "CONNECTED (PORT 8000)";
            document.getElementById('hdr-api-status').className = "value status-ok";
            document.getElementById('sys-pulse').className = "pulse-indicator online";
        }

        const eventsRes = await fetch(`${API_BASE}/api/events?limit=20`);
        if (eventsRes.ok) {
            const data = await eventsRes.json();
            if (data.events && data.events.length > 0) {
                renderEventFeed(data.events);
            }
        }

        const areasRes = await fetch(`${API_BASE}/api/vru-areas`);
        if (areasRes.ok) {
            const geojson = await areasRes.json();
            if (geojson.features && geojson.features.length > 0) {
                loadGeoJSON(geojson);
            }
        }
    } catch (e) {
        // Fallback gracefully to offline simulation data
        document.getElementById('hdr-api-status').textContent = "OFFLINE LAB CACHE";
        document.getElementById('hdr-api-status').className = "value";
        renderEventFeed(fallbackEvents);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initMap();
    fetchTelemetry();

    document.getElementById('btn-recenter').addEventListener('click', () => {
        map.flyTo([DEFAULT_LAT, DEFAULT_LON], DEFAULT_ZOOM);
    });

    document.getElementById('btn-export-geojson').addEventListener('click', () => {
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(fallbackGeoJSON, null, 2));
        const dlAnchor = document.createElement('a');
        dlAnchor.setAttribute("href", dataStr);
        dlAnchor.setAttribute("download", "vulnerable_areas.geojson");
        dlAnchor.click();
    });

    // Poll every 5s if server is running
    setInterval(fetchTelemetry, 5000);
});
