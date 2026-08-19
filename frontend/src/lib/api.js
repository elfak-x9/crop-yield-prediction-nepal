const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON, keep statusText
    }
    throw new Error(detail);
  }

  return res.json();
}

export function getCrops() {
  return request("/crops");
}

export function getDistricts() {
  return request("/districts");
}

export function getYears(district) {
  return request(`/years/${encodeURIComponent(district)}`);
}

export function getStats() {
  return request("/stats");
}

export function predictYield({ crop, district, year, landArea }) {
  return request("/predict", {
    method: "POST",
    body: JSON.stringify({
      crop,
      district,
      year: year ? Number(year) : null,
      land_area: landArea ? Number(landArea) : null,
    }),
  });
}

export const PLOT_URLS = {
  trainingHistory: `${API_BASE_URL}/static/models/training_history.png`,
  actualVsPredicted: `${API_BASE_URL}/static/models/actual_vs_predicted.png`,
};
