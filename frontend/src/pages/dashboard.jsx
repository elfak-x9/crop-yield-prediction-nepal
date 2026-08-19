import { useEffect, useState } from "react";
import { getStats, PLOT_URLS } from "../lib/api";

const metricBar = (value, max, color) => {
  const widthPct = Math.max(Math.min((value / max) * 100, 100), 4);
  return (
    <div className="w-full bg-gray-100 rounded-full h-3">
      <div
        className={`${color} h-3 rounded-full transition-all duration-500`}
        style={{ width: `${widthPct}%` }}
      />
    </div>
  );
};

function StatsPage() {
  const [stats, setStats] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(null);

  useEffect(() => {
    getStats()
      .then(setStats)
      .catch((e) =>
        setError(
          e.message === "Failed to fetch"
            ? "Can't reach the backend. Make sure it's running on http://localhost:8000."
            : e.message || "Failed to load model statistics."
        )
      )
      .finally(() => setLoading(false));
  }, []);

  const maxR2 = 1.0;
  const maxRmsle = Math.max(...stats.map((s) => s.rmse_mt_per_ha), 0.5) * 1.2;
  const maxMae = Math.max(...stats.map((s) => s.mae_mt_per_ha), 0.4) * 1.2;

  return (
    <section className="min-h-screen bg-green-50 py-20">
      <div className="max-w-5xl mx-auto px-6">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-green-700">Model Statistics</h1>
          <p className="text-gray-600 mt-2">
            Performance of the dual-branch 1D CNN-LSTM models measured on the
            held-out validation split (20% of each crop's dataset).
          </p>
        </div>

        {loading && (
          <p className="text-center text-gray-500">Loading model statistics...</p>
        )}

        {error && (
          <div className="bg-red-100 text-red-700 rounded-lg p-4 mb-6">{error}</div>
        )}

        {!loading && !error && (
          <>
            <div className="grid gap-6">
              {stats.map((s) => (
                <div key={s.crop} className="bg-white rounded-xl shadow-lg overflow-hidden">
                  <div className="flex items-center justify-between px-6 py-4 bg-green-600 text-white">
                    <div>
                      <h2 className="text-xl font-bold">{s.crop_name}</h2>
                      <span className="text-sm opacity-80">{s.crop}</span>
                    </div>
                    <button
                      onClick={() => setExpanded(expanded === s.crop ? null : s.crop)}
                      className="text-sm underline hover:text-green-200"
                    >
                      {expanded === s.crop ? "Hide architecture" : "View architecture"}
                    </button>
                  </div>

                  <div className="grid md:grid-cols-3 gap-4 p-6">
                    <div className="text-center">
                      <p className="text-3xl font-bold text-green-700">{s.r2.toFixed(3)}</p>
                      <p className="text-sm text-gray-500">R² Score</p>
                      <div className="mt-2">{metricBar(s.r2, maxR2, "bg-green-600")}</div>
                    </div>
                    <div className="text-center">
                      <p className="text-3xl font-bold text-red-600">{s.rmse_mt_per_ha} mt/ha</p>
                      <p className="text-sm text-gray-500">RMSE</p>
                      <div className="mt-2">{metricBar(s.rmse_mt_per_ha, maxRmsle, "bg-red-500")}</div>
                    </div>
                    <div className="text-center">
                      <p className="text-3xl font-bold text-amber-600">{s.mae_mt_per_ha} mt/ha</p>
                      <p className="text-sm text-gray-500">Mean Absolute Error</p>
                      <div className="mt-2">{metricBar(s.mae_mt_per_ha, maxMae, "bg-amber-500")}</div>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-6 px-6 pb-4 text-sm text-gray-600">
                    <span>Mean actual yield: <b>{s.mean_actual_mt_per_ha} mt/ha</b></span>
                    <span>Samples: <b>{s.n_samples.toLocaleString()}</b></span>
                    <span>Validation: <b>{s.n_validation}</b></span>
                    <span>Model parameters: <b>{s.model_parameters.toLocaleString()}</b></span>
                  </div>

                  {expanded === s.crop && (
                    <div className="px-6 pb-6">
                      <pre className="bg-gray-900 text-green-300 rounded-lg p-4 text-xs overflow-x-auto">
                        {s.architecture}
                      </pre>
                    </div>
                  )}
                </div>
              ))}
            </div>

            <h2 className="text-2xl font-bold text-green-700 mt-12 mb-4">
              Training Curves
            </h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white rounded-xl shadow-lg p-4">
                <p className="font-semibold text-center mb-2">Loss Curves</p>
                <img
                  src={PLOT_URLS.trainingHistory}
                  alt="Training history"
                  className="w-full rounded-lg"
                  loading="lazy"
                />
              </div>
              <div className="bg-white rounded-xl shadow-lg p-4">
                <p className="font-semibold text-center mb-2">Actual vs Predicted</p>
                <img
                  src={PLOT_URLS.actualVsPredicted}
                  alt="Actual vs predicted"
                  className="w-full rounded-lg"
                  loading="lazy"
                />
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  );
}

export default StatsPage;