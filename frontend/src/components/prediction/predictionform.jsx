import { useEffect, useState } from "react";
import { getCrops, getDistricts, predictYield } from "../../lib/api";

function PredictionForm() {
  const [crops, setCrops] = useState([]);
  const [districts, setDistricts] = useState([]);

  const [formData, setFormData] = useState({
    crop: "",
    district: "",
    landArea: "",
  });

  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [initError, setInitError] = useState("");
  const [initLoading, setInitLoading] = useState(true);

  useEffect(() => {
    Promise.all([getCrops(), getDistricts()])
      .then(([cropsData, districtsData]) => {
        setCrops(cropsData);
        setDistricts(districtsData);
      })
      .catch(() =>
        setInitError(
          "Can't reach the backend. Make sure it's running (uvicorn backend.app.main:app --reload --port 8000) and refresh this page."
        )
      )
      .finally(() => setInitLoading(false));
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setPrediction(null);
    setLoading(true);

    try {
      const result = await predictYield({
        crop: formData.crop,
        district: formData.district,
        landArea: formData.landArea,
      });
      setPrediction(result);
    } catch (err) {
      const msg =
        err.message === "Failed to fetch"
          ? "Can't reach the backend. Make sure it's running on http://localhost:8000."
          : err.message || "Prediction failed.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const confidenceColor = (pct) =>
    pct >= 80 ? "text-green-600" : pct >= 60 ? "text-amber-600" : "text-red-600";

  return (
    <section className="min-h-screen bg-green-50 py-20">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-green-700">
            Crop Yield Prediction
          </h1>
          <p className="text-gray-600 mt-2">
            Pick a crop, pick a district, and enter your land area in hectares.
            The CNN-LSTM model predicts the expected yield with a confidence
            score based on its validated performance.
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-8">
          {initLoading && (
            <p className="text-center text-gray-500 mb-6">Loading form options...</p>
          )}

          {initError && (
            <div className="mb-6 bg-red-100 text-red-700 rounded-lg p-4 text-sm">
              {initError}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Crop */}
            <div>
              <label className="block mb-2 font-semibold">Crop</label>
              <select
                name="crop"
                value={formData.crop}
                onChange={handleChange}
                className="w-full border rounded-lg p-3"
                required
              >
                <option value="">Select Crop</option>
                {crops.map((c) => (
                  <option key={c.code} value={c.code}>
                    {c.name}
                  </option>
                ))}
              </select>
            </div>

            {/* District */}
            <div>
              <label className="block mb-2 font-semibold">District</label>
              <select
                name="district"
                value={formData.district}
                onChange={handleChange}
                className="w-full border rounded-lg p-3 capitalize"
                required
              >
                <option value="">Select District</option>
                {districts.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>

            {/* Land Area */}
            <div>
              <label className="block mb-2 font-semibold">
                Land Area (Hectares)
              </label>
              <input
                type="number"
                name="landArea"
                value={formData.landArea}
                onChange={handleChange}
                placeholder="Example: 2.5"
                className="w-full border rounded-lg p-3"
                min="0"
                step="any"
                required
              />
            </div>

            <button
              type="submit"
              disabled={loading || initLoading || !!initError}
              className="w-full bg-green-600 text-white py-3 rounded-lg hover:bg-green-700 transition disabled:opacity-60"
            >
              {loading ? "Predicting..." : "Predict Yield"}
            </button>
          </form>

          {error && (
            <div className="mt-6 bg-red-100 text-red-700 rounded-lg p-4">
              {error}
            </div>
          )}
        </div>

        {prediction && (
          <div className="mt-8 bg-white rounded-xl shadow-lg p-8 border-t-4 border-green-600">
            <h2 className="text-2xl font-bold text-green-700 mb-6 text-center">
              Prediction Result
            </h2>

            <div className="grid grid-cols-2 gap-4 mb-6 text-center">
              <div className="bg-green-50 rounded-lg p-4">
                <p className="text-gray-500 text-sm">Crop</p>
                <p className="font-semibold">{prediction.crop_name}</p>
              </div>
              <div className="bg-green-50 rounded-lg p-4">
                <p className="text-gray-500 text-sm">District</p>
                <p className="font-semibold capitalize">{prediction.district}</p>
              </div>
            </div>

            <div className="bg-green-600 text-white rounded-lg p-6 text-center">
              <p className="text-lg font-medium">Predicted Yield</p>
              <p className="text-4xl font-bold mt-1">
                {prediction.predicted_yield_mt_per_ha} mt/ha
              </p>
              <p className="text-green-100 text-sm mt-1">
                for climate year {prediction.year}
              </p>
            </div>

            <div className="mt-4 rounded-lg p-5 text-center border">
              <p className="text-lg font-semibold">Confidence</p>
              <p className={`text-4xl font-bold mt-1 ${confidenceColor(prediction.confidence_pct)}`}>
                {prediction.confidence_pct}%
              </p>
              <p className="text-gray-500 text-sm mt-1">
                typical error ± {prediction.error_margin_mt_per_ha} mt/ha
              </p>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-4 text-center">
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-gray-500 text-sm">Land Area</p>
                <p className="font-semibold text-lg">
                  {prediction.land_area_ha} ha
                </p>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-gray-500 text-sm">Total Production</p>
                <p className="font-semibold text-lg">
                  {prediction.predicted_total_yield_mt} mt
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

export default PredictionForm;