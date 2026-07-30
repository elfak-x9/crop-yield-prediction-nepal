import { useState } from "react";
import { predictYield } from "../api/predictionApi";


function PredictionForm() {
  const [formData, setFormData] = useState({
    crop: "",
    location: "",
    landArea: "",
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

 const handleSubmit = async (e) => {
  e.preventDefault();

  try {
    const result = await predictYield({
      crop: formData.crop,
      location: formData.location,
      land_area: formData.landArea,
    });

    setPrediction(result);
  } catch (error) {
    console.error(error);
    alert("Failed to connect to Django backend.");
  }
};

  return (
    <section className="min-h-screen bg-green-50 py-20">
      <div className="max-w-2xl mx-auto bg-white rounded-xl shadow-lg p-8">

        <h1 className="text-3xl font-bold text-center text-green-700 mb-8">
          Crop Yield Prediction
        </h1>

        <form onSubmit={handleSubmit} className="space-y-6">

          {/* Crop */}
          <div>
            <label className="block mb-2 font-semibold">
              Crop Name
            </label>

            <select
              name="crop"
              value={formData.crop}
              onChange={handleChange}
              className="w-full border rounded-lg p-3"
              required
            >
              <option value="">Select Crop</option>
              <option value="Paddy">Paddy</option>
              <option value="Maize">Maize</option>
              <option value="Wheat">Wheat</option>
              <option value="Millet">Millet</option>
              <option value="Buckwheat">Buckwheat</option>
              <option value="Barlay">Barlay</option>
            </select>
          </div>

          {/* Location */}
          <div>
            <label className="block mb-2 font-semibold">
              Location
            </label>

            <input
              type="text"
              name="location"
              value={formData.location}
              onChange={handleChange}
              placeholder="Enter District or City"
              className="w-full border rounded-lg p-3"
              required
            />
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
              placeholder="Example: 2"
              className="w-full border rounded-lg p-3"
              required
            />
          </div>

          <button
            type="submit"
            className="w-full bg-green-600 text-white py-3 rounded-lg hover:bg-green-700 transition"
          >
            Predict Yield
          </button>

        </form>

        {prediction && (
          <div className="mt-8 border-t pt-6">
            <h2 className="text-2xl font-bold text-green-700 mb-4">
              Prediction Result
            </h2>

            <p>
              <strong>Crop:</strong> {prediction.crop}
            </p>

            <p>
              <strong>Location:</strong> {prediction.location}
            </p>

            <p>
              <strong>Land Area:</strong> {prediction.land_area} Hectares
            </p>

            <div className="mt-6 bg-green-100 rounded-lg p-6 text-center">
              <h3 className="text-lg font-semibold">
                Estimated Yield
              </h3>

              <p className="text-3xl font-bold text-green-700 mt-2">
                {prediction.predicted_yield}
              </p>
            </div>
          </div>
        )}

      </div>
    </section>
  );
}

export default PredictionForm;