import { Link } from "react-router-dom";
import { FaSeedling, FaCloudSun, FaChartLine, FaBrain } from "react-icons/fa";

function About() {
  return (
    <section className="min-h-screen bg-green-50 py-20">
      <div className="max-w-4xl mx-auto px-6">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-green-700">About This Project</h1>
          <p className="text-gray-600 mt-2">
            Machine-learning based crop yield prediction for Nepal using a
            dual-branch CNN-LSTM model.
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-green-700 mb-4">The Model</h2>
          <p className="text-gray-700 leading-relaxed">
            Each crop (Paddy, Maize, Wheat) has its own trained
            <b> 1D CNN + Bidirectional LSTM </b>
            model that fuses two inputs:
          </p>
          <ul className="mt-4 space-y-2 text-gray-700">
            <li className="flex gap-2">
              <FaCloudSun className="text-green-600 mt-1 shrink-0" />
              <span>
                <b>Climate sequence</b> — the 12 monthly records of rainfall,
                humidity, temperature (min/max/mean) and wind speed for the
                chosen district and year.
              </span>
            </li>
            <li className="flex gap-2">
              <FaSeedling className="text-green-600 mt-1 shrink-0" />
              <span>
                <b>Soil profile</b> — pH, organic matter, nitrogen, phosphorus,
                potassium, micronutrients, texture, and soil type of the
                district.
              </span>
            </li>
          </ul>
          <p className="text-gray-700 leading-relaxed mt-4">
            The CNN layers learn short-term weather patterns, the LSTM captures
            seasonal trends, and a dense regression head combines the climate
            and soil branches to estimate yield in metric tons per hectare
            (mt/ha).
          </p>
        </div>

        <div className="mt-8 grid md:grid-cols-3 gap-6">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <FaChartLine className="text-4xl text-green-600 mb-3" />
            <h3 className="text-lg font-bold">Prediction</h3>
            <p className="text-gray-600 mt-1 text-sm">
              Choose a crop, district, and year to get the predicted yield with
              a confidence score.
            </p>
            <Link to="/prediction" className="text-green-700 font-semibold text-sm hover:underline">
              Go to prediction →
            </Link>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <FaBrain className="text-4xl text-green-600 mb-3" />
            <h3 className="text-lg font-bold">Statistics</h3>
            <p className="text-gray-600 mt-1 text-sm">
              Review each model's R², RMSE, MAE, sample counts, and architecture.
            </p>
            <Link to="/statistics" className="text-green-700 font-semibold text-sm hover:underline">
              View statistics →
            </Link>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <FaSeedling className="text-4xl text-green-600 mb-3" />
            <h3 className="text-lg font-bold">Data</h3>
            <p className="text-gray-600 mt-1 text-sm">
              Trained on 45+ years of district climate data, soil surveys, and
              annual crop yield records (1979–2024).
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default About;