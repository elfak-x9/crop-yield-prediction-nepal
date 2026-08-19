import { FaSeedling, FaChartLine, FaCloudSun } from "react-icons/fa";

function Features() {
  return (
    <section className="py-20 bg-green-50">
      <div className="max-w-6xl mx-auto px-6">
        <h2 className="text-4xl font-bold text-center text-green-800">
          Why Choose Our System?
        </h2>

        <div className="grid md:grid-cols-3 gap-8 mt-12">
          <div className="bg-white p-8 rounded-xl shadow">
            <FaSeedling className="text-5xl text-green-600 mb-4" />
            <h3 className="text-xl font-bold">Crop Prediction</h3>
            <p className="mt-2 text-gray-600">
              Predict crop yield using AI.
            </p>
          </div>

          <div className="bg-white p-8 rounded-xl shadow">
            <FaChartLine className="text-5xl text-green-600 mb-4" />
            <h3 className="text-xl font-bold">Analytics</h3>
            <p className="mt-2 text-gray-600">
              View prediction reports.
            </p>
          </div>

          <div className="bg-white p-8 rounded-xl shadow">
            <FaCloudSun className="text-5xl text-green-600 mb-4" />
            <h3 className="text-xl font-bold">Weather Data</h3>
            <p className="mt-2 text-gray-600">
              Uses climate information.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Features;
