import { FaSeedling, FaChartLine, FaCloudSun } from "react-icons/fa";

function Features() {
  const features = [
    {
      icon: <FaSeedling className="text-5xl text-green-600 mb-4" />,
      title: "Paddy, Maize & Wheat",
      desc: "Dedicated CNN-LSTM models trained per crop on historical yield records.",
    },
    {
      icon: <FaCloudSun className="text-5xl text-green-600 mb-4" />,
      title: "Climate + Soil Fusion",
      desc: "Blends monthly climate sequences with district soil properties for prediction.",
    },
    {
      icon: <FaChartLine className="text-5xl text-green-600 mb-4" />,
      title: "Confidence & Statistics",
      desc: "Every prediction ships with a confidence score and model performance metrics.",
    },
  ];

  return (
    <section className="py-20 bg-green-50">
      <div className="max-w-6xl mx-auto px-6">
        <h2 className="text-4xl font-bold text-center text-green-800">
          Why Choose Our System?
        </h2>

        <div className="grid md:grid-cols-3 gap-8 mt-12">
          {features.map((f) => (
            <div key={f.title} className="bg-white p-8 rounded-xl shadow">
              {f.icon}
              <h3 className="text-xl font-bold">{f.title}</h3>
              <p className="mt-2 text-gray-600">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default Features;
